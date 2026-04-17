"""Blender-side worker for VLM-guided object placement.

Runs inside Blender:
    blender -b scene.blend -P scripts/vlm_place_worker.py -- --mode <mode> [args]

Modes:
    discover_candidates  Find top-K flat surface positions via raycast grid.
    place_and_render     Place object at a given position, render EEVEE/Cycles preview.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Ensure src/ is importable when running inside Blender
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bpy
from mathutils import Vector

from src.assets import pick_first_file
from src.blender.objects import (
    auto_fix_orientation,
    auto_scale,
    fix_missing_textures,
    import_object,
    normalize_to_metric,
    parent_and_center,
    restore_hidden,
    set_hidden,
)
from src.blender.bbox import get_world_bbox, project_bbox_2d
from src.blender.collision import check_collisions
from src.blender.camera import (
    compute_view_coverage,
    find_valid_azimuths,
    find_valid_azimuths_with_elevation,
    setup_camera,
)
from src.blender.grounding import snap_to_floor, verify_grounded
from src.blender.gpu import configure_gpu
from src.blender.rendering import render_image
from src.blender.sky import set_nishita_sky
from src.blender.surfaces import (
    compute_scene_camera_view,
    find_candidates_multi_height,
    find_flat_candidates,
    get_camera_ground_target,
)

# ---------------------------------------------------------------------------
# Result output protocol
# ---------------------------------------------------------------------------
RESULT_PREFIX = "###RESULT###"


def apply_scene_scale(scene_scale: float, imported_names: set) -> None:
    """Scale scene geometry AND compensate light intensity.

    Non-imported, top-level objects get their position/scale multiplied by
    scene_scale. For point/area/spot lights (power in Watts), illuminance
    varies as 1/r², so we multiply data.energy by scene_scale² to keep the
    scene at roughly the same brightness after shrinking. Sun lights use
    directional irradiance (W/m²) and are left alone.
    """
    if abs(scene_scale - 1.0) < 0.01:
        return
    print(f"Applying scene_scale override: {scene_scale}")
    for obj in bpy.data.objects:
        if obj.name in imported_names:
            continue
        if obj.parent is None:
            obj.scale = tuple(s * scene_scale for s in obj.scale)
            obj.location = tuple(v * scene_scale for v in obj.location)
    # Compensate light intensity: point/area/spot → energy *= s²; sun unchanged.
    energy_factor = scene_scale * scene_scale
    n_adjusted = 0
    for light in bpy.data.lights:
        if light.type == "SUN":
            continue
        old = light.energy
        light.energy = old * energy_factor
        n_adjusted += 1
        print(f"  Light '{light.name}' ({light.type}): "
              f"{old:.1f}W -> {light.energy:.1f}W")
    if n_adjusted:
        print(f"  Adjusted {n_adjusted} non-sun light(s) by x{energy_factor:.3f}")
    bpy.context.view_layer.update()


def emit_result(data: dict) -> None:
    print(f"{RESULT_PREFIX}{json.dumps(data)}", flush=True)


def emit_error(msg: str) -> None:
    emit_result({"status": "error", "error": msg})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    idx = argv.index("--") + 1 if "--" in argv else len(argv)
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["discover_candidates", "place_and_render"])
    p.add_argument("--object_file", required=True)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--min_candidate_distance", type=float, default=2.0)
    # place_and_render specific
    p.add_argument("--position", type=float, nargs=3, default=None)
    p.add_argument("--rotation_x", type=float, default=0.0, help="Object X rotation in degrees")
    p.add_argument("--rotation_y", type=float, default=0.0, help="Object Y rotation in degrees")
    p.add_argument("--rotation_z", type=float, default=0.0, help="Object Z rotation in degrees")
    p.add_argument("--camera_radius", type=float, default=5.0)
    p.add_argument("--camera_elevation", type=float, default=30.0, help="Degrees above horizon")
    p.add_argument("--camera_azimuth", type=float, default=30.0, help="Degrees horizontal")
    p.add_argument("--output_image", type=str, default="/tmp/vlm_preview.png")
    p.add_argument("--scale", type=float, default=0.0,
                    help="Override object scale (0 = auto)")
    p.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="EEVEE")
    p.add_argument("--resolution", type=int, nargs=2, default=[512, 384])
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--sky_strength", type=float, default=0.3)
    p.add_argument("--camera_azimuths", type=float, nargs="*", default=None,
                   help="Multiple azimuth angles for multi-view rendering")
    p.add_argument("--gpu_backend", default="AUTO",
                   choices=["AUTO", "OPTIX", "CUDA", "CPU"])
    p.add_argument("--gpu_devices", type=int, nargs="*", default=None)
    p.add_argument("--scene_preview_path", type=str, default=None,
                   help="If set and the file exists, use it as the scene preview "
                        "instead of rendering a new one.")
    p.add_argument("--shot_types", nargs="*", default=None,
                   help="Shot types to render: full, wide, mid. Default: just full.")
    p.add_argument("--scene_scale", type=float, default=1.0,
                   help="Scale the scene by this factor (used for retry on invisible objects)")
    return p.parse_args(argv[idx:])


# ---------------------------------------------------------------------------
# Mode: discover_candidates
# ---------------------------------------------------------------------------


def mode_discover_candidates(args):
    obj_file = pick_first_file(Path(args.object_file))
    if obj_file is None:
        emit_error(f"Object not found: {args.object_file}")
        return

    imported = import_object(obj_file)
    if not imported:
        emit_error("No objects imported")
        return

    root = parent_and_center(imported)

    # Get scene meshes (exclude imported)
    imported_names = {o.name for o in imported} | {root.name}
    scene_meshes = [
        o for o in bpy.context.scene.objects
        if o.type == "MESH" and o.name not in imported_names
    ]

    # Object-anchored metric normalization
    auto_rot = auto_fix_orientation(root, imported)
    scene_correction = normalize_to_metric(
        root, imported, scene_meshes,
        object_path=args.object_file,
        exclude_names=imported_names,
    )
    # Apply scene scale override (used when retrying with different scale)
    if hasattr(args, 'scene_scale'):
        apply_scene_scale(args.scene_scale, set(imported_names))
    # Only apply VLM override scale if explicitly requested
    applied_scale = 1.0
    if hasattr(args, 'scale') and args.scale > 0 and abs(args.scale - 1.0) > 0.01:
        applied_scale = auto_scale(root, imported, scene_meshes,
                                   override_scale=args.scale)
    bpy.context.view_layer.update()

    bbox = get_world_bbox(imported)
    if bbox is None:
        emit_error("Could not compute object bbox")
        return
    obj_size = bbox[1] - bbox[0]

    # Hide imported for raycasting
    hidden = set_hidden(imported + [root], True)
    bpy.context.view_layer.update()

    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bounds = get_world_bbox(scene_meshes)
        if bounds is None:
            emit_error("No scene geometry found")
            return

        # Use scene camera to determine where objects should be placed
        cam_target, cam_z = get_camera_ground_target(
            bpy.context.scene, depsgraph
        )
        if cam_target:
            print(f"Camera ground target: [{cam_target.x:.2f}, {cam_target.y:.2f}, {cam_target.z:.2f}]")

        # Compute the scene camera's visible region (for visibility filtering
        # and centroid-based scoring inside find_flat_candidates).
        scene_cam_pos, visible_hits, visible_centroid = compute_scene_camera_view(
            bpy.context.scene, depsgraph, n_rays=14,
        )
        if scene_cam_pos is not None:
            print(f"Scene camera at [{scene_cam_pos.x:.2f}, {scene_cam_pos.y:.2f}, "
                  f"{scene_cam_pos.z:.2f}], {len(visible_hits)} visible hit points")
        if visible_centroid is not None:
            print(f"Visible centroid: [{visible_centroid.x:.2f}, "
                  f"{visible_centroid.y:.2f}, {visible_centroid.z:.2f}]")

        obj_seed = hash(args.object_file) & 0xFFFFFFFF

        candidates = find_flat_candidates(
            bpy.context.scene,
            depsgraph,
            bounds[0],
            bounds[1],
            obj_size,
            ignore_names=imported_names,
            top_k=args.top_k,
            min_distance=args.min_candidate_distance,
            camera_target=cam_target,
            object_seed=obj_seed,
            scene_camera_pos=scene_cam_pos,
            visible_centroid=visible_centroid,
        )

        # Check if all candidates are on ceiling — retry with floor-level target
        if candidates and all(c["position"][2] > bounds[0].z + 1.0 for c in candidates):
            print(f"All candidates above floor (z>{bounds[0].z + 1.0:.2f}), retrying from floor level...")
            floor_target = Vector(((bounds[0].x + bounds[1].x) / 2,
                                   (bounds[0].y + bounds[1].y) / 2,
                                   bounds[0].z))
            candidates = find_flat_candidates(
                bpy.context.scene, depsgraph, bounds[0], bounds[1], obj_size,
                ignore_names=imported_names,
                top_k=args.top_k, min_distance=args.min_candidate_distance,
                camera_target=floor_target, object_seed=obj_seed,
                scene_camera_pos=scene_cam_pos,
                visible_centroid=visible_centroid,
            )

        # Retry with relaxed tolerance if nothing found
        if not candidates:
            print("No candidates with strict tolerance, retrying relaxed...")
            candidates = find_flat_candidates(
                bpy.context.scene,
                depsgraph,
                bounds[0],
                bounds[1],
                obj_size,
                ignore_names=imported_names,
                top_k=args.top_k,
                min_distance=args.min_candidate_distance,
                height_tol=0.5,
                max_slope_deg=25.0,
                camera_target=cam_target,
                object_seed=obj_seed,
                scene_camera_pos=scene_cam_pos,
                visible_centroid=visible_centroid,
            )

        # Multi-height raycast fallback for indoor scenes
        if not candidates:
            print("Trying multi-height raycast for indoor scenes...")
            candidates = find_candidates_multi_height(
                bpy.context.scene,
                depsgraph,
                bounds[0],
                bounds[1],
                obj_size,
                ignore_names=imported_names,
                top_k=args.top_k,
                min_distance=args.min_candidate_distance,
                object_seed=obj_seed,
                scene_camera_pos=scene_cam_pos,
                visible_centroid=visible_centroid,
            )

        # Last resort: drop the visibility gate. Some scenes have no scene
        # camera or a camera pointed away from any usable surface.
        if not candidates and scene_cam_pos is not None:
            print("No visible candidates — falling back without visibility gate.")
            candidates = find_flat_candidates(
                bpy.context.scene, depsgraph, bounds[0], bounds[1], obj_size,
                ignore_names=imported_names,
                top_k=args.top_k, min_distance=args.min_candidate_distance,
                height_tol=0.5, max_slope_deg=25.0,
                camera_target=cam_target, object_seed=obj_seed,
                # scene_camera_pos omitted on purpose
            )

        # Terrain mode: very relaxed tolerances for outdoor/landscape scenes
        # where ground is rough, hilly, or uneven.
        if not candidates:
            print("Trying terrain mode (very relaxed tolerances)...")
            candidates = find_flat_candidates(
                bpy.context.scene, depsgraph, bounds[0], bounds[1], obj_size,
                ignore_names=imported_names,
                top_k=args.top_k, min_distance=args.min_candidate_distance,
                height_tol=2.0, max_slope_deg=45.0,
                camera_target=cam_target, object_seed=obj_seed,
            )

        # Absolute last resort: place at the camera ground target directly.
        # The scene camera artist chose a good spot to look at — place there.
        if not candidates and cam_target is not None:
            print("All discovery failed — using camera ground target as candidate.")
            candidates = [{
                "position": [cam_target.x, cam_target.y, cam_target.z],
                "normal": [0, 0, 1],
                "score": 0.0,
            }]
    finally:
        restore_hidden(hidden)

    emit_result({
        "status": "ok",
        "candidates": candidates,
        "object_size": [obj_size.x, obj_size.y, obj_size.z],
        "applied_scale": applied_scale,
        "scene_correction": scene_correction,
        "auto_rotation": list(auto_rot),
        "scene_bounds": [
            [bounds[0].x, bounds[0].y, bounds[0].z],
            [bounds[1].x, bounds[1].y, bounds[1].z],
        ] if bounds else None,
    })


# ---------------------------------------------------------------------------
# Mode: place_and_render
# ---------------------------------------------------------------------------


def mode_place_and_render(args):
    if args.position is None:
        emit_error("--position is required for place_and_render mode")
        return

    # Scene preview: use cached path if provided & exists, otherwise render
    scene_preview_path = None
    if args.scene_preview_path and os.path.isfile(args.scene_preview_path):
        scene_preview_path = args.scene_preview_path
        print(f"Scene preview cached: {scene_preview_path}")
    else:
        scene_cam = bpy.context.scene.camera
        if scene_cam:
            scene_preview_path = (
                args.scene_preview_path
                or os.path.splitext(args.output_image)[0] + "_scene.png"
            )
            os.makedirs(os.path.dirname(scene_preview_path), exist_ok=True)
            scene = bpy.context.scene
            scene.render.resolution_x = args.resolution[0]
            scene.render.resolution_y = args.resolution[1]
            scene.render.resolution_percentage = 100
            scene.render.image_settings.file_format = "PNG"
            scene.render.filepath = scene_preview_path
            scene.render.engine = "CYCLES"
            scene.cycles.samples = max(4, args.samples // 4)
            scene.cycles.use_denoising = True
            if args.engine == "CYCLES":
                configure_gpu(
                    scene,
                    disable_gpu=(args.gpu_backend == "CPU"),
                    gpu_backend=args.gpu_backend,
                    gpu_devices=args.gpu_devices,
                )
            bpy.ops.render.render(write_still=True)
            print(f"Scene preview rendered: {scene_preview_path}")

    obj_file = pick_first_file(Path(args.object_file))
    if obj_file is None:
        emit_error(f"Object not found: {args.object_file}")
        return

    imported = import_object(obj_file)
    if not imported:
        emit_error("No objects imported")
        return

    root = parent_and_center(imported)

    imported_names = {o.name for o in imported} | {root.name}
    scene_meshes = [
        o for o in bpy.context.scene.objects
        if o.type == "MESH" and o.name not in imported_names
    ]

    # Auto-fix orientation first (e.g. Y-up OBJ -> Z-up)
    auto_rot = auto_fix_orientation(root, imported)

    # Object-anchored metric normalization
    scene_correction = normalize_to_metric(
        root, imported, scene_meshes,
        object_path=args.object_file,
        exclude_names=imported_names,
    )

    # Apply scene scale override (used when retrying with different scale)
    if hasattr(args, 'scene_scale'):
        apply_scene_scale(args.scene_scale, set(imported_names))
    # Only apply VLM override scale if explicitly different from 1.0
    applied_scale = 1.0
    if args.scale > 0 and abs(args.scale - 1.0) > 0.01:
        applied_scale = auto_scale(root, imported, scene_meshes,
                                   override_scale=args.scale)

    # Place object at candidate position with rotation
    position = Vector(args.position)
    root.location = position
    rx = math.radians(auto_rot[0] + args.rotation_x)
    ry = math.radians(auto_rot[1] + args.rotation_y)
    rz = math.radians(auto_rot[2] + args.rotation_z)
    root.rotation_euler = (rx, ry, rz)
    bpy.context.view_layer.update()

    # Re-center bottom to sit on the surface
    bbox_after = get_world_bbox(imported)
    if bbox_after:
        z_bottom = bbox_after[0].z
        root.location.z += position.z - z_bottom
        bpy.context.view_layer.update()

    # Snap to floor via raycast (much faster than physics simulation)
    snapped = snap_to_floor(root, imported, scene_meshes, max_search_dist=1.0)
    if not snapped:
        # Fallback: small offset above raycast position
        root.location.z += 0.05
        bpy.context.view_layer.update()
        print("snap_to_floor: no surface found, using raycast position")

    # Verify grounding — programmatic check for floating/embedded
    grounding_info = verify_grounded(imported, scene_meshes, max_gap=0.08)
    print(f"Grounding: {grounding_info}")

    # Fix textures
    fix_missing_textures()

    # Only add sky lighting if the scene has no existing lights or world shader
    has_lights = any(o.type == 'LIGHT' for o in bpy.data.objects if o.name not in imported_names)
    has_world = False
    world = bpy.context.scene.world
    if world and world.use_nodes:
        for node in world.node_tree.nodes:
            if node.type in ('TEX_SKY', 'TEX_ENVIRONMENT', 'TEX_IMAGE'):
                has_world = True
                break
            if node.type == 'BACKGROUND':
                c = node.inputs['Color']
                if c.is_linked or sum(c.default_value[:3]) > 0.01:
                    has_world = True
                    break
    if not has_lights and not has_world:
        print("No scene lighting found, adding Nishita sky")
        set_nishita_sky(args.sky_strength)
    else:
        print(f"Using scene lighting (lights={has_lights}, world={has_world})")

    # Get object size after scaling
    bbox = get_world_bbox(imported)
    if bbox is None:
        emit_error("Could not compute object bbox after placement")
        return
    obj_size = bbox[1] - bbox[0]

    # Check collisions with scene geometry
    collisions = check_collisions(imported, scene_meshes)
    if collisions:
        print(f"Collisions detected: {collisions}")

    # Determine camera azimuths — validate line of sight for each
    preferred = args.camera_azimuths or [args.camera_azimuth]
    resolution = tuple(args.resolution)

    obj_diag = obj_size.length
    cam_radius = args.camera_radius
    if cam_radius is None or cam_radius <= 0:
        cam_radius = max(obj_diag * 3.0, obj_diag * 2.0)

    imported_names = {o.name for o in imported} | {root.name}
    # If exactly 1 azimuth is specified, treat it as a fixed override (final render mode)
    if len(preferred) == 1:
        azim_elev_pairs = [(preferred[0], args.camera_elevation, 1.0)]
        print(f"Fixed camera azimuth override: {preferred[0]}")
    else:
        # Find 6-8 diverse views: varied azimuths AND elevations/distances
        azim_elev_pairs = find_valid_azimuths_with_elevation(
            position, obj_size, cam_radius, args.camera_elevation,
            preferred_azimuths=preferred,
            ignore_names=imported_names,
            num_needed=6,
            min_separation=15.0,
            elevation_choices=[10, 20, 35, 50],
        )
        if not azim_elev_pairs:
            azim_elev_pairs = [(preferred[0], args.camera_elevation, 0.0)]
            print(f"WARNING: no valid camera angles found, using fallback azimuth {preferred[0]}")
        else:
            print(f"Valid camera views ({len(azim_elev_pairs)}): {azim_elev_pairs}")

    # Configure GPU once
    if args.engine == "CYCLES":
        configure_gpu(
            bpy.context.scene,
            disable_gpu=(args.gpu_backend == "CPU"),
            gpu_backend=args.gpu_backend,
            gpu_devices=args.gpu_devices,
        )

    os.makedirs(os.path.dirname(args.output_image), exist_ok=True)

    # Vary camera distance per view: close-up, standard, wide establishing shots.
    # With the 2m minimum radius, even small objects get proper wide shots.
    radius_multipliers = [1.0, 3.0, 0.5, 2.0, 0.7, 4.0, 1.5, 0.4]

    # Render from each view
    views = []
    primary_result = {}
    for view_idx, ae in enumerate(azim_elev_pairs):
        azim, elev, _pre_cov = ae
        r_mult = radius_multipliers[view_idx % len(radius_multipliers)]
        view_radius = cam_radius * r_mult
        cam = setup_camera(
            position,
            obj_size,
            radius=view_radius,
            elevation_deg=elev,
            azimuth_deg=azim,
        )
        bpy.context.view_layer.update()

        # Coverage check: skip cameras that mostly look at empty space
        coverage = compute_view_coverage(cam, n_rays=8)
        print(f"View {view_idx} azim={azim} elev={elev}: coverage={coverage:.2f}")

        # Output path: primary view uses the original path, extra views get suffix
        if view_idx == 0:
            view_path = args.output_image
        else:
            base, ext = os.path.splitext(args.output_image)
            view_path = f"{base}_v{view_idx}{ext}"

        if coverage < 0.4:
            print(f"  → SKIP (coverage too low: {coverage:.2f})")
            views.append({
                "image_path": None,
                "azimuth": azim,
                "coverage": coverage,
                "skipped": True,
                "skip_reason": "low_coverage",
            })
            continue

        render_image(
            view_path,
            engine=args.engine,
            resolution=resolution,
            samples=args.samples,
        )

        cam_mat = cam.matrix_world
        cam_forward = Vector(cam_mat.col[2][:3]) * -1
        cam_up = Vector(cam_mat.col[1][:3])
        cam_pos = Vector(cam_mat.col[3][:3])

        view_data = {
            "image_path": view_path,
            "azimuth": azim,
            "elevation": elev,
            "coverage": coverage,
            "camera_position": [cam_pos.x, cam_pos.y, cam_pos.z],
            "camera_forward": [cam_forward.x, cam_forward.y, cam_forward.z],
            "camera_up": [cam_up.x, cam_up.y, cam_up.z],
            "camera_radius": (cam_pos - position).length,
        }
        views.append(view_data)

        if view_idx == 0:
            bbox_2d = project_bbox_2d(imported, resolution)
            primary_result = view_data.copy()
            primary_result["object_bbox_2d"] = bbox_2d

    # Render additional shot types (wide, mid) if requested — for final renders
    # Shot variety is built into the views (varied radius/elevation per azimuth)

    scene_bounds = get_world_bbox(scene_meshes)

    result = {
        "status": "ok",
        "image_path": args.output_image,
        "scene_preview": scene_preview_path,
        "object_position": list(args.position),
        "object_size": [obj_size.x, obj_size.y, obj_size.z],
        "object_bbox_2d": primary_result.get("object_bbox_2d"),
        "camera_position": primary_result.get("camera_position"),
        "camera_forward": primary_result.get("camera_forward"),
        "camera_up": primary_result.get("camera_up"),
        "camera_radius": primary_result.get("camera_radius"),
        "collisions": collisions,
        "grounding": grounding_info,
        "applied_scale": applied_scale,
        "auto_rotation": list(auto_rot),
        "applied_rotation": [
            auto_rot[0] + args.rotation_x,
            auto_rot[1] + args.rotation_y,
            auto_rot[2] + args.rotation_z,
        ],
        "scene_bounds": [
            [scene_bounds[0].x, scene_bounds[0].y, scene_bounds[0].z],
            [scene_bounds[1].x, scene_bounds[1].y, scene_bounds[1].z],
        ] if scene_bounds else None,
        "views": views,
    }
    emit_result(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    try:
        if args.mode == "discover_candidates":
            mode_discover_candidates(args)
        elif args.mode == "place_and_render":
            mode_place_and_render(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        emit_error(str(e))


if __name__ == "__main__":
    main()
