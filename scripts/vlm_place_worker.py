"""Blender-side worker for VLM-guided object placement.

Runs inside Blender:
    blender -b scene.blend -P scripts/vlm_place_worker.py -- --mode <mode> [args]

Modes:
    discover_candidates  Find top-K flat surface positions via raycast grid.
    place_and_render     Place object at a given position, render EEVEE/Cycles preview.
    serve                Long-lived RPC mode: scene loaded once, reads JSON
                         commands from stdin, processes each via the same
                         logic as discover_candidates / place_and_render,
                         emits ###RESULT### per command. Exits on {"cmd":"exit"}.
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


def normalize_render_settings() -> None:
    """Force consistent color management across scenes.

    Different BlenderKit assets ship with different view_transforms:
      - AgX (Blender 4.0+ default): good highlight roll-off
      - Filmic (older default): good but slightly desaturated
      - Standard: clips anything >1.0 to pure white → blown-out renders
      - ACES/custom: often unavailable in our Blender, silently falls back

    For predictable renders, force AgX (or Filmic as fallback), reset
    exposure, and cap any absurd world-background strengths.
    """
    scene = bpy.context.scene
    vs = scene.view_settings
    preferred = ["AgX", "Filmic"]
    valid = [t.identifier for t in vs.bl_rna.properties["view_transform"].enum_items]
    for choice in preferred:
        if choice in valid:
            if vs.view_transform != choice:
                print(f"Normalizing view_transform: {vs.view_transform} -> {choice}")
                vs.view_transform = choice
            break
    if abs(vs.exposure) > 0.5:
        print(f"Normalizing exposure: {vs.exposure:.2f} -> 0")
        vs.exposure = 0.0
    if abs(vs.gamma - 1.0) > 0.05:
        print(f"Normalizing gamma: {vs.gamma:.2f} -> 1.0")
        vs.gamma = 1.0
    # Cap world-background strength: some scenes ship with strength=11 HDRIs.
    world = scene.world
    if world and world.use_nodes:
        for node in world.node_tree.nodes:
            if node.type == "BACKGROUND":
                s = node.inputs["Strength"].default_value
                if s > 3.0:
                    print(f"Capping world Background strength: {s:.1f} -> 2.0")
                    node.inputs["Strength"].default_value = 2.0


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


def disable_heavy_modifiers_for_raycast() -> list:
    """Disable procedural-geometry modifiers for raycasting speed.

    Raycasting over procedural geometry (grass/flower particles, subsurf
    meshes, geometry-nodes grass generators) can be 40x+ slower than over
    base meshes (measured: 95ms/ray vs 2.4ms/ray on Picnic). Discovery
    only needs the base terrain surface, so we toggle these modifiers off.

    Returns a list of (obj_name, modifier_name, old_show_viewport,
    old_show_render) tuples so the caller can restore (used in serve mode
    to make the disable reversible between commands; one-shot mode just
    discards the list).
    """
    touched = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for m in obj.modifiers:
            if m.type in ("PARTICLE_SYSTEM", "SUBSURF", "NODES"):
                touched.append((obj.name, m.name, m.show_viewport, m.show_render))
                m.show_viewport = False
                m.show_render = False
    if touched:
        print(f"Disabled {len(touched)} heavy modifiers for fast raycasting")
        bpy.context.view_layer.update()
    return touched


def restore_heavy_modifiers(touched: list) -> None:
    """Reverse disable_heavy_modifiers_for_raycast (serve mode cleanup)."""
    for obj_name, mod_name, sv, sr in touched:
        obj = bpy.data.objects.get(obj_name)
        if not obj:
            continue
        m = obj.modifiers.get(mod_name)
        if not m:
            continue
        m.show_viewport = sv
        m.show_render = sr
    if touched:
        bpy.context.view_layer.update()


def image_bbox_to_world_region(scene, depsgraph, image_bbox):
    """Map an image-fraction bbox [x_lo, y_lo, x_hi, y_hi] to a world-space XY region.

    Raycasts from the scene camera through a 3x3 grid of samples inside the
    image bbox; uses the min/max of hits to form a world-space rectangle.
    Returns {"x_lo", "x_hi", "y_lo", "y_hi"} or None if no hits.
    """
    cam = scene.camera
    if cam is None or not image_bbox:
        return None
    x_lo, y_lo, x_hi, y_hi = image_bbox
    cam_matrix = cam.matrix_world
    cam_pos = cam_matrix.translation.copy()
    cam_data = cam.data
    aspect = scene.render.resolution_x / max(1, scene.render.resolution_y)
    if cam_data.type == "PERSP":
        half_fov_y = cam_data.angle / 2
        half_fov_x = math.atan(math.tan(half_fov_y) * aspect)
    else:
        half_fov_x = half_fov_y = 0.5
    rot = cam_matrix.to_quaternion()

    hits = []
    # 3x3 grid inside the image bbox for a robust min/max envelope
    for iy in range(3):
        for ix in range(3):
            xf = x_lo + (x_hi - x_lo) * ((ix + 0.5) / 3.0)
            yf = y_lo + (y_hi - y_lo) * ((iy + 0.5) / 3.0)
            # Image (0,0) is top-left; Blender frustum v=+1 is top in camera space.
            # In compute_scene_camera_view: u,v = (ix+0.5)/n*2-1 with y going bottom-to-top.
            # We invert y so image-y=0 maps to top (v=+1), image-y=1 maps to bottom (v=-1).
            u = xf * 2 - 1
            v = -(yf * 2 - 1)
            local_dir = Vector((
                u * math.tan(half_fov_x),
                v * math.tan(half_fov_y),
                -1.0,
            )).normalized()
            world_dir = rot @ local_dir
            hit, loc, _, _, _, _ = scene.ray_cast(depsgraph, cam_pos, world_dir)
            if hit:
                hits.append(loc)
    if not hits:
        return None
    xs = [h.x for h in hits]
    ys = [h.y for h in hits]
    return {
        "x_lo": min(xs), "x_hi": max(xs),
        "y_lo": min(ys), "y_hi": max(ys),
    }


_current_cmd_id = None  # set by mode_serve_scene to tag responses


def emit_result(data: dict) -> None:
    if _current_cmd_id is not None and isinstance(data, dict) and "id" not in data:
        data = {"id": _current_cmd_id, **data}
    print(f"{RESULT_PREFIX}{json.dumps(data)}", flush=True)


def emit_error(msg: str) -> None:
    emit_result({"status": "error", "error": msg})


# ---------------------------------------------------------------------------
# Scene-state snapshot/restore (used by serve mode to roll back per-command
# mutations so the next command starts from the same clean scene state).
# ---------------------------------------------------------------------------


def snapshot_scene_state() -> dict:
    """Capture the parts of scene/data state that any mode might mutate.

    This is conservative — we record more than strictly necessary to avoid
    silent drift across commands.
    """
    scene = bpy.context.scene
    vs = scene.view_settings
    rs = scene.render
    snap = {
        "object_names": {o.name for o in bpy.data.objects},
        # View settings (touched by normalize_render_settings)
        "view_transform": vs.view_transform,
        "exposure": vs.exposure,
        "gamma": vs.gamma,
        # Scene camera (touched by setup_camera)
        "scene_camera_name": scene.camera.name if scene.camera else None,
        # Render settings (touched by mode_place_and_render)
        "render_engine": rs.engine,
        "render_resx": rs.resolution_x,
        "render_resy": rs.resolution_y,
        "render_filepath": rs.filepath,
        "cycles_samples": getattr(scene.cycles, "samples", None) if hasattr(scene, "cycles") else None,
        # Per-light energy + per-non-imported-root (loc, scale) snapshot for
        # apply_scene_scale rollback.
        "light_energies": {l.name: l.energy for l in bpy.data.lights},
        "object_transforms": {
            o.name: (tuple(o.location), tuple(o.scale))
            for o in bpy.data.objects if o.parent is None
        },
        # World background strength (touched by normalize_render_settings)
        "world_bg_strengths": _capture_world_bg_strengths(scene),
    }
    return snap


def _capture_world_bg_strengths(scene) -> list:
    out = []
    world = scene.world
    if world and world.use_nodes:
        for node in world.node_tree.nodes:
            if node.type == "BACKGROUND":
                try:
                    out.append((node.name, node.inputs["Strength"].default_value))
                except Exception:
                    pass
    return out


def restore_scene_state(snap: dict) -> None:
    """Roll back to the state recorded by snapshot_scene_state.

    Order matters: undo mutations roughly in reverse order of likely apply.
    """
    scene = bpy.context.scene

    # 1. Remove objects added since snapshot (imported objects, PreviewCam,
    #    parent_and_center root, etc.).
    pre_names = snap["object_names"]
    to_remove = [o for o in bpy.data.objects if o.name not in pre_names]
    for o in to_remove:
        try:
            bpy.data.objects.remove(o, do_unlink=True)
        except Exception:
            pass

    # 2. Purge orphans (data-blocks left behind: meshes, materials, images).
    try:
        bpy.ops.outliner.orphans_purge(do_recursive=True)
    except Exception:
        pass

    # 3. Restore object transforms (apply_scene_scale rollback).
    for name, (loc, scl) in snap["object_transforms"].items():
        o = bpy.data.objects.get(name)
        if o is None:
            continue
        try:
            o.location = loc
            o.scale = scl
        except Exception:
            pass

    # 4. Restore light energies (apply_scene_scale s² compensation rollback).
    for lname, energy in snap["light_energies"].items():
        l = bpy.data.lights.get(lname)
        if l:
            try:
                l.energy = energy
            except Exception:
                pass

    # 5. Restore view settings.
    vs = scene.view_settings
    try:
        vs.view_transform = snap["view_transform"]
        vs.exposure = snap["exposure"]
        vs.gamma = snap["gamma"]
    except Exception:
        pass

    # 6. Restore world background strengths.
    world = scene.world
    if world and world.use_nodes:
        for name, strength in snap["world_bg_strengths"]:
            node = world.node_tree.nodes.get(name)
            if node and node.type == "BACKGROUND":
                try:
                    node.inputs["Strength"].default_value = strength
                except Exception:
                    pass

    # 7. Restore scene camera reference.
    cam_name = snap.get("scene_camera_name")
    if cam_name:
        cam = bpy.data.objects.get(cam_name)
        if cam:
            scene.camera = cam

    # 8. Restore render settings.
    rs = scene.render
    try:
        rs.engine = snap["render_engine"]
        rs.resolution_x = snap["render_resx"]
        rs.resolution_y = snap["render_resy"]
        rs.filepath = snap["render_filepath"]
    except Exception:
        pass
    if snap.get("cycles_samples") is not None and hasattr(scene, "cycles"):
        try:
            scene.cycles.samples = snap["cycles_samples"]
        except Exception:
            pass

    bpy.context.view_layer.update()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser(require_object_file: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=False,
                   choices=["discover_candidates", "place_and_render", "serve"])
    p.add_argument("--object_file", required=require_object_file)
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
    p.add_argument("--interest_region", type=str, default=None,
                   help="JSON of world-space interest region {x_lo,x_hi,y_lo,y_hi}. "
                        "Produced by the orchestrator from a VLM pass over the scene preview.")
    p.add_argument("--interest_target_xy", type=float, nargs=2, default=None,
                   help="World XY of the interest-region centroid; biases camera "
                        "azimuth selection so views look toward the interesting part.")
    return p


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    idx = argv.index("--") + 1 if "--" in argv else len(argv)
    p = _build_arg_parser(require_object_file=False)  # check mode-conditionally below
    args = p.parse_args(argv[idx:])
    if not args.mode:
        p.error("--mode is required")
    if args.mode != "serve" and not args.object_file:
        p.error("--object_file is required for one-shot modes")
    return args


def args_from_command(cmd: dict) -> argparse.Namespace:
    """Build an argparse.Namespace for a single serve-mode command.

    The command dict mirrors CLI kwargs (e.g. {"object_file": "...",
    "position": [x, y, z], "scene_scale": 0.5, "camera_azimuths": [0, 120, 240]}).
    Missing fields fall back to argparse defaults.
    """
    p = _build_arg_parser(require_object_file=False)
    args = p.parse_args([])  # populate defaults
    for k, v in cmd.items():
        if k in ("cmd", "id"):
            continue
        # interest_region needs to be a JSON STRING in CLI form; in the command
        # dict it may already be a dict — convert here if so.
        if k == "interest_region" and isinstance(v, dict):
            v = json.dumps(v)
        setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# Mode: discover_candidates
# ---------------------------------------------------------------------------


def mode_discover_candidates(args):
    normalize_render_settings()
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
    # Disable particle systems and subsurf before raycasting — huge speedup
    # on outdoor scenes with procedural grass/trees.
    disable_heavy_modifiers_for_raycast()
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

        # Parse the (optional) VLM-provided interest region. Two forms accepted:
        #   world-space: {"x_lo", "x_hi", "y_lo", "y_hi"}
        #   image-frac:  {"image_bbox": [x_lo, y_lo, x_hi, y_hi]} in [0,1],
        #                y top-to-bottom (standard image convention).
        # Image-frac forms are converted via raycasting through the scene camera.
        interest_region = None
        if getattr(args, "interest_region", None):
            try:
                raw = json.loads(args.interest_region)
            except Exception as e:
                print(f"  Could not parse --interest_region: {e}")
                raw = None
            if isinstance(raw, dict):
                if "image_bbox" in raw:
                    mapped = image_bbox_to_world_region(
                        bpy.context.scene, depsgraph, raw["image_bbox"]
                    )
                    if mapped:
                        interest_region = mapped
                        print(f"  Using VLM interest region (from image_bbox): {mapped}")
                    else:
                        print(f"  image_bbox provided but no camera raycasts hit; ignoring")
                elif all(k in raw for k in ("x_lo", "x_hi", "y_lo", "y_hi")):
                    interest_region = raw
                    print(f"  Using VLM interest region (world): {raw}")
                else:
                    print(f"  Ignoring malformed interest_region: {raw}")

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
            visible_hits=visible_hits,
            scene_meshes=scene_meshes,
            interest_region=interest_region,
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
                visible_hits=visible_hits,
                scene_meshes=scene_meshes,
                interest_region=interest_region,
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
                visible_hits=visible_hits,
                scene_meshes=scene_meshes,
                interest_region=interest_region,
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
                visible_hits=visible_hits,
                scene_meshes=scene_meshes,
                interest_region=interest_region,
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
                scene_meshes=scene_meshes,
                interest_region=interest_region,
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
                scene_meshes=scene_meshes,
                interest_region=interest_region,
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
        # Resolved world-space interest region (if image_bbox was provided):
        # downstream uses this to bias camera azimuths during render.
        "interest_world_region": interest_region,
    })


# ---------------------------------------------------------------------------
# Mode: place_and_render
# ---------------------------------------------------------------------------


def mode_place_and_render(args):
    normalize_render_settings()
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
    grounding_info = verify_grounded(imported, scene_meshes, max_gap=0.10)
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
        interest_target = None
        if getattr(args, "interest_target_xy", None):
            tx, ty = args.interest_target_xy
            interest_target = Vector((float(tx), float(ty), position.z))
        azim_elev_pairs = find_valid_azimuths_with_elevation(
            position, obj_size, cam_radius, args.camera_elevation,
            preferred_azimuths=preferred,
            ignore_names=imported_names,
            num_needed=6,
            min_separation=15.0,
            elevation_choices=[10, 20, 35, 50],
            interest_target=interest_target,
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

        if view_idx == 0 or not primary_result:
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


def mode_serve_scene():
    """Long-lived RPC mode. The scene is loaded once at process start.

    Reads one JSON command per line from stdin. Each command must include
    `cmd` (one of: "discover", "place_and_render", "ping", "exit") and an
    optional `id` to correlate with the response. Per-command state mutations
    are rolled back via snapshot_scene_state / restore_scene_state so the
    next command starts from clean state.
    """
    global _current_cmd_id
    snap = snapshot_scene_state()
    print(f"###SERVE_READY### {{\"object_count\": {len(snap['object_names'])}}}",
          flush=True)
    import traceback
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except Exception as e:
            print(f"{RESULT_PREFIX}{json.dumps({'status': 'error', 'error': f'bad json: {e}'})}",
                  flush=True)
            continue
        cmd_kind = cmd.get("cmd")
        _current_cmd_id = cmd.get("id")
        if cmd_kind == "exit":
            print(f"{RESULT_PREFIX}{json.dumps({'id': _current_cmd_id, 'status': 'bye'})}",
                  flush=True)
            break
        if cmd_kind == "ping":
            emit_result({"status": "ok"})
            continue
        try:
            if cmd_kind == "discover":
                args = args_from_command(cmd)
                args.mode = "discover_candidates"
                mode_discover_candidates(args)
            elif cmd_kind == "place_and_render":
                args = args_from_command(cmd)
                args.mode = "place_and_render"
                mode_place_and_render(args)
            else:
                emit_error(f"unknown cmd: {cmd_kind!r}")
        except Exception as e:
            traceback.print_exc()
            emit_error(f"{type(e).__name__}: {e}")
        finally:
            try:
                restore_scene_state(snap)
            except Exception as e:
                traceback.print_exc()
                # Don't die on restore failure; log and continue
                print(f"WARN: restore_scene_state raised {e}", file=sys.stderr, flush=True)
            _current_cmd_id = None


def main():
    args = parse_args()
    try:
        if args.mode == "discover_candidates":
            mode_discover_candidates(args)
        elif args.mode == "place_and_render":
            mode_place_and_render(args)
        elif args.mode == "serve":
            mode_serve_scene()
    except Exception as e:
        import traceback
        traceback.print_exc()
        emit_error(str(e))


if __name__ == "__main__":
    main()
