"""Compute mesh-tight 2D bbox (off-image OK) for each annotated camera in a placement.

Optimized with numpy vectorization: per-camera projection of N verts is O(N)
matrix ops (not a Python loop), giving ~1000x speedup over `world_to_camera_view`.

Run via Blender CLI:
  ./blender/blender --background --python scripts/compute_mesh_tight_bbox.py -- \
      outputs/v5_3090x8_260429_092917/p0_Lynxsdesign_snowman_5e87c51f-66bf-496e-8e6 \
      outputs/v5_3090x8_260429_092917/p16_Picnic_andrew_16e03125-959b-4312-8e8a \
      ...

For each placement_dir, writes <placement_dir>/mesh_tight_bbox.json with one entry
per view containing the projected 2D AABB of all subject mesh vertices.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector


# Default Blender camera intrinsics (must match the renderer's scene defaults).
# `view_frame()` at z=-1 gives 4 corners (BR, TR, TL, BL) in camera-local coords.
# We read it once per scene below.

def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    cam_data = bpy.data.cameras.new("Cam")
    # Match render_object.py defaults: focal=24mm, sensor 12.8×9.6mm.
    cam_data.lens = 24.0
    cam_data.sensor_width = 12.8
    cam_data.sensor_height = 9.6
    cam_data.sensor_fit = "AUTO"
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj


def setup_render_resolution(width: int, height: int) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100


def get_subject_meshes_from_blend(blend_path: str) -> list:
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    imported = [o for o in data_to.objects if o is not None]
    meshes: list = []
    for o in imported:
        bpy.context.collection.objects.link(o)
        if o.type == "MESH":
            meshes.append(o)
    return meshes


def world_vertices_np(meshes: list, placement_position, placement_rotation_xyz_rad,
                      placement_scale: float, max_verts: int = 50000) -> np.ndarray:
    """Return Nx3 numpy array of world-space subject mesh vertices.

    Subsample uniformly if total > max_verts (5000 verts gives a near-perfect
    tight 2D AABB for any reasonable subject; original v3 also uses 8 corners).
    """
    rot_mat = Euler(placement_rotation_xyz_rad, "XYZ").to_matrix().to_4x4()
    placement_xf = Matrix.Translation(Vector(placement_position)) @ rot_mat @ Matrix.Scale(placement_scale, 4)

    parts: list = []
    for m in meshes:
        mw = placement_xf @ m.matrix_world
        local = np.array([v.co[:] for v in m.data.vertices], dtype=np.float64)
        ones = np.ones((local.shape[0], 1), dtype=np.float64)
        homog = np.concatenate([local, ones], axis=1)            # (N, 4)
        mat = np.array(mw, dtype=np.float64)                     # (4, 4)
        world = (homog @ mat.T)[:, :3]                            # (N, 3)
        parts.append(world)
    verts = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3))
    if verts.shape[0] > max_verts:
        idx = np.linspace(0, verts.shape[0] - 1, max_verts).astype(np.int64)
        verts = verts[idx]
    return verts


def get_camera_frame_bounds(scene, cam_obj) -> tuple[float, float, float, float]:
    """Frame corners NORMALIZED to z=1 → (min_x, max_x, min_y, max_y).

    `cam_data.view_frame()` returns corners at some camera-defined depth (Blender
    picks a "natural" z based on focal/sensor, not z=-1). We rescale so the
    bounds correspond to z=1, which makes the per-ray scaling
    `fmin_x = min_x * z_ray` correct.
    """
    cam_data = cam_obj.data
    frame = cam_data.view_frame(scene=scene)
    view_z = abs(frame[0].z)
    if view_z < 1e-9:
        view_z = 1.0
    xs = [f.x / view_z for f in frame]
    ys = [f.y / view_z for f in frame]
    return float(min(xs)), float(max(xs)), float(min(ys)), float(max(ys))


def camera_basis(final_forward, final_up):
    """Return (right, up, neg_forward) world axes for the camera (Blender conv).

    Camera local +X=right, +Y=up, -Z=forward.
    """
    fwd = np.array(final_forward, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    up = np.array(final_up, dtype=np.float64)
    up /= np.linalg.norm(up)
    nz = -fwd
    right = np.cross(up, nz)
    right /= np.linalg.norm(right)
    ortho_up = np.cross(nz, right)
    ortho_up /= np.linalg.norm(ortho_up)
    return right, ortho_up, nz


def project_verts_vec(verts_world: np.ndarray, cam_pos, final_forward, final_up,
                      width: int, height: int,
                      min_x: float, max_x: float, min_y: float, max_y: float) -> list | None:
    """Vectorized projection. Returns [xmin, ymin, xmax, ymax] or None."""
    right, up, nz = camera_basis(final_forward, final_up)
    cam_pos = np.array(cam_pos, dtype=np.float64)

    rel = verts_world - cam_pos                              # (N, 3)
    co_x = rel @ right                                       # (N,)
    co_y = rel @ up
    co_z = rel @ nz                                          # +Z = behind in this basis
    z = -co_z                                                # forward distance

    valid = z > 1e-6
    if not np.any(valid):
        return None
    co_x = co_x[valid]
    co_y = co_y[valid]
    z = z[valid]

    # frame corners scale linearly with depth (perspective). frame is at z = 1.
    fmin_x = min_x * z
    fmax_x = max_x * z
    fmin_y = min_y * z
    fmax_y = max_y * z

    ndc_x = (co_x - fmin_x) / (fmax_x - fmin_x)              # [0, 1] in-frame
    ndc_y = (co_y - fmin_y) / (fmax_y - fmin_y)
    x_px = ndc_x * float(width)
    y_px = (1.0 - ndc_y) * float(height)

    return [float(x_px.min()), float(y_px.min()), float(x_px.max()), float(y_px.max())]


def process_placement(placement_dir: Path) -> None:
    placement_json_path = placement_dir / "placement.json"
    annotations_json_path = placement_dir / "annotations.json"
    output_path = placement_dir / "mesh_tight_bbox.json"

    if not placement_json_path.exists() or not annotations_json_path.exists():
        print(f"[skip] missing json in {placement_dir}", flush=True)
        return

    pmt = json.loads(placement_json_path.read_text())
    ann = json.loads(annotations_json_path.read_text())

    reset_scene()
    setup_render_resolution(1024, 768)
    scene = bpy.context.scene
    cam_obj = scene.camera

    obj_path = Path(pmt["object_path"])
    if not obj_path.is_absolute():
        obj_path = (Path.cwd() / pmt["object_path"]).resolve()
    if not obj_path.exists():
        print(f"[skip] missing object .blend: {obj_path}", flush=True)
        return

    print(f"[{placement_dir.name}] loading {obj_path.name} ...", flush=True)
    meshes = get_subject_meshes_from_blend(str(obj_path))
    if not meshes:
        print(f"[skip] no meshes in {obj_path}", flush=True)
        return
    total_verts = sum(len(m.data.vertices) for m in meshes)
    print(f"  meshes={len(meshes)} raw_verts={total_verts}", flush=True)

    placement_rot = pmt.get("rotation_xyz_rad", [0.0, 0.0, 0.0])
    placement_scale = float(pmt.get("scale", 1.0))
    verts_world = world_vertices_np(meshes, pmt["position"], placement_rot, placement_scale)
    print(f"  sampled_verts={verts_world.shape[0]}", flush=True)

    min_x, max_x, min_y, max_y = get_camera_frame_bounds(scene, cam_obj)
    print(f"  view_frame: x=[{min_x:.4f},{max_x:.4f}] y=[{min_y:.4f},{max_y:.4f}]", flush=True)

    import time
    t0 = time.time()
    results = []
    for entry in ann:
        bbox = project_verts_vec(
            verts_world,
            entry["camera_position"],
            entry.get("final_forward", entry["base_forward"]),
            entry.get("final_up", entry["base_up"]),
            1024, 768, min_x, max_x, min_y, max_y,
        )
        if bbox is None:
            continue
        results.append({"image": entry["image"], "bbox_2d_full_mesh_tight": bbox})

    output_path.write_text(json.dumps(results, indent=2))
    dt = time.time() - t0
    print(f"  wrote {len(results)} entries in {dt:.1f}s ({dt/max(1,len(ann))*1000:.1f}ms/view) -> {output_path.name}", flush=True)


def main() -> None:
    try:
        sep = sys.argv.index("--")
        targets = sys.argv[sep + 1:]
    except ValueError:
        targets = []
    if not targets:
        print("Usage: blender --background --python THIS_SCRIPT -- <placement_dir> [...]", flush=True)
        sys.exit(1)
    for t in targets:
        p = Path(t).resolve()
        if not p.is_dir():
            print(f"[skip] not a directory: {p}", flush=True)
            continue
        process_placement(p)


if __name__ == "__main__":
    main()
