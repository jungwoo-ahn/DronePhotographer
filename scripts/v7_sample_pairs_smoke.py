#!/usr/bin/env python3
"""Layer-2 smoke for v7 pair generation.

Runs inside Blender. Loads ONE placement JSON, opens the scene, places the
object, then exercises src.policy.data.sampling on it without rendering any
images. Writes:

  outputs/v7_pair_smoke/<placement_name>.json    — accepted pairs + 32f traj
  outputs/v7_pair_smoke/<placement_name>_3d.png  — matplotlib viz
  outputs/v7_pair_smoke/<placement_name>_radius.csv

Usage:
  blender/blender -b -P scripts/v7_sample_pairs_smoke.py -- \
      --placement-json data/vlm_object_placing_v6_260428_061326/<file>.json \
      --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.policy.data import sampling as S  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Argparse against the args after the Blender `--` separator."""
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser(description="v7 pair-sampling Blender smoke")
    p.add_argument(
        "--placement-json",
        default=None,
        help="Path to a single placement JSON. Defaults to the first one in "
        "data/vlm_object_placing_v6_260428_061326/.",
    )
    p.add_argument("--placement-idx", type=int, default=0,
                   help="Which entry in placements[] to use (default 0).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "outputs" / "v7_pair_smoke"),
    )
    p.add_argument("--assets-root", default=str(REPO_ROOT),
                   help="Base path for resolving relative scene_file/object_file.")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip matplotlib 3D viz PNG.")
    p.add_argument("--render", dest="render", action="store_true", default=True,
                   help="Render each trajectory frame to JPEG (default on).")
    p.add_argument("--no-render", dest="render", action="store_false",
                   help="Skip rendering; report will have no thumbnails.")
    p.add_argument("--render-width", type=int, default=640)
    p.add_argument("--render-height", type=int, default=480)
    p.add_argument("--render-samples", type=int, default=32,
                   help="Cycles samples per render (default 32; project default 64).")
    p.add_argument("--render-stride", type=int, default=1,
                   help="Render every Nth frame (default 1 = all 32). "
                        "Ignored if --render-num-frames is set.")
    p.add_argument("--render-num-frames", type=int, default=None,
                   help="Render exactly this many frames per clip, evenly "
                        "spaced via linspace(0, N_FRAMES-1, num). Endpoints "
                        "are always included. Overrides --render-stride.")
    p.add_argument("--gpu-index", type=int, default=None,
                   help="Use only this physical GPU index for rendering. "
                        "Leave unset to fall back to CPU (single-GPU policy).")
    p.add_argument("--focal-length", type=float, default=24.0,
                   help="Camera focal length in mm (project default 24).")
    p.add_argument("--sensor-width", type=float, default=12.8,
                   help="Camera sensor width in mm (project default 12.8).")
    p.add_argument("--sensor-height", type=float, default=9.6,
                   help="Camera sensor height in mm (project default 9.6).")
    p.add_argument("--sky-strength", type=float, default=0.1,
                   help="Nishita sky strength (project default 0.1).")
    p.add_argument("--min-occupancy", type=float, default=0.01,
                   help="Reject pair if start OR end frame has subject bbox "
                        "occupancy < this (default 0.01 = 1%). Midpoints not checked.")
    return p.parse_args(argv)


def default_placement_json() -> Path:
    d = REPO_ROOT / "data" / "vlm_object_placing_v6_260428_061326"
    candidates = sorted(d.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No placement JSONs in {d}")
    return candidates[0]


def load_placement(path: Path, idx: int) -> dict:
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    placements = doc.get("placements", [])
    if not placements:
        raise ValueError(f"{path.name} has no placements")
    if idx >= len(placements):
        raise IndexError(
            f"{path.name} has {len(placements)} placement(s); idx {idx} out of range"
        )
    chosen = placements[idx]
    if not chosen.get("accepted", True):
        print(f"[smoke] note: placements[{idx}].accepted is False — proceeding anyway")
    return {
        "name": path.stem,
        "scene_file": doc["scene_file"],
        "object_file": doc["object_file"],
        "scene_scale": float(doc.get("scene_scale", 1.0)),
        "position": np.asarray(chosen["position"], dtype=np.float64),
        "rotation": [float(v) for v in chosen.get("rotation", [0.0, 0.0, 0.0])],
        "scale": float(chosen.get("scale", 1.0)),
    }


def setup_blender_scene(
    p: dict,
    assets_root: Path,
    *,
    max_verts: int = 5000,
) -> dict:
    """Open scene, scale it, place the object. Mirrors render_v5 pipeline.

    Returns metadata: imported subject object names + axis-aligned bbox center
    (the cinematic "subject center" we orbit around) + bbox z extent, and
    a cached numpy array of subject mesh vertices in world space
    (subsampled to ``max_verts``) for tight 2D-bbox projection in the
    in-frame visibility check.
    """
    import bpy
    from mathutils import Vector

    from src.scenes.scene import open_scene
    from render_object import apply_scene_scale, place_imported_object

    scene_file = assets_root / p["scene_file"]
    object_file = assets_root / p["object_file"]
    scene = open_scene(str(scene_file))
    apply_scene_scale(scene, p["scene_scale"])

    pre = set(o.name for o in bpy.data.objects)
    place_imported_object(
        scene,
        str(object_file),
        position=p["position"].tolist(),
        rotation_xyz_rad=p["rotation"],
        scale=p["scale"],
    )
    post = set(o.name for o in bpy.data.objects)
    subject_names = sorted(post - pre)

    bbox_min = np.full(3, np.inf)
    bbox_max = np.full(3, -np.inf)
    vert_chunks: list[np.ndarray] = []
    for name in subject_names:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            wc = obj.matrix_world @ Vector(corner)
            v = np.array([wc.x, wc.y, wc.z])
            bbox_min = np.minimum(bbox_min, v)
            bbox_max = np.maximum(bbox_max, v)
        mw = np.array(obj.matrix_world, dtype=np.float64)        # (4, 4)
        local = np.array([v.co[:] for v in obj.data.vertices], dtype=np.float64)
        if local.size == 0:
            continue
        homog = np.concatenate(
            [local, np.ones((local.shape[0], 1), dtype=np.float64)], axis=1
        )
        world = (homog @ mw.T)[:, :3]                            # (N, 3)
        vert_chunks.append(world)

    if vert_chunks:
        verts = np.concatenate(vert_chunks, axis=0)
        if verts.shape[0] > max_verts:
            idx = np.linspace(0, verts.shape[0] - 1, max_verts).astype(np.int64)
            verts = verts[idx]
    else:
        verts = np.zeros((0, 3), dtype=np.float64)

    if np.isfinite(bbox_min).all():
        subject_center = ((bbox_min + bbox_max) / 2.0).astype(np.float64)
        subject_height = float(bbox_max[2] - bbox_min[2])
    else:
        subject_center = p["position"].astype(np.float64)
        subject_height = 1.7
    return {
        "subject_names": subject_names,
        "subject_center": subject_center,
        "subject_height": subject_height,
        "subject_foot": p["position"].astype(np.float64),
        "subject_verts_world": verts,
    }


def build_is_valid(O: np.ndarray, ignore_names: list[str]):
    """Return a closure mapping cam_pos (np.array) → bool via Blender ray-cast.

    Floor-below check is DISABLED (we already bound camera height to a sane
    range above the placement floor at sample time, so `no_floor_below` was
    rejecting useful poses inside narrow/open scenes where the ground mesh
    doesn't extend under the entire orbit).

    Also exposes `is_valid.diagnose(pos) -> reason_str` for failed positions.
    """
    import bpy
    from mathutils import Vector

    from src.blender.camera import is_camera_valid

    target = Vector(O.tolist())
    ignore_set = set(ignore_names)

    def is_valid(pos: np.ndarray) -> bool:
        return bool(
            is_camera_valid(
                Vector(pos.tolist()), target,
                ignore_names=ignore_set,
                check_floor=False,
            )
        )

    def diagnose(pos: np.ndarray) -> str:
        cam_pos = Vector(pos.tolist())
        scene = bpy.context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        test_dirs = [
            Vector((1, 0, 0)), Vector((-1, 0, 0)),
            Vector((0, 1, 0)), Vector((0, -1, 0)),
            Vector((0, 0, 1)),
            Vector((1, 1, 0)).normalized(), Vector((-1, 1, 0)).normalized(),
            Vector((1, -1, 0)).normalized(), Vector((-1, -1, 0)).normalized(),
            Vector((1, 0, 1)).normalized(), Vector((-1, 0, 1)).normalized(),
            Vector((0, 1, 1)).normalized(), Vector((0, -1, 1)).normalized(),
        ]
        close_hits = 0
        for d in test_dirs:
            hit, loc, _, _, _, _ = scene.ray_cast(depsgraph, cam_pos, d)
            if hit and (loc - cam_pos).length < 0.8:
                close_hits += 1
        if close_hits >= 4:
            return f"occlusion_sphere({close_hits}/14)"
        direction = cam_pos - target
        distance = direction.length
        if distance < 0.01:
            return "at_target"
        direction_norm = direction.normalized()
        hit, loc, _, _, obj, _ = scene.ray_cast(depsgraph, target, direction_norm)
        if not hit:
            return "ok_but_outer_returned_false"
        hit_distance = (loc - target).length
        if hit_distance < distance - 0.8:
            obj_name = obj.name if obj else "?"
            if obj_name in ignore_set:
                return "ok_but_outer_returned_false"
            return f"line_of_sight_blocked_by:{obj_name}@{hit_distance:.2f}m_vs_{distance:.2f}m"
        return "ok_but_outer_returned_false"

    is_valid.diagnose = diagnose
    return is_valid


def try_pair(
    O: np.ndarray,
    floor_z: float,
    rng: np.random.Generator,
    is_valid,
    in_frame_check=None,
    min_occupancy: float = 0.0,
    debug=None,
) -> tuple[dict | None, str | None]:
    s = S.sample_pose(O, rng, floor_z=floor_z)
    e = S.sample_pose(O, rng, floor_z=floor_z)
    if s is None or e is None:
        return None, "singular_lookat"
    if not is_valid(s["pos"]):
        if debug is not None and hasattr(is_valid, "diagnose"):
            d = is_valid.diagnose(s["pos"])
            debug[d] = debug.get(d, 0) + 1
        return None, "start_invalid"
    if not is_valid(e["pos"]):
        if debug is not None and hasattr(is_valid, "diagnose"):
            d = is_valid.diagnose(e["pos"])
            debug[d] = debug.get(d, 0) + 1
        return None, "end_invalid"
    sep = S.spherical_angle(s["pos"] - O, e["pos"] - O)
    if sep < np.deg2rad(S.MIN_ANG_SEP_DEG):
        return None, "ang_sep_too_small"
    if in_frame_check is not None:
        s_pos, s_fwd, s_up = S.pose_after_jitter(s)
        ok, occ_s, _ = in_frame_check(s_pos, s_fwd, s_up)
        s["bbox_occupancy"] = occ_s
        if not ok or occ_s < min_occupancy:
            if debug is not None:
                key = f"start_off_frame(occ={occ_s:.3f})"
                debug[key] = debug.get(key, 0) + 1
            return None, "start_off_frame"
        e_pos, e_fwd, e_up = S.pose_after_jitter(e)
        ok, occ_e, _ = in_frame_check(e_pos, e_fwd, e_up)
        e["bbox_occupancy"] = occ_e
        if not ok or occ_e < min_occupancy:
            if debug is not None:
                key = f"end_off_frame(occ={occ_e:.3f})"
                debug[key] = debug.get(key, 0) + 1
            return None, "end_off_frame"
    if s["r"] >= e["r"]:
        C_far, C_near = s["pos"], e["pos"]
        c_far_is_start = True
    else:
        C_far, C_near = e["pos"], s["pos"]
        c_far_is_start = False
    try:
        E = S.solve_ellipse(O, C_far, C_near)
    except S.DegenerateEllipse as exc:
        return None, f"degenerate:{exc}"
    for t in S.MIDPOINT_TS:
        theta = (1.0 - t) * E["theta_far"] + t * E["theta_near"]
        if not is_valid(S.ellipse_at(E, float(theta))):
            return None, "midpoint_invalid"
    return {
        "start": s,
        "end": e,
        "C_far": C_far,
        "C_near": C_near,
        "c_far_is_start": c_far_is_start,
        "ellipse": E,
    }, None


def sample_placement(
    O: np.ndarray,
    floor_z: float,
    is_valid,
    rng: np.random.Generator,
    in_frame_check=None,
    min_occupancy: float = 0.0,
):
    accepted = []
    rejections: dict[str, int] = {}
    sub_reasons: dict[str, int] = {}
    attempts = 0
    for _ in range(S.MAX_ATTEMPTS):
        attempts += 1
        if len(accepted) >= S.K_CLIPS_PER_PLACEMENT:
            break
        pair, reason = try_pair(
            O, floor_z, rng, is_valid,
            in_frame_check=in_frame_check, min_occupancy=min_occupancy,
            debug=sub_reasons,
        )
        if pair is not None:
            accepted.append(pair)
        else:
            rejections[reason] = rejections.get(reason, 0) + 1
    return accepted, rejections, attempts, sub_reasons


def _jitter_endpoints_for_pair(pair: dict) -> tuple[float, float, float, float]:
    """Map start/end pose jitter to (pitch_far, pitch_near, yaw_far, yaw_near).

    trajectory_frames lerps from C_far at frame 0 to C_near at frame N-1.
    """
    s_pitch = float(pair["start"].get("pitch_jitter_deg", 0.0))
    e_pitch = float(pair["end"].get("pitch_jitter_deg", 0.0))
    s_yaw = float(pair["start"].get("yaw_jitter_deg", 0.0))
    e_yaw = float(pair["end"].get("yaw_jitter_deg", 0.0))
    if pair["c_far_is_start"]:
        return s_pitch, e_pitch, s_yaw, e_yaw
    return e_pitch, s_pitch, e_yaw, s_yaw


def _pitch_endpoints_for_pair(pair: dict) -> tuple[float, float]:
    p_far, p_near, _, _ = _jitter_endpoints_for_pair(pair)
    return p_far, p_near


def serialize_pair(pair: dict) -> dict:
    """Convert numpy + frames into JSON-friendly dict."""
    E = pair["ellipse"]
    pitch_far_deg, pitch_near_deg, yaw_far_deg, yaw_near_deg = (
        _jitter_endpoints_for_pair(pair)
    )
    frames = S.trajectory_frames(
        E,
        n=S.N_FRAMES,
        pitch_start_deg=pitch_far_deg,
        pitch_end_deg=pitch_near_deg,
        yaw_start_deg=yaw_far_deg,
        yaw_end_deg=yaw_near_deg,
    )
    if frames is None:
        frame_list = None
    else:
        def _alpha(i: int) -> float:
            return (i / (S.N_FRAMES - 1)) if S.N_FRAMES > 1 else 0.0

        frame_list = [
            {
                "pos": [float(x) for x in pos],
                "forward": [float(x) for x in fwd],
                "up": [float(x) for x in up],
                "pitch_deg": float(
                    (1.0 - _alpha(idx)) * pitch_far_deg
                    + _alpha(idx) * pitch_near_deg
                ),
                "yaw_deg": float(
                    (1.0 - _alpha(idx)) * yaw_far_deg
                    + _alpha(idx) * yaw_near_deg
                ),
            }
            for idx, (pos, fwd, up) in enumerate(frames)
        ]

    def _pose(p: dict) -> dict:
        return {
            "pos": [float(x) for x in p["pos"]],
            "forward": [float(x) for x in p["forward"]],
            "up": [float(x) for x in p["up"]],
            "r": p["r"],
            "az_deg": float(np.rad2deg(p["az"])),
            "elev_deg": float(np.rad2deg(p["elev"])),
            "pitch_jitter_deg": float(p.get("pitch_jitter_deg", 0.0)),
            "yaw_jitter_deg": float(p.get("yaw_jitter_deg", 0.0)),
            "bbox_occupancy": float(p.get("bbox_occupancy", 0.0)),
        }

    return {
        "start": _pose(pair["start"]),
        "end": _pose(pair["end"]),
        "C_far": [float(x) for x in pair["C_far"]],
        "C_near": [float(x) for x in pair["C_near"]],
        "c_far_is_start": pair["c_far_is_start"],
        "pitch_far_deg": float(pitch_far_deg),
        "pitch_near_deg": float(pitch_near_deg),
        "yaw_far_deg": float(yaw_far_deg),
        "yaw_near_deg": float(yaw_near_deg),
        "ellipse": {
            "a": float(E["a"]),
            "b": float(E["b"]),
            "theta_far_deg": float(np.rad2deg(E["theta_far"])),
            "theta_near_deg": float(np.rad2deg(E["theta_near"])),
            "u": [float(x) for x in E["u"]],
            "v": [float(x) for x in E["v"]],
            "O": [float(x) for x in E["O"]],
        },
        "trajectory_32f": frame_list,
    }


def _camera_matrix_from_forward_up(pos, forward, up):
    """Blender camera world matrix: local -Z = forward, +Y = up, +X = right."""
    from mathutils import Matrix, Vector
    fwd = np.asarray(forward, dtype=np.float64)
    fwd = fwd / np.linalg.norm(fwd)
    upn = np.asarray(up, dtype=np.float64)
    upn = upn / np.linalg.norm(upn)
    right = np.cross(fwd, upn)
    right = right / np.linalg.norm(right)
    upn = np.cross(right, fwd)
    upn = upn / np.linalg.norm(upn)
    rot = Matrix((
        (float(right[0]), float(upn[0]), float(-fwd[0])),
        (float(right[1]), float(upn[1]), float(-fwd[1])),
        (float(right[2]), float(upn[2]), float(-fwd[2])),
    ))
    return Matrix.Translation(Vector([float(p) for p in pos])) @ rot.to_4x4()


def configure_renderer(
    width: int,
    height: int,
    samples: int,
    *,
    focal_length: float,
    sensor_width: float,
    sensor_height: float,
    sky_strength: float,
    gpu_index: int | None = None,
) -> "bpy.types.Object":  # type: ignore[name-defined]
    """Create + register a camera, configure Cycles for fast smoke renders.

    Camera lens/sensor are set to project defaults so framing matches what the
    placement search saw. Nishita sky is applied per ``sky_strength``.

    If ``gpu_index`` is given, restrict Cycles to that one physical GPU.
    Otherwise fall back to CPU.
    """
    import bpy
    scene = bpy.context.scene

    cam_data = bpy.data.cameras.new("V7SmokeCam")
    cam_data.lens = float(focal_length)
    cam_data.sensor_width = float(sensor_width)
    cam_data.sensor_height = float(sensor_height)
    cam_obj = bpy.data.objects.new("V7SmokeCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 82
    scene.render.film_transparent = False
    scene.render.use_persistent_data = True

    cycles = scene.cycles
    cycles.samples = int(samples)
    cycles.max_bounces = 4
    cycles.diffuse_bounces = 2
    cycles.glossy_bounces = 2
    cycles.transmission_bounces = 2
    if hasattr(cycles, "use_denoising"):
        cycles.use_denoising = True

    if sky_strength > 0:
        try:
            from src.scenes.scene import set_nishita_sky
            set_nishita_sky(float(sky_strength))
        except Exception as exc:
            print(f"[smoke] Nishita sky setup failed: {exc}")

    if gpu_index is None:
        cycles.device = "CPU"
        print("[smoke] render device: CPU (no --gpu-index)")
        return cam_obj

    cycles_addon = bpy.context.preferences.addons.get("cycles")
    if cycles_addon is None:
        cycles.device = "CPU"
        print("[smoke] render device: CPU (cycles addon missing)")
        return cam_obj
    prefs = cycles_addon.preferences

    chosen_label = None
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
        except Exception:
            continue
        backend_devs = [d for d in prefs.devices if d.type == backend]
        if not backend_devs:
            continue
        if gpu_index >= len(backend_devs):
            print(f"[smoke] --gpu-index {gpu_index} >= {len(backend_devs)} {backend} "
                  "devices; trying next backend")
            continue
        for d in prefs.devices:
            d.use = False
        backend_devs[gpu_index].use = True
        cycles.device = "GPU"
        chosen_label = f"{backend}:{gpu_index} ({backend_devs[gpu_index].name})"
        break

    if chosen_label is None:
        cycles.device = "CPU"
        print("[smoke] render device: CPU (no usable GPU backend)")
    else:
        print(f"[smoke] render device: {chosen_label}")
    return cam_obj


def _get_camera_frame_bounds(scene, cam_obj):
    """Return (min_x, max_x, min_y, max_y) of the cam frame normalized to z=1.

    ``cam_data.view_frame()`` returns 4 corners at Blender's view depth; we
    rescale to z=1 so per-vertex projection ``fmin_x = min_x * z_ray`` is
    correct under perspective.
    """
    frame = cam_obj.data.view_frame(scene=scene)
    view_z = abs(frame[0].z)
    if view_z < 1e-9:
        view_z = 1.0
    xs = [f.x / view_z for f in frame]
    ys = [f.y / view_z for f in frame]
    return float(min(xs)), float(max(xs)), float(min(ys)), float(max(ys))


def _camera_basis_np(forward, up):
    """Right/up/-forward axes (Blender camera local: +X=right, +Y=up, -Z=fwd)."""
    fwd = np.asarray(forward, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    upv = np.asarray(up, dtype=np.float64)
    upv /= np.linalg.norm(upv)
    nz = -fwd
    right = np.cross(upv, nz)
    right /= np.linalg.norm(right)
    ortho_up = np.cross(nz, right)
    ortho_up /= np.linalg.norm(ortho_up)
    return right, ortho_up, nz


def make_in_frame_check(verts_world, frame_bounds, resolution):
    """Closure: (pos, fwd, up) -> (in_frame: bool, occupancy: float, bbox or None).

    Vectorized perspective projection of the subject's actual mesh vertices
    (subsampled to ~5k) — NOT the 8-corner bound_box. Computes a tight 2D
    AABB, clips it to the image, and returns occupancy = clipped_area /
    (W*H). Reject (False, 0, None) when:
      - no vertices land in front of the camera (subject fully behind us); OR
      - the projected bbox is entirely off-image.
    """
    W, H = int(resolution[0]), int(resolution[1])
    min_x, max_x, min_y, max_y = frame_bounds
    verts = np.asarray(verts_world, dtype=np.float64)
    if verts.shape[0] == 0:
        def check_empty(pos, fwd, up):
            return False, 0.0, None
        return check_empty

    def check(pos, fwd, up):
        right, upv, nz = _camera_basis_np(fwd, up)
        cam_pos = np.asarray(pos, dtype=np.float64)
        rel = verts - cam_pos                                    # (N, 3)
        co_x = rel @ right                                       # (N,)
        co_y = rel @ upv
        co_z = rel @ nz                                          # +Z = behind
        z = -co_z                                                # forward dist
        valid = z > 1e-6
        if not np.any(valid):
            return False, 0.0, None
        co_x = co_x[valid]; co_y = co_y[valid]; z = z[valid]
        fmin_x = min_x * z; fmax_x = max_x * z
        fmin_y = min_y * z; fmax_y = max_y * z
        ndc_x = (co_x - fmin_x) / (fmax_x - fmin_x)
        ndc_y = (co_y - fmin_y) / (fmax_y - fmin_y)
        x_px = ndc_x * float(W)
        y_px = (1.0 - ndc_y) * float(H)
        xmin = float(x_px.min()); xmax = float(x_px.max())
        ymin = float(y_px.min()); ymax = float(y_px.max())
        # clip to image
        cx1 = max(0.0, xmin); cx2 = min(float(W), xmax)
        cy1 = max(0.0, ymin); cy2 = min(float(H), ymax)
        if cx2 <= cx1 or cy2 <= cy1:
            return False, 0.0, [xmin, ymin, xmax, ymax]
        occ = (cx2 - cx1) * (cy2 - cy1) / (float(W) * float(H))
        return True, float(occ), [xmin, ymin, xmax, ymax]

    return check


def render_trajectories(
    cam_obj,
    accepted: list[dict],
    out_dir: Path,
    *,
    stride: int = 1,
    num_frames: int | None = None,
) -> list[list[dict]]:
    """Render trajectory frames for every accepted pair.

    If ``num_frames`` is set, pick ``num_frames`` indices via
    ``np.linspace(0, N_FRAMES-1, num_frames)`` so both endpoints (0 and
    N_FRAMES-1) are always included. Otherwise fall back to ``stride``.

    Returns frame_records[pair_idx] = list of {frame_idx, path_rel}.
    """
    import bpy
    renders_dir = out_dir / "renders"
    renders_dir.mkdir(exist_ok=True)
    scene = bpy.context.scene

    if num_frames is not None and num_frames > 0:
        n = max(2, int(num_frames))
        idxs = sorted(set(
            int(round(x)) for x in np.linspace(0, S.N_FRAMES - 1, n)
        ))
    else:
        step = max(1, int(stride))
        idxs = list(range(0, S.N_FRAMES, step))
        if (S.N_FRAMES - 1) not in idxs:
            idxs.append(S.N_FRAMES - 1)

    records: list[list[dict]] = []
    t_total = 0.0
    n_rendered = 0
    for i, pair in enumerate(accepted):
        pitch_far_deg, pitch_near_deg, yaw_far_deg, yaw_near_deg = (
            _jitter_endpoints_for_pair(pair)
        )
        frames = S.trajectory_frames(
            pair["ellipse"],
            n=S.N_FRAMES,
            pitch_start_deg=pitch_far_deg,
            pitch_end_deg=pitch_near_deg,
            yaw_start_deg=yaw_far_deg,
            yaw_end_deg=yaw_near_deg,
        )
        if frames is None:
            records.append([])
            continue
        pair_recs: list[dict] = []
        for j in idxs:
            pos, fwd, up = frames[j]
            cam_obj.matrix_world = _camera_matrix_from_forward_up(pos, fwd, up)
            bpy.context.view_layer.update()
            rel = f"renders/pair_{i:02d}_frame_{j:02d}.jpg"
            out_path = out_dir / rel
            scene.render.filepath = str(out_path)
            t0 = time.time()
            bpy.ops.render.render(write_still=True)
            t_total += time.time() - t0
            n_rendered += 1
            pair_recs.append({"frame_idx": j, "path_rel": rel})
        records.append(pair_recs)
    print(f"[smoke] rendered {n_rendered} frames in {t_total:.1f}s "
          f"({t_total / max(1, n_rendered):.2f}s/frame)")
    return records


def write_radius_csv(path: Path, pairs: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_idx", "r_start", "r_end", "a", "b"])
        for i, p in enumerate(pairs):
            E = p["ellipse"]
            w.writerow([
                i,
                f"{p['start']['r']:.4f}",
                f"{p['end']['r']:.4f}",
                f"{E['a']:.4f}",
                f"{E['b']:.4f}",
            ])


def draw_3d_viz(path: Path, O: np.ndarray, pairs: list[dict]) -> bool:
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as exc:
        print(f"[smoke] matplotlib unavailable ({exc}); skipping 3d viz")
        return False

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([O[0]], [O[1]], [O[2]], c="red", s=80, marker="*", label="subject O")
    cmap = plt.get_cmap("hsv")
    for i, p in enumerate(pairs):
        E = p["ellipse"]
        thetas = np.linspace(0.0, 2.0 * np.pi, 200)
        pts = np.array([S.ellipse_at(E, float(t)) for t in thetas])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=cmap(i / max(1, len(pairs))), alpha=0.35, lw=0.8)
        arc_t = np.linspace(E["theta_far"], E["theta_near"], 32)
        arc = np.array([S.ellipse_at(E, float(t)) for t in arc_t])
        ax.plot(arc[:, 0], arc[:, 1], arc[:, 2],
                color=cmap(i / max(1, len(pairs))), alpha=1.0, lw=2.0)
        s_pos = p["start"]["pos"]
        e_pos = p["end"]["pos"]
        ax.scatter([s_pos[0]], [s_pos[1]], [s_pos[2]],
                   color=cmap(i / max(1, len(pairs))), marker="o", s=30)
        ax.scatter([e_pos[0]], [e_pos[1]], [e_pos[2]],
                   color=cmap(i / max(1, len(pairs))), marker="x", s=30)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"{path.stem}\n{len(pairs)} accepted trajectories")
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    placement_path = Path(args.placement_json) if args.placement_json else default_placement_json()
    assets_root = Path(args.assets_root)
    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    placement = load_placement(placement_path, args.placement_idx)
    out_dir = base_out_dir / placement["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] placement={placement['name']} idx={args.placement_idx} "
          f"foot={placement['position'].tolist()}")

    t0 = time.time()
    meta = setup_blender_scene(placement, assets_root)
    t_setup = time.time() - t0
    print(
        f"[smoke] scene setup: {t_setup:.2f}s "
        f"subject_names={meta['subject_names']} "
        f"center={meta['subject_center'].tolist()} "
        f"h={meta['subject_height']:.2f}m"
    )

    O = meta["subject_center"]
    floor_z = float(meta["subject_foot"][2])   # subject's feet sit on the floor
    is_valid = build_is_valid(O, ignore_names=meta["subject_names"])

    # Set up camera + render settings BEFORE sampling, so the in-frame check
    # can reuse the same scene.camera. If --no-render, we still need the camera
    # for the visibility check; just keep gpu_index=None so Cycles stays on CPU
    # (we won't call render.render anyway).
    cam_obj = configure_renderer(
        args.render_width,
        args.render_height,
        args.render_samples,
        focal_length=args.focal_length,
        sensor_width=args.sensor_width,
        sensor_height=args.sensor_height,
        sky_strength=args.sky_strength,
        gpu_index=args.gpu_index if args.render else None,
    )
    # Build the subject-mesh in-frame check (endpoint-only per design).
    # Uses vectorized mesh-vertex projection (tight 2D AABB), not the loose
    # 8-corner bound_box that `project_bbox_2d` would give.
    import bpy
    frame_bounds = _get_camera_frame_bounds(bpy.context.scene, cam_obj)
    in_frame_check = make_in_frame_check(
        meta["subject_verts_world"],
        frame_bounds,
        (args.render_width, args.render_height),
    )

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    accepted, rejections, attempts, sub_reasons = sample_placement(
        O, floor_z, is_valid, rng,
        in_frame_check=in_frame_check,
        min_occupancy=args.min_occupancy,
    )
    t_sample = time.time() - t0
    K = S.K_CLIPS_PER_PLACEMENT

    serialized = [serialize_pair(p) for p in accepted]
    r_starts = [p["start"]["r"] for p in accepted]
    r_ends = [p["end"]["r"] for p in accepted]
    theta_near_degs = [float(np.rad2deg(p["ellipse"]["theta_near"])) for p in accepted]

    render_records: list[list[dict]] = []
    t_render = 0.0
    if args.render and accepted:
        t0 = time.time()
        render_records = render_trajectories(
            cam_obj, accepted, out_dir,
            stride=args.render_stride,
            num_frames=args.render_num_frames,
        )
        t_render = time.time() - t0

    json_path = out_dir / "data.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "placement": placement["name"],
                "placement_idx": args.placement_idx,
                "scene_file": placement["scene_file"],
                "object_file": placement["object_file"],
                "subject_foot": placement["position"].tolist(),
                "subject_center": O.tolist(),
                "subject_height": meta["subject_height"],
                "subject_names": meta["subject_names"],
                "seed": args.seed,
                "K_target": K,
                "K_accepted": len(accepted),
                "attempts_used": attempts,
                "time_sample_s": float(t_sample),
                "time_setup_s": float(t_setup),
                "time_render_s": float(t_render),
                "rejections_by_reason": rejections,
                "sub_reasons": sub_reasons,
                "accepted_pairs": serialized,
                "render_records": render_records,
                "render_width": args.render_width if args.render else None,
                "render_height": args.render_height if args.render else None,
                "render_samples": args.render_samples if args.render else None,
                "render_stride": args.render_stride if args.render else None,
                "render_num_frames": args.render_num_frames if args.render else None,
            },
            f,
            indent=2,
        )

    csv_path = out_dir / "radius.csv"
    write_radius_csv(csv_path, accepted)

    viz_ok = False
    if not args.no_viz and accepted:
        viz_path = out_dir / "3d.png"
        viz_ok = draw_3d_viz(viz_path, O, accepted)

    report_path: Path | None = None
    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from make_v7_pair_smoke_report import build_report  # type: ignore
        report_path = build_report(out_dir)
    except Exception as exc:
        print(f"[smoke] report builder failed: {exc}")

    r_range = (
        (min(min(r_starts), min(r_ends)), max(max(r_starts), max(r_ends)))
        if accepted
        else (0.0, 0.0)
    )
    theta_range = (
        (min(theta_near_degs), max(theta_near_degs)) if accepted else (0.0, 0.0)
    )
    print(
        f"[smoke] placement={placement['name']} "
        f"accepted={len(accepted)}/{K} attempts={attempts} "
        f"acceptance_rate={(len(accepted) / max(1, attempts)):.2f} "
        f"r_range=[{r_range[0]:.2f}, {r_range[1]:.2f}] "
        f"theta_near_deg_range=[{theta_range[0]:.1f}, {theta_range[1]:.1f}] "
        f"viz={'yes' if viz_ok else 'no'}"
    )
    print(f"[smoke] rejections={rejections}")
    if sub_reasons:
        print(f"[smoke] sub_reasons={sub_reasons}")
    files = [json_path.name, csv_path.name]
    if viz_ok:
        files.append("3d.png")
    if report_path is not None:
        files.append(report_path.name)
    if render_records:
        files.append(f"renders/ ({sum(len(r) for r in render_records)} jpg)")
    print(f"[smoke] dir={out_dir} files={files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
