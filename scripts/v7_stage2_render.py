#!/usr/bin/env python3
"""V7 Stage 2 — render the 32-frame trajectories stored by Stage 1.

For each placement assigned to this slice:
  1. Read ``<stage1-dir>/<placement>/data.json`` (Stage 1 output;
     contains ``accepted_pairs[].trajectory_32f``).
  2. Look up ``scene_scale`` / ``placement.rotation`` / ``placement.scale``
     from ``<placements-v6-dir>/<placement>.json`` (Stage 1 does not
     serialize these — they live in the original v6 placement JSON).
  3. Open scene + place object via the same helpers Stage 1 used
     (``scripts/v7_sample_pairs_smoke.py``).
  4. For each accepted pair p, for each frame f in ``trajectory_32f``:
     - ``cam.matrix_world = _camera_matrix_from_forward_up(...)``
     - render to ``<out>/<placement>/renders/pair_<p:02d>_frame_<f:02d>.jpg``
  5. Write ``<out>/<placement>/data.json`` (Stage 1 fields + ``render_records``)
     and ``done.flag``.

Run inside Blender:

    blender/blender -b -P scripts/v7_stage2_render.py -- \\
        --stage1-dir outputs/v7_stage1_sample \\
        --placements-v6-dir data/vlm_object_placing_v6_260428_061326 \\
        --assets-root /home/nas1/jungwooahn/projects/DronePhotographer \\
        --out-dir outputs/v7_stage2_renders \\
        --assignment-file splits/v7_stage2_assignments.json \\
        --side jungwooahn --slice-index 0 --slice-count 7 --gpu-index 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse all the heavy lifting from the Stage 1 smoke script.
SMOKE = REPO_ROOT / "scripts" / "v7_sample_pairs_smoke.py"
if not SMOKE.exists():
    raise SystemExit(f"missing dependency: {SMOKE}")
sys.path.insert(0, str(SMOKE.parent))
import v7_sample_pairs_smoke as smoke  # noqa: E402

import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1-dir", default="outputs/v7_stage1_sample")
    p.add_argument(
        "--placements-v6-dir",
        default="data/vlm_object_placing_v6_260428_061326",
        help="Directory of v6 placement JSONs (scene_scale + rotation + scale lookup).",
    )
    p.add_argument(
        "--assets-root",
        default=str(REPO_ROOT),
        help="Base path for resolving scene_file / object_file. "
        "Defaults to the v7 repo root; pass the main DronePhotographer repo "
        "if scenes/objects live there.",
    )
    p.add_argument("--out-dir", default="outputs/v7_stage2_renders")
    p.add_argument(
        "--assignment-file",
        default="splits/v7_stage2_assignments.json",
        help="Manifest produced by v7_stage2_make_split.py.",
    )
    p.add_argument(
        "--side",
        default="jungwooahn",
        help="Which side of the split to consume (key under assignment.sides).",
    )
    p.add_argument("--slice-index", type=int, default=0)
    p.add_argument("--slice-count", type=int, default=1,
                   help="Take placements[i::slice_count] from this side's list.")
    p.add_argument("--only-placement", default=None,
                   help="Render exactly one named placement; ignores slice/pilot.")
    p.add_argument("--pilot-count", type=int, default=0,
                   help="If >0, take the first K placements from this side's list "
                        "before slicing (for end-to-end pilots).")
    p.add_argument("--frames-per-pair", type=int, default=32,
                   help="Number of frames to render per pair. 32 = all of "
                        "trajectory_32f; lower picks evenly-spaced indices "
                        "(endpoints always included).")
    p.add_argument("--render-samples", type=int, default=32)
    p.add_argument("--resolution", nargs=2, type=int, default=[1024, 768],
                   metavar=("W", "H"))
    p.add_argument("--focal-length", type=float, default=24.0)
    p.add_argument("--sensor-width", type=float, default=12.8)
    p.add_argument("--sensor-height", type=float, default=9.6)
    p.add_argument("--sky-strength", type=float, default=0.1)
    p.add_argument("--gpu-index", type=int, default=None,
                   help="Physical GPU index. Omit for CPU.")
    p.add_argument("--resume", action="store_true",
                   help="Skip placements whose <out>/<name>/done.flag exists.")
    return p.parse_args(argv)


def load_assignment(path: Path, side: str) -> list[str]:
    doc = json.loads(path.read_text())
    sides = doc.get("sides", {})
    if side not in sides:
        raise KeyError(f"side '{side}' not in {sorted(sides)} ({path})")
    return list(sides[side]["placements"])


def select_slice(
    names: list[str],
    slice_index: int,
    slice_count: int,
    pilot_count: int,
) -> list[str]:
    if pilot_count and pilot_count > 0:
        names = names[:pilot_count]
    if slice_count <= 1:
        return names
    return names[slice_index::slice_count]


def load_v6_placement(v6_path: Path, placement_idx: int) -> dict:
    """Return {scene_scale, rotation, scale, position} from the v6 placement JSON."""
    doc = json.loads(v6_path.read_text())
    placements = doc.get("placements", [])
    if placement_idx >= len(placements):
        raise IndexError(
            f"{v6_path.name}: placement_idx {placement_idx} out of range "
            f"({len(placements)} placements)"
        )
    chosen = placements[placement_idx]
    return {
        "scene_scale": float(doc.get("scene_scale", 1.0)),
        "position": [float(v) for v in chosen["position"]],
        "rotation": [float(v) for v in chosen.get("rotation", [0.0, 0.0, 0.0])],
        "scale": float(chosen.get("scale", 1.0)),
    }


def build_placement_dict(stage1_data: dict, v6: dict) -> dict:
    """Build the dict expected by smoke.setup_blender_scene()."""
    return {
        "name": stage1_data["placement"],
        "scene_file": stage1_data["scene_file"],
        "object_file": stage1_data["object_file"],
        "scene_scale": v6["scene_scale"],
        "position": np.asarray(v6["position"], dtype=np.float64),
        "rotation": v6["rotation"],
        "scale": v6["scale"],
    }


def frame_indices(num_frames: int, traj_len: int) -> list[int]:
    if num_frames >= traj_len:
        return list(range(traj_len))
    n = max(2, int(num_frames))
    return sorted({int(round(x)) for x in np.linspace(0, traj_len - 1, n)})


def render_one_placement(
    name: str,
    stage1_dir: Path,
    v6_dir: Path,
    assets_root: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """Render every pair × frame in trajectory_32f for one placement."""
    placement_out = out_dir / name
    done_flag = placement_out / "done.flag"
    if args.resume and done_flag.exists():
        return {"name": name, "status": "skip"}

    stage1_data = json.loads((stage1_dir / name / "data.json").read_text())
    v6 = load_v6_placement(v6_dir / f"{name}.json",
                           int(stage1_data.get("placement_idx", 0)))

    placement_out.mkdir(parents=True, exist_ok=True)
    failed_flag = placement_out / "failed.flag"
    if failed_flag.exists():
        failed_flag.unlink()

    placement = build_placement_dict(stage1_data, v6)

    import bpy

    # Clean slate between placements within the same Blender session.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    t0 = time.time()
    meta = smoke.setup_blender_scene(placement, assets_root)
    t_setup = time.time() - t0

    cam_obj = smoke.configure_renderer(
        int(args.resolution[0]),
        int(args.resolution[1]),
        int(args.render_samples),
        focal_length=float(args.focal_length),
        sensor_width=float(args.sensor_width),
        sensor_height=float(args.sensor_height),
        sky_strength=float(args.sky_strength),
        gpu_index=args.gpu_index,
    )

    renders_dir = placement_out / "renders"
    renders_dir.mkdir(exist_ok=True)
    scene = bpy.context.scene

    # Mesh-tight subject bbox projector (deterministic, no GroundingDINO).
    frame_bounds = smoke._get_camera_frame_bounds(scene, cam_obj)
    in_frame_check = smoke.make_in_frame_check(
        meta["subject_verts_world"], frame_bounds,
        (int(args.resolution[0]), int(args.resolution[1])),
    )

    accepted_pairs = stage1_data.get("accepted_pairs", []) or []
    traj_len_default = 32
    if accepted_pairs and accepted_pairs[0].get("trajectory_32f"):
        traj_len_default = len(accepted_pairs[0]["trajectory_32f"])

    render_records: list[list[dict]] = []
    t_render = 0.0
    n_rendered = 0
    for i, pair in enumerate(accepted_pairs):
        traj = pair.get("trajectory_32f") or []
        if not traj:
            render_records.append([])
            continue
        idxs = frame_indices(int(args.frames_per_pair), len(traj))
        pair_recs: list[dict] = []
        for j in idxs:
            frame = traj[j]
            cam_obj.matrix_world = smoke._camera_matrix_from_forward_up(
                frame["pos"], frame["forward"], frame["up"]
            )
            bpy.context.view_layer.update()
            in_frame, occ, bbox = in_frame_check(
                frame["pos"], frame["forward"], frame["up"]
            )
            rel = f"renders/pair_{i:02d}_frame_{j:02d}.jpg"
            scene.render.filepath = str(placement_out / rel)
            ts = time.time()
            bpy.ops.render.render(write_still=True)
            t_render += time.time() - ts
            n_rendered += 1
            pair_recs.append({
                "frame_idx": j,
                "path_rel": rel,
                "bbox_xyxy_full": (
                    [float(v) for v in bbox] if bbox is not None else None
                ),
                "occupancy_clipped": float(occ),
                "in_frame": bool(in_frame),
            })
        render_records.append(pair_recs)

    out_data = dict(stage1_data)
    out_data["render_records"] = render_records
    out_data["render_width"] = int(args.resolution[0])
    out_data["render_height"] = int(args.resolution[1])
    out_data["render_samples"] = int(args.render_samples)
    out_data["render_frames_per_pair"] = int(args.frames_per_pair)
    out_data["stage2_time_setup_s"] = float(t_setup)
    out_data["stage2_time_render_s"] = float(t_render)
    out_data["stage2_n_rendered"] = int(n_rendered)
    out_data["stage2_traj_len"] = int(traj_len_default)

    (placement_out / "data.json").write_text(
        json.dumps(out_data, indent=2), encoding="utf-8"
    )
    done_flag.write_text(
        f"rendered={n_rendered}  setup={t_setup:.2f}s  render={t_render:.2f}s\n",
        encoding="utf-8",
    )
    return {
        "name": name,
        "status": "ok",
        "n_rendered": n_rendered,
        "t_setup": t_setup,
        "t_render": t_render,
    }


def main() -> int:
    args = parse_args()
    stage1_dir = (REPO_ROOT / args.stage1_dir).resolve()
    v6_dir = (REPO_ROOT / args.placements_v6_dir).resolve()
    assets_root = Path(args.assets_root).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only_placement:
        names = [args.only_placement]
    else:
        names = load_assignment(REPO_ROOT / args.assignment_file, args.side)
        names = select_slice(
            names,
            args.slice_index,
            args.slice_count,
            args.pilot_count,
        )

    print(
        f"[stage2] side={args.side} slice={args.slice_index}/{args.slice_count} "
        f"pilot={args.pilot_count} placements={len(names)} "
        f"out={out_dir} gpu={args.gpu_index}"
    )

    t_start = time.time()
    n_ok = n_skip = n_fail = 0
    for k, name in enumerate(names):
        try:
            res = render_one_placement(
                name, stage1_dir, v6_dir, assets_root, out_dir, args
            )
        except Exception:
            tb = traceback.format_exc()
            (out_dir / name).mkdir(parents=True, exist_ok=True)
            (out_dir / name / "failed.flag").write_text(tb, encoding="utf-8")
            print(f"[stage2] FAIL {name}\n{tb}", file=sys.stderr)
            n_fail += 1
            continue
        if res["status"] == "skip":
            n_skip += 1
            print(f"[stage2] {k+1}/{len(names)} skip {name}")
        else:
            n_ok += 1
            print(
                f"[stage2] {k+1}/{len(names)} ok {name} "
                f"frames={res['n_rendered']} "
                f"setup={res['t_setup']:.1f}s render={res['t_render']:.1f}s"
            )

    elapsed = time.time() - t_start
    print(
        f"[stage2] done in {elapsed/60:.1f}min  "
        f"ok={n_ok} skip={n_skip} fail={n_fail}"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
