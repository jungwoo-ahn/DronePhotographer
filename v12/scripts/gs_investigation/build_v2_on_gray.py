"""Diagnostic: benchmark v2 episodes replayed through the 3DGS compositor with a gray background.
Poses are moved into each subject's local frame (foot at origin; v2 subjects are unrotated); the goal
labels are v2's own (translation-invariant). Only the first chunk is used (orbit direction)."""
import ast, json, sys
from pathlib import Path
import numpy as np
V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
sys.path.insert(0, V12)
from src.common.annotations import iter_goal_start_windows
OUT = Path(f"{V12}/runs/bench_gs_probe/v2gray"); (OUT / "_specs").mkdir(parents=True, exist_ok=True)
tmpl = json.load(open(f"{V12}/runs/bench_gs/_specs/garden__00_Redhead-Girl-in-а-red-dress.json"))
man = json.load(open(f"{V12}/configs/eval_episodes_v2.json"))
eps = []
for ep in man["episodes"]:
    pl = ep["placement"]; obj = pl.split("__", 1)[1]
    ri = json.load(open(f"{V12}/runs/bench_v2/_run_info/{pl}.run_info.json"))
    opts = ri["options"] if isinstance(ri["options"], dict) else ast.literal_eval(ri["options"])
    assert all(abs(v) < 1e-9 for v in ri["rotation_xyz_rad"]), pl
    p0 = np.array(opts["object_position"], float)
    spec_path = OUT / "_specs" / f"{pl}.json"
    if not spec_path.exists():
        spec = dict(tmpl); spec.update({"placement": pl, "scene": pl.split("__")[0], "object": obj,
                                         "blend_dir": str(Path(ri["input_object"]).parent), "rotation_z_deg": 0.0})
        spec_path.write_text(json.dumps(spec, indent=1))
    w = next(w for w in iter_goal_start_windows(ep["data_json"], chunk_size=8, max_per_pair=2)
             if (w.pair_idx, w.start_frame_idx, w.goal_frame.frame_idx) == (ep["pair_idx"], ep["start_frame_idx"], ep["goal_frame_idx"]))
    loc = lambda f: {"position": [float(x) for x in np.asarray(f.camera_position) - p0], "forward": [float(x) for x in f.camera_forward], "up": [float(x) for x in f.camera_up]}
    eps.append({"key": ep["key"], "placement": pl, "scene": pl.split("__")[0], "object": obj, "spec": str(spec_path),
                "traj": ep["pair_idx"], "start_frame_idx": ep["start_frame_idx"], "goal_frame_idx": ep["goal_frame_idx"],
                "start_pose": loc(w.start), "goal_pose": loc(w.goal_frame),
                "subject_center": [float(x) for x in np.asarray(w.goal_frame.subject_center) - p0],
                "subject_height": float(w.goal_frame.subject_height), "goal_raw": dict(w.goal_frame.raw),
                "goal_vec": ep["goal_vec"] if "goal_vec" in ep else None, "goal_keys": ep.get("goal_keys"),
                "d_start": ep["d_start"], "bearing": ep["bearing"], "sector": ep["sector"], "delta": ep["delta"],
                "start_image": w.start.image, "goal_image": w.goal_frame.image})
from src.common.goal_space import DEFAULT_GOAL_KEYS, goal_vector
for e in eps:
    e["goal_keys"] = list(DEFAULT_GOAL_KEYS)
    e["goal_vec"] = [float(x) for x in goal_vector(e["goal_raw"], DEFAULT_GOAL_KEYS, object_key=e["object"])]
(OUT / "episodes.json").write_text(json.dumps({"kind": "gs", "chunk_size": 8, "episodes": eps}, indent=1))
print(len(eps), "episodes ->", OUT / "episodes.json")
