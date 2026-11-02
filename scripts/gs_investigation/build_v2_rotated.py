"""v2-on-gray episodes with the WHOLE episode turned by THETA about the subject's foot: subject
rotation_z = THETA, every camera pose rotated by THETA, goal labels recomputed (bearing unchanged by
construction). On a gray background with uniform light this must not change anything; if it does,
something in the 3DGS pipeline is not rotation-correct."""
import json, math, sys
from pathlib import Path
import numpy as np
V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
sys.path.insert(0, V12)
from src.common.goal_space import DEFAULT_GOAL_KEYS, goal_vector
THETA = float(sys.argv[1])
SRC = Path(f"{V12}/runs/bench_gs_probe/v2gray"); OUT = Path(f"{V12}/runs/bench_gs_probe/v2gray_rot{int(THETA)}")
(OUT / "_specs").mkdir(parents=True, exist_ok=True)
a = math.radians(THETA); R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
rot = lambda v: [float(x) for x in R @ np.asarray(v, float)]
doc = json.load(open(SRC / "episodes.json"))
out = []
for e in doc["episodes"]:
    spec = json.load(open(e["spec"])); spec["rotation_z_deg"] = THETA
    sp = OUT / "_specs" / Path(e["spec"]).name; sp.write_text(json.dumps(spec, indent=1))
    f = dict(e); f["spec"] = str(sp)
    for k in ("start_pose", "goal_pose"):
        f[k] = {kk: rot(vv) for kk, vv in e[k].items()}
    f["subject_center"] = rot(e["subject_center"])
    raw = dict(e["goal_raw"])
    raw["cam_to_obj_azimuth_deg"] = (float(raw["cam_to_obj_azimuth_deg"]) + THETA + 180) % 360 - 180
    raw["placement_yaw_deg"] = THETA
    f["goal_raw"] = raw
    gv = goal_vector(raw, DEFAULT_GOAL_KEYS, object_key=e["object"])
    assert abs(((gv[2] - e["goal_vec"][2]) + 180) % 360 - 180) < 0.51, (e["key"], gv[2], e["goal_vec"][2])
    f["goal_vec"] = [float(x) for x in gv]
    out.append(f)
(OUT / "episodes.json").write_text(json.dumps({"kind": "gs", "chunk_size": 8, "episodes": out}, indent=1))
print(len(out), "episodes ->", OUT / "episodes.json")
