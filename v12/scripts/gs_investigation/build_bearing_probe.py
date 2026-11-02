"""For each start frame, the same goal prompt with the bearing set to (true start bearing + D),
D in {-60,-30,+30,+60}. A policy that knows where it stands must orbit by sign(-D) every time."""
import json, math, sys
from pathlib import Path
import numpy as np
V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
sys.path.insert(0, V12)
from src.common.facing import front_azimuth, subject_bearing_deg
from src.common.goal_space import DEFAULT_GOAL_KEYS, goal_vector
from src.scoring.projection import cam_to_subject_angles
src, out, every = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
eps = json.load(open(src))["episodes"][::every]
res = []
for e in eps:
    yaw = float(e["goal_raw"].get("placement_yaw_deg", 0.0))
    s = np.array(e["start_pose"]["position"]); c = np.array(e["subject_center"])
    sb = subject_bearing_deg(cam_to_subject_angles(s, c)[0], e["object"], yaw_deg=yaw)
    fe = front_azimuth(e["object"]) + yaw
    for D in (-60, -30, 30, 60):
        f = json.loads(json.dumps(e))
        raw = f["goal_raw"]
        target = (sb + D) % 360
        raw["cam_to_obj_azimuth_deg"] = float(((fe - target) + 180) % 360 - 180)     # bearing = fe - az
        raw["placement_yaw_deg"] = yaw
        gv = goal_vector(raw, DEFAULT_GOAL_KEYS, object_key=e["object"])
        assert abs(((gv[2] - target) + 180) % 360 - 180) < 1e-3, (gv[2], target)
        f["goal_vec"] = [float(x) for x in gv]; f["bearing"] = float(target)
        f["key"] = f"{e['key']}:d{D:+d}"; f["probe_delta"] = D; f["probe_start_bearing"] = float(sb)
        f["traj"] = int(e["traj"]) + {-60: 100, -30: 200, 30: 300, 60: 400}[D]     # unique eval key per variant
        res.append(f)
res.sort(key=lambda x: x["placement"])
(out / "episodes.json").write_text(json.dumps({"kind": "gs", "chunk_size": 8, "episodes": res}, indent=1))
print(len(res), "probe episodes ->", out / "episodes.json")
