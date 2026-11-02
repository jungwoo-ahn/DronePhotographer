import json, math, os, re
import numpy as np
from collections import defaultdict
V = "/home/nas_main/jungwooahn/projects/DronePhotographerV12/runs/bench_gs_probe"
def ang(p, c): return math.degrees(math.atan2(p[1] - c[1], p[0] - c[0]))
def wrap(a): return (a + 180) % 360 - 180
OFF = {1: -60, 2: -30, 3: 30, 4: 60}
for b in ("v2", "gs"):
    p = f"{V}/bearing_{b}/probe.json"; p = p if os.path.exists(p) else p.replace(".json", ".partial.json")
    if not os.path.exists(p): print(b, "pending"); continue
    per_d, per_frame = defaultdict(list), defaultdict(dict)
    for e in json.load(open(p))["episodes"]:
        if not e["chunk_steps"]: continue
        traj = int(re.search(r":p(\d+):", e["key"]).group(1)); D = OFF[traj // 100]
        c = np.array(e["subject_center"]); s = np.array(e["step_poses"][0]["position"]); t = np.array(e["step_poses"][e["chunk_steps"][0]]["position"])
        done = wrap(ang(t, c) - ang(s, c))
        if abs(done) > 2: per_d[D].append(np.sign(done) == np.sign(-D))
        per_frame[e["key"].split(":p")[0] + ":" + e["key"].split(":", 2)[2]][D] = done
    consistent = [all(np.sign(v[d]) == np.sign(-d) for d in v) for v in per_frame.values() if len(v) == 4]
    print(f"{b}: " + " | ".join(f"D={d:+d}: {np.mean(per_d[d]):.2f} (n={len(per_d[d])})" for d in sorted(per_d)) +
          f" || start frames with all 4 correct: {np.mean(consistent) if consistent else float('nan'):.2f} (n={len(consistent)})")
