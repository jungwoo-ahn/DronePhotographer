"""First-chunk orbit-direction agreement for each 3DGS probe variant (+ the full run's first chunk)."""
import json, math, os
import numpy as np
V = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
def ang(p, c): return math.degrees(math.atan2(p[1] - c[1], p[0] - c[0]))
def wrap(a): return (a + 180) % 360 - 180
def agree(path):
    src = path if os.path.exists(path) else path.replace(".json", ".partial.json")
    if not os.path.exists(src): return None
    eps = json.load(open(src))["episodes"]
    a, rad, need_l, done_l = [], [], [], []
    for e in eps:
        if not e["chunk_steps"]: continue
        k = e["chunk_steps"][0]; c = np.array(e["subject_center"])
        s = np.array(e["step_poses"][0]["position"]); t = np.array(e["step_poses"][k]["position"]); g = np.array(e["goal_pose"]["position"])
        need, done = wrap(ang(g, c) - ang(s, c)), wrap(ang(t, c) - ang(s, c))
        if abs(need) > 15 and abs(done) > 2: a.append(np.sign(need) == np.sign(done))
        rad.append((np.linalg.norm((g - c)[:2]) - np.linalg.norm((s - c)[:2]), np.linalg.norm((t - c)[:2]) - np.linalg.norm((s - c)[:2])))
    rad = np.array(rad)
    return np.mean(a), len(a), len(eps), np.corrcoef(rad[:, 0], rad[:, 1])[0, 1], np.mean([e["d_end"] < e["d_start"] for e in eps])
rows = [("full run (first chunk)", f"{V}/runs/bench_gs/rollout_ours_v6_45k.json")] + \
       [(n, f"{V}/runs/bench_gs_probe/{n}.json") for n in ("p0_base", "p1_gray", "p2_agx", "p3_gray_uniform_agx")]
for n, p in rows:
    r = agree(p)
    print(f"{n:24s} " + ("pending" if r is None else f"orbit agreement {r[0]:.2f} (n={r[1]}, eps {r[2]}) | radial corr {r[3]:+.2f} | closer after 1 chunk {100*r[4]:.0f}%"))
