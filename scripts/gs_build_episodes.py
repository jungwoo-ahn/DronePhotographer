"""3DGS benchmark episodes (paper Table 5): the training sampler's camera paths around each placed
subject, kept only where the reconstruction can be trusted, labelled by the training scorer.

The placements are the 3DGS session's (`data/gs_scenes/composite_v5`: 5 scenes x 10 subjects,
one preview still each). An episode needs more: a start and a goal on one camera path, the goal's
shot profile in exactly the form the policies were trained on, and both pictures. So, per
placement, in its local metric frame L (feet at the origin, z up):
  * paths    `src/policy/data/sampling.py`, the sampler that generated the v7 training data:
             two poses around the subject's look-at point (75% of its height), an elliptical arc
             from the far one to the near one, 32 frames with the endpoints' jitter
  * trust    a splat is a photograph only near where it was photographed. Every endpoint and
             arc midpoint must (a) not sit inside geometry (2nd-percentile depth >= 0.6 m),
             (b) see the subject (at most one of 18 body points hidden), and (c) be within
             --cov-max of a capture camera, scored as metres away + 3 m per unit of viewing-
             direction mismatch -- prepare_jobs.py's own view score. A placement that cannot
             fill its paths at --cov-max is retried at --cov-relax, and says so.
  * labels   `PlacementGeometry.profile` over the subject's mesh (`gs_subject_blender.py dump`):
             the stage-3 scorer of the training data. `placement_yaw_deg` carries the subject's
             turn, so `goal_vector` derives its bearing through the facing map like any v7 goal.
  * windows  v7's rules: goal well framed (occupancy 10-80, >= 35% of the body's height in
             frame, centre on screen) and, as in benchmark v2, head in frame; start shows the
             subject; 8-31 frames apart, either direction; at most two per path. Start and goal
             must also LOOK AT what the capture photographed (`Placement.support` >= 0.85): the
             coverage test above admits views whose background is a smear -- ground magnified
             under a low camera, needle Gaussians seen broadside, a wall nobody faced
  * pictures start and goal composites from `gs_render_server.Compositor`, the renderer the
             rollouts use, for a few candidate windows per placement; the detector pass and the
             final pick are `scripts/gs_select_episodes.py`.

    .venv-gs/bin/python scripts/gs_build_episodes.py specs
    .venv-gs/bin/python scripts/gs_build_episodes.py geom          # Blender, CPU
    .venv-gs/bin/python scripts/gs_build_episodes.py candidates    # GPU: sbatch_gs_build.sh
    (~4 min a placement on one GPU, so shard it: --shard i --nshards 4)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
SHARED = "/home/nas_main/jungwooahn/projects/DronePhotographer"
GS = f"{SHARED}/data/gs_scenes"
sys.path[:0] = [V12, f"{V12}/scripts", f"{GS}/_tools/composite"]

from src.common.facing import sector8  # noqa: E402
from src.common.goal_space import DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY, goal_vector  # noqa: E402
from src.common.reward import _geometry_distance, pose_to_geometry  # noqa: E402
from src.goal_authoring.pose_solver import PlacementGeometry  # noqa: E402

_spec = importlib.util.spec_from_file_location("sampling", f"{SHARED}/src/policy/data/sampling.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=("specs", "geom", "candidates"))
ap.add_argument("--out", default=f"{V12}/runs/bench_gs")
ap.add_argument("--composite", default=f"{GS}/composite_v5")
ap.add_argument("--scenes", default=f"{GS}/_tools/composite/scenes_v1.json")
ap.add_argument("--cov-max", type=float, default=3.0)
ap.add_argument("--cov-relax", type=float, default=4.5)
ap.add_argument("--paths", type=int, default=8, help="accepted camera paths per placement")
ap.add_argument("--min-paths", type=int, default=3, help="below this, retry at --cov-relax")
ap.add_argument("--attempts", type=int, default=600)
ap.add_argument("--max-empty", type=float, default=0.35,
                help="largest share of a start or goal frame the splat may leave empty (sky)")
ap.add_argument("--min-support", type=float, default=0.85,
                help="start and goal: share of the background a capture camera saw (Placement.support)")
ap.add_argument("--per-placement", type=int, default=10, help="candidate windows rendered")
ap.add_argument("--samples", type=int, default=64, help="Cycles samples for the subject")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1,
                help="candidates: shard i takes every Nth placement of the sorted list (so each "
                     "shard still loads a scene's splat once) and writes candidates_<i>of<N>.json; "
                     "gs_select_episodes.py merges them")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--only", nargs="*", default=None, help="placement-name substrings")
args = ap.parse_args()

# Absolute, always: the spec paths written below are read again by Blender, which runs with the
# shared repo as its working directory, so a relative --out sends it looking in the wrong tree.
OUT = Path(args.out) if Path(args.out).is_absolute() else Path(V12) / args.out
SPECS, GEOM, FRAMES = OUT / "_specs", OUT / "_geom", OUT / "frames"
LW, LH = 128, 96
BINS_LONG_FROM = 16
CAP_W = 128                     # capture-view depth maps for Placement.support
SUPPORT_DEPTH_TOL = 0.10        # a capture view "saw" a point if its depth agrees within 10%
SUPPORT_ANGLE_DEG = 60.0
SUPPORT_DIST_RATIO = 3.0


def placements() -> list[Path]:
    ps = sorted(SPECS.glob("*.json"))
    if args.only:
        ps = [p for p in ps if any(o in p.stem for o in args.only)]
    return ps


def stage_specs() -> int:
    scenes = json.loads(Path(args.scenes).read_text())
    SPECS.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in sorted(Path(args.composite).iterdir()):
        if not (d / "jobs.json").exists():
            continue
        cfg = scenes[d.name]
        for job in json.loads((d / "jobs.json").read_text()):
            name = f"{d.name}__{job['tag']}"
            spec = {"placement": name, "scene": d.name, "object": job["object"],
                    "ply": cfg["ply"], "cams": cfg["cams"], "mpu": cfg["mpu"], "up": cfg["up"],
                    "spot_splat": job["spot_splat"], "azimuth_deg": job["azimuth_deg"],
                    "rotation_z_deg": job["rotation_z_deg"], "env": job["env"],
                    "blend_dir": job["blend_dir"], "height_m": job["height_m"],
                    "samples": args.samples, "preview": job["camera"], "preview_out": job["out"]}
            (SPECS / f"{name}.json").write_text(json.dumps(spec, indent=1))
            n += 1
    print(f"[specs] {n} placements -> {SPECS}")
    return 0


def stage_geom() -> int:
    GEOM.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{SHARED}/blender/syslibs/lib:" + env.get("LD_LIBRARY_PATH", "")
    todo = [p for p in placements() if not (GEOM / f"{p.stem}.npz").exists()]

    def dump(p: Path):
        r = subprocess.run([f"{SHARED}/blender/blender", "-b", "-P", f"{V12}/scripts/gs_subject_blender.py",
                            "--", "dump", str(p), str(GEOM / f"{p.stem}.npz")],
                           cwd=SHARED, env=env, capture_output=True, text=True, timeout=900)
        return p.stem, (GEOM / f"{p.stem}.npz").exists(), r.stdout[-300:] + r.stderr[-600:]

    failed = []
    with ThreadPoolExecutor(args.workers) as pool:
        for name, ok, log in pool.map(dump, todo):
            print(f"[geom] {'ok ' if ok else 'FAIL'} {name}" + ("" if ok else f"\n{log}"), flush=True)
            if not ok:
                failed.append(name)
    return 1 if failed else 0


class Placement:
    """A placement's frame, splat views and capture coverage, in L metres."""

    def __init__(self, spec_path: Path, splat_cache: dict):
        from gscommon import K_hfov, Splat, load_cameras
        from gs_render_server import HFOV, local_frame

        self.spec_path = spec_path
        self.spec = json.loads(spec_path.read_text())
        ply = self.spec["ply"]
        if ply not in splat_cache:
            splat_cache.clear()
            cams = load_cameras(*self.spec["cams"])
            splat_cache[ply] = {"splat": Splat(ply), "C": np.stack([c[1][:3, 3] for c in cams]),
                                "Cf": np.stack([c[1][:3, 2] for c in cams]),
                                "w2c": np.stack([np.linalg.inv(c[1]) for c in cams]),
                                "K": np.stack([c[2] for c in cams]),
                                "size": np.array([c[3] for c in cams], float)}
        self.cache = splat_cache[ply]
        self.splat, self.C, self.Cf = self.cache["splat"], self.cache["C"], self.cache["Cf"]
        self.mpu = float(self.spec["mpu"])
        self.Sp = np.asarray(self.spec["spot_splat"], float)
        self.Rl = local_frame(self.spec["up"], self.spec["azimuth_deg"])
        self.KL = K_hfov(HFOV, LW, LH)
        self.geom = PlacementGeometry.load(GEOM / f"{spec_path.stem}.npz")
        self.h = float(self.geom.subject_height)

    def c2w(self, pos, fwd, up) -> np.ndarray:
        from gs_render_server import camera_axes
        f, r, u = camera_axes(fwd, up)
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2] = self.Rl @ r, -(self.Rl @ u), self.Rl @ f
        m[:3, 3] = self.Sp + self.Rl @ np.asarray(pos, float) / self.mpu
        return m

    def check(self, pos, fwd, up) -> tuple[bool, int, float, float]:
        """(clear of geometry, hidden body points out of 18, coverage score in metres, share of
        the frame the splat leaves empty)."""
        c2w = self.c2w(pos, fwd, up)
        _, dz, al = self.splat.render(c2w, self.KL, LW, LH, near=0.05 / self.mpu)
        solid = al > 0.5
        clear = not (solid.any() and np.percentile(dz[solid], 2) * self.mpu < 0.6)
        w2c = np.linalg.inv(c2w)[:3]
        f = np.asarray(fwd, float)
        side = np.cross([0.0, 0.0, 1.0], f)
        side = side / (np.linalg.norm(side) or 1.0)
        blocked = 0
        for t in (0.05, 0.25, 0.45, 0.65, 0.85, 1.0):
            for s_ in (-1, 0, 1):
                P = np.array([0.0, 0.0, t * self.h]) + s_ * 0.18 * side
                pc = w2c @ np.append(self.Sp + self.Rl @ P / self.mpu, 1)
                if pc[2] <= 0:
                    blocked += 1
                    continue
                px = self.KL @ pc
                xi, yi = int(px[0] / px[2]), int(px[1] / px[2])
                if 0 <= xi < LW and 0 <= yi < LH and al[yi, xi] > 0.5 and dz[yi, xi] < pc[2] - 0.25 / self.mpu:
                    blocked += 1
        p_s, f_s = c2w[:3, 3], c2w[:3, 2]
        cov = float(np.min(np.linalg.norm(self.C - p_s, axis=1) * self.mpu + 3.0 * (1 - self.Cf @ f_s)))
        return clear, blocked, cov, float((al < 0.5).mean())


    def _capture_depths(self):
        """Each capture camera's own view of the splat, at 128 px wide: the depth where it saw
        a surface (inf where it saw nothing). Built once per scene."""
        import torch
        if "depths" not in self.cache:
            c = self.cache
            w, h = c["size"][0]
            ch = int(round(CAP_W * h / w))
            Ks = c["K"].copy()
            Ks[:, 0, :] *= CAP_W / c["size"][:, 0:1]
            Ks[:, 1, :] *= ch / c["size"][:, 1:2]
            D = np.zeros((len(Ks), ch, CAP_W), np.float32)
            for j in range(len(Ks)):
                _, dz, al = self.splat.render(np.linalg.inv(c["w2c"][j]), Ks[j], CAP_W, ch,
                                              near=0.05 / self.mpu)
                D[j] = np.where(al > 0.5, dz, np.inf)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)
            c["depths"] = {"D": T(D), "K": T(Ks), "h": ch, "R": T(c["w2c"][:, :3, :3]),
                           "t": T(c["w2c"][:, :3, 3]), "C": T(c["C"]), "dev": dev}
        return self.cache["depths"]

    def support(self, pos, fwd, up) -> float:
        """Share of the visible background some capture camera actually photographed.

        `check`'s coverage score asks whether the CAMERA stands near a capture camera; what a
        splat renders well is decided by what it LOOKS AT. Every background pixel is back-
        projected through the splat's depth and counts as supported when a capture camera (a)
        had the point in frame, (b) saw a surface there itself -- its own rendered depth agrees
        within SUPPORT_DEPTH_TOL, which rejects floaters and needle Gaussians that the capture
        views look through edge-on, and points they had occluded -- (c) from within
        SUPPORT_ANGLE_DEG of this view's ray, and (d) from at most SUPPORT_DIST_RATIO times this
        camera's distance, which rejects ground a metre in front of a low camera that walkers
        photographed from 3-5 m. Calibrated by eye on the 236 candidates of the first build:
        below ~0.6 nearly every frame is a smear, 0.7-0.85 is mixed (smeared paths and wall
        tops still get through), above 0.85 they are clean. Sky (alpha < 0.5) and the
        subject's box are left out; a frame the subject nearly fills counts as supported."""
        import torch
        cap = self._capture_depths()
        c2w = self.c2w(pos, fwd, up)
        _, dz, al = self.splat.render(c2w, self.KL, LW, LH, near=0.05 / self.mpu)
        keep = al > 0.5
        box = self.geom.full_bbox(np.asarray(pos), np.asarray(fwd), np.asarray(up))
        if box is not None:
            sx, sy = LW / self.geom.width, LH / self.geom.height
            keep[max(0, int(box[1] * sy)):min(LH, int(np.ceil(box[3] * sy))),
                 max(0, int(box[0] * sx)):min(LW, int(np.ceil(box[2] * sx)))] = False
        ys, xs = np.nonzero(keep)
        if len(ys) < 0.05 * LW * LH:
            return 1.0
        dev = cap["dev"]
        z = dz[ys, xs].astype(np.float64)
        rays = np.linalg.inv(self.KL) @ np.stack([xs + 0.5, ys + 0.5, np.ones(len(xs))])
        X = torch.from_numpy(c2w[:3, :3] @ (rays * z) + c2w[:3, 3:4]).float().to(dev)   # (3, N)
        zr = torch.from_numpy(z).float().to(dev)
        d_r = X - torch.from_numpy(c2w[:3, 3]).float().to(dev)[:, None]
        d_r = d_r / d_r.norm(dim=0, keepdim=True)
        cos_min = math.cos(math.radians(SUPPORT_ANGLE_DEG))
        ok = torch.zeros(X.shape[1], dtype=torch.bool, device=dev)
        for j0 in range(0, cap["R"].shape[0], 64):
            sl = slice(j0, j0 + 64)
            Xj = cap["R"][sl] @ X + cap["t"][sl][:, :, None]                            # (m, 3, N)
            zj = Xj[:, 2]
            K = cap["K"][sl]
            u = K[:, 0, 0:1] * Xj[:, 0] / zj + K[:, 0, 2:3]
            v = K[:, 1, 1:2] * Xj[:, 1] / zj + K[:, 1, 2:3]
            inside = (zj > 0) & (u >= 0) & (u < CAP_W) & (v >= 0) & (v < cap["h"])
            seen = cap["D"][sl][torch.arange(zj.shape[0], device=dev)[:, None],
                                v.clamp(0, cap["h"] - 1).long(), u.clamp(0, CAP_W - 1).long()]
            d_j = X[None] - cap["C"][sl][:, :, None]
            cos = (d_r[None] * d_j).sum(1) / d_j.norm(dim=1)
            ok |= (inside & ((zj - seen).abs() <= SUPPORT_DEPTH_TOL * zj) & (cos >= cos_min)
                   & (zj <= SUPPORT_DIST_RATIO * zr[None])).any(0)
        return float(ok.float().mean())


def raw_of(pl: Placement, pos, fwd, up, idx: int, pitch: float, yaw: float) -> dict | None:
    prof = pl.geom.profile(np.asarray(pos), np.asarray(fwd), np.asarray(up))
    if prof is None:
        return None
    raw = {k: ([float(x) for x in v] if isinstance(v, (list, tuple, np.ndarray)) else float(v))
           for k, v in prof.items()}
    raw.update({"frame_idx": float(idx), "pitch_deg": float(pitch), "yaw_deg": float(yaw),
                "placement_yaw_deg": float(pl.spec["rotation_z_deg"]), "in_frame": 1.0,
                "occupancy_clipped": raw["occupancy"] / 100.0})
    return raw


def sample_paths(pl: Placement, rng: np.random.Generator) -> tuple[list, float]:
    O = np.array([0.0, 0.0, 0.75 * pl.h])
    paths = []
    used = args.cov_max
    for cov_max in (args.cov_max, args.cov_relax):
        used = cov_max
        for _ in range(args.attempts):
            if len(paths) >= args.paths:
                break
            s, e = S.sample_pose(O, rng, floor_z=0.0), S.sample_pose(O, rng, floor_z=0.0)
            if s is None or e is None:
                continue
            if S.spherical_angle(s["pos"] - O, e["pos"] - O) < np.deg2rad(S.MIN_ANG_SEP_DEG):
                continue
            ok, ends = True, {}
            for tag, pose in (("s", s), ("e", e)):
                pos, fwd, up = S.pose_after_jitter(pose)
                clear, blocked, cov, empty = pl.check(pos, fwd, up)
                prof = pl.geom.profile(pos, fwd, up)
                if (not clear or blocked > 1 or cov > cov_max or empty > args.max_empty
                        or prof is None or prof["occupancy"] < 1):
                    ok = False
                    break
            if not ok:
                continue
            far, near = (s, e) if s["r"] >= e["r"] else (e, s)
            try:
                E = S.solve_ellipse(O, far["pos"], near["pos"])
            except S.DegenerateEllipse:
                continue
            for t in S.MIDPOINT_TS:
                pos = S.ellipse_at(E, (1 - t) * E["theta_far"] + t * E["theta_near"])
                look = S._lookat(pos, O)
                if look is None:
                    ok = False
                    break
                clear, _, cov, _ = pl.check(pos, *look)
                if not clear or cov > cov_max + 1.0:
                    ok = False
                    break
            if not ok:
                continue
            pf, pn = float(far.get("pitch_jitter_deg", 0)), float(near.get("pitch_jitter_deg", 0))
            yf, yn = float(far.get("yaw_jitter_deg", 0)), float(near.get("yaw_jitter_deg", 0))
            frames = S.trajectory_frames(E, n=S.N_FRAMES, pitch_start_deg=pf, pitch_end_deg=pn,
                                         yaw_start_deg=yf, yaw_end_deg=yn)
            if frames is None:
                continue
            alpha = [i / (S.N_FRAMES - 1) for i in range(S.N_FRAMES)]
            recs = []
            for i, (pos, fwd, up) in enumerate(frames):
                recs.append({"pose": {"position": [float(v) for v in pos], "forward": [float(v) for v in fwd],
                                      "up": [float(v) for v in up]},
                             "raw": raw_of(pl, pos, fwd, up, i, (1 - alpha[i]) * pf + alpha[i] * pn,
                                           (1 - alpha[i]) * yf + alpha[i] * yn)})
            paths.append(recs)
        if len(paths) >= args.min_paths:
            break
    return paths, used


def windows(pl: Placement, paths: list, rng: np.random.Generator, cov_max: float) -> list[dict]:
    """Up to two windows per path. The path checks covered its ends and three arc midpoints;
    the start and goal are frames in between, so each is checked again where it stands."""
    out = []
    for t, recs in enumerate(paths):
        def good_goal(r):
            return (r is not None and 10 <= r["occupancy"] <= 80 and r.get("visible_frac", 0) >= 0.35
                    and r.get("head_in_frame") == 1.0 and 0 <= r["object_center_x"] <= 1024
                    and 0 <= r["object_center_y"] <= 768)
        pairs = [(s, g) for g in range(len(recs)) if good_goal(recs[g]["raw"])
                 for s in range(len(recs)) if 8 <= abs(g - s) <= 31
                 and recs[s]["raw"] is not None and recs[s]["raw"]["occupancy"] > 1.0]
        if not pairs:
            continue
        kept = 0
        memo: dict = {}                 # frame -> (check, support); a frame recurs across pairs

        def judge(i):
            if i not in memo:
                p_ = recs[i]["pose"]
                c, b, cov, e = pl.check(p_["position"], p_["forward"], p_["up"])
                good = c and b <= 1 and cov <= cov_max + 1.0 and e <= args.max_empty
                sup = pl.support(p_["position"], p_["forward"], p_["up"]) if good else 0.0
                memo[i] = ((c, b, cov, e), sup, good and sup >= args.min_support)
            return memo[i]

        for k in rng.permutation(len(pairs)):
            if kept >= 2:
                break
            s, g = pairs[int(k)]
            if not (judge(s)[2] and judge(g)[2]):
                continue
            trust = [memo[s][0], memo[g][0]]
            sup = [memo[s][1], memo[g][1]]
            raw = recs[g]["raw"]
            gv = goal_vector(raw, DEFAULT_GOAL_KEYS, object_key=pl.spec["object"])
            if not np.isfinite(gv).all():
                continue
            bearing = float(gv[DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)])
            sp, gp = recs[s]["pose"], recs[g]["pose"]
            kw = {"subject_center": pl.geom.subject_center, "subject_height": pl.geom.subject_height}
            d0 = float(_geometry_distance(
                pose_to_geometry(sp["position"], sp["forward"], sp["up"], **kw),
                pose_to_geometry(gp["position"], gp["forward"], gp["up"], **kw)))
            delta = abs(g - s)
            out.append({"key": f"{pl.spec['placement']}:p{t:02d}:s{s:02d}:g{g:02d}",
                        "placement": pl.spec["placement"], "scene": pl.spec["scene"],
                        "object": pl.spec["object"], "spec": str(pl.spec_path),
                        "traj": t, "start_frame_idx": s, "goal_frame_idx": g,
                        "start_pose": sp, "goal_pose": gp,
                        "subject_center": [float(v) for v in pl.geom.subject_center],
                        "subject_height": float(pl.geom.subject_height),
                        "goal_raw": raw, "goal_vec": [float(x) for x in gv], "goal_keys": list(DEFAULT_GOAL_KEYS),
                        "delta": delta, "n_chunks": int(math.ceil(delta / 8)),
                        "delta_bin": "long" if delta >= BINS_LONG_FROM else "short",
                        "sector": sector8(bearing), "bearing": bearing, "d_start": d0,
                        "start_cov_m": trust[0][2], "goal_cov_m": trust[1][2],
                        "start_empty": trust[0][3], "goal_empty": trust[1][3],
                        "start_support": sup[0], "goal_support": sup[1]})
            kept += 1
    return out


def pick_superset(cands: list[dict], k: int) -> list[dict]:
    """Up to k windows, spread over sectors and paths, for the detector pass to choose from."""
    order = ("front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left")
    by = {s: [c for c in cands if c["sector"] == s] for s in order}
    chosen, trajs = [], set()
    while len(chosen) < k and any(by.values()):
        for s in order:
            if not by[s] or len(chosen) >= k:
                continue
            by[s].sort(key=lambda c: (c["traj"] in trajs))
            c = by[s].pop(0)
            chosen.append(c)
            trajs.add(c["traj"])
    return chosen


def stage_candidates() -> int:
    from PIL import Image
    from gs_render_server import Compositor

    FRAMES.mkdir(parents=True, exist_ok=True)
    dest = OUT / ("candidates.json" if args.nshards == 1
                  else f"candidates_{args.shard}of{args.nshards}.json")
    splat_cache: dict = {}
    comp = Compositor()
    all_c, report = [], {}
    t_all = time.time()
    try:
        for sp in placements()[args.shard::args.nshards]:
            t0 = time.time()
            pl = Placement(sp, splat_cache)
            seed = int(hashlib.md5(sp.stem.encode()).hexdigest()[:8], 16) ^ args.seed
            rng = np.random.default_rng(seed)
            paths, cov_used = sample_paths(pl, rng)
            cands = windows(pl, paths, rng, cov_used)
            chosen = pick_superset(cands, args.per_placement)
            for c in chosen:
                for which in ("start", "goal"):
                    idx = c[f"{which}_frame_idx"]
                    out = FRAMES / sp.stem / f"t{c['traj']:02d}_f{idx:02d}.png"
                    if not out.exists():
                        out.parent.mkdir(parents=True, exist_ok=True)
                        pose = c[f"{which}_pose"]
                        Image.fromarray(comp.render(str(sp), pose["position"], pose["forward"],
                                                    pose["up"])).save(out)
                    c[f"{which}_image"] = str(out)
                c["cov_max_used"] = cov_used
            all_c += chosen
            report[sp.stem] = {"paths": len(paths), "windows": len(cands), "rendered": len(chosen),
                               "cov_max_used": cov_used}
            print(f"[cand] {sp.stem[:48]:48s} paths {len(paths)} (cov<= {cov_used}) windows "
                  f"{len(cands)} rendered {len(chosen)} ({time.time()-t0:.0f}s, "
                  f"{(time.time()-t_all)/60:.1f} min)", flush=True)
            dest.write_text(json.dumps(
                {"report": report, "candidates": all_c}, indent=1))
    finally:
        comp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit({"specs": stage_specs, "geom": stage_geom, "candidates": stage_candidates}[args.stage]())
