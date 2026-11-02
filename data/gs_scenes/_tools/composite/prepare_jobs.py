"""Place the 10 eval subjects in a splat scene and write everything Blender needs to render them.

For each scene (scenes_v1.json) this picks 10 subject spots, and for each (spot, subject):
  * a preview camera with the policy lens (24 mm on 12.8 x 9.6 mm, 4:3, 1024x768) aimed at mid-body,
    far enough out that the whole body fills ~80% of the frame height, from the azimuth / distance /
    height whose view is closest to a capture camera's (position + viewing direction) among those
    that see every one of 18 points on the body unblocked and are not inside geometry;
  * the splat's RGB and planar depth (metres) from that camera, for the composite;
  * a 360 deg environment map rendered from the splat at the subject's chest, in Blender's
    equirect convention (image centre = +x, top row = +z), to light the subject like its scene;
  * the camera in a local metric frame L: origin at the subject's feet, z up, +x toward the
    camera, so Blender only needs the subject at the origin rotated to face +x.

Spots come from where people actually stood: every training camera within 4 m of the ground
casts its bottom-centre pixels into the splat; the hit is snapped to the local ground and kept
if the ground is flat (std < 6 cm over 0.5 m), nothing body-height (0.25-2 m) stands within
0.9 m, no step or slope rises 8-25 cm within 0.8 m, there is no surface 0.25-1 m below it (a
table top, bench, stair or ledge has one; the ground does not), and many training cameras see
it. Ten are drawn by farthest-point sampling. Once a spot has its preview camera, the feet are
moved onto the splat surface as that camera sees it: the lowest point on the vertical through the
feet that is less than half hidden by the splat in front of it and nearer than its expected depth
(the test compose.py applies). The median of nearby Gaussian centres sat 10-15 cm off it beside
steps and on gravel, floating or burying the feet, and a straight-down render is no better: from
above, ground captured at grazing angles is a metre-thick haze of needle Gaussians.

  prepare_jobs.py --scenes scenes_v1.json --subjects ../../subjects_v1.json --out DIR [--only NAME ...]

Re-placing single subjects: `--redo SCENE:i,j ...` keeps every other subject of DIR/SCENE/jobs.json
and writes --k candidate spots per listed subject to --cand-out (tags `IIcK_name`), taken from the
best-covered candidates (most capture cameras within 10 m; --cover-q) at least --keep-gap from the
kept subjects; merge_redo.py then swaps the chosen ones in.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from gscommon import K_hfov, Splat, draw_spots, load_cameras, look_at

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", required=True)
ap.add_argument("--subjects", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--cam-h", type=float, default=1.7, help="preview camera height above the feet (m)")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--redo", nargs="*", default=None, help="SCENE:i,j,... subjects to re-place; the rest of --out stays")
ap.add_argument("--k", type=int, default=4, help="candidate spots per re-placed subject")
ap.add_argument("--cand-out", default=None, help="where the re-placement candidates go")
ap.add_argument("--cover-q", type=float, default=0.5, help="candidates keep this coverage quantile and up")
ap.add_argument("--keep-gap", type=float, default=2.0, help="min distance (m) from the kept subjects")
ap.add_argument("--cand-gap", type=float, default=1.5, help="min distance (m) between candidates")
a = ap.parse_args()

W, H, HFOV = 1024, 768, 29.9
LW, LH = 128, 96                                               # visibility-check renders
KL = K_hfov(HFOV, LW, LH)
TAN_V = (H / 2) / K_hfov(HFOV, W, H)[1, 1]                     # tan(VFOV / 2)
scenes = json.loads(Path(a.scenes).read_text())
subjects = json.loads(Path(a.subjects).read_text())["subjects"]
out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)
YAW_OFFSETS = [0, 20, -20, 35, -35, 10, -10, 25, -25, 0]      # small turns away from the camera


def frame_axes(up, toward):
    x = toward - (toward @ up) * up; x /= np.linalg.norm(x)
    return x, np.cross(up, x), up


class Scene:
    def __init__(self, name, cfg):
        self.name, self.cfg = name, cfg
        self.sp = Splat(cfg["ply"])
        self.cams = load_cameras(*cfg["cams"])
        self.mpu = float(cfg["mpu"]); self.u = 1.0 / self.mpu
        up = np.asarray(cfg["up"], float); self.up = up / np.linalg.norm(up)
        self.C = np.stack([c[1][:3, 3] for c in self.cams])
        self.Cf = np.stack([c[1][:3, 2] for c in self.cams])  # capture viewing directions
        self.W2C = np.stack([np.linalg.inv(c[1])[:3] for c in self.cams])
        self.Kc = np.stack([np.asarray(c[2], float) for c in self.cams])
        self.size = np.array([c[3] for c in self.cams], float)
        self.G = self.sp.xyz[self.sp.opacity > 0.5]
        self.tree = cKDTree(self.G)
        e1 = self.hor(np.array([1.0, 0, 0]) if abs(self.up[0]) < 0.9 else np.array([0, 1.0, 0]))
        self.e1 = e1 / np.linalg.norm(e1); self.e2 = np.cross(self.up, self.e1)
        self.cam_h = float(cfg.get("cam_h_m", a.cam_h))

    def hor(self, v):
        return v - np.outer(v @ self.up, self.up) if v.ndim == 2 else v - (v @ self.up) * self.up

    def hdist_m(self, A, B):
        """Horizontal distances (m) between point sets A (n,3) and B (m,3)."""
        d = A[:, None, :] - B[None, :, :]
        d = d - (d @ self.up)[..., None] * self.up
        return np.linalg.norm(d, axis=2) * self.mpu


def ground_hits(sc):
    """Bottom-centre hits of the near-ground training cameras, densified around each."""
    sel = [i for i, c in enumerate(sc.cams) if not sc.cfg.get("ground_cams") or re.search(sc.cfg["ground_cams"], c[0])]
    hits = []
    for i in np.array(sel)[np.linspace(0, len(sel) - 1, min(200, len(sel))).astype(int)]:
        _, c2w, K, (cw, ch) = sc.cams[i]
        s = 160 / max(cw, ch); w, h = max(16, round(cw * s)), max(16, round(ch * s))
        Ks = np.array(K, float).copy(); Ks[:2] *= s
        _, depth, alpha = sc.sp.render(c2w, Ks, w, h)
        ys, xs = np.mgrid[int(h * 0.70):int(h * 0.95), int(w * 0.35):int(w * 0.65)]
        ok = alpha[ys, xs] > 0.9
        if ok.sum() < 8:
            continue
        rays = np.linalg.inv(Ks) @ np.stack([xs[ok], ys[ok], np.ones(ok.sum())])
        P = (c2w[:3, :3] @ (rays * depth[ys, xs][ok])).T + c2w[:3, 3]
        X = np.median(P, 0)
        hc = float((c2w[:3, 3] - X) @ sc.up) * sc.mpu
        if 0.4 < hc < 4.0:
            hits.append(X)
    hits = np.array(hits)
    # a camera's bottom-centre hit is one point of open ground; look around it too
    ring = [r * (math.cos(t) * sc.e1 + math.sin(t) * sc.e2) for r in (0.75, 1.5) for t in np.arange(8) * np.pi / 4]
    return np.concatenate([hits] + [hits + d * sc.u for d in ring])


def candidates(sc):
    """-> (spots (n,3), capture cameras seeing each within 25 m, ... within 10 m)."""
    G, u, mpu, up = sc.G, sc.u, sc.mpu, sc.up
    flat_max = float(sc.cfg.get("flat_max_m", 0.06))
    body_max = int(sc.cfg.get("body_max", 15)); step_max = int(sc.cfg.get("step_max", 40))
    hits = ground_hits(sc)
    out, vis, near10 = [], [], []
    for X in hits:
        nb = G[sc.tree.query_ball_point(X, r=0.8 * u)]
        if len(nb) < 20:
            continue
        rel = nb - X; hz = rel @ up; rh = np.linalg.norm(sc.hor(rel), axis=1)
        near = (rh < 0.5 * u) & (np.abs(hz) < 0.3 * u)
        if near.sum() < 10:
            continue
        g = float(np.median(hz[near]))
        if float(np.std(hz[near] - g)) * mpu > flat_max:
            continue
        Xg = X + g * up
        rel = G[sc.tree.query_ball_point(Xg, r=2.2 * u)] - Xg
        hz = (rel @ up) * mpu; rh = np.linalg.norm(sc.hor(rel), axis=1) * mpu
        body = int(((hz > 0.25) & (hz < 2.0) & (rh < 0.9)).sum())
        step = int(((hz > 0.08) & (hz < 0.25) & (rh < 0.8)).sum())
        gnd = int(((np.abs(hz) < 0.06) & (rh < 0.5)).sum())
        under = int(((hz < -0.25) & (hz > -1.0) & (rh < 0.3)).sum())
        if body > body_max or step > step_max or under > 0.3 * max(gnd, 1):
            continue                                          # crowded, beside a step, or standing on something
        pc = sc.W2C @ np.append(Xg + 1.0 * u * up, 1)          # chest point in every capture camera
        px = np.einsum("nij,nj->ni", sc.Kc, pc)
        with np.errstate(divide="ignore", invalid="ignore"):
            px = px[:, :2] / px[:, 2:3]
        seen = ((pc[:, 2] > 1.0 * u) & (pc[:, 2] < 25 * u) & (px[:, 0] >= 0) & (px[:, 0] < sc.size[:, 0])
                & (px[:, 1] >= 0) & (px[:, 1] < sc.size[:, 1]))
        out.append(Xg); vis.append(int(seen.sum())); near10.append(int((seen & (pc[:, 2] * mpu < 10)).sum()))
    print(f"  {len(hits)} camera ground hits -> {len(out)} flat, clear candidates")
    if not out:
        raise SystemExit(f"{sc.name}: no subject spots")
    return np.stack(out), np.array(vis), np.array(near10)


def settle(sc, S, v):
    """Move the feet (and the view with them) onto the splat surface as the view's camera sees it:
    the lowest point on the vertical through the feet that is both less than half hidden by the
    splat in front of it (what the eye sees) and nearer than the splat's expected depth (the test
    compose.py applies). -> (S, view, metres moved up); +-0.4 m means no surface was found."""
    u, up = sc.u, sc.up
    eye = v["eye"]
    c2w = look_at(eye, v["tgt"], up); Ks = K_hfov(HFOV, 256, 192)
    _, depth, alpha = sc.sp.render(c2w, Ks, 256, 192, near=0.05 * u)
    w2c = np.linalg.inv(c2w)[:3]
    K16 = K_hfov(4.0, 16, 16)

    def seen(P):
        pc = w2c @ np.append(P, 1); px = Ks @ pc
        xi, yi = int(px[0] / px[2]), int(px[1] / px[2])
        if 0 <= xi < 256 and 0 <= yi < 192 and alpha[yi, xi] > 0.5 and pc[2] > depth[yi, xi]:
            return False
        z = float(np.linalg.norm(P - eye))                     # opacity stacked up in front of P
        _, _, al = sc.sp.render(look_at(eye, P, up), K16, 16, 16, near=0.05 * u, far=z - 0.02 * u)
        return float(al[6:10, 6:10].mean()) < 0.5

    ts = np.arange(-0.4, 0.401, 0.02)
    lows = []
    for s_ in (-0.12, 0.0, 0.12):                              # across the feet
        at = lambda k: S + ts[k] * u * up + s_ * u * c2w[:3, 0]
        if not seen(at(len(ts) - 1)):
            lows.append(ts[-1]); continue
        if seen(at(0)):
            lows.append(ts[0]); continue
        lo, hi = 0, len(ts) - 1                                # hidden at lo, seen at hi
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if seen(at(mid)) else (mid, hi)
        lows.append(ts[hi])
    t = float(np.median(lows))
    v = dict(v, eye=v["eye"] + t * u * up, tgt=v["tgt"] + t * u * up)
    return S + t * u * up, v, t


def body_check(sc, eye, tgt, S, h_m):
    """-> (blocked body points, whole body inside the frame, camera clear of geometry)."""
    u, up = sc.u, sc.up
    c2w = look_at(eye, tgt, up)
    _, dz, al = sc.sp.render(c2w, KL, LW, LH, near=0.05 * u)
    if (al > 0.5).mean() > 0 and np.percentile(dz[al > 0.5], 2) < 0.6 * u:
        return 99, False, False
    side = np.cross(up, c2w[:3, 2]); side /= np.linalg.norm(side)
    w2c = np.linalg.inv(c2w)[:3]
    blocked, inside = 0, True
    for t in (0.05, 0.25, 0.45, 0.65, 0.85, 1.0):
        for s_ in (-1, 0, 1):
            Pt = S + t * h_m * u * up + s_ * 0.18 * u * side
            pc = w2c @ np.append(Pt, 1)
            px = KL @ pc; px = px[:2] / px[2]
            if t in (0.05, 1.0):
                inside &= bool(0.04 * LW <= px[0] < 0.96 * LW and 0.04 * LH <= px[1] < 0.96 * LH)
            xi, yi = int(px[0]), int(px[1])
            if not (0 <= xi < LW and 0 <= yi < LH):
                continue
            if al[yi, xi] > 0.5 and dz[yi, xi] < pc[2] - 0.25 * u:
                blocked += 1
    return blocked, inside, True


def best_view(sc, S, h_m):
    """-> view dict (level 0 = whole body seen and in frame, 1 = seen, 2 = partly blocked) or None."""
    u, up = sc.u, sc.up
    tgt = S + 0.5 * h_m * u * up
    d_fit = max(3.0, 0.5 * h_m * 1.24 / TAN_V)                 # body = ~80% of the frame height
    best = None
    for dist in (d_fit, 1.15 * d_fit, 1.3 * d_fit, 0.85 * d_fit, 0.7 * d_fit):
        for ch in (sc.cam_h, sc.cam_h + 0.7):
            for az in range(0, 360, 15):
                hd = math.cos(math.radians(az)) * sc.e1 + math.sin(math.radians(az)) * sc.e2
                eye = S + dist * u * hd + ch * u * up
                blocked, inside, clear = body_check(sc, eye, tgt, S, h_m)
                if not clear:
                    continue
                level = 0 if (blocked < 1 and inside) else 1 if blocked < 1 else 2 if blocked < 4 else None
                if level is None:
                    continue
                f = (tgt - eye) / np.linalg.norm(tgt - eye)
                dpos = np.linalg.norm(sc.C - eye, axis=1) * sc.mpu
                score = float(np.min(dpos + 3.0 * (1 - sc.Cf @ f)))  # 3 m per unit of view-direction mismatch
                if best is None or (level, score) < (best["level"], best["score"]):
                    best = {"level": level, "score": score, "eye": eye, "tgt": tgt, "az": az, "dist": dist,
                            "ch": ch, "dmin": float(dpos.min())}
    return best


def write_job(sc, od, i, tag, S, subj, v):
    """Render the splat background + environment for one (spot, subject, view) and return its job."""
    u, up, mpu, sp = sc.u, sc.up, sc.mpu, sc.sp
    eye = v["eye"]
    c2w = look_at(eye, v["tgt"], up)
    rgb, depth, alpha = sp.render(c2w, K_hfov(HFOV, W, H), W, H, near=0.05 * u)
    Image.fromarray(rgb).save(od / f"{tag}_bg.png")
    np.save(od / f"{tag}_bgdepth_m.npy", (depth * mpu).astype(np.float32))
    np.save(od / f"{tag}_bgalpha.npy", alpha.astype(np.float16))
    # local metric frame L: origin at the feet, +x toward the camera, z up
    x, y, z = frame_axes(up, eye - S)
    Rl = np.stack([x, y, z], 1)                                  # columns = L axes in splat coords
    cam_loc = Rl.T @ (eye - S) * mpu
    r, d_, f = c2w[:3, 0], c2w[:3, 1], c2w[:3, 2]
    cam_rot = np.stack([Rl.T @ r, -(Rl.T @ d_), -(Rl.T @ f)], 1)    # Blender: right, up, -forward
    # environment: cube map at the chest, resampled to Blender's equirect
    faces = {}
    for key, fwd, upv in [("+x", x, z), ("-x", -x, z), ("+y", y, z), ("-y", -y, z), ("+z", z, -x), ("-z", -z, x)]:
        cw2 = np.eye(4); rr = np.cross(fwd, upv); rr /= np.linalg.norm(rr)
        cw2[:3, 0] = rr; cw2[:3, 1] = -upv; cw2[:3, 2] = fwd; cw2[:3, 3] = S + 1.0 * u * up
        frgb, _, fal = sp.render(cw2, K_hfov(90.0, 256, 256), 256, 256, near=0.05 * u)
        sky = np.array([205, 220, 240], np.float32)
        faces[key] = (frgb.astype(np.float32) * fal[..., None] + sky * (1 - fal[..., None]), cw2)
    EW, EH = 512, 256
    jj, ii = np.mgrid[0:EH, 0:EW]
    el = np.pi / 2 - np.pi * (jj + 0.5) / EH
    azm = (0.5 - (ii + 0.5) / EW) * 2 * np.pi
    dirs = np.stack([np.cos(el) * np.cos(azm), np.cos(el) * np.sin(azm), np.sin(el)], -1) @ Rl.T
    env = np.zeros((EH, EW, 3), np.float32); best_dot = np.full((EH, EW), -2.0)
    for img, cw2 in faces.values():
        dot = dirs @ cw2[:3, 2]
        m = dot > best_dot
        pc = dirs @ cw2[:3, :3]                                  # camera coords (x right, y down, z fwd)
        with np.errstate(divide="ignore", invalid="ignore"):
            px = 128 + 128 * pc[..., 0] / pc[..., 2]; py = 128 + 128 * pc[..., 1] / pc[..., 2]
        px = np.clip(px, 0, 255).astype(int); py = np.clip(py, 0, 255).astype(int)
        env[m] = img[py[m], px[m]]; best_dot[m] = dot[m]
    Image.fromarray(env.clip(0, 255).astype(np.uint8)).save(od / f"{tag}_env.png")
    yaw = YAW_OFFSETS[i % len(YAW_OFFSETS)]
    print(f"  {tag}: level {v['level']} az {v['az']:3d} dist {v['dist']:.1f} m h {v['ch']:.1f} m  "
          f"capture score {v['score']:.1f} m (nearest cam {v['dmin']:.1f} m)")
    return {"scene": sc.name, "index": i, "tag": tag, "object": subj["object"],
            "blend_dir": subj["blend_dir"], "facing_world_deg": subj["facing_world_deg"],
            "height_m": subj["height_m"],
            # rotate the asset so its front faces +x (the camera), then turn by `yaw`
            "rotation_z_deg": float(yaw - subj["facing_world_deg"]),
            "camera": {"location": cam_loc.tolist(), "rotation": cam_rot.tolist(),
                       "lens_mm": 24.0, "sensor_w_mm": 12.8, "sensor_h_mm": 9.6, "res": [W, H]},
            "env": str(od / f"{tag}_env.png"), "bg": str(od / f"{tag}_bg.png"),
            "bg_depth": str(od / f"{tag}_bgdepth_m.npy"), "bg_alpha": str(od / f"{tag}_bgalpha.npy"),
            "fg": str(od / f"{tag}_fg.png"), "fg_depth": str(od / f"{tag}_fgdepth.exr"),
            "out": str(od / f"{tag}_composite.jpg"),
            "spot_splat": S.tolist(), "azimuth_deg": v["az"], "nearest_capture_cam_m": round(v["dmin"], 2),
            "view": {"level": v["level"], "capture_score_m": round(v["score"], 2),
                     "dist_m": round(v["dist"], 2), "cam_h_m": round(v["ch"], 2)}}


def overview(sc, od, marks):
    """Top-down-ish view of the site with the marks drawn; the clean render and camera are kept
    so merge_redo.py can redraw the marks after a swap."""
    P = np.stack([m[0] for m in marks]); ctr = P.mean(0)
    rad = max(float(np.max(np.linalg.norm(sc.hor(P - ctr), axis=1))) * sc.mpu, 5.0)
    hgt = rad * 2.2
    eye = ctr + hgt * sc.u * sc.up + 0.35 * hgt * sc.u * sc.e1
    c2w = look_at(eye, ctr, sc.up); K = K_hfov(60, W, H)
    rgb, _, _ = sc.sp.render(c2w, K, W, H, near=0.05 * sc.u)
    img = Image.fromarray(rgb)
    img.save(od / "overview_clean.jpg", quality=88)
    (od / "overview_cam.json").write_text(json.dumps({"c2w": c2w.tolist(), "K": K.tolist()}))
    draw_spots(img, c2w, K, marks)
    img.save(od / "overview.jpg", quality=88)


def short(subj):
    return subj["object"].split("_")[0][:28]


def place_all(sc, P, vis):
    od = out_root / sc.name; od.mkdir(parents=True, exist_ok=True)
    keep = vis >= 0.3 * vis.max()
    P, vis = P[keep], vis[keep]
    chosen = [int(np.argmax(vis))]
    while len(chosen) < min(a.n, len(P)):
        d = sc.hdist_m(P, P[chosen]).min(1)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))                      # farthest-point: spreads as far as the site allows
    spots = [P[k] for k in chosen]
    while len(spots) < a.n:                                   # tiny scenes: reuse spots
        spots.append(spots[len(spots) % len(chosen)])
    jobs = []
    for i, (S, subj) in enumerate(zip(spots, subjects)):
        v = best_view(sc, S, float(subj["height_m"]))
        if v is None:
            print(f"  spot {i}: no usable view, skipped"); continue
        S, v, t = settle(sc, S, v)
        if abs(t) > 0.3:
            print(f"  spot {i}: feet moved {t:+.2f} m to reach the surface")
        jobs.append(write_job(sc, od, i, f"{i:02d}_{short(subj)}", S, subj, v))
        jobs[-1]["settle_m"] = round(t, 3)
    overview(sc, od, [(np.array(j["spot_splat"]), str(j["index"]), (255, 40, 40)) for j in jobs])
    (od / "jobs.json").write_text(json.dumps(jobs, indent=1))
    print(f"  wrote {len(jobs)} jobs -> {od / 'jobs.json'}")


def place_redo(sc, P, near10, idxs):
    base = json.loads((out_root / sc.name / "jobs.json").read_text())
    kept = [j for j in base if j["index"] not in idxs]
    old = np.array([j["spot_splat"] for j in base if j["index"] in idxs])
    od = Path(a.cand_out) / sc.name; od.mkdir(parents=True, exist_ok=True)
    ok = near10 >= np.quantile(near10, a.cover_q)              # the better-covered part
    if kept:
        ok &= sc.hdist_m(P, np.array([j["spot_splat"] for j in kept])).min(1) >= a.keep_gap
    ok &= sc.hdist_m(P, old).min(1) >= 1.0
    order = [int(k) for k in np.argsort(-near10) if ok[k]]
    print(f"  {len(order)} candidates left for {len(idxs)} subjects")
    taken, jobs = [], []
    for i in idxs:
        subj = subjects[i]; k = 0
        for c in order:
            if k == a.k:
                break
            if taken and sc.hdist_m(P[c:c + 1], np.array(taken)).min() < a.cand_gap:
                continue
            v = best_view(sc, P[c], float(subj["height_m"]))
            if v is None or v["level"] > 0:
                continue
            S, v, t = settle(sc, P[c], v)
            if abs(t) > 0.15:
                continue                                      # the surface is not where the Gaussians say
            taken.append(P[c])
            job = write_job(sc, od, i, f"{i:02d}c{k}_{short(subj)}", S, subj, v)
            job.update({"cand": k, "near10": int(near10[c]), "settle_m": round(t, 3)})
            jobs.append(job); k += 1
        print(f"  subject {i}: {k} candidates")
    overview(sc, od, [(np.array(j["spot_splat"]), str(j["index"]), (255, 255, 255)) for j in kept]
             + [(np.array(j["spot_splat"]), f"{j['index']}.{j['cand']}", (255, 40, 40)) for j in jobs])
    (od / "jobs.json").write_text(json.dumps(jobs, indent=1))
    print(f"  wrote {len(jobs)} candidate jobs -> {od / 'jobs.json'}")


redo = {}
for r in a.redo or []:
    s, idx = r.split(":"); redo[s] = [int(x) for x in idx.split(",")]
if redo and not a.cand_out:
    raise SystemExit("--redo needs --cand-out")
for name, cfg in scenes.items():
    if (redo and name not in redo) or (not redo and a.only and name not in a.only):
        continue
    print(f"== {name}")
    sc = Scene(name, cfg)
    P, vis, near10 = candidates(sc)
    if redo:
        place_redo(sc, P, near10, redo[name])
    else:
        place_all(sc, P, vis)
