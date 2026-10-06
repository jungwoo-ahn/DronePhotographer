"""PhotoFlow's global anchor bank.

The paper builds anchors from "scene-bounding-box heuristics, blueprint look-toward targets,
object visibility anchors, and scout-view relocation anchors", each carrying an initial
camera position, look-at target, approximate lens choice, aspect-ratio hint, and a prior
score. Our renderer is fixed at 24 mm / 4:3 (the harness metric cannot see lens or aspect),
so `lens_mm` and `aspect` are recorded for faithfulness but not searched.

Radii come from the lens, not from the goal: the distance at which a subject of height `H`
fills a target fraction of a frame with vertical FOV `vfov` is H / (2 * frac * tan(vfov/2)).
That is a photographic heuristic available before any scoring, which is the point -- the
anchor bank is built once, up front, with no access to the metric.
"""

from __future__ import annotations

import math

import numpy as np

VFOV_DEG = 22.62                 # the render camera; see scripts/rollout_server.py
FILL_FRACTIONS = (0.35, 0.55, 0.80)        # wide / medium / tight
ELEVATIONS_DEG = (-20.0, 0.0, 15.0)        # high angle, eye level, low angle
N_BEARINGS = 8                             # one per view sector, matching the manifest's 8


def framing_radius(subject_height: float, fill: float, vfov_deg: float = VFOV_DEG) -> float:
    h = max(float(subject_height), 0.2)
    return h / (2.0 * max(fill, 1e-3) * math.tan(math.radians(vfov_deg) / 2.0))


def _clip_into(p, extents, margin=0.3):
    if not extents:
        return np.asarray(p, float)
    lo = np.asarray(extents["min"], float) + margin
    hi = np.asarray(extents["max"], float) - margin
    return np.clip(np.asarray(p, float), np.minimum(lo, hi), np.maximum(lo, hi))


def _anchor(pos, look, prior, source):
    return {"position": [float(v) for v in pos], "look_at": [float(v) for v in look],
            "lens_mm": 24.0, "aspect": "4:3", "prior": float(prior), "source": source}


def build_anchor_bank(scout: dict, blueprint: dict | None = None, scout_views=()) -> list[dict]:
    """Assemble the four anchor families described in the paper."""
    subj = (scout or {}).get("subject", {}) or {}
    target = np.asarray(subj.get("center") or [0.0, 0.0, 0.0], float)
    height = float(subj.get("height") or 1.7)
    extents = (scout or {}).get("extents") or {}
    out: list[dict] = []

    # 1. scene-bounding-box heuristics: a ring per (fill, elevation), aimed at the subject.
    for fill in FILL_FRACTIONS:
        r = framing_radius(height, fill)
        for el in ELEVATIONS_DEG:
            for i in range(N_BEARINGS):
                a = 2.0 * math.pi * i / N_BEARINGS
                rad = math.radians(el)
                p = target + np.array([r * math.cos(rad) * math.cos(a),
                                       r * math.cos(rad) * math.sin(a),
                                       r * math.sin(rad)], float)
                p = _clip_into(p, extents)
                # mid fill and eye level are the safer photographic bets -> higher prior
                prior = 0.5 - 0.25 * abs(fill - 0.55) / 0.25 - 0.15 * abs(el) / 20.0
                out.append(_anchor(p, target, max(prior, 0.05), "scene_bbox"))

    # 2. blueprint look-toward targets: same ring radius, but aimed where the blueprint says.
    for lt in (blueprint or {}).get("look_toward", []) or []:
        try:
            look = np.asarray(lt, float)
            if look.shape != (3,):
                continue
        except (TypeError, ValueError):
            continue
        r = framing_radius(height, 0.55)
        for i in range(4):
            a = 2.0 * math.pi * i / 4
            p = _clip_into(look + np.array([r * math.cos(a), r * math.sin(a), 0.4], float), extents)
            out.append(_anchor(p, look, 0.55, "blueprint"))

    # 3. object visibility anchors: stand off a large object, look at the subject.
    for ob in ((scout or {}).get("objects") or [])[:8]:
        c = np.asarray(ob.get("center") or [0, 0, 0], float)
        v = target - c
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            continue
        r = framing_radius(height, 0.55)
        p = _clip_into(target - (v / n) * r, extents)
        out.append(_anchor(p, target, 0.35, f"visibility:{ob.get('name', '?')}"))

    # 4. scout-view relocation anchors: wherever the low-sample scout renders were taken.
    for sv in scout_views or ():
        out.append(_anchor(sv["position"], sv.get("look_at", target), 0.30, "scout_view"))

    return out


__all__ = ["build_anchor_bank", "framing_radius", "VFOV_DEG", "FILL_FRACTIONS",
           "ELEVATIONS_DEG", "N_BEARINGS"]
