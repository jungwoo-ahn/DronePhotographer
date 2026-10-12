"""PhotoFlow: the Director-Reviewer-Reflector round loop (arXiv:2605.23771).

Reimplemented from the paper description -- the authors' agent code is unreleased (their
repo ships only the project page and VPhotoBench), so prompts and JSON schemas here are
ours while every stated constant is theirs: 6 rounds x 4 candidates, J's six weights,
m2's 0.45 falloff, cells at max(0.12*sceneScale, 0.9), promising at .68/.70, and the
relocation score. Treat the row it produces as "PhotoFlow-style", not as the authors' own.

Scope in our sim: the search is over (position, look_at). The paper also searches focal
length, aperture and aspect ratio; our renderer is fixed at 24 mm / 4:3 and the harness
metric cannot see those, so varying them would be unmeasurable rather than faithful.

What this is NOT: an incremental policy. It proposes absolute camera states anywhere in the
scene and keeps the best one, so it sees far more of the scene than a policy that takes
`n_chunks` steps from a start pose. Compare it accordingly -- `render_budget` is reported in
the summary for exactly that reason.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.policy.photoflow.anchors import _clip_into, build_anchor_bank, framing_radius
from src.policy.photoflow.memory import RegionMemory
from src.policy.photoflow.reviewer import (SEMANTIC_KEY, combined_score, pairwise_choose,
                                           rule_composition_distance, rule_in_requested_half)

ROUNDS = 6
CANDIDATES_PER_ROUND = 4
SCOUT_VIEWS = 4          # "low-sample preview renders from a small set of canonical cameras"
STAGNATION_ROUNDS = 2            # rounds with no incumbent change before a high-explore jump
MAX_CRITIQUE_WORKERS = 4         # = candidates per round; keeps the gateway load flat


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def pose_from_look_at(position, look_at, world_up=(0.0, 0.0, 1.0)):
    """(position, look_at) -> (pos, forward, up), the pose triple the renderer wants."""
    pos = np.asarray(position, dtype=np.float32)
    fwd = _unit(np.asarray(look_at, float) - np.asarray(position, float))
    wu = np.asarray(world_up, float)
    right = np.cross(fwd, wu)
    if float(np.linalg.norm(right)) < 1e-6:            # looking straight up/down
        right = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
    right = _unit(right)
    up = _unit(np.cross(right, fwd))
    return pos, fwd.astype(np.float32), up.astype(np.float32)


class PhotoFlowAgent:
    """Finite-horizon feedback-driven search over absolute camera states."""

    search_space = "global"        # the harness drives this through `search()`, not `act()`

    def __init__(self, vlm, rounds=ROUNDS, candidates=CANDIDATES_PER_ROUND,
                 render_budget=None):
        self.vlm = vlm             # .ask(messages) -> text ; .ask_json(...) -> dict | None
        self.rounds = int(rounds)
        self.candidates = int(candidates)
        # A hard cap so a render-matched ablation is one argument, not a different code path.
        self.render_budget = int(render_budget) if render_budget else self.rounds * self.candidates
        self.calls = self.failures = 0

    # ---------------------------------------------------------------- Director
    def scout_views(self, render, scout, n=SCOUT_VIEWS):
        """Canonical establishing renders, so the Director SEES the scene before planning.

        The paper's Director conditions on "global scout views ... low-sample preview renders
        from a small set of canonical or visibility-oriented cameras". Without them it plans an
        entire mission from a text list of bounding boxes, which leaves it unable to tell which
        way the subject faces -- and the brief is phrased in the subject's own frame.

        These are establishing shots, not candidates: they are never scored by the Reviewer and
        never become the incumbent, so they do not consume the preview budget.
        """
        subj = (scout or {}).get("subject", {}) or {}
        target = np.asarray(subj.get("center") or [0.0, 0.0, 0.0], float)
        r = framing_radius(float(subj.get("height") or 1.7), 0.35)
        views = []
        for i in range(n):
            a = 2.0 * math.pi * i / n
            p = target + np.array([r * math.cos(a), r * math.sin(a), 0.35 * r], float)
            p = _clip_into(p, (scout or {}).get("extents") or {})
            try:
                img, _, _ = render(p.tolist(), target.tolist())
            except Exception:
                continue
            views.append({"position": [round(float(v), 2) for v in p],
                          "look_at": [round(float(v), 2) for v in target],
                          "world_azimuth_deg": round(math.degrees(a), 1), "image": img})
        return views

    def blueprint(self, scout: dict, brief: str, views=()) -> dict:
        """Soft photographic blueprint: subject, context, cues, angle/zone/axis preferences."""
        d = self.vlm.ask_json(
            system=(
                "You are a photography director planning a shot in a 3D scene. Read the scene "
                "summary and the photographic brief, then emit a SOFT blueprint as strict JSON:\n"
                '{"primary_subject":"<name>","context_objects":["<name>",...],'
                '"composition_cues":"<short>","angle_preference":"high|eye|low",'
                '"zone_preference":"<short, where in the scene to shoot from>",'
                '"look_toward":[[x,y,z],...],"axis_preference":"<short>",'
                '"symmetry_preference":"none|weak|strong","semantic_vibe":"<short>",'
                '"negative_preferences":["<what to avoid>",...]}\n'
                "Preferences are soft hints for a search, not hard constraints."),
            user=f"SCENE SUMMARY:\n{json.dumps(scout)[:6000]}\n\nBRIEF:\n{brief}",
            images=[(v["image"], f"scout view from world azimuth {v['world_azimuth_deg']} deg, "
                                 f"camera at {v['position']}") for v in views])
        self.calls += 1
        if not isinstance(d, dict):
            self.failures += 1
            return {}
        return d

    def propose(self, scout, brief, blueprint, incumbent, memory, anchors, k, views=()):
        """Propose k absolute camera states, biased by the incumbent's critique and memory."""
        dead = [list(key) for key, r in memory.regions.items() if r.dead][:40]
        inc = None
        if incumbent:
            inc = {"position": incumbent["position"], "look_at": incumbent["look_at"],
                   "J": round(incumbent["J"], 3), "critique": incumbent.get("critique", "")[:400]}
        top = [{"position": a["position"], "look_at": a["look_at"], "source": a["source"]}
               for a in anchors[:8]]
        d = self.vlm.ask_json(
            system=(
                f"Propose exactly {k} DIVERSE candidate camera states for the brief, as strict "
                'JSON: {"candidates":[{"position":[x,y,z],"look_at":[x,y,z],"why":"<short>"},...]}\n'
                "Camera is 24 mm on a 4:3 frame: horizontal FOV 29.9 deg, vertical 22.6 deg. Keep "
                "the subject inside that narrow frame. Stay inside the scene extents and away from "
                "the listed dead cells. Spread the candidates out -- different sides, distances and "
                "heights -- rather than clustering near the incumbent."),
            user=(f"SCENE SUMMARY:\n{json.dumps(scout)[:4000]}\n\nBRIEF:\n{brief}\n\n"
                  f"BLUEPRINT:\n{json.dumps(blueprint)[:1500]}\n\n"
                  f"SUGGESTED ANCHORS (ranked, you may use or ignore):\n{json.dumps(top)}\n\n"
                  f"CURRENT BEST:\n{json.dumps(inc)}\n\n"
                  f"DEAD CELLS (cell index, side {memory.h:.2f} m):\n{json.dumps(dead)}"),
            images=[(v["image"], f"scout view from world azimuth {v['world_azimuth_deg']} deg")
                    for v in views])
        self.calls += 1
        cands = (d or {}).get("candidates")
        out = []
        for c in cands or []:
            try:
                p = [float(v) for v in c["position"]]
                l = [float(v) for v in c["look_at"]]
            except (KeyError, TypeError, ValueError):
                continue
            if len(p) == 3 and len(l) == 3:
                out.append({"position": p, "look_at": l, "why": str(c.get("why", ""))[:200]})
        if not out:
            self.failures += 1
        return out[:k]

    # ---------------------------------------------------------------- Reviewer
    def critique(self, image_path, brief, blueprint) -> dict:
        """m3-m6 plus a short written critique the Director sees next round."""
        d = self.vlm.ask_json(
            system=("Score this photograph against the brief. Strict JSON, each score in [0,1]:\n"
                    '{"m3_composition":<f>,"m4_technical":<f>,"m5_aesthetic":<f>,'
                    '"m6_alignment":<f>,"critique":"<one or two sentences: what is wrong and '
                    'which way the camera should move>"}\n'
                    "m3 composition quality; m4 technical image quality; m5 aesthetic quality and "
                    "photographic appeal; m6 alignment with the brief and the blueprint."),
            user=f"BRIEF:\n{brief}\n\nBLUEPRINT:\n{json.dumps(blueprint)[:1200]}",
            image=image_path)
        self.calls += 1
        if not isinstance(d, dict):
            self.failures += 1
            return {"m3": 0.0, "m4": 0.0, "m5": 0.0, "m6": 0.0, "critique": ""}
        g = lambda k: float(np.clip(float(d.get(k, 0.0) or 0.0), 0.0, 1.0))  # noqa: E731
        return {"m3": g("m3_composition"), "m4": g("m4_technical"),
                "m5": g("m5_aesthetic"), "m6": g("m6_alignment"),
                "critique": str(d.get("critique", ""))[:400]}

    def _critique_many(self, images, brief, blueprint):
        """Critique a slate concurrently, returning results in the SAME order as `images`."""
        if not images:
            return []
        if len(images) == 1:
            return [self.critique(images[0], brief, blueprint)]
        with ThreadPoolExecutor(max_workers=min(len(images), MAX_CRITIQUE_WORKERS)) as ex:
            return list(ex.map(lambda im: self.critique(im, brief, blueprint), images))

    @staticmethod
    def rule_scores(achieved, in_frame, requested_half, target_point, resolution):
        """m1, m2 from the rendered preview's measured subject position."""
        w, h = float(resolution[0]), float(resolution[1])
        cx = achieved.get("object_center_x")
        cy = achieved.get("object_center_y")
        if cx is None or cy is None:
            return 0.0, 0.0, None
        uv = (float(cx) / w, float(cy) / h)
        return (rule_in_requested_half(uv, in_frame, requested_half),
                rule_composition_distance(uv, target_point), uv)

    # ---------------------------------------------------------------- loop
    def search(self, render, scout, brief, requested_half=None, target_point=(0.5, 0.5),
               resolution=(1024, 768)):
        """Run the round loop. `render(position, look_at) -> (image, achieved, in_frame)`.

        Returns (best, log) where `best` carries the chosen position/look_at and J.
        """
        memory = RegionMemory(float((scout or {}).get("scene_scale") or 1.0))
        views = self.scout_views(render, scout)
        bp = self.blueprint(scout, brief, views)
        anchors = build_anchor_bank(scout, bp, scout_views=views)
        subject = np.asarray((scout.get("subject") or {}).get("center") or [0, 0, 0], float)
        incumbent, log, used, stalled = None, [], 0, 0

        for rnd in range(self.rounds):
            if used >= self.render_budget:
                break
            target_pos = np.asarray(incumbent["position"], float) if incumbent else subject
            ranked = memory.rank_anchors(anchors, target_pos)
            k = min(self.candidates, self.render_budget - used)
            cands = self.propose(scout, brief, bp, incumbent, memory, ranked, k, views)
            if not cands:                                  # Director failed -> fall back to anchors
                cands = [{"position": a["position"], "look_at": a["look_at"],
                          "why": f"anchor:{a['source']}"} for a in ranked[:k]]
            if stalled >= STAGNATION_ROUNDS and ranked:
                # high-explore relocation: force the top-ranked live anchor into the slate
                cands[-1] = {"position": ranked[0]["position"], "look_at": ranked[0]["look_at"],
                             "why": f"high-explore:{ranked[0]['source']}"}
                stalled = 0

            # Render the slate serially (one Blender server, file handshake), then critique the
            # previews CONCURRENTLY: m3-m6 are per-candidate and independent. Selection and
            # memory stay strictly sequential below, in slate order, so this is a pure latency
            # win with identical semantics -- it is ~4x of the per-round wall clock.
            previews = []
            for c in cands:
                if used >= self.render_budget:
                    break
                if memory.is_forbidden(c["position"]):     # dead regions are forbidden zones
                    log.append({"round": rnd, "skipped": "dead_region", **c})
                    continue
                image, achieved, in_frame = render(c["position"], c["look_at"])
                used += 1
                m1, m2, uv = self.rule_scores(achieved, in_frame, requested_half,
                                              target_point, resolution)
                previews.append({**c, "image": image, "achieved": achieved,
                                 "in_frame": bool(in_frame), "uv": uv, "m1": m1, "m2": m2})

            critiques = self._critique_many([q["image"] for q in previews], brief, bp)

            replaced_this_round = False
            for q, crit in zip(previews, critiques):
                m = {"m1": q["m1"], "m2": q["m2"], **crit}
                cand = {**{k: v for k, v in q.items() if k not in ("m1", "m2")},
                        "m": m, "J": combined_score(m), "critique": m.get("critique", ""),
                        "round": rnd}
                before = incumbent
                incumbent, replaced = pairwise_choose(incumbent, cand)
                replaced_this_round |= replaced
                memory.visit(q["position"], cand["J"], m.get(SEMANTIC_KEY, 0.0),
                             improved=replaced)
                log.append({"round": rnd, "J": round(cand["J"], 4),
                            "m": {k2: round(v, 3) for k2, v in m.items() if k2 != "critique"},
                            "in_frame": q["in_frame"], "position": q["position"],
                            "look_at": q["look_at"], "replaced": replaced,
                            "incumbent_J": round(incumbent["J"], 4),
                            "was": round(before["J"], 4) if before else None})
            stalled = 0 if replaced_this_round else stalled + 1

        return incumbent, {"log": log, "renders": used, "scout_views": len(views), "blueprint": bp,
                           "anchors": len(anchors), "cells": len(memory.regions),
                           "dead_cells": sum(1 for r in memory.regions.values() if r.dead),
                           "vlm_calls": self.calls, "vlm_failures": self.failures}


__all__ = ["PhotoFlowAgent", "pose_from_look_at", "ROUNDS", "CANDIDATES_PER_ROUND"]
