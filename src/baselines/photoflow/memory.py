"""Region memory, dead-zone suppression and high-explore relocation (PhotoFlow Reflector).

Constants are the paper's (arXiv:2605.23771): cubic cells of side
`h = max(0.12 * sceneScale, 0.9)`; a region is *promising* once its best internal score
reaches .68, its best semantic score reaches .70, or it takes a promising hit; it becomes
*dead* after repeated low-score visits or repeated stagnation without improvement; dead
regions become forbidden zones. Relocation ranks anchors by

    s(a) = pi(a) + u(a) + min(||p_a - p_t|| / (2h), 2.0) - 0.35 * n(a) - 0.40 * 1[a in promising]

with u(a) = 1.2 for an unknown region else 0.25. Anchors in dead regions are skipped
before ranking. The negative weight on `promising` is deliberate: this ranking exists to
*escape* a stalled region, so it pushes away from where the search has already settled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CELL_SCALE = 0.12
CELL_MIN = 0.9
PROMISING_SCORE = 0.68
PROMISING_SEMANTIC = 0.70
POOR_HITS_DEAD = 3                 # "repeated low-score visits"
STAGNATION_DEAD = 3                # "repeated stagnation without improvement"
POOR_SCORE = 0.40
UNKNOWN_BONUS = 1.2
KNOWN_BONUS = 0.25
VISIT_PENALTY = 0.35
PROMISING_PENALTY = 0.40
DISTANCE_CAP = 2.0


def cell_size(scene_scale: float) -> float:
    return max(CELL_SCALE * float(scene_scale), CELL_MIN)


@dataclass
class Region:
    visits: int = 0
    best_score: float = 0.0
    best_semantic: float = 0.0
    poor_hits: int = 0
    promising_hits: int = 0
    improvement_hits: int = 0
    stagnation_hits: int = 0

    @property
    def promising(self) -> bool:
        return (self.best_score >= PROMISING_SCORE
                or self.best_semantic >= PROMISING_SEMANTIC
                or self.promising_hits > 0)

    @property
    def dead(self) -> bool:
        return self.poor_hits >= POOR_HITS_DEAD or self.stagnation_hits >= STAGNATION_DEAD


class RegionMemory:
    """Cubic-cell memory over camera positions."""

    def __init__(self, scene_scale: float = 1.0) -> None:
        self.h = cell_size(scene_scale)
        self.regions: dict[tuple[int, int, int], Region] = {}

    def key(self, p) -> tuple[int, int, int]:
        q = np.floor(np.asarray(p, dtype=np.float64) / self.h).astype(int)
        return (int(q[0]), int(q[1]), int(q[2]))

    def get(self, p) -> Region | None:
        return self.regions.get(self.key(p))

    def visit(self, p, score: float, semantic: float, improved: bool) -> Region:
        r = self.regions.setdefault(self.key(p), Region())
        r.visits += 1
        improved_here = score > r.best_score
        r.best_score = max(r.best_score, float(score))
        r.best_semantic = max(r.best_semantic, float(semantic))
        if score < POOR_SCORE:
            r.poor_hits += 1
        if score >= PROMISING_SCORE or semantic >= PROMISING_SEMANTIC:
            r.promising_hits += 1
        if improved:
            r.improvement_hits += 1
        elif not improved_here:
            r.stagnation_hits += 1
        return r

    def is_forbidden(self, p) -> bool:
        r = self.get(p)
        return bool(r and r.dead)

    def relocation_score(self, anchor_pos, prior: float, target_pos) -> float:
        """s(a); callers must drop forbidden anchors first (see `rank_anchors`)."""
        r = self.get(anchor_pos)
        u = KNOWN_BONUS if r is not None else UNKNOWN_BONUS
        d = float(np.linalg.norm(np.asarray(anchor_pos, float) - np.asarray(target_pos, float)))
        far = min(d / (2.0 * self.h), DISTANCE_CAP)
        visits = r.visits if r else 0
        promising = 1.0 if (r and r.promising) else 0.0
        return (float(prior) + u + far
                - VISIT_PENALTY * visits - PROMISING_PENALTY * promising)

    def rank_anchors(self, anchors, target_pos):
        """Live anchors, best first. `anchors` are dicts with 'position' and 'prior'."""
        live = [a for a in anchors if not self.is_forbidden(a["position"])]
        return sorted(live, key=lambda a: -self.relocation_score(
            a["position"], a.get("prior", 0.0), target_pos))


__all__ = ["Region", "RegionMemory", "cell_size", "CELL_SCALE", "CELL_MIN",
           "PROMISING_SCORE", "PROMISING_SEMANTIC"]
