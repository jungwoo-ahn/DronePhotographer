"""PhotoFlow's deterministic core: the paper's constants, pairwise selection, memory, scope.

The VLM-driven roles (blueprint / propose / critique) are mocked -- what is pinned here is
every number quoted from arXiv:2605.23771, so a later refactor cannot quietly drift off the
published algorithm.
"""

import json
import math

import numpy as np
import pytest

from src.policy.photoflow.agent import (CANDIDATES_PER_ROUND, ROUNDS, SCOUT_VIEWS,
                                        PhotoFlowAgent, pose_from_look_at)
from src.policy.photoflow.anchors import build_anchor_bank, framing_radius
from src.policy.photoflow.memory import (CELL_MIN, CELL_SCALE, PROMISING_SCORE,
                                         PROMISING_SEMANTIC, RegionMemory, cell_size)
from src.policy.photoflow.reviewer import (M2_FALLOFF, WEIGHTS, combined_score, pairwise_choose,
                                           rule_composition_distance, rule_in_requested_half)

SCOUT = {"scene": "t", "scene_scale": 10.0,
         "extents": {"min": [-20, -20, 0], "max": [20, 20, 8]},
         "subject": {"names": ["person"], "center": [0.0, 0.0, 1.0], "height": 1.8},
         "objects": [{"name": "wall", "center": [6.0, 0.0, 1.5], "size": [1, 8, 3]}],
         "camera": {"hfov_deg": 29.86, "vfov_deg": 22.62}}


# ---------------------------------------------------------------- published constants
def test_budget_is_six_rounds_of_four():
    assert (ROUNDS, CANDIDATES_PER_ROUND) == (6, 4)
    assert PhotoFlowAgent(vlm=None).render_budget == 24


def test_reviewer_weights_are_the_published_ones():
    assert WEIGHTS == {"m1": 0.10, "m2": 0.10, "m3": 0.15, "m4": 0.15, "m5": 0.25, "m6": 0.25}
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    assert M2_FALLOFF == 0.45
    assert combined_score({k: 1.0 for k in WEIGHTS}) == pytest.approx(1.0)


def test_cell_size_follows_the_published_rule():
    assert (CELL_SCALE, CELL_MIN) == (0.12, 0.9)
    assert cell_size(100.0) == pytest.approx(12.0)      # 0.12 * sceneScale dominates
    assert cell_size(1.0) == pytest.approx(0.9)         # the 0.9 floor dominates


def test_m2_falls_off_linearly_and_clamps_at_zero():
    assert rule_composition_distance((0.5, 0.5), (0.5, 0.5)) == pytest.approx(1.0)
    assert rule_composition_distance((0.5 + 0.45, 0.5), (0.5, 0.5)) == pytest.approx(0.0)
    assert rule_composition_distance((0.5 + 0.9, 0.5), (0.5, 0.5)) == 0.0    # no negatives
    assert rule_composition_distance((0.5 + 0.225, 0.5), (0.5, 0.5)) == pytest.approx(0.5)


def test_m1_needs_in_frame_and_the_requested_half():
    assert rule_in_requested_half((0.8, 0.5), True, "right") == 1.0
    assert rule_in_requested_half((0.2, 0.5), True, "right") == 0.0
    assert rule_in_requested_half((0.8, 0.5), False, "right") == 0.0   # out of frame
    assert rule_in_requested_half((1.3, 0.5), True, None) == 0.0       # off-screen centre
    assert rule_in_requested_half((0.2, 0.5), True, None) == 1.0       # no half requested


# ---------------------------------------------------------------- pairwise selection
def _c(J, **m):
    base = {"m1": 1.0, "m2": 0.5, "m3": 0.5, "m4": 0.5, "m5": 0.5, "m6": 0.5}
    return {"J": J, "m": {**base, **m}}


def test_pairwise_takes_the_first_candidate_then_needs_a_higher_J():
    c = _c(0.4)
    assert pairwise_choose(None, c) == (c, True)
    assert pairwise_choose(c, _c(0.3))[1] is False
    assert pairwise_choose(c, _c(0.5))[1] is True


def test_pairwise_refuses_a_higher_J_that_drops_the_framing_rule():
    """The trade the paper's pairwise step exists to refuse: prettier, subject lost."""
    inc = _c(0.40, m1=1.0)
    better_but_unframed = _c(0.80, m1=0.0, m5=1.0, m6=1.0)
    winner, replaced = pairwise_choose(inc, better_but_unframed)
    assert replaced is False and winner is inc


def test_pairwise_refuses_a_higher_J_that_regresses_two_dimensions_badly():
    inc = _c(0.50, m3=0.9, m4=0.9)
    unstable = _c(0.55, m3=0.1, m4=0.1, m5=1.0, m6=1.0)
    assert pairwise_choose(inc, unstable)[1] is False


# ---------------------------------------------------------------- region memory
def test_region_promising_thresholds():
    m = RegionMemory(1.0)
    assert m.visit([0, 0, 0], PROMISING_SCORE, 0.0, improved=True).promising
    assert m.visit([9, 9, 9], 0.0, PROMISING_SEMANTIC, improved=True).promising
    assert not m.visit([20, 20, 20], 0.5, 0.5, improved=True).promising


def test_repeated_poor_visits_kill_a_region_and_forbid_it():
    m = RegionMemory(1.0)
    p = [0.1, 0.1, 0.1]
    for _ in range(3):
        m.visit(p, 0.1, 0.1, improved=False)
    assert m.get(p).dead and m.is_forbidden(p)
    assert not m.is_forbidden([50, 50, 50])


def test_stagnation_without_improvement_kills_a_region():
    m = RegionMemory(1.0)
    p = [0.1, 0.1, 0.1]
    m.visit(p, 0.9, 0.9, improved=True)          # a good first hit, so poor_hits stays 0
    for _ in range(3):
        m.visit(p, 0.5, 0.5, improved=False)     # never beats the best -> stagnation
    assert m.get(p).poor_hits == 0 and m.get(p).dead


def test_relocation_prefers_unknown_and_distant_and_penalises_revisits():
    m = RegionMemory(1.0)
    here = [0.0, 0.0, 0.0]
    for _ in range(2):
        m.visit(here, 0.5, 0.5, improved=False)
    unknown_far = m.relocation_score([10.0, 0.0, 0.0], 0.0, here)
    known_near = m.relocation_score(here, 0.0, here)
    assert unknown_far > known_near
    # u(a): unknown 1.2 vs known 0.25, distance capped at 2.0, visits -0.35 each
    assert unknown_far == pytest.approx(1.2 + 2.0)
    assert known_near == pytest.approx(0.25 + 0.0 - 0.35 * 2)


def test_rank_anchors_drops_forbidden_regions():
    m = RegionMemory(1.0)
    bad = [0.1, 0.1, 0.1]
    for _ in range(3):
        m.visit(bad, 0.1, 0.1, improved=False)
    anchors = [{"position": bad, "prior": 9.0, "source": "dead"},
               {"position": [9.0, 9.0, 9.0], "prior": 0.0, "source": "live"}]
    ranked = m.rank_anchors(anchors, [0, 0, 0])
    assert [a["source"] for a in ranked] == ["live"]     # high prior does not revive a dead cell


# ---------------------------------------------------------------- anchors & pose
def test_framing_radius_matches_the_lens_geometry():
    # a 1.8 m subject filling half of a 22.62 deg vertical frame
    r = framing_radius(1.8, 0.5, 22.62)
    assert r == pytest.approx(1.8 / (2 * 0.5 * math.tan(math.radians(22.62) / 2)), rel=1e-6)
    assert framing_radius(1.8, 0.8) < framing_radius(1.8, 0.35)   # tighter fill = closer


def test_anchor_bank_is_built_without_any_goal_information():
    bank = build_anchor_bank(SCOUT, {"look_toward": [[1.0, 0.0, 1.0]]})
    assert len(bank) >= 8 * 3 * 3
    assert {"position", "look_at", "lens_mm", "aspect", "prior", "source"} <= set(bank[0])
    assert {a["lens_mm"] for a in bank} == {24.0}          # lens is fixed, not searched
    srcs = {a["source"].split(":")[0] for a in bank}
    assert {"scene_bbox", "blueprint", "visibility"} <= srcs
    lo = np.asarray(SCOUT["extents"]["min"], float)
    hi = np.asarray(SCOUT["extents"]["max"], float)
    for a in bank:                                         # every anchor inside the scene
        assert np.all(np.asarray(a["position"]) >= lo - 1e-6)
        assert np.all(np.asarray(a["position"]) <= hi + 1e-6)


def test_pose_from_look_at_aims_at_the_target_with_level_up():
    pos, fwd, up = pose_from_look_at([5.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    assert np.allclose(fwd, [-1.0, 0.0, 0.0], atol=1e-6)
    assert np.dot(fwd, up) == pytest.approx(0.0, abs=1e-6)
    assert up[2] > 0.99                                    # no induced roll
    _, f2, u2 = pose_from_look_at([0.0, 0.0, 9.0], [0.0, 0.0, 0.0])   # straight down
    assert np.allclose(f2, [0.0, 0.0, -1.0], atol=1e-6)
    assert np.dot(f2, u2) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- the loop
class _VLM:
    """Scripted stand-in. Scores a preview by the IMAGE it is given, never by call order --
    the critiques run concurrently, so an order-dependent mock would make these tests vacuous."""

    # Off the symmetry axis on purpose: at (3,0,1) the candidates at (0,3,1) and (0,-3,1) are
    # equidistant and score identically, which would hide a slate-mixing bug.
    def __init__(self, sweet=(3.0, 1.0, 1.0)):
        self.sweet = np.asarray(sweet, float)
        self.by_image: dict[str, np.ndarray] = {}

    def ask_json(self, system, user, image=None, images=()):
        # Dispatch on the IMAGE, not on prompt text: the critique prompt mentions "blueprint",
        # so matching that word sent critiques to the blueprint branch and quietly zeroed
        # m3-m6 in every test that used this mock.
        if image is None:
            return self._plan(system)
        pos = self.by_image[image]                      # KeyError here = a critique lost its image
        q = float(np.exp(-np.linalg.norm(pos - self.sweet)))
        return {"m3_composition": q, "m4_technical": q, "m5_aesthetic": q,
                "m6_alignment": q, "critique": "move in"}

    def _plan(self, system):
        if "candidate camera states" in system:
            return {"candidates": [{"position": [3.0, 0.0, 1.0], "look_at": [0, 0, 1], "why": "a"},
                                   {"position": [0.0, 3.0, 1.0], "look_at": [0, 0, 1], "why": "b"},
                                   {"position": [-3.0, 0.0, 1.0], "look_at": [0, 0, 1], "why": "c"},
                                   {"position": [0.0, -3.0, 1.0], "look_at": [0, 0, 1], "why": "d"}]}
        return {"primary_subject": "person", "look_toward": [[0.0, 0.0, 1.0]]}


def _runner(vlm, budget=None):
    rendered = []

    def render(position, look_at):
        # a distinct image per preview, so the mock can score position-faithfully
        img = f"/tmp/pf_{len(rendered)}.jpg"
        rendered.append(list(position))
        vlm.by_image[img] = np.asarray(position, float)
        return img, {"object_center_x": 512, "object_center_y": 384}, True

    agent = PhotoFlowAgent(vlm, render_budget=budget)
    best, info = agent.search(render, SCOUT, "a medium shot from the front")
    return best, info, rendered


def test_search_respects_the_render_budget_and_returns_the_best_state():
    vlm = _VLM()
    best, info, rendered = _runner(vlm)
    # `renders` counts PREVIEWS, which is what the budget caps. The Director's establishing
    # scout views are separate renders and are never scored or selected, so they sit outside
    # it -- but they are still renders, so total cost is the sum.
    assert info["renders"] <= 24
    assert info["scout_views"] == SCOUT_VIEWS
    assert len(rendered) == info["renders"] + info["scout_views"]
    assert np.allclose(best["position"], [3.0, 0.0, 1.0])     # the scripted sweet spot
    assert best["J"] == pytest.approx(combined_score(best["m"]))


def test_scout_views_are_establishing_shots_not_candidates():
    """They must never be scored, selected, or counted against the preview budget."""
    vlm = _VLM()
    best, info, rendered = _runner(vlm, budget=4)
    scored = [e for e in info["log"] if "J" in e]
    assert len(scored) == 4                       # only the 4 previews were reviewed
    # the first SCOUT_VIEWS renders are the establishing ring, none of them a scored candidate
    ring = rendered[:SCOUT_VIEWS]
    assert len(ring) == SCOUT_VIEWS
    for e in scored:
        assert not any(np.allclose(e["position"], r) for r in ring)


def test_every_preview_is_scored_from_its_own_image():
    """Guards the concurrency: a slate of 4 must get 4 distinct J values, not 4 copies."""
    vlm = _VLM()
    best, info, _ = _runner(vlm, budget=4)
    js = [e["J"] for e in info["log"] if "J" in e]
    assert len(js) == 4 and len(set(js)) == 4, f"previews not scored independently: {js}"
    assert max(js) == pytest.approx(js[0])            # [3,0,1] is the sweet spot, proposed first
    assert np.allclose(best["position"], [3.0, 0.0, 1.0])


def test_render_matched_budget_caps_renders_for_the_fair_ablation():
    _, info, rendered = _runner(_VLM(), budget=4)
    assert info["renders"] == 4
    assert len(rendered) == 4 + SCOUT_VIEWS


def test_search_survives_a_vlm_that_returns_nothing():
    class _Dead:
        def ask_json(self, system, user, image=None, images=()):
            return None

    def render(position, look_at):
        return "/tmp/x.jpg", {"object_center_x": 512, "object_center_y": 384}, True

    agent = PhotoFlowAgent(_Dead(), render_budget=6)
    best, info = agent.search(render, SCOUT, "a shot")
    assert info["renders"] == 6          # falls back to the anchor bank instead of stalling
    assert best is not None and info["vlm_failures"] > 0


# ---------------------------------------------------------------- bearing convention
def test_bearing_convention_places_the_camera_on_the_correct_side():
    """The instruction handed to the Director must reproduce the goal pose, not its antipode.

    `pose_to_geometry`'s `az` is the CAMERA->SUBJECT direction, so the camera stands at
    az + 180. An earlier version of this instruction omitted the 180 and would have sent the
    Director to the opposite side of every subject.
    """
    import math

    from src.policy.common.reward import pose_to_geometry

    subject = np.array([0.043, 5.747, 0.421])
    camera = np.array([-3.01, 9.05, 0.48])          # a real manifest goal pose
    geo = pose_to_geometry(camera, subject - camera, [0, 0, 1], subject, 0.84)
    bearing = 137.0
    front_az = (bearing + math.degrees(geo["az"])) % 360.0

    theta = math.radians((front_az - bearing + 180.0) % 360.0)
    r = float(np.linalg.norm((subject - camera)[:2]))
    placed = subject + np.array([r * math.cos(theta), r * math.sin(theta), 0.0])

    assert np.allclose(placed[:2], camera[:2], atol=1e-6), "camera placed on the wrong side"
    # and the naive version (no +180) must land on the opposite side, i.e. be wrong
    theta_bad = math.radians((front_az - bearing) % 360.0)
    bad = subject + np.array([r * math.cos(theta_bad), r * math.sin(theta_bad), 0.0])
    assert np.linalg.norm(bad[:2] - camera[:2]) > r, "the naive convention should be ~2r away"
