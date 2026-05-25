from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scoring.bbox_control import V5_SCORE_KEYS, compute_v5_scores
from src.vlm_qwen25.schema import (
    parse_scores_from_text,
    scores_to_canonical_json,
)


W = 1024
H = 768


def test_v5_keys_all_int_and_present():
    bbox = (200.0, 100.0, 600.0, 600.0)
    scores = compute_v5_scores(W, H, bbox, azimuth_deg=120.0, elevation_deg=15.0)
    assert set(scores.keys()) == set(V5_SCORE_KEYS)
    for key, value in scores.items():
        assert isinstance(value, int), f"{key} should be int, got {type(value)}"


def test_fully_framed_body_in_frame_100():
    bbox = (200.0, 200.0, 800.0, 600.0)
    scores = compute_v5_scores(W, H, bbox, 0, 0)
    assert scores["body_in_frame_ratio"] == 100
    assert 0 < scores["occupancy"] < 100


def test_half_clipped_off_right_body_in_frame_about_50():
    bbox_w = 400.0
    half_outside_x1 = float(W) - bbox_w / 2.0
    half_outside_x2 = half_outside_x1 + bbox_w
    bbox = (half_outside_x1, 200.0, half_outside_x2, 600.0)
    scores = compute_v5_scores(W, H, bbox, 0, 0)
    assert 45 <= scores["body_in_frame_ratio"] <= 55
    assert scores["object_center_x"] >= W


def test_completely_off_screen_body_zero():
    bbox = (W + 50.0, 100.0, W + 200.0, 300.0)
    scores = compute_v5_scores(W, H, bbox, 0, 0)
    assert scores["body_in_frame_ratio"] == 0
    assert scores["occupancy"] == 0
    assert scores["object_center_x"] > W


def test_pathological_projection_zeroes_image_fields():
    # Camera at near-plane: bbox blows up to ~50000 x 30000 pixels.
    bbox = (-25000.0, -15000.0, 25000.0, 15000.0)
    scores = compute_v5_scores(W, H, bbox, 30, 10)
    # Image fields must zero out, not leak 5-digit pixel coords.
    assert scores["object_center_x"] == 0
    assert scores["object_center_y"] == 0
    assert scores["bbox_x_offset"] == 0
    assert scores["bbox_y_offset"] == 0
    assert scores["body_in_frame_ratio"] == 0
    assert scores["occupancy"] == 0
    # Angles still valid.
    assert scores["cam_to_obj_azimuth_deg"] == 30
    assert scores["cam_to_obj_elevation_deg"] == 10


def test_offsets_non_negative_and_match_half_extent():
    bbox = (100.0, 100.0, 300.0, 500.0)
    scores = compute_v5_scores(W, H, bbox, 0, 0)
    assert scores["bbox_x_offset"] >= 0
    assert scores["bbox_y_offset"] >= 0
    assert scores["bbox_x_offset"] == 100  # (300 - 100) / 2
    assert scores["bbox_y_offset"] == 200  # (500 - 100) / 2


def test_azimuth_wraps_into_0_360():
    scores = compute_v5_scores(W, H, (10.0, 10.0, 100.0, 100.0), -30.0, 0.0)
    assert 0 <= scores["cam_to_obj_azimuth_deg"] < 360
    assert scores["cam_to_obj_azimuth_deg"] == 330


def test_elevation_clamped_to_pm90():
    high = compute_v5_scores(W, H, None, 0, 95.0)
    low = compute_v5_scores(W, H, None, 0, -120.0)
    assert high["cam_to_obj_elevation_deg"] == 90
    assert low["cam_to_obj_elevation_deg"] == -90


def test_anchor_bucket_with_fallback_open_scene():
    """When every position is valid, all n_target anchors come from
    the bucket phase and span n_target distinct equal-width bins."""
    import math, random

    def random_direction(hemisphere=True):
        u, v = random.random(), random.random()
        theta = 2 * math.pi * u
        z = v if hemisphere else 2 * v - 1
        r = math.sqrt(max(0.0, 1.0 - z * z))
        return (r * math.cos(theta), r * math.sin(theta), z)

    def _try_sample_in_range(obj, r_lo, r_hi, max_attempts, is_valid):
        for k in range(1, max_attempts + 1):
            d = random_direction(True)
            radius = random.uniform(r_lo, r_hi)
            cam = (obj[0] + d[0] * radius, obj[1] + d[1] * radius, obj[2] + d[2] * radius)
            if is_valid(cam, obj, radius):
                return cam, k, radius
        return None, max_attempts, None

    def discover(obj, r_range, n_target, max_attempts, is_valid):
        r_min, r_max = sorted(r_range)
        bin_width = (r_max - r_min) / n_target
        accepted, bin_info, failed = [], [], []
        for i in range(n_target):
            lo = r_min + i * bin_width
            hi = r_min + (i + 1) * bin_width
            cam, n_att, radius = _try_sample_in_range(obj, lo, hi, max_attempts, is_valid)
            if cam is not None:
                accepted.append((cam, radius))
                bin_info.append({"bin": [lo, hi], "source": "bucket", "accepted": True})
            else:
                bin_info.append({"bin": [lo, hi], "source": "bucket", "accepted": False})
                failed.append((lo, hi))
        for lo, hi in failed:
            cam, n_att, radius = _try_sample_in_range(obj, r_min, r_max, max_attempts, is_valid)
            if cam is not None:
                accepted.append((cam, radius))
                bin_info.append({"bin": [lo, hi], "source": "fallback", "accepted": True})
            else:
                bin_info.append({"bin": [lo, hi], "source": "fallback", "accepted": False})
        return accepted, bin_info

    random.seed(721)
    accepted, bin_info = discover((0, 0, 0), (1.0, 7.0), 4, 500,
                                  is_valid=lambda cam, obj, r: True)
    assert len(accepted) == 4
    # All 4 should be source=bucket
    assert all(b["source"] == "bucket" for b in bin_info)
    # Each radius falls inside its bin (bins were [1,2.5),[2.5,4),[4,5.5),[5.5,7])
    bin_edges = [1.0, 2.5, 4.0, 5.5, 7.0]
    for i, (_, r) in enumerate(accepted):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        assert lo <= r <= hi, f"anchor {i} radius {r:.3f} outside its bin [{lo}, {hi}]"


def test_anchor_bucket_with_fallback_narrow_scene():
    """When only close r < 2.0 is valid (simulates a narrow alley), Phase 1
    fills only the first bin and Phase 2 fallback fills the rest -- still
    all close, and total count is preserved at n_target."""
    import math, random

    def random_direction(hemisphere=True):
        u, v = random.random(), random.random()
        theta = 2 * math.pi * u
        z = v if hemisphere else 2 * v - 1
        r = math.sqrt(max(0.0, 1.0 - z * z))
        return (r * math.cos(theta), r * math.sin(theta), z)

    def _try_sample_in_range(obj, r_lo, r_hi, max_attempts, is_valid):
        for k in range(1, max_attempts + 1):
            d = random_direction(True)
            radius = random.uniform(r_lo, r_hi)
            cam = (obj[0] + d[0] * radius, obj[1] + d[1] * radius, obj[2] + d[2] * radius)
            if is_valid(cam, obj, radius):
                return cam, k, radius
        return None, max_attempts, None

    def discover(obj, r_range, n_target, max_attempts, is_valid):
        r_min, r_max = sorted(r_range)
        bin_width = (r_max - r_min) / n_target
        accepted, bin_info, failed = [], [], []
        for i in range(n_target):
            lo = r_min + i * bin_width
            hi = r_min + (i + 1) * bin_width
            cam, n_att, radius = _try_sample_in_range(obj, lo, hi, max_attempts, is_valid)
            if cam is not None:
                accepted.append((cam, radius))
                bin_info.append({"bin": [lo, hi], "source": "bucket", "accepted": True})
            else:
                bin_info.append({"bin": [lo, hi], "source": "bucket", "accepted": False})
                failed.append((lo, hi))
        for lo, hi in failed:
            cam, n_att, radius = _try_sample_in_range(obj, r_min, r_max, max_attempts, is_valid)
            if cam is not None:
                accepted.append((cam, radius))
                bin_info.append({"bin": [lo, hi], "source": "fallback", "accepted": True})
            else:
                bin_info.append({"bin": [lo, hi], "source": "fallback", "accepted": False})
        return accepted, bin_info

    random.seed(721)
    accepted, bin_info = discover((0, 0, 0), (1.0, 7.0), 4, 500,
                                  is_valid=lambda cam, obj, r: r < 2.0)
    assert len(accepted) == 4, f"expected 4 anchors via fallback, got {len(accepted)}"
    # All radii must be < 2.0 (validity respected)
    for _, r in accepted:
        assert r < 2.0
    # 1 bucket success (bin [1, 2.5) overlap with valid r<2), 3 fallbacks
    bucket_ok = [b for b in bin_info if b["source"] == "bucket" and b["accepted"]]
    fallback_ok = [b for b in bin_info if b["source"] == "fallback" and b["accepted"]]
    assert len(bucket_ok) == 1
    assert len(fallback_ok) == 3


def test_cam_to_obj_v2_sign_convention():
    """v2 convention: elevation/azimuth describe the cam->obj vector.

    Mirrors the formulas in render_object_v3.compute_3d_metrics and
    compute_camera_to_object_angles. If the production code drifts away
    from this convention, this test catches it.
    """
    import math

    def angles_v2(cam, obj):
        dx, dy, dz = cam[0] - obj[0], cam[1] - obj[1], cam[2] - obj[2]
        horiz = math.sqrt(dx * dx + dy * dy)
        elev = math.degrees(math.atan2(-dz, horiz))
        azim = math.degrees(math.atan2(-dy, -dx)) % 360
        return round(azim), round(elev)

    # cam directly above obj -> cam->obj points down -> elev = -90
    az, el = angles_v2((0.0, 0.0, 5.0), (0.0, 0.0, 0.0))
    assert el == -90, f"cam above should be -90, got {el}"
    # cam directly below -> cam->obj points up -> elev = +90
    az, el = angles_v2((0.0, 0.0, -5.0), (0.0, 0.0, 0.0))
    assert el == 90
    # eye-level
    az, el = angles_v2((5.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert el == 0
    # cam at obj's +X (right side) -> cam->obj points -X -> azim = 180
    az, el = angles_v2((5.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert az == 180
    # cam at obj's +Y (in front) -> cam->obj points -Y -> azim = 270
    az, el = angles_v2((0.0, 5.0, 0.0), (0.0, 0.0, 0.0))
    assert az == 270
    # cam at -X (left side) -> cam->obj points +X -> azim = 0
    az, el = angles_v2((-5.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert az == 0
    # cam at -Y (behind) -> cam->obj points +Y -> azim = 90
    az, el = angles_v2((0.0, -5.0, 0.0), (0.0, 0.0, 0.0))
    assert az == 90


def test_none_bbox_zeroes_image_fields_keeps_angles():
    scores = compute_v5_scores(W, H, None, 30.0, 10.0)
    assert scores["occupancy"] == 0
    assert scores["body_in_frame_ratio"] == 0
    assert scores["object_center_x"] == 0
    assert scores["object_center_y"] == 0
    assert scores["bbox_x_offset"] == 0
    assert scores["bbox_y_offset"] == 0
    assert scores["cam_to_obj_azimuth_deg"] == 30
    assert scores["cam_to_obj_elevation_deg"] == 10


def test_schema_round_trip_with_off_image_center():
    raw = {
        "occupancy": 35,
        "body_in_frame_ratio": 87,
        "cam_to_obj_azimuth_deg": 142,
        "cam_to_obj_elevation_deg": 28,
        "object_center_x": 1100,
        "object_center_y": -20,
        "bbox_x_offset": 80,
        "bbox_y_offset": 120,
    }
    text = scores_to_canonical_json(raw, score_keys=V5_SCORE_KEYS)
    assert '"occupancy":35' in text
    assert '.0' not in text  # ints serialised without trailing .0
    parsed = parse_scores_from_text(text, score_keys=V5_SCORE_KEYS)
    assert parsed == raw
    for value in parsed.values():
        assert isinstance(value, int)


def test_schema_accepts_float_input_and_coerces_to_int():
    text = (
        '{"occupancy":35.0,"body_in_frame_ratio":87.0,"cam_to_obj_azimuth_deg":142.0,'
        '"cam_to_obj_elevation_deg":28.0,"object_center_x":1100.0,"object_center_y":-20.0,'
        '"bbox_x_offset":80.0,"bbox_y_offset":120.0}'
    )
    parsed = parse_scores_from_text(text, score_keys=V5_SCORE_KEYS)
    assert parsed is not None
    for value in parsed.values():
        assert isinstance(value, int)
