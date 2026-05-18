from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vlm_qwen25.dataset import DroneActionScoreDataset
from src.vlm_qwen25.rotation_utils import (
    batch_relative_rotation_angle_deg,
    relative_rotation_matrix,
    rotation_matrix_to_rotvec,
)


# ---------------------------------------------------------------------------
# batch_relative_rotation_angle_deg
# ---------------------------------------------------------------------------


def _single_angle_deg(fwd_i, up_i, fwd_j, up_j):
    rotation = relative_rotation_matrix(fwd_i, up_i, fwd_j, up_j)
    rotvec = rotation_matrix_to_rotvec(rotation)
    return float(np.degrees(np.linalg.norm(rotvec)))


def test_batch_angle_identity_is_zero():
    fwd = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    angs = batch_relative_rotation_angle_deg(
        fwd, up, np.tile(fwd, (5, 1)), np.tile(up, (5, 1))
    )
    assert np.allclose(angs, 0.0, atol=1e-3)


def test_batch_angle_90deg_yaw():
    fwd_a = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    up_a = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fwd_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    angs = batch_relative_rotation_angle_deg(fwd_a, up_a, fwd_b[None, :], up_b[None, :])
    assert abs(float(angs[0]) - 90.0) < 0.1


def test_batch_angle_matches_single_pair():
    rng = np.random.default_rng(0)
    n = 64
    fwd_i = rng.standard_normal(3).astype(np.float32)
    fwd_i /= np.linalg.norm(fwd_i)
    up_i = rng.standard_normal(3).astype(np.float32)
    up_i = up_i - np.dot(up_i, fwd_i) * fwd_i
    up_i /= np.linalg.norm(up_i)

    fwds = rng.standard_normal((n, 3)).astype(np.float32)
    fwds /= np.linalg.norm(fwds, axis=1, keepdims=True)
    ups = rng.standard_normal((n, 3)).astype(np.float32)
    ups = ups - (ups * fwds).sum(axis=1, keepdims=True) * fwds
    ups /= np.linalg.norm(ups, axis=1, keepdims=True)

    batch = batch_relative_rotation_angle_deg(fwd_i, up_i, fwds, ups)
    single = np.array(
        [_single_angle_deg(fwd_i, up_i, fwds[k], ups[k]) for k in range(n)]
    )
    assert np.max(np.abs(batch - single)) < 1e-3


# ---------------------------------------------------------------------------
# DroneActionScoreDataset filters + per-placement grouping
# ---------------------------------------------------------------------------


def _make_synthetic_placement(out_dir: Path, placement_id: str, n_views: int, *, seed: int = 0):
    """Create a fake placement dir with <n_views> trivial views.

    Camera positions are random in [0, 1)^3 (so distances are bounded < 1.73).
    Forward = -Z, Up = +Y for all views (so all rotation pair-angles are 0,
    keeping rotation filter inert and isolating distance-binning behaviour).
    Each view writes a 4x4 RGB PNG image and a v5 score block.
    """
    from PIL import Image

    from src.scoring.bbox_control import compute_v5_scores

    rng = np.random.default_rng(seed)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for k in range(n_views):
        img_name = f"img_{k:04d}.png"
        Image.new("RGB", (4, 4), (k % 256, 0, 0)).save(images_dir / img_name)
        x, y, z = rng.uniform(0.0, 1.0, size=3).tolist()
        bbox = (100.0, 100.0, 300.0, 400.0)
        v5 = compute_v5_scores(1024, 768, bbox, azimuth_deg=0.0, elevation_deg=0.0)
        item = {
            "image": f"images/{img_name}",
            "camera_position": [float(x), float(y), float(z)],
            "final_forward": [0.0, 0.0, -1.0],
            "final_up": [0.0, 1.0, 0.0],
            "detections": [{"bbox": list(bbox), "score": 0.9, "label": "obj"}],
            "bbox_2d_full_projected": list(bbox),
        }
        for key, val in v5.items():
            item[f"score_{key}"] = val
        items.append(item)
    (out_dir / "annotations.json").write_text(json.dumps(items))


def _build_dataset(annotations_path: Path, **overrides) -> DroneActionScoreDataset:
    base = dict(
        annotations_path=str(annotations_path),
        action_frame="camera_local",
        rotation_representation="orientation_6d",
        distance_threshold=1.0,
        rotation_threshold_deg=180.0,  # disable rotation filter by default
        pair_distance_distribution="natural",
        n_distance_bins=5,
        min_pair_distance=0.05,
        max_pairs_per_image=64,
        zero_action_ratio=0.0,
        target_score_keys=[
            "occupancy",
            "body_in_frame_ratio",
            "cam_to_obj_azimuth_deg",
            "cam_to_obj_elevation_deg",
            "object_center_x",
            "object_center_y",
            "bbox_x_offset",
            "bbox_y_offset",
        ],
        seed=0,
    )
    base.update(overrides)
    return DroneActionScoreDataset(**base)


def test_pairs_stay_within_placement(tmp_path):
    root = tmp_path / "fake_run"
    _make_synthetic_placement(root / "p0_sceneA", "p0_sceneA", n_views=20, seed=1)
    _make_synthetic_placement(root / "p1_sceneB", "p1_sceneB", n_views=20, seed=2)

    ds = _build_dataset(root)
    assert len(ds) > 0
    for pair in ds.pairs:
        pid_i = ds.views[pair.index_i].placement_id
        pid_j = ds.views[pair.index_j].placement_id
        assert pid_i == pid_j, f"cross-placement pair leaked: {pid_i} vs {pid_j}"


def test_log_uniform_binning_flattens_distance_distribution(tmp_path):
    root = tmp_path / "fake_run"
    # One placement with many views so binning has data per bin.
    _make_synthetic_placement(root / "p0_scene", "p0_scene", n_views=300, seed=3)

    ds = _build_dataset(
        root,
        pair_distance_distribution="log_uniform",
        n_distance_bins=4,
        min_pair_distance=0.05,
        distance_threshold=1.0,
        max_pairs_per_image=4,  # 1 per bin target
    )

    edges = np.exp(np.linspace(np.log(0.05), np.log(1.0), 5))
    dists = []
    for pair in ds.pairs:
        if pair.index_i == pair.index_j:
            continue
        p_i = ds.views[pair.index_i].camera_position
        p_j = ds.views[pair.index_j].camera_position
        dists.append(float(np.linalg.norm(p_i - p_j)))
    hist, _ = np.histogram(dists, bins=edges)
    # With ~1-per-bin cap and dense uniform sources, every bin should be populated.
    assert (hist > 0).all(), f"some bins empty under log_uniform binning: {hist}"
    # The ratio between max and min bin should be modest (not the natural skew).
    assert hist.max() / max(1, hist.min()) < 4, f"binning too uneven: {hist}"


def test_natural_distribution_reproduces_old_behavior(tmp_path):
    root = tmp_path / "fake_run"
    _make_synthetic_placement(root / "p0_scene", "p0_scene", n_views=80, seed=4)
    ds = _build_dataset(
        root,
        pair_distance_distribution="natural",
        max_pairs_per_image=8,
    )
    # Natural cap is per-source-view, so pair count <= n_views * max_pairs_per_image.
    n_views = sum(1 for v in ds.views)
    assert len(ds.pairs) <= n_views * 8


def test_rotation_filter_blocks_far_orientations(tmp_path):
    root = tmp_path / "fake_run"
    pdir = root / "p0_scene"
    pdir.mkdir(parents=True)
    images_dir = pdir / "images"
    images_dir.mkdir()
    from PIL import Image

    from src.scoring.bbox_control import compute_v5_scores

    bbox = (100.0, 100.0, 300.0, 400.0)
    v5_scores = compute_v5_scores(1024, 768, bbox, azimuth_deg=0.0, elevation_deg=0.0)

    def _v5_block(item: dict) -> dict:
        for key, val in v5_scores.items():
            item[f"score_{key}"] = val
        return item

    items = []
    # Two views very close in position but with opposite forward.
    for k, (pos, fwd) in enumerate(
        [
            ((0.0, 0.0, 0.0), [0, 0, -1]),
            ((0.05, 0.0, 0.0), [0, 0, 1]),
        ]
    ):
        Image.new("RGB", (4, 4)).save(images_dir / f"img_{k:04d}.png")
        items.append(
            _v5_block(
                {
                    "image": f"images/img_{k:04d}.png",
                    "camera_position": list(pos),
                    "final_forward": fwd,
                    "final_up": [0.0, 1.0, 0.0],
                    "detections": [{"bbox": list(bbox), "score": 0.9, "label": "x"}],
                    "bbox_2d_full_projected": list(bbox),
                }
            )
        )
    (pdir / "annotations.json").write_text(json.dumps(items))

    ds = _build_dataset(
        root,
        rotation_threshold_deg=30.0,
        distance_threshold=1.0,
    )
    # No pair should connect view 0 (-z forward) and view 1 (+z forward) since
    # their relative rotation is 180 deg > 30 deg threshold.
    forbidden = {(0, 1), (1, 0)}
    for pair in ds.pairs:
        assert (pair.index_i, pair.index_j) not in forbidden
