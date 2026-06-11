"""End-to-end integration test for v7 Stage-2/3 data through the Cosmos policy.

Builds a synthetic mini-placement directory matching the v7 schema (per the
handoff at `docs/v7_handoff_jooyeol.md` and the verified `data.json` schema on
the v7_data_for_cosmos_policy branch), then runs:

  - v7 annotation iterator → window count + frame indexing
  - CosmosDroneDataset(data_mode="v7") → tensor shapes
  - CosmosWorldActionPolicy.compute_loss with a mock backbone → joint
    EDM loss with per-component breakdown
  - CosmosWorldActionPolicy.sample with a mock backbone → action chunk + value

We don't run the real Cosmos-Predict2.5-2B backbone — that's a 2B-param model
and not available in this test environment. The mock backbone has the exact
same call signature so the data path is fully exercised.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")
from torch import nn

from src.policy.common.action_repr import ACTION_DIM
from src.policy.common.annotations import iter_windows
from src.policy.cosmos.dataset import CosmosDroneDataset
from src.policy.cosmos.model import CosmosWorldActionPolicy
from src.policy.cosmos.vae import CosmosVAEWrapper


# --- Synthetic v7 data generator -------------------------------------------------


def _make_dummy_frame(theta: float, radius: float = 3.0, height: float = 1.0) -> dict:
    """One trajectory_32f frame: camera orbiting the origin on a circle of radius r at height h."""
    pos = [radius * float(np.cos(theta)), radius * float(np.sin(theta)), height]
    # Camera looking back at origin
    forward = [-float(np.cos(theta)), -float(np.sin(theta)), 0.0]
    up = [0.0, 0.0, 1.0]
    return {
        "pos": pos,
        "forward": forward,
        "up": up,
        "pitch_deg": 5.0,
        "yaw_deg": float(np.degrees(theta)),
    }


def _make_accepted_pair(seed: int, traj_len: int = 32) -> dict:
    """One accepted_pairs[i] entry with a 32-frame circular trajectory."""
    rng = np.random.default_rng(seed)
    theta0 = float(rng.uniform(0, np.pi))
    span = float(rng.uniform(np.pi / 3, np.pi / 2))
    thetas = np.linspace(theta0, theta0 + span, traj_len)
    traj = [_make_dummy_frame(float(t)) for t in thetas]
    return {
        "start": {"pos": traj[0]["pos"], "forward": traj[0]["forward"], "up": traj[0]["up"],
                  "r": 3.0, "az_deg": 0.0, "elev_deg": 0.0,
                  "pitch_jitter_deg": 0.0, "yaw_jitter_deg": 0.0, "bbox_occupancy": 0.2},
        "end": {"pos": traj[-1]["pos"], "forward": traj[-1]["forward"], "up": traj[-1]["up"],
                "r": 3.0, "az_deg": 0.0, "elev_deg": 0.0,
                "pitch_jitter_deg": 0.0, "yaw_jitter_deg": 0.0, "bbox_occupancy": 0.3},
        "C_far": traj[0]["pos"], "C_near": traj[-1]["pos"],
        "c_far_is_start": True,
        "pitch_far_deg": 5.0, "pitch_near_deg": 5.0,
        "yaw_far_deg": float(thetas[0]), "yaw_near_deg": float(thetas[-1]),
        "ellipse": {"a": 3.0, "b": 2.0, "theta_far_deg": 0.0, "theta_near_deg": float(span),
                    "u": [1, 0, 0], "v": [0, 1, 0], "O": [0, 0, 0]},
        "trajectory_32f": traj,
    }


def _build_dummy_v7_placement(
    placement_dir: Path,
    *,
    n_accepted_pairs: int = 2,
    frames_per_pair: int = 32,
    image_resolution: tuple[int, int] = (16, 16),  # (H, W) — tiny for tests
) -> Path:
    """Build a synthetic placement dir with data.json + small JPEGs.

    Layout matches v7:
      <placement_dir>/data.json
      <placement_dir>/renders/pair_<pp>_frame_<ff>.jpg
      <placement_dir>/done.flag
      <placement_dir>/scored.flag
    """
    from PIL import Image

    placement_dir.mkdir(parents=True, exist_ok=True)
    (placement_dir / "renders").mkdir(exist_ok=True)

    accepted_pairs = [_make_accepted_pair(seed=i, traj_len=frames_per_pair)
                      for i in range(n_accepted_pairs)]
    render_records: list[list[dict]] = []
    for i in range(n_accepted_pairs):
        pair_recs = []
        for j in range(frames_per_pair):
            rel = f"renders/pair_{i:02d}_frame_{j:02d}.jpg"
            # Write a tiny JPEG with a deterministic per-frame color
            img = Image.new(
                "RGB", (image_resolution[1], image_resolution[0]),
                color=(i * 20 % 255, j * 5 % 255, 128),
            )
            img.save(placement_dir / rel, "JPEG")
            pair_recs.append({
                "frame_idx": j,
                "path_rel": rel,
                "bbox_xyxy_full": [100.0, 100.0, 300.0, 400.0],
                "occupancy_clipped": 0.2,
                "in_frame": True,
                # 8 V5 scores (integers) — Stage 3 output
                "scores": {
                    "occupancy": 20 + j,
                    "body_in_frame_ratio": 95,
                    "cam_to_obj_azimuth_deg": (j * 10) % 360 - 180,
                    "cam_to_obj_elevation_deg": -10,
                    "object_center_x": 419 + j,
                    "object_center_y": 282 + j,
                    "bbox_x_offset": 117,
                    "bbox_y_offset": 310,
                },
            })
        render_records.append(pair_recs)

    data = {
        "placement": placement_dir.name,
        "placement_idx": 0,
        "scene_file": "data/scenes/Test-scene_0000abcd/scene.blend",
        "object_file": "data/objects/Test-object_1111efgh/obj.blend",
        "scene": "Test-scene_0000abcd",
        "object": "Test-object_1111efgh",
        "scene_scale": 1.0,
        "subject_foot": [0.0, 0.0, 0.0],
        "subject_center": [0.0, 0.0, 0.9],
        "subject_height": 1.8,
        "subject_names": ["Person"],
        "seed": 0,
        "K_target": n_accepted_pairs,
        "K_accepted": n_accepted_pairs,
        "attempts_used": n_accepted_pairs * 6,
        "time_sample_s": 1.0,
        "time_setup_s": 0.5,
        "time_render_s": 30.0,
        "rejections_by_reason": {},
        "sub_reasons": {},
        "accepted_pairs": accepted_pairs,
        "render_records": render_records,
        "render_width": image_resolution[1],
        "render_height": image_resolution[0],
        "render_samples": 32,
        "render_stride": 1,
        "render_num_frames": frames_per_pair,
        "render_frames_per_pair": frames_per_pair,
    }
    (placement_dir / "data.json").write_text(json.dumps(data, indent=2))
    (placement_dir / "done.flag").write_text("ok")
    (placement_dir / "scored.flag").write_text("ok")
    return placement_dir / "data.json"


@pytest.fixture
def v7_placement(tmp_path) -> Path:
    return _build_dummy_v7_placement(
        tmp_path / "Test-scene_0000abcd__Test-object_1111efgh",
        n_accepted_pairs=2,
        frames_per_pair=32,
    )


@pytest.fixture
def v7_two_placements(tmp_path) -> tuple[Path, Path]:
    p1 = _build_dummy_v7_placement(tmp_path / "scene_a__obj_x", n_accepted_pairs=1, frames_per_pair=32)
    p2 = _build_dummy_v7_placement(tmp_path / "scene_b__obj_y", n_accepted_pairs=2, frames_per_pair=32)
    return p1, p2


# --- iter_windows -------------------------------------------------------------


def test_iter_windows_count(v7_placement):
    # 2 accepted_pairs × 32 frames; chunk=8, stride=1 → 24 windows each → 48 total
    windows = list(iter_windows(v7_placement, chunk_size=8, stride=1))
    assert len(windows) == 2 * (32 - 8)


def test_iter_windows_carries_scores(v7_placement):
    windows = list(iter_windows(v7_placement, chunk_size=4, stride=1))
    w = windows[0]
    # End frame's raw dict has the 8 V5 score keys merged in
    assert "occupancy" in w.end.raw
    assert "cam_to_obj_azimuth_deg" in w.end.raw
    assert "body_in_frame_ratio" in w.end.raw


def test_iter_windows_image_path_resolves(v7_placement):
    windows = list(iter_windows(v7_placement, chunk_size=4, stride=1))
    w = windows[0]
    assert Path(w.start.image).exists()
    assert Path(w.end.image).exists()


def test_iter_windows_action_chunk_decoded_from_trajectory(v7_placement):
    windows = list(iter_windows(v7_placement, chunk_size=8, stride=1))
    w = windows[0]
    # Camera position changes between consecutive frames in the trajectory,
    # so the start vs end positions differ
    assert w.start.camera_position != w.end.camera_position


# --- CosmosDroneDataset --------------------------------------------------------


def test_cosmos_dataset_loads_v7(v7_placement):
    ds = CosmosDroneDataset(
        [v7_placement],
        goal_score_keys=["occupancy", "cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg"],
        chunk_size=4,
        stride=1,
        max_samples=8,
        target_resolution=(16, 16),
    )
    assert len(ds) > 0
    sample = ds[0]
    assert sample["state_image"].shape == (3, 16, 16)
    assert sample["next_state_image"].shape == (3, 16, 16)
    assert sample["action_chunk"].shape == (4, ACTION_DIM)
    assert sample["goal_vec"].shape == (3,)
    assert sample["meta"]["chunk_size"] == 4
    assert "pair_idx" in sample["meta"]


def test_clamped_goal_windows_are_filtered(v7_placement):
    """Windows whose goal frame hit the scorer's off-screen sentinel are dropped."""
    import json as _json

    # Zero-clamp the scores of frames >= 24 in pair 0 (mimics close-range blow-up)
    doc = _json.loads(Path(v7_placement).read_text())
    for rec in doc["render_records"][0]:
        if rec["frame_idx"] >= 24:
            rec["scores"] = {k: 0 for k in rec["scores"]}
    clamped_path = Path(v7_placement).parent / "data.json"
    clamped_path.write_text(_json.dumps(doc))

    from src.policy.common.dataset_base import BasePolicyDataset

    K = 8
    with_filter = BasePolicyDataset([clamped_path], chunk_size=K, stride=1)
    without = BasePolicyDataset([clamped_path], chunk_size=K, stride=1, filter_clamped_goals=False)
    # pair 0 windows with end >= 24 must be gone; pair 1 untouched
    assert len(with_filter) < len(without)
    for i in range(len(with_filter)):
        s = with_filter[i]
        if s.start.pair_idx == 0:
            assert s.end.frame_idx < 24


def test_value_is_pose_based_and_finite_even_with_clamped_scores(v7_placement):
    """The value no longer depends on the bbox score pixels at all."""
    import json as _json

    doc = _json.loads(Path(v7_placement).read_text())
    # corrupt ALL bbox-derived scores in pair 1 except azimuth/elevation
    for rec in doc["render_records"][1]:
        rec["scores"]["object_center_x"] = 0
        rec["scores"]["object_center_y"] = 0
        rec["scores"]["bbox_x_offset"] = 0
    Path(v7_placement).write_text(_json.dumps(doc))

    from src.policy.common.dataset_base import BasePolicyDataset

    ds = BasePolicyDataset([v7_placement], chunk_size=4, stride=4)
    pair1 = [ds[i] for i in range(len(ds)) if ds[i].start.pair_idx == 1]
    assert pair1
    import numpy as _np
    vals = _np.array([s.value for s in pair1])
    # all finite, strictly negative (start != goal on a moving trajectory)
    assert _np.isfinite(vals).all()
    assert (vals < 0).all()


def test_cosmos_dataset_full_v5_goal(v7_placement):
    """All 8 V5 keys are present in v7; default goal_score_keys yields finite samples."""
    ds = CosmosDroneDataset(
        [v7_placement],
        chunk_size=2, stride=1, max_samples=4, target_resolution=(16, 16),
    )
    assert len(ds) > 0
    sample = ds[0]
    assert sample["goal_vec"].shape == (8,)
    assert torch.isfinite(sample["goal_vec"]).all()


# --- End-to-end through CosmosWorldActionPolicy -------------------------------


@dataclass
class _DiffusersOutput:
    sample: torch.Tensor


class _MockBackbone(nn.Module):
    def __init__(self, latent_channels: int = 16, model_dim: int = 1024):
        super().__init__()
        self.conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)
        self.cond_proj = nn.Linear(model_dim, latent_channels)

    def forward(self, hidden_states, timestep, encoder_hidden_states, condition_mask=None, padding_mask=None, return_dict=True):
        c_emb = self.cond_proj(encoder_hidden_states.mean(dim=1))
        x = self.conv(hidden_states) + c_emb[:, :, None, None, None]
        return _DiffusersOutput(sample=x) if return_dict else (x,)


class _IdentityVAE(nn.Module):
    """Stub VAE: encode (B, C, T, H, W) → identity-style 16-channel latent."""

    def __init__(self):
        super().__init__()
        self.encode_proj = nn.Conv3d(3, 16, kernel_size=1)
        self.decode_proj = nn.Conv3d(16, 3, kernel_size=1)

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return self.encode_proj(video)

    def decode(self, latent: torch.Tensor):
        class _R: pass
        r = _R()
        r.sample = self.decode_proj(latent)
        return r


def test_v7_end_to_end_compute_loss(v7_two_placements):
    """The full data → encode → flow loss pipeline runs without errors on v7 data."""
    from torch.utils.data import DataLoader

    p1, p2 = v7_two_placements
    ds = CosmosDroneDataset(
        [p1, p2], chunk_size=4, stride=4, max_samples=16, target_resolution=(16, 16),
    )
    assert len(ds) > 0

    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    vae = CosmosVAEWrapper(_IdentityVAE())
    backbone = _MockBackbone()
    policy = CosmosWorldActionPolicy(backbone, chunk_size=4, freeze_backbone=False)

    state_img = batch["state_image"]
    next_img = batch["next_state_image"]
    clip = vae.assemble_clip(state_img, next_img)               # (B, 3, T=4, H, W)
    image_latent = vae.encode(clip)                              # (B, 16, T=4, H, W)
    loss_out = policy.compute_loss(
        image_latent=image_latent,
        action_chunk=batch["action_chunk"],
        goal_vec=batch["goal_vec"],
        value_target=batch["value_target"],
    )
    assert loss_out.total.dim() == 0
    assert torch.isfinite(loss_out.total)
    assert loss_out.world.dim() == 0
    assert loss_out.action.dim() == 0
    assert loss_out.value is not None
    loss_out.total.backward()
    # Conditioner got gradient — confirms goal vector influences the loss
    has_cond_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in policy.conditioner.parameters())
    assert has_cond_grad


def test_v7_end_to_end_sample(v7_placement):
    """Diffusion sampling on v7-loaded latents produces an action chunk + value."""
    ds = CosmosDroneDataset(
        [v7_placement], chunk_size=4, stride=4, max_samples=2, target_resolution=(16, 16),
    )
    sample = ds[0]
    vae = CosmosVAEWrapper(_IdentityVAE())
    backbone = _MockBackbone()
    policy = CosmosWorldActionPolicy(backbone, chunk_size=4).eval()

    state_img = sample["state_image"].unsqueeze(0)
    next_img = sample["next_state_image"].unsqueeze(0)
    clip = vae.assemble_clip(state_img, next_img)
    image_latent = vae.encode(clip)
    out = policy.sample(image_latent=image_latent, goal_vec=sample["goal_vec"].unsqueeze(0), n_steps=4)
    assert out.pred_action_chunk.shape == (1, 4, ACTION_DIM)
    assert out.pred_value.shape == (1,)
    assert torch.isfinite(out.pred_action_chunk).all()
