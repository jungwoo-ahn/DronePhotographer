"""Resume plumbing for CosmosPolicyTrainer.

Validates the checkpoint round-trip and resume bookkeeping without running the
full training loop (the loop's end-to-end resume is exercised by the login
smoke run against the real model). Covers:
  - save_checkpoint persists optimizer/scheduler/iteration/EMA/RNG state
  - _resolve_resume accepts a run dir or a .pt path
  - a resumed run continues in the original run dir (TB/log/ckpt contiguity)
  - the saved optimizer state reloads into a fresh optimizer
"""

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from src.policy.cosmos.model import CosmosWorldActionPolicy
from src.policy.cosmos.trainer import CosmosPolicyTrainer, TrainerConfig


@dataclass
class _DiffusersOutput:
    sample: torch.Tensor


class _MockBackbone(nn.Module):
    """Minimal diffusers-style backbone: enough trainable params for AdamW."""

    def __init__(self, latent_channels: int = 16, model_dim: int = 1024):
        super().__init__()
        self.conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)
        self.cond_proj = nn.Linear(model_dim, latent_channels)

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                condition_mask=None, padding_mask=None, return_dict=True):
        x = self.conv(hidden_states)
        c = self.cond_proj(encoder_hidden_states.mean(dim=1))
        x = x + c[:, :, None, None, None]
        return _DiffusersOutput(sample=x) if return_dict else (x,)


class _StubVAE:
    """Trainer stores .vae but never touches it outside fit()."""

    def to(self, *_a, **_k):
        return self


def _cfg(tmp_path, **over):
    base = dict(
        output_root=tmp_path, run_name="rtest", device="cpu", dtype="float32",
        max_iter=3, grad_accum=1, warmup_iter=1, save_iter=1, val_iter=0,
    )
    base.update(over)
    return TrainerConfig(**base)


def test_save_checkpoint_persists_full_training_state(tmp_path):
    policy = CosmosWorldActionPolicy(_MockBackbone(), chunk_size=1, freeze_backbone=False)
    trainer = CosmosPolicyTrainer(policy, _StubVAE(), _cfg(tmp_path))

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
    (sum(p.sum() for p in policy.parameters() if p.requires_grad)).backward()
    opt.step()  # populate optimizer state so state_dict is non-trivial
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

    path = trainer.save_checkpoint(
        3, name="ckpt_last.pt", optimizer=opt, scheduler=sched,
        loss_ema=1.5, best_ema=1.5, epoch=2,
    )
    ckpt = torch.load(path, weights_only=False)

    assert ckpt["iteration"] == 3
    assert ckpt["epoch"] == 2
    assert ckpt["loss_ema"] == 1.5 and ckpt["best_ema"] == 1.5
    assert "optimizer_state" in ckpt and "scheduler_state" in ckpt
    assert "policy_state" in ckpt and "torch_rng_state" in ckpt
    # bf16/no-scaler run → no scaler state saved
    assert "scaler_state" not in ckpt


def test_weights_only_snapshot_omits_optimizer(tmp_path):
    policy = CosmosWorldActionPolicy(_MockBackbone(), chunk_size=1, freeze_backbone=False)
    trainer = CosmosPolicyTrainer(policy, _StubVAE(), _cfg(tmp_path))
    ckpt = torch.load(trainer.save_checkpoint(1, name="weights.pt"), weights_only=False)
    assert "policy_state" in ckpt
    assert "optimizer_state" not in ckpt  # not passed → weights-only snapshot


def test_resume_resolves_dir_and_continues_in_place(tmp_path):
    policy = CosmosWorldActionPolicy(_MockBackbone(), chunk_size=1, freeze_backbone=False)
    trainer = CosmosPolicyTrainer(policy, _StubVAE(), _cfg(tmp_path))
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
    (sum(p.sum() for p in policy.parameters() if p.requires_grad)).backward()
    opt.step()
    ckpt_path = trainer.save_checkpoint(3, name="ckpt_last.pt", optimizer=opt)

    # Resume from the run DIR → resolves to ckpt_last.pt, reuses the same dir.
    trainer2 = CosmosPolicyTrainer(
        CosmosWorldActionPolicy(_MockBackbone(), chunk_size=1, freeze_backbone=False),
        _StubVAE(),
        _cfg(tmp_path, resume_from=str(trainer.run_dir)),
    )
    assert trainer2.resume_ckpt_path == trainer.run_dir / "ckpt_last.pt"
    assert trainer2.run_dir == trainer.run_dir  # continues in place, no new dir

    # Resume from an explicit .pt path resolves identically.
    trainer3 = CosmosPolicyTrainer(
        CosmosWorldActionPolicy(_MockBackbone(), chunk_size=1, freeze_backbone=False),
        _StubVAE(),
        _cfg(tmp_path, resume_from=str(ckpt_path)),
    )
    assert trainer3.resume_ckpt_path == ckpt_path

    # The saved optimizer state reloads into a fresh optimizer without error.
    fresh = torch.optim.AdamW([p for p in trainer2.policy.parameters() if p.requires_grad], lr=1e-4)
    fresh.load_state_dict(torch.load(ckpt_path, weights_only=False)["optimizer_state"])


def test_missing_resume_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CosmosPolicyTrainer(
            CosmosWorldActionPolicy(_MockBackbone(), chunk_size=1, freeze_backbone=False),
            _StubVAE(),
            _cfg(tmp_path, resume_from=str(tmp_path / "does_not_exist")),
        )
