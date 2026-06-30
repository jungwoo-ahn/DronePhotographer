"""Flow-matching sigma sampling + schedule (family-agnostic).

Shared by every policy family that trains a flow-matching head — the Cosmos
world-action model and the VLA / diffusion-policy ablation baselines all use the
same convention so their losses are directly comparable:

  - sigma in [0, 1]; x_sigma = (1 - sigma) * x0 + sigma * eps
  - the network predicts the flow velocity v = eps - x0
  - matches Cosmos-Predict2.5's pretraining (prediction_type=flow_prediction,
    use_flow_sigmas=true)

`BALANCED_TWO_HEADS` carries over into flow space: a fraction of samples is
forced near sigma=1 (noise head) and near sigma=0 (clean head), so both ends of
the denoiser stay well supervised.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FlowConfig:
    """Sigma sampling for flow-matching training + sampling constants.

    `p_mean/p_std` parameterize a logit-normal over sigma (the flow-matching
    standard; the analog of EDM's log-normal).
    """

    p_mean: float = 0.0
    p_std: float = 1.0
    use_balanced_two_heads: bool = True
    high_sigma_ratio: float = 0.25
    low_sigma_ratio: float = 0.25
    high_band: tuple[float, float] = (0.85, 1.0)
    low_band: tuple[float, float] = (0.0, 0.15)
    # timestep given to pinned conditioning frames at sampling (pipeline default)
    cond_timestep: float = 0.1


def sample_flow_sigma(batch_size: int, config: FlowConfig, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Sample (B,) flow sigmas in [0, 1]: logit-normal base + balanced tails."""
    sigma = torch.sigmoid(torch.randn(batch_size, device=device) * config.p_std + config.p_mean)
    if config.use_balanced_two_heads:
        shape = sigma.shape
        hi_lo, hi_hi = config.high_band
        mask_high = torch.rand(shape, device=device) < config.high_sigma_ratio
        sigma = torch.where(mask_high, torch.rand(shape, device=device) * (hi_hi - hi_lo) + hi_lo, sigma)
        lo_lo, lo_hi = config.low_band
        mask_low = torch.rand(shape, device=device) < config.low_sigma_ratio
        sigma = torch.where(mask_low, torch.rand(shape, device=device) * (lo_hi - lo_lo) + lo_lo, sigma)
    return sigma


def flow_sigma_schedule(n_steps: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Linear sigma schedule 1 -> 0 for Euler sampling (n_steps + 1 entries)."""
    return torch.linspace(1.0, 0.0, n_steps + 1, device=device)


__all__ = ["FlowConfig", "sample_flow_sigma", "flow_sigma_schedule"]
