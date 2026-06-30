"""EDM σ sampling + preconditioning + `BALANCED_TWO_HEADS_V1` strategy.

Ports the upstream cosmos-policy training recipe:

  - σ sampling: log-normal `log(σ) ~ N(p_mean=-1.2, p_std=1.2)` (Karras 2022 default).
    Source: `_src/imaginaire/modules/edm_sde.py::EDMSDE.sample_t`.
  - BALANCED_TWO_HEADS_V1: after base σ sampling, replace `high_sigma_ratio`
    of samples with log-uniform [200, 100000] AND replace `low_sigma_ratio` with
    uniform [1e-5, 2.0]. Keeps both EDM denoiser heads (eps at high σ, x0 at low σ)
    well-supervised when log-normal alone would undersample the extremes.
    Source: `_src/predict2/models/video2world_model.py:118`.
  - EDM scaling coefficients (Karras 2022 eq. 7):
        c_skip = σ_data² / (σ² + σ_data²)
        c_out  = σ · σ_data / sqrt(σ² + σ_data²)
        c_in   = 1 / sqrt(σ² + σ_data²)
        c_noise = 0.25 · log(σ)
  - Per-σ loss weight (Karras 2022, EDM scaling):
        w(σ) = (σ² + σ_data²) / (σ · σ_data)²
    Source: `_src/predict2/models/text2world_model.py::get_per_sigma_loss_weights`.
  - Karras σ schedule for inference (Algorithm 1, Karras 2022):
        σ_i = (σ_max^(1/ρ) + i/(N-1) · (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ, ρ = 7.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


LOG_200 = math.log(200.0)
LOG_100000 = math.log(100000.0)


@dataclass
class EDMConfig:
    """Hyperparameters for EDM σ sampling + preconditioning.

    Defaults match cosmos-policy's `HybridEDMSDE` defaults
    (`_src/imaginaire/modules/edm_sde.py` + `modules/hybrid_edm_sde.py`).
    """

    sigma_data: float = 1.0
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    p_mean: float = -1.2
    p_std: float = 1.2
    # BALANCED_TWO_HEADS_V1 knobs (defaults conservative; cosmos-policy's
    # production 14B uses similar values though the exact ratios vary by config).
    use_balanced_two_heads: bool = True
    high_sigma_ratio: float = 0.25
    low_sigma_ratio: float = 0.25
    # Karras inference schedule
    rho: float = 7.0


def sample_sigma(batch_size: int, config: EDMConfig, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Sample (B,) σ values per the upstream training distribution."""
    log_sigma = torch.randn(batch_size, device=device) * config.p_std + config.p_mean
    sigma = log_sigma.exp()
    if config.use_balanced_two_heads:
        sigma = apply_balanced_two_heads(
            sigma,
            high_sigma_ratio=config.high_sigma_ratio,
            low_sigma_ratio=config.low_sigma_ratio,
        )
    return sigma


def apply_balanced_two_heads(
    sigma: torch.Tensor,
    *,
    high_sigma_ratio: float,
    low_sigma_ratio: float,
    high_log_min: float = LOG_200,
    high_log_max: float = LOG_100000,
    low_min: float = 1e-5,
    low_max: float = 2.0,
) -> torch.Tensor:
    """Apply the BALANCED_TWO_HEADS_V1 σ adjustment.

    With probability `high_sigma_ratio`, replace σ with log-uniform [high_log_min, high_log_max].
    Independently, with probability `low_sigma_ratio`, replace σ with uniform [low_min, low_max].
    (Upstream applies the low-σ pass *after* the high-σ pass; same here.)
    """
    device = sigma.device
    shape = sigma.shape

    # High σ replacement (e.g. 25% of samples → log-uniform [200, 100000])
    mask_high = torch.rand(shape, device=device) < high_sigma_ratio
    log_high = torch.rand(shape, device=device) * (high_log_max - high_log_min) + high_log_min
    sigma = torch.where(mask_high, log_high.exp().to(sigma.dtype), sigma)

    # Low σ replacement (e.g. 25% of samples → uniform [1e-5, 2.0])
    mask_low = torch.rand(shape, device=device) < low_sigma_ratio
    low_vals = (torch.rand(shape, device=device) * (low_max - low_min) + low_min).to(sigma.dtype)
    sigma = torch.where(mask_low, low_vals, sigma)

    return sigma


def edm_scaling(sigma: torch.Tensor, sigma_data: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (c_skip, c_out, c_in, c_noise) per Karras 2022 eq. 7, EDM parameterization."""
    sigma2 = sigma * sigma
    denom = sigma2 + sigma_data * sigma_data
    sqrt_denom = denom.sqrt()
    c_skip = (sigma_data * sigma_data) / denom
    c_out = (sigma * sigma_data) / sqrt_denom
    c_in = 1.0 / sqrt_denom
    c_noise = 0.25 * sigma.log()
    return c_skip, c_out, c_in, c_noise


def per_sigma_weight(sigma: torch.Tensor, sigma_data: float, scaling: str = "edm") -> torch.Tensor:
    """Per-σ loss weight applied to (x0 - x0_pred)² (Karras 2022, Table 1)."""
    if scaling == "edm":
        return (sigma * sigma + sigma_data * sigma_data) / ((sigma * sigma_data) ** 2)
    if scaling == "rectified_flow":
        return ((1.0 + sigma) ** 2) / (sigma * sigma)
    raise ValueError(f"unknown scaling: {scaling}")


def karras_sigma_schedule(
    n_steps: int,
    config: EDMConfig,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Karras 2022 σ schedule for sampling (Algorithm 1)."""
    steps = torch.arange(n_steps + 1, dtype=torch.float64, device=device)
    inv = 1.0 / config.rho
    sigmas = (
        config.sigma_max ** inv
        + steps / max(n_steps, 1) * (config.sigma_min ** inv - config.sigma_max ** inv)
    ) ** config.rho
    # Ensure the final σ goes to 0 for the last "step"
    sigmas[-1] = 0.0
    return sigmas.to(torch.float32)


__all__ = [
    "EDMConfig",
    "LOG_200",
    "LOG_100000",
    "apply_balanced_two_heads",
    "edm_scaling",
    "karras_sigma_schedule",
    "per_sigma_weight",
    "sample_sigma",
]


# --- Flow-matching (Cosmos-Predict2.5's actual convention) ---------------------
#
# Moved to `src/policy/common/flow.py` so the VLA / diffusion-policy ablation
# baselines can share the exact same flow convention (comparable losses).
# Re-exported here for backward compatibility — existing cosmos imports unchanged.
from src.policy.common.flow import (  # noqa: E402,F401
    FlowConfig,
    flow_sigma_schedule,
    sample_flow_sigma,
)
