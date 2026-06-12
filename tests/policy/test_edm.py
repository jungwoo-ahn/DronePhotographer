"""Tests for src/policy/cosmos/edm.py.

Distribution-shape tests for σ sampling under various configs, plus property
tests for the EDM scaling coefficients and the per-σ loss weight.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from src.policy.cosmos.edm import (
    EDMConfig,
    LOG_100000,
    LOG_200,
    apply_balanced_two_heads,
    edm_scaling,
    karras_sigma_schedule,
    per_sigma_weight,
    sample_sigma,
)


def test_sample_sigma_default_distribution_is_lognormal_ish():
    """Without BALANCED_TWO_HEADS, σ should be log-normal around p_mean."""
    cfg = EDMConfig(use_balanced_two_heads=False)
    torch.manual_seed(0)
    sigma = sample_sigma(10000, cfg)
    log_sigma = sigma.log()
    # Sample mean of log(σ) should be ≈ p_mean = -1.2
    assert abs(log_sigma.mean().item() - cfg.p_mean) < 0.1
    # Sample std ≈ p_std = 1.2
    assert abs(log_sigma.std().item() - cfg.p_std) < 0.1


def test_balanced_two_heads_pushes_high_and_low_extremes():
    """With high=low=0.5, ~25% of samples should land at the high σ extreme
    (since the subsequent low-σ pass overwrites half of them)."""
    cfg = EDMConfig(
        use_balanced_two_heads=True,
        high_sigma_ratio=0.5,
        low_sigma_ratio=0.5,
    )
    torch.manual_seed(0)
    sigma = sample_sigma(20000, cfg)
    # High pass injects 50%, low pass overwrites half of those → ~25% remain in high
    high_frac = ((sigma >= 200) & (sigma <= 100000)).float().mean().item()
    assert 0.2 < high_frac < 0.30
    # The low pass guarantees ≥50% land in [1e-5, 2.0]; baseline log-normal
    # (mean log σ = -1.2) also frequently lands there, so the observed fraction is higher.
    low_frac = ((sigma >= 1e-5) & (sigma <= 2.0)).float().mean().item()
    assert low_frac >= 0.5


def test_balanced_two_heads_pure_baseline_ignores_extremes():
    """Without BALANCED_TWO_HEADS, σ stays log-normal and rarely touches the extremes."""
    cfg = EDMConfig(use_balanced_two_heads=False)
    torch.manual_seed(0)
    sigma = sample_sigma(20000, cfg)
    high_frac = ((sigma >= 200)).float().mean().item()
    assert high_frac < 0.001  # essentially zero under log-normal


def test_balanced_two_heads_all_high():
    """With high_ratio=1.0, low_ratio=0.0 → every sigma in [200, 100000]."""
    sigma = torch.tensor([0.1, 1.0, 10.0, 100.0])
    out = apply_balanced_two_heads(sigma, high_sigma_ratio=1.0, low_sigma_ratio=0.0)
    assert (out >= 200).all()
    assert (out <= 100000).all()


def test_balanced_two_heads_all_low():
    """With high_ratio=0.0, low_ratio=1.0 → every sigma in [1e-5, 2.0]."""
    sigma = torch.tensor([0.1, 1.0, 10.0, 100.0])
    out = apply_balanced_two_heads(sigma, high_sigma_ratio=0.0, low_sigma_ratio=1.0)
    assert (out >= 1e-5).all()
    assert (out <= 2.0).all()


def test_balanced_two_heads_none_is_identity():
    sigma = torch.tensor([0.1, 1.0, 10.0, 100.0])
    out = apply_balanced_two_heads(sigma, high_sigma_ratio=0.0, low_sigma_ratio=0.0)
    torch.testing.assert_close(out, sigma)


def test_edm_scaling_satisfies_consistency_at_low_sigma():
    """At very small σ, c_skip→1 (preserve x_t as the prediction)."""
    sigma = torch.tensor([1e-6])
    c_skip, c_out, c_in, c_noise = edm_scaling(sigma, sigma_data=1.0)
    assert c_skip.item() > 0.99
    assert c_out.item() < 1e-3
    assert c_in.item() < 1.001 and c_in.item() > 0.999


def test_edm_scaling_satisfies_consistency_at_high_sigma():
    """At very large σ, c_skip→0 (net_out drives the prediction)."""
    sigma = torch.tensor([1e6])
    c_skip, c_out, c_in, _ = edm_scaling(sigma, sigma_data=1.0)
    assert c_skip.item() < 1e-6
    # c_in ≈ 1/σ
    assert abs(c_in.item() - 1e-6) < 1e-8


def test_per_sigma_weight_edm_shape():
    """For EDM scaling, w(σ) = 1/σ_d² + 1/σ² (decreasing in σ, asymptote 1/σ_d²)."""
    sigma_data = 1.0
    sigma_grid = torch.tensor([0.1, 1.0, 10.0, 100.0])
    w = per_sigma_weight(sigma_grid, sigma_data, scaling="edm")
    # Monotonically decreasing in σ
    diffs = w[1:] - w[:-1]
    assert (diffs <= 0).all()
    # Exact values: w(σ_d=1) = 1 + 1 = 2; w(σ→∞) → 1
    assert abs(w[1].item() - 2.0) < 1e-5
    assert abs(w[3].item() - 1.0) < 1e-3


def test_per_sigma_weight_rectified_flow_shape():
    """For rectified flow, w(σ) = (1+σ)²/σ². Asymptotes to 1 at large σ."""
    sigma = torch.tensor([0.1, 1.0, 10.0])
    w = per_sigma_weight(sigma, sigma_data=1.0, scaling="rectified_flow")
    # w(1) = 4
    assert abs(w[1].item() - 4.0) < 1e-5
    # w → 1 at large σ
    assert abs(w[2].item() - 1.21) < 1e-3


def test_per_sigma_weight_unknown_scaling_raises():
    with pytest.raises(ValueError):
        per_sigma_weight(torch.tensor([1.0]), 1.0, scaling="garbage")


def test_karras_schedule_monotonically_decreasing():
    cfg = EDMConfig(sigma_min=0.002, sigma_max=80.0, rho=7.0)
    schedule = karras_sigma_schedule(n_steps=16, config=cfg)
    # σ_0 ≈ σ_max; σ_N = 0
    assert abs(schedule[0].item() - cfg.sigma_max) < 1e-4
    assert schedule[-1].item() == 0.0
    # Monotonically decreasing
    diffs = schedule[1:] - schedule[:-1]
    assert (diffs <= 1e-5).all()


def test_karras_schedule_length():
    cfg = EDMConfig()
    schedule = karras_sigma_schedule(n_steps=8, config=cfg)
    assert schedule.shape == (9,)  # n_steps + 1


def test_log_constants_match_upstream():
    """Cosmos-policy uses log(200), log(100000) for BALANCED_TWO_HEADS_V1 bounds."""
    assert abs(LOG_200 - math.log(200)) < 1e-12
    assert abs(LOG_100000 - math.log(100000)) < 1e-12
