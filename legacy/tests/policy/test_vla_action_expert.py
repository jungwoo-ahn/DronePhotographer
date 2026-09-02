"""Shape + flow round-trip tests for the VLA action expert."""

import pytest

torch = pytest.importorskip("torch")

from src.policy.common.flow import FlowConfig, flow_sigma_schedule, sample_flow_sigma
from src.policy.vla.action_expert import ActionExpert, sigma_time_embedding


def _expert(action_dim=5, ctx_dim=64, dim=32, depth=2, n_heads=4, chunk_size=8):
    return ActionExpert(action_dim=action_dim, ctx_dim=ctx_dim, dim=dim, depth=depth,
                        n_heads=n_heads, chunk_size=chunk_size)


def test_velocity_shape():
    e = _expert()
    a = torch.randn(3, 8, 5)
    sigma = torch.rand(3)
    ctx = torch.randn(3, 17, 64)
    v = e(a, sigma, ctx)
    assert v.shape == (3, 8, 5)


def test_output_zero_init():
    # out_proj zero-init → initial velocity is exactly zero (stable training start)
    e = _expert()
    v = e(torch.randn(2, 8, 5), torch.rand(2), torch.randn(2, 10, 64))
    assert torch.count_nonzero(v) == 0


def test_context_key_padding_mask_accepted():
    e = _expert()
    ctx = torch.randn(2, 12, 64)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[:, 8:] = True  # ignore the last 4 context tokens
    v = e(torch.randn(2, 8, 5), torch.rand(2), ctx, context_key_padding_mask=mask)
    assert v.shape == (2, 8, 5)


def test_gradients_flow_to_all_params():
    e = _expert()
    # break the zero-init so output is non-zero and grads are meaningful
    torch.nn.init.normal_(e.out_proj.weight, std=0.1)
    a = torch.randn(2, 8, 5)
    e(a, torch.rand(2), torch.randn(2, 10, 64)).pow(2).sum().backward()
    missing = [n for n, p in e.named_parameters() if p.grad is None]
    assert not missing, f"no grad for: {missing}"


def test_sigma_time_embedding_shape_and_finite():
    emb = sigma_time_embedding(torch.rand(5), 32)
    assert emb.shape == (5, 32)
    assert torch.isfinite(emb).all()
    emb_odd = sigma_time_embedding(torch.rand(5), 31)
    assert emb_odd.shape == (5, 31)


def test_flow_roundtrip_overfits_constant_action():
    """A capacity check: with a constant context the expert can learn the velocity
    field for a single fixed action chunk (eps - a0), driving the flow loss to ~0."""
    torch.manual_seed(0)
    e = _expert(dim=64, depth=3)
    torch.nn.init.normal_(e.out_proj.weight, std=0.02)
    opt = torch.optim.Adam(e.parameters(), lr=1e-3)
    a0 = torch.randn(1, 8, 5)
    ctx = torch.randn(1, 4, 64)
    cfg = FlowConfig()
    losses = []
    for _ in range(300):
        sigma = sample_flow_sigma(16, cfg)
        a0b = a0.expand(16, -1, -1)
        ctxb = ctx.expand(16, -1, -1)
        eps = torch.randn_like(a0b)
        a_sigma = (1 - sigma[:, None, None]) * a0b + sigma[:, None, None] * eps
        v_pred = e(a_sigma, sigma, ctxb)
        loss = (v_pred - (eps - a0b)).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.25 * losses[0], f"flow loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_euler_integration_recovers_overfit_action():
    """After overfitting one (a0, ctx), integrating sigma:1->0 from noise recovers a0."""
    torch.manual_seed(0)
    e = _expert(dim=64, depth=3)
    torch.nn.init.normal_(e.out_proj.weight, std=0.02)
    opt = torch.optim.Adam(e.parameters(), lr=1e-3)
    a0 = torch.randn(1, 8, 5)
    ctx = torch.randn(1, 4, 64)
    cfg = FlowConfig()
    for _ in range(600):
        sigma = sample_flow_sigma(32, cfg)
        a0b = a0.expand(32, -1, -1); ctxb = ctx.expand(32, -1, -1)
        eps = torch.randn_like(a0b)
        a_sigma = (1 - sigma[:, None, None]) * a0b + sigma[:, None, None] * eps
        loss = (e(a_sigma, sigma, ctxb) - (eps - a0b)).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    e.eval()
    with torch.no_grad():
        x = torch.randn(1, 8, 5)
        sigmas = flow_sigma_schedule(20)
        for i in range(20):
            s = sigmas[i].expand(1)
            v = e(x, s, ctx)
            x = x + (sigmas[i + 1] - sigmas[i]) * v
    assert (x - a0).abs().mean() < 0.2
