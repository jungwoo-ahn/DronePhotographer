"""Shape + DDPM-overfit tests for the Diffusion Policy ConditionalUnet1D denoiser."""

import pytest

torch = pytest.importorskip("torch")

from src.policy.diffusion_policy.denoiser import ConditionalUnet1D


def _unet(action_dim=5, global_cond_dim=64, chunk_size=8):
    return ConditionalUnet1D(action_dim, global_cond_dim, diffusion_step_embed_dim=64,
                             down_dims=(64, 128, 256))


def test_output_shape_matches_input():
    net = _unet()
    x = torch.randn(3, 8, 5)
    t = torch.randint(0, 100, (3,))
    cond = torch.randn(3, 64)
    out = net(x, t, cond)
    assert out.shape == (3, 8, 5)


def test_scalar_timestep_is_broadcast():
    net = _unet()
    out = net(torch.randn(2, 8, 5), torch.tensor(7), torch.randn(2, 64))
    assert out.shape == (2, 8, 5)


def test_gradients_flow_to_all_params():
    net = _unet()
    out = net(torch.randn(2, 8, 5), torch.randint(0, 100, (2,)), torch.randn(2, 64))
    out.pow(2).sum().backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing, f"no grad for: {missing}"


def test_cond_changes_output():
    net = _unet().eval()
    x = torch.randn(1, 8, 5)
    t = torch.tensor([10])
    o1 = net(x, t, torch.zeros(1, 64))
    o2 = net(x, t, torch.ones(1, 64))
    assert not torch.allclose(o1, o2)


def test_ddpm_overfits_constant_action():
    """Capacity check: with a fixed cond the U-Net learns the epsilon field for a
    single fixed action chunk under the DDPM forward process, driving loss -> ~0."""
    from diffusers import DDPMScheduler

    torch.manual_seed(0)
    net = _unet()
    sched = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    a0 = torch.randn(1, 8, 5)
    cond = torch.randn(1, 64)
    losses = []
    for _ in range(300):
        b = 16
        a0b = a0.expand(b, -1, -1)
        condb = cond.expand(b, -1)
        t = torch.randint(0, 100, (b,))
        noise = torch.randn_like(a0b)
        noisy = sched.add_noise(a0b, noise, t)
        pred = net(noisy, t, condb)
        loss = (pred - noise).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], f"DDPM loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
