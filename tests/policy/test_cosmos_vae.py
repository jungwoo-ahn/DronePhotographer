"""Behavior tests for src/policy/cosmos/vae.py using a stub VAE.

We can't load the real Cosmos VAE (2B-parameter backbone), so we substitute a
tiny `StubVAE` that mimics the two upstream conventions:
  (a) Native Cosmos VAE: `encode(video) -> torch.Tensor` directly.
  (b) Diffusers-wrapped: `encode(video) -> obj.latent_dist.sample()`.
"""

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from src.policy.cosmos.vae import CosmosVAEWrapper


# --- Stubs ---------------------------------------------------------------

class _NativeStubVAE(nn.Module):
    """Encodes (B, C, T, H, W) → (B, 16, T, H/4, W/4) with a fixed-scale conv."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Conv3d(3, 16, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.dec = nn.ConvTranspose3d(16, 3, kernel_size=(1, 4, 4), stride=(1, 4, 4))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)


@dataclass
class _LatentDist:
    mean: torch.Tensor

    def sample(self) -> torch.Tensor:
        return self.mean


@dataclass
class _EncoderOutput:
    latent_dist: _LatentDist


class _DiffusersStubVAE(nn.Module):
    """Mimics diffusers' AutoencoderKL with .latent_dist + scaling_factor."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Conv3d(3, 16, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.dec = nn.ConvTranspose3d(16, 3, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        # diffusers exposes .config.scaling_factor
        self.config = type("cfg", (), {"scaling_factor": 0.18215})()

    def encode(self, x: torch.Tensor):
        return _EncoderOutput(latent_dist=_LatentDist(mean=self.enc(x)))

    def decode(self, z: torch.Tensor):
        class _R:
            sample = self.dec(z)
        return _R()


# --- Tests ---------------------------------------------------------------

def test_assemble_clip_layout():
    state = torch.zeros(2, 3, 8, 8)
    nxt = torch.ones(2, 3, 8, 8)
    clip = CosmosVAEWrapper.assemble_clip(state, nxt, T=4)
    assert clip.shape == (2, 3, 4, 8, 8)
    assert torch.all(clip[:, :, 0] == 0)
    assert torch.all(clip[:, :, 1] == 1)
    # The remaining frames repeat next_state (not zeros) per docstring
    assert torch.all(clip[:, :, 2] == 1)
    assert torch.all(clip[:, :, 3] == 1)


def test_assemble_clip_rejects_shape_mismatch():
    s = torch.zeros(2, 3, 8, 8)
    n = torch.zeros(2, 3, 4, 4)
    with pytest.raises(ValueError):
        CosmosVAEWrapper.assemble_clip(s, n)


def test_native_vae_encode_returns_tensor_directly():
    w = CosmosVAEWrapper(_NativeStubVAE())
    video = torch.randn(1, 3, 4, 16, 16)
    z = w.encode(video)
    assert z.shape == (1, 16, 4, 4, 4)


def test_diffusers_vae_encode_applies_scaling():
    stub = _DiffusersStubVAE()
    w = CosmosVAEWrapper(stub)
    video = torch.ones(1, 3, 4, 16, 16)
    z_with = w.encode(video)
    # Bypass the wrapper and encode the same input directly to recover the raw
    # latent without the scaling_factor multiplication.
    z_raw = stub.encode(video).latent_dist.sample()
    expected = z_raw * stub.config.scaling_factor
    assert torch.allclose(z_with, expected)


def test_encode_pair_runs_end_to_end():
    w = CosmosVAEWrapper(_NativeStubVAE())
    state = torch.randn(2, 3, 16, 16)
    nxt = torch.randn(2, 3, 16, 16)
    z = w.encode_pair(state, nxt, T=4)
    assert z.shape == (2, 16, 4, 4, 4)


def test_decode_matches_encode_layout():
    w = CosmosVAEWrapper(_NativeStubVAE())
    video = torch.randn(1, 3, 4, 16, 16)
    z = w.encode(video)
    out = w.decode(z)
    assert out.shape == video.shape
