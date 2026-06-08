"""Inject/extract round-trip tests for the latent-frame action/value design.

Action chunks are tiled into a `(C, H, W)` latent frame and averaged across the
repeats at decode. With no noise added, encode→decode is exact.
"""

import pytest

torch = pytest.importorskip("torch")

from src.policy.cosmos.action_latent import (
    action_capacity,
    extract_action_chunk,
    extract_value,
    inject_action_chunk,
    inject_value,
)


def test_action_inject_extract_roundtrip_clean():
    b, c, h, w = 2, 16, 4, 4   # 256 elements per frame
    frame = torch.zeros(b, c, h, w)
    chunk = torch.randn(b, 1, 5)
    filled = inject_action_chunk(frame, chunk)
    decoded = extract_action_chunk(filled, chunk_size=1, action_dim=5)
    torch.testing.assert_close(decoded, chunk, atol=1e-5, rtol=0)


def test_action_inject_extract_roundtrip_chunked():
    b, c, h, w = 2, 16, 8, 8   # 1024 elements per frame
    frame = torch.zeros(b, c, h, w)
    chunk = torch.randn(b, 4, 5)  # 20 floats
    filled = inject_action_chunk(frame, chunk)
    decoded = extract_action_chunk(filled, chunk_size=4, action_dim=5)
    torch.testing.assert_close(decoded, chunk, atol=1e-5, rtol=0)


def test_action_inject_fills_entire_latent_via_repeats():
    b, c, h, w = 1, 4, 4, 4    # 64 elements
    chunk = torch.arange(8, dtype=torch.float32).view(1, 2, 4)  # 8 floats
    filled = inject_action_chunk(torch.zeros(b, c, h, w), chunk)
    # Should be 64/8 = 8 repeats of the same flat pattern
    flat = filled.reshape(1, -1)
    for i in range(8):
        torch.testing.assert_close(flat[0, i*8:(i+1)*8], chunk.reshape(-1), atol=1e-5, rtol=0)


def test_action_extract_averages_noisy_repeats():
    """Decode should average the noisy repeats and recover the underlying chunk."""
    b, c, h, w = 1, 16, 16, 16   # 4096 elements
    chunk = torch.randn(b, 4, 5)
    filled = inject_action_chunk(torch.zeros(b, c, h, w), chunk)
    # Add zero-mean noise to all the repeats
    noisy = filled + torch.randn_like(filled) * 0.01
    decoded = extract_action_chunk(noisy, chunk_size=4, action_dim=5)
    # Averaging across many repeats should keep error small
    torch.testing.assert_close(decoded, chunk, atol=5e-3, rtol=0)


def test_action_rejects_oversized_chunk():
    frame = torch.zeros(1, 1, 2, 2)   # 4 elements
    chunk = torch.zeros(1, 2, 5)      # 10 floats
    with pytest.raises(ValueError):
        inject_action_chunk(frame, chunk)
    with pytest.raises(ValueError):
        extract_action_chunk(frame, chunk_size=2, action_dim=5)


def test_value_inject_extract_roundtrip():
    b, c, h, w = 3, 16, 4, 4
    frame = torch.zeros(b, c, h, w)
    value = torch.tensor([1.0, -2.5, 0.3])
    filled = inject_value(frame, value)
    # Whole frame is filled with the scalar
    for i in range(b):
        assert torch.all(filled[i] == value[i])
    decoded = extract_value(filled)
    torch.testing.assert_close(decoded, value, atol=1e-6, rtol=0)


def test_value_decode_averages_noisy_frame():
    frame = torch.ones(1, 16, 8, 8) * 2.5 + torch.randn(1, 16, 8, 8) * 0.05
    v = extract_value(frame)
    assert abs(float(v) - 2.5) < 0.01


def test_action_capacity_reports_complete_repeats():
    # (16, 8, 8) = 1024 elements; chunk_size·action_dim = 20 → 51 complete repeats
    assert action_capacity((1, 16, 8, 8), chunk_size=4, action_dim=5) == 1024 // 20
    # Tight fit
    assert action_capacity((1, 4, 4, 4), chunk_size=2, action_dim=4) == 8
