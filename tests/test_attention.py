from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.blocks import GatedSpatialMixer2D, SelfAttention


def test_sdpa_and_chunked_attention_match_on_cpu() -> None:
    torch.manual_seed(0)
    sdpa = SelfAttention(dim=16, num_heads=4, dropout=0.0).eval()
    chunked = SelfAttention(dim=16, num_heads=4, dropout=0.0, chunk_size=3).eval()
    chunked.load_state_dict(sdpa.state_dict())
    inputs = torch.randn(2, 7, 16)
    mask = torch.tensor(
        [[True, True, True, True, True, False, False], [True] * 7],
        dtype=torch.bool,
    )
    with torch.no_grad():
        expected = sdpa(inputs, attention_mask=mask)
        actual = chunked(inputs, attention_mask=mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_attention_falls_back_when_sdpa_is_unavailable(monkeypatch) -> None:
    torch.manual_seed(2)
    attention = SelfAttention(dim=8, num_heads=2, dropout=0.0).eval()
    reference = SelfAttention(dim=8, num_heads=2, dropout=0.0, chunk_size=2).eval()
    reference.load_state_dict(attention.state_dict())
    inputs = torch.randn(1, 4, 8)
    with torch.no_grad():
        expected = reference(inputs)
        monkeypatch.delattr(F, "scaled_dot_product_attention")
        actual = attention(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_attention_padding_mask_blocks_padded_keys() -> None:
    torch.manual_seed(1)
    attention = SelfAttention(dim=12, num_heads=3, dropout=0.0).eval()
    inputs = torch.randn(1, 5, 12)
    changed = inputs.clone()
    changed[:, 3:] = 1_000.0
    mask = torch.tensor([[True, True, True, False, False]])
    with torch.no_grad():
        first = attention(inputs, attention_mask=mask)
        second = attention(changed, attention_mask=mask)
    torch.testing.assert_close(first[:, :3], second[:, :3], rtol=1e-5, atol=1e-6)


def test_gated_spatial_mixer_is_honest_2d_mixer() -> None:
    mixer = GatedSpatialMixer2D(dim=8, expand=2, kernel_size=3).eval()
    inputs = torch.randn(2, 6, 8)
    with torch.no_grad():
        outputs = mixer(inputs, spatial_shape=(2, 3))
    assert outputs.shape == inputs.shape
    parameter_names = set(dict(mixer.named_parameters()))
    assert "depthwise.weight" in parameter_names
    assert not any(name in parameter_names for name in {"A_log", "D", "x_proj.weight", "dt_proj.weight"})
