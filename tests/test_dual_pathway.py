"""Tests for the dual-pathway (image + SVG vector) architecture."""

from __future__ import annotations

import math

import pytest
import torch

from src.data.strokes import (
    STROKE_FEATURE_DIM,
    normalize_strokes,
    pad_stroke_batch,
    sample_strokes,
    stroke_type_ids,
    transform_normalized_strokes,
)
from src.models import build_model
from src.models.config import ModelConfig, VectorBranchConfig
from src.models.vector_encoder import VectorEncoder


# ── Stroke tokenization ────────────────────────────────────────────────────────


class TestNormalizeStrokes:
    def test_empty(self) -> None:
        result = normalize_strokes([], (512, 512))
        assert result.shape == (0, STROKE_FEATURE_DIM)

    def test_normalizes_endpoints_by_image_size(self) -> None:
        strokes = [[256.0, 128.0, 512.0, 384.0] + [0.0] * 8]
        result = normalize_strokes(strokes, (512, 256))
        assert result.shape == (1, STROKE_FEATURE_DIM)
        assert result[0, 0] == pytest.approx(0.5)  # x / width
        assert result[0, 1] == pytest.approx(0.5)  # y / height
        assert result[0, 2] == pytest.approx(1.0)
        assert result[0, 3] == pytest.approx(1.5)  # y beyond height stays linear

    def test_radius_uses_geometric_mean(self) -> None:
        strokes = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 64.0] + [0.0] * 5]
        result = normalize_strokes(strokes, (256, 1024))
        assert result[0, 6] == pytest.approx(64.0 / math.sqrt(256 * 1024))

    def test_angle_features_pass_through(self) -> None:
        cos_sin = [1.0, 0.0, -1.0, 0.0, 1.0]
        strokes = [[1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0] + cos_sin[:4] + [1.0]]
        result = normalize_strokes(strokes, (100, 100))
        assert result[0, 7:11].tolist() == [1.0, 0.0, -1.0, 0.0]

    def test_rejects_wrong_dim(self) -> None:
        with pytest.raises(ValueError):
            normalize_strokes([[1.0, 2.0]], (100, 100))

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError):
            normalize_strokes([[float("nan")] * 12], (100, 100))

    def test_rejects_bad_image_size(self) -> None:
        with pytest.raises(ValueError):
            normalize_strokes([[0.0] * 12], (0, 100))


def test_transform_normalized_strokes_tracks_flip_and_resize() -> None:
    tokens = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.1, 1.0, 0.0, 0.0, 1.0, 0.0]]
    )
    # Horizontal flip in 100x100, then resize to 200x200.
    matrix = torch.tensor(
        [[-2.0, 0.0, 200.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
    )
    result = transform_normalized_strokes(tokens, matrix, (100, 100), (200, 200))
    assert torch.allclose(
        result[0, :6], torch.tensor([0.9, 0.2, 0.7, 0.4, 0.8, 0.3])
    )
    assert result[0, 6] == pytest.approx(0.1)
    assert torch.allclose(result[0, 7:9], torch.tensor([-1.0, 0.0]))
    assert torch.allclose(result[0, 9:11], torch.tensor([0.0, 1.0]))


class TestStrokeTypeIds:
    def test_line_and_arc(self) -> None:
        strokes = [
            [1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        ]
        ids = stroke_type_ids(strokes)
        assert ids.tolist() == [0, 1]


class TestSampleStrokes:
    def test_no_sampling_below_limit(self) -> None:
        tokens = torch.randn(10, STROKE_FEATURE_DIM)
        result = sample_strokes(tokens, 32)
        assert torch.equal(result, tokens)

    def test_subsamples_to_limit(self) -> None:
        tokens = torch.randn(100, STROKE_FEATURE_DIM)
        result = sample_strokes(tokens, 10, generator=__import__("random").Random(0))
        assert result.shape == (10, STROKE_FEATURE_DIM)
        # Every kept row must come from the original set.
        originals = {tuple(row.tolist()) for row in tokens}
        assert all(tuple(row.tolist()) in originals for row in result)

    def test_deterministic_with_seeded_generator(self) -> None:
        import random

        tokens = torch.randn(50, STROKE_FEATURE_DIM)
        first = sample_strokes(tokens, 5, generator=random.Random(7))
        second = sample_strokes(tokens, 5, generator=random.Random(7))
        assert torch.equal(first, second)

    def test_length_weighted_keeps_long_strokes(self) -> None:
        import random

        # One long stroke, 99 zero-length strokes; sampling 10 should keep the
        # long one with overwhelming probability under length weighting.
        tokens = torch.zeros((100, STROKE_FEATURE_DIM))
        tokens[0, 2] = 1.0  # long stroke: x1 - x0 = 1
        kept_count = sum(
            bool(sample_strokes(tokens, 10, generator=random.Random(seed))[0, 2] == 1.0)
            for seed in range(20)
        )
        assert kept_count >= 18  # floor allows rare misses

    def test_exact_count_after_weighted_sampling(self) -> None:
        import random

        tokens = torch.rand(200, STROKE_FEATURE_DIM)
        result = sample_strokes(tokens, 50, generator=random.Random(0))
        assert result.shape == (50, STROKE_FEATURE_DIM)
        # Unique rows preserved (no duplicated indices).
        originals = {tuple(row.tolist()) for row in tokens}
        assert len({tuple(row.tolist()) for row in result}) == 50
        assert all(tuple(row.tolist()) in originals for row in result)

    def test_rejects_nonpositive_n_max(self) -> None:
        with pytest.raises(ValueError):
            sample_strokes(torch.randn(4, STROKE_FEATURE_DIM), 0)


class TestPadStrokeBatch:
    def test_pads_and_masks(self) -> None:
        tokens_a = torch.randn(3, STROKE_FEATURE_DIM)
        tokens_b = torch.randn(5, STROKE_FEATURE_DIM)
        tokens, valid = pad_stroke_batch([tokens_a, tokens_b], 8)
        assert tokens.shape == (2, 8, STROKE_FEATURE_DIM)
        assert valid.tolist()[0] == [True, True, True, False, False, False, False, False]
        assert valid.tolist()[1] == [True] * 5 + [False] * 3
        assert torch.equal(tokens[0, :3], tokens_a)
        assert torch.all(tokens[0, 3:] == 0)

    def test_empty_sample_yields_all_pad_row(self) -> None:
        tokens, valid = pad_stroke_batch([torch.zeros((0, STROKE_FEATURE_DIM))], 4)
        assert valid.shape == (1, 4)
        assert not bool(valid.any())

    def test_truncates_overlong_sample(self) -> None:
        tokens = torch.randn(10, STROKE_FEATURE_DIM)
        padded, valid = pad_stroke_batch([tokens], 5)
        assert padded.shape == (1, 5, STROKE_FEATURE_DIM)
        assert valid.sum() == 5


# ── Vector encoder ─────────────────────────────────────────────────────────────


class TestVectorEncoder:
    def test_output_shape(self) -> None:
        encoder = VectorEncoder(model_dim=64, num_heads=4, depth=2)
        tokens = torch.randn(2, 16, STROKE_FEATURE_DIM)
        mask = torch.ones(2, 16, dtype=torch.bool)
        out = encoder(tokens, mask)
        assert out.shape == (2, 16, 64)

    def test_pad_positions_zeroed(self) -> None:
        encoder = VectorEncoder(model_dim=64, num_heads=4, depth=1)
        tokens = torch.randn(1, 8, STROKE_FEATURE_DIM)
        mask = torch.tensor([[True] * 4 + [False] * 4])
        out = encoder(tokens, mask)
        assert torch.all(out[0, 4:] == 0)
        assert torch.any(out[0, :4] != 0)

    def test_mask_changes_output(self) -> None:
        torch.manual_seed(0)
        encoder = VectorEncoder(model_dim=64, num_heads=4, depth=1)
        encoder.eval()
        tokens = torch.randn(1, 6, STROKE_FEATURE_DIM)
        full = torch.ones(1, 6, dtype=torch.bool)
        partial = torch.tensor([[True, True, True, False, False, False]])
        with torch.no_grad():
            out_full = encoder(tokens, full)
            out_partial = encoder(tokens, partial)
        assert not torch.allclose(out_full[0, :3], out_partial[0, :3])

    def test_validation_errors(self) -> None:
        encoder = VectorEncoder(model_dim=64, num_heads=4)
        with pytest.raises(ValueError):
            encoder(torch.randn(1, 4, 8), torch.ones(1, 4, dtype=torch.bool))
        with pytest.raises(ValueError):
            encoder(torch.randn(1, 4, 12), torch.ones(1, 5, dtype=torch.bool))

    def test_rejects_bad_config(self) -> None:
        with pytest.raises(ValueError):
            VectorEncoder(model_dim=60, num_heads=8)
        with pytest.raises(ValueError):
            VectorEncoder(depth=-1)


# ── Presets and integration ────────────────────────────────────────────────────


class TestDualPathwayPresets:
    def test_dual_pathway_preset_builds(self) -> None:
        model = build_model(preset="dual_pathway")
        assert model.vector_enabled
        assert model.vector_n_max == 1024
        assert model.vector_encoder is not None

    def test_base_preset_has_no_vector_branch(self) -> None:
        model = build_model(preset="floorplan_base")
        assert not model.vector_enabled
        assert model.vector_encoder is None

    def test_vector_config_serializes(self) -> None:
        config = ModelConfig(vector=VectorBranchConfig(enabled=True, n_max=512))
        data = config.to_dict()
        assert data["vector"]["enabled"] is True
        assert data["vector"]["n_max"] == 512
        restored = ModelConfig.from_dict(data)
        assert restored.vector.enabled
        assert restored.vector.n_max == 512

    def test_budget_discipline(self) -> None:
        base = sum(p.numel() for p in build_model(preset="floorplan_base").parameters())
        dual = sum(p.numel() for p in build_model(preset="dual_pathway").parameters())
        # Vector branch adds capacity; document the overhead budget (+50% cap).
        assert dual > base
        assert dual < base * 1.5

    def test_control_preset_has_no_conditioning(self) -> None:
        model = build_model(preset="dual_no_fusion")
        assert model.config.conditioner.kind == "none"
        assert model.vector_enabled


class TestDetectorVectorForward:
    @pytest.fixture()
    def dual_model(self) -> torch.nn.Module:
        model = build_model(preset="dual_pathway")
        model.eval()
        return model

    @staticmethod
    def _strokes(batch: int, length: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.rand(batch, length, STROKE_FEATURE_DIM)
        mask = torch.ones(batch, length, dtype=torch.bool)
        mask[0, length // 2:] = False
        return tokens, mask

    def test_selected_class_forward(self, dual_model) -> None:
        image = torch.randn(2, 3, 512, 512)
        tokens, mask = self._strokes(2)
        out = dual_model(image, class_ids=torch.tensor([3, 5]), stroke_tokens=tokens, stroke_mask=mask)
        assert out["center_heatmap"].shape == (2, 1, 64, 64)

    def test_all_class_forward(self, dual_model) -> None:
        image = torch.randn(1, 3, 512, 512)
        tokens, mask = self._strokes(1)
        out = dual_model(image, stroke_tokens=tokens, stroke_mask=mask, class_chunk_size=8)
        assert out["center_heatmap"].shape == (1, 35, 64, 64)

    def test_strokes_change_output(self, dual_model) -> None:
        torch.manual_seed(1)
        image = torch.randn(1, 3, 512, 512)
        tokens, mask = self._strokes(1)
        with torch.no_grad():
            with_strokes = dual_model(image, class_ids=torch.tensor([4]), stroke_tokens=tokens, stroke_mask=mask)
            without = dual_model(image, class_ids=torch.tensor([4]), stroke_tokens=None, stroke_mask=None)
        assert not torch.allclose(with_strokes["center_heatmap"], without["center_heatmap"])

    def test_requires_strokes_when_enabled(self, dual_model) -> None:
        from train import _forward_query_batch

        image = torch.randn(1, 3, 512, 512)
        with pytest.raises(ValueError):
            _forward_query_batch(dual_model, image, torch.tensor([2]), ["text"])

    def test_batch_mismatch_rejected(self, dual_model) -> None:
        image = torch.randn(2, 3, 512, 512)
        tokens, mask = self._strokes(1)
        with pytest.raises(ValueError):
            dual_model(image, class_ids=torch.tensor([1, 2]), stroke_tokens=tokens, stroke_mask=mask)

    def test_base_model_ignores_strokes(self) -> None:
        model = build_model(preset="floorplan_base")
        model.eval()
        image = torch.randn(1, 3, 512, 512)
        tokens, mask = self._strokes(1)
        with torch.no_grad():
            out = model(image, class_ids=torch.tensor([4]), stroke_tokens=tokens, stroke_mask=mask)
        assert out["center_heatmap"].shape == (1, 1, 64, 64)

    def test_backward_flows_through_vector_branch(self) -> None:
        model = build_model(preset="dual_pathway")
        image = torch.randn(1, 3, 512, 512)
        tokens, mask = self._strokes(1)
        out = model(image, class_ids=torch.tensor([4]), stroke_tokens=tokens, stroke_mask=mask)
        out["center_logits"].sum().backward()
        grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith("vector_encoder") and parameter.grad is not None
        ]
        assert grads, "no gradients reached the vector encoder"


class TestClassConditionedVectorFusion:
    """dual_pathway_v2: FiLM(class embedding) over vector tokens before fusion."""

    @pytest.fixture()
    def v2_model(self) -> torch.nn.Module:
        model = build_model(preset="dual_pathway_v2")
        model.eval()
        return model

    @staticmethod
    def _strokes(batch: int, length: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.rand(batch, length, STROKE_FEATURE_DIM)
        mask = torch.ones(batch, length, dtype=torch.bool)
        return tokens, mask

    def test_preset_config(self, v2_model) -> None:
        assert v2_model.vector_enabled
        assert v2_model.vector_class_conditioned
        assert v2_model.config.conditioner.kind == "class_embedding"

    def test_selected_class_forward(self, v2_model) -> None:
        image = torch.randn(2, 3, 512, 512)
        tokens, mask = self._strokes(2)
        out = v2_model(
            image, class_ids=torch.tensor([3, 5]), stroke_tokens=tokens, stroke_mask=mask
        )
        assert out["center_heatmap"].shape == (2, 1, 64, 64)

    def test_all_class_forward(self, v2_model) -> None:
        image = torch.randn(1, 3, 512, 512)
        tokens, mask = self._strokes(1)
        out = v2_model(
            image, stroke_tokens=tokens, stroke_mask=mask, class_chunk_size=8
        )
        assert out["center_heatmap"].shape == (1, 35, 64, 64)

    def test_class_identity_changes_fusion(self, v2_model) -> None:
        torch.manual_seed(2)
        image = torch.randn(1, 3, 512, 512)
        tokens, mask = self._strokes(1)
        with torch.no_grad():
            out_a = v2_model(
                image, class_ids=torch.tensor([4]), stroke_tokens=tokens, stroke_mask=mask
            )
            out_b = v2_model(
                image, class_ids=torch.tensor([9]), stroke_tokens=tokens, stroke_mask=mask
            )
        assert not torch.allclose(out_a["center_heatmap"], out_b["center_heatmap"])

    def test_backward_reaches_film_and_encoder(self) -> None:
        model = build_model(preset="dual_pathway_v2")
        image = torch.randn(1, 3, 512, 512)
        tokens, mask = self._strokes(1)
        out = model(image, class_ids=torch.tensor([4]), stroke_tokens=tokens, stroke_mask=mask)
        out["center_logits"].sum().backward()
        film = [n for n, p in model.named_parameters() if "vector_film" in n and p.grad is not None]
        encoder = [
            n for n, p in model.named_parameters()
            if n.startswith("vector_encoder") and p.grad is not None
        ]
        assert film and encoder

    def test_requires_conditioner(self) -> None:
        from src.models.config import ConditionerConfig, ModelConfig, VectorBranchConfig

        # class_conditioned fusion with the "none" conditioner is rejected at
        # model construction (the FiLM would have no input).
        config = ModelConfig(
            vector=VectorBranchConfig(enabled=True, class_conditioned=True),
            fusion_mode="none",
            conditioner=ConditionerConfig(kind="none"),
        )
        with pytest.raises(ValueError):
            build_model(config)

    def test_class_conditioned_serializes(self) -> None:
        config = ModelConfig(
            vector=VectorBranchConfig(enabled=True, class_conditioned=True)
        )
        data = config.to_dict()
        assert data["vector"]["class_conditioned"] is True
        restored = ModelConfig.from_dict(data)
        assert restored.vector.class_conditioned

    def test_budget_discipline_v2(self) -> None:
        base = sum(p.numel() for p in build_model(preset="floorplan_base").parameters())
        v2 = sum(p.numel() for p in build_model(preset="dual_pathway_v2").parameters())
        assert base < v2 < base * 1.5
