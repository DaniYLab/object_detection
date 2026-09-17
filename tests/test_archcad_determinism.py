"""Deterministic-mode plumbing and capacity flags for the ArchCAD pilot trainer.

The 2026-09-16 and 2026-09-17 seed-1337 runs recorded identical seeds, data and
code but produced different epoch-20 validation metrics, so the control
comparison they were built on was not reproducible. These tests pin the pieces
that make a rerun verifiable; they cannot test the GPU kernels themselves.
"""
import json
import os
import sys

import pytest
import torch

from scripts.research.train_archcad_primitive_pilot import enforce_determinism, select_samples
from src.models.archcad_primitive_baseline import ArchCADPrimitiveBaseline


def test_enforce_determinism_configures_torch_and_reports_settings(monkeypatch):
    saved = {
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'flash_sdp': torch.backends.cuda.flash_sdp_enabled(),
        'mem_efficient_sdp': torch.backends.cuda.mem_efficient_sdp_enabled(),
        'math_sdp': torch.backends.cuda.math_sdp_enabled(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
    }
    # Present already: the configuration path must not restart the process.
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.setattr(os, 'execv', lambda *a: pytest.fail('must not restart when the variable is present'))
    try:
        settings = enforce_determinism()
        assert settings['enabled'] is True
        assert settings['deterministic_algorithms'] is True
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cuda.flash_sdp_enabled() is False
        assert torch.backends.cuda.mem_efficient_sdp_enabled() is False
        assert torch.backends.cuda.math_sdp_enabled() is True
        assert settings['cublas_workspace_config'] == ':4096:8'
    finally:
        torch.use_deterministic_algorithms(saved['deterministic_algorithms'])
        torch.backends.cudnn.benchmark = saved['cudnn_benchmark']
        torch.backends.cudnn.deterministic = saved['cudnn_deterministic']
        torch.backends.cuda.enable_flash_sdp(saved['flash_sdp'])
        torch.backends.cuda.enable_mem_efficient_sdp(saved['mem_efficient_sdp'])
        torch.backends.cuda.enable_math_sdp(saved['math_sdp'])


def test_enforce_determinism_fails_loudly_when_the_workspace_variable_is_absent(monkeypatch):
    """Silently running non-deterministically would invalidate every control comparison."""
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    with pytest.raises(RuntimeError, match='CUBLAS_WORKSPACE_CONFIG'):
        enforce_determinism()


def test_capacity_flags_reach_the_resolved_model_config():
    model = ArchCADPrimitiveBaseline(dim=32, num_queries=4, heads=8, encoder_layers=3, decoder_layers=5)
    assert model.config == {'dim': 32, 'num_queries': 4, 'heads': 8, 'encoder_layers': 3, 'decoder_layers': 5}
    # Wider/deeper variants must remain buildable and forwardable on CPU.
    features = torch.randn(6, 12)
    output = model(features)
    assert output['class_logits'].shape == (4, 31)
    assert output['mask_logits'].shape == (4, 6)


class _StubDataset:
    """Smallest thing select_samples accepts: image_ids, root with json/, and indexing."""

    def __init__(self, root, image_ids):
        self.root = root
        self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # 'image' is present because select_samples drops it for the primitive-only control.
        return {'image_id': self.image_ids[index], 'features': torch.randn(2, 12),
                'image': torch.zeros(3, 4, 4)}


def _line_entity(semantic, instance=None):
    entity = {'type': 'LINE', 'start': [0, 0], 'end': [1, 1], 'semantic': semantic}
    if instance is not None:
        entity['instance'] = instance
    return entity


def _stub_root(tmp_path):
    # Four images: two eligible, one over the primitive budget, one over the segment budget.
    # Segments are per thing instance and per stuff class, so exceeding the segment budget
    # needs distinct instances, not repeated primitives.
    payloads = {
        'ok_a': {'entities': [_line_entity(1, 'i'), _line_entity(1, 'i')]},
        'ok_b': {'entities': [_line_entity(20)]},
        'too_many_primitives': {'entities': [_line_entity(1, 'i') for _ in range(9)]},
        'too_many_segments': {'entities': [_line_entity(1, f'i{k}') for k in range(4)]},
    }
    (tmp_path / 'json').mkdir()
    for image_id, payload in payloads.items():
        (tmp_path / 'json' / f'{image_id}.json').write_text(json.dumps(payload), encoding='utf-8')
    return tmp_path


def test_select_samples_counts_a_requested_prefix(tmp_path):
    dataset = _StubDataset(_stub_root(tmp_path), ['ok_a', 'ok_b', 'too_many_primitives', 'too_many_segments'])
    selected, truths, skips = select_samples(dataset, 1, seed=1337, max_primitives=8, num_queries=3)
    assert len(selected) == 1 and len(truths) == 1
    # skips only counts candidates examined before the quota was filled, so it may be empty.


def test_select_samples_negative_count_collects_every_eligible_image(tmp_path):
    """The full-split arms cannot know the eligible count up front, so -1 must resolve it."""
    dataset = _StubDataset(_stub_root(tmp_path), ['ok_a', 'ok_b', 'too_many_primitives', 'too_many_segments'])
    selected, truths, skips = select_samples(dataset, -1, seed=1337, max_primitives=8, num_queries=3)
    assert len(selected) == 2 == len(truths)
    assert skips == {'primitive_budget': 1, 'segment_budget': 1}
    # A negative count is a superset of any prefix of the same shuffle.
    prefix, _, _ = select_samples(dataset, 1, seed=1337, max_primitives=8, num_queries=3)
    assert prefix[0]['image_id'] in {item['image_id'] for item in selected}
    # Requesting more than exist must still fail loudly rather than silently truncating.
    with pytest.raises(ValueError, match='only 2 eligible'):
        select_samples(dataset, 3, seed=1337, max_primitives=8, num_queries=3)
