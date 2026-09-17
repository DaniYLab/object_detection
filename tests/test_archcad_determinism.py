"""Deterministic-mode plumbing and capacity flags for the ArchCAD pilot trainer.

The 2026-09-16 and 2026-09-17 seed-1337 runs recorded identical seeds, data and
code but produced different epoch-20 validation metrics, so the control
comparison they were built on was not reproducible. These tests pin the pieces
that make a rerun verifiable; they cannot test the GPU kernels themselves.
"""
import os
import sys

import pytest
import torch

from scripts.research.train_archcad_primitive_pilot import enforce_determinism
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


def test_enforce_determinism_restarts_before_cuda_init_when_the_variable_is_absent(monkeypatch):
    """cuBLAS must see the variable before the first handle, so in-process setting is not enough."""
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    calls = []

    def fake_execv(path, argv):
        calls.append((path, argv))
        raise RuntimeError('re-exec requested')

    monkeypatch.setattr(os, 'execv', fake_execv)
    try:
        with pytest.raises(RuntimeError, match='re-exec requested'):
            enforce_determinism()
        assert len(calls) == 1
        assert calls[0][0] == sys.executable
        # The variable is exported before restarting, so the child does not loop.
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
        assert calls[0][1][:2] == [sys.executable, sys.argv[0]]
    finally:
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)


def test_capacity_flags_reach_the_resolved_model_config():
    model = ArchCADPrimitiveBaseline(dim=32, num_queries=4, heads=8, encoder_layers=3, decoder_layers=5)
    assert model.config == {'dim': 32, 'num_queries': 4, 'heads': 8, 'encoder_layers': 3, 'decoder_layers': 5}
    # Wider/deeper variants must remain buildable and forwardable on CPU.
    features = torch.randn(6, 12)
    output = model(features)
    assert output['class_logits'].shape == (4, 31)
    assert output['mask_logits'].shape == (4, 6)
