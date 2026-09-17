"""Decoder contract for the opt-in consistent semantic-map thresholds.

The default decode path is the frozen decoder of the 2026-09-16/17 runs, so the
new policy must be provably inert unless explicitly requested. Instances are
built from query class scores and must never depend on the semantic map.
"""
import numpy as np
import torch

from src.models.archcad_primitive_baseline import ArchCADPrimitiveBaseline, decode_primitive_masks


def test_decode_default_path_is_unaffected_by_the_consistent_semantic_option():
    torch.manual_seed(7)
    model = ArchCADPrimitiveBaseline(dim=16, num_queries=4, encoder_layers=1, decoder_layers=1)
    output = model(torch.randn(9, 12))
    default = decode_primitive_masks(output, 'sample')
    explicit = decode_primitive_masks(output, 'sample', consistent_semantic_thresholds=False)
    assert np.array_equal(default['semantic_labels'], explicit['semantic_labels'])
    assert [p['semantic_id'] for p in default['instances']] == [p['semantic_id'] for p in explicit['instances']]
    # The consistent policy may only replace labels, never instances or their order.
    consistent = decode_primitive_masks(output, 'sample', consistent_semantic_thresholds=True)
    assert [p['semantic_id'] for p in consistent['instances']] == [p['semantic_id'] for p in default['instances']]
    assert len(consistent['semantic_labels']) == len(default['semantic_labels'])
    assert set(np.unique(consistent['semantic_labels'])).issubset(set(np.unique(default['semantic_labels'])) | {30})
