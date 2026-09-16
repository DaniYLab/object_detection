import torch
import pytest

from src.models.archcad_primitive_baseline import ArchCADPrimitiveBaseline, primitive_mask_loss, decode_primitive_masks


def test_hungarian_loss_is_invariant_to_target_order_and_backpropagates():
    torch.manual_seed(3)
    model = ArchCADPrimitiveBaseline(dim=16, num_queries=4, encoder_layers=1, decoder_layers=1)
    output = model(torch.randn(6, 12))
    target = {'labels': torch.tensor([1, 20]), 'masks': torch.tensor([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.bool)}
    result = primitive_mask_loss(output, target)
    swapped = primitive_mask_loss(output, {'labels': target['labels'].flip(0), 'masks': target['masks'].flip(0)})
    torch.testing.assert_close(result['loss'], swapped['loss'])
    result['loss'].backward()
    assert model.embedding[0].weight.grad.norm() > 0
    assert model.classes.weight.grad.norm() > 0
    assert model.mask_queries.weight.grad.norm() > 0


def test_decoder_produces_disjoint_masks_and_merges_stuff():
    logits = torch.full((3, 31), -20.)
    logits[0, 20] = logits[1, 20] = logits[2, 1] = 20
    output = {'class_logits': logits, 'mask_logits': torch.tensor([[20., -20, -20], [-20, 20., -20], [-20, -20, 20.]])}
    pred = decode_primitive_masks(output, 'drawing')
    assert len(pred['instances']) == 2
    assert pred['semantic_labels'].tolist() == [20, 20, 1]
    assert pred['instances'][0]['mask'].tolist() == [True, True, False]


def test_query_budget_cannot_drop_targets():
    model = ArchCADPrimitiveBaseline(dim=16, num_queries=1, decoder_layers=1)
    with pytest.raises(ValueError, match='more target segments'):
        primitive_mask_loss(model(torch.randn(2, 12)), {'labels': torch.tensor([1, 2]), 'masks': torch.eye(2, dtype=torch.bool)})
