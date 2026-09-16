import torch
from scripts.research.train_archcad_primitive_pilot import evaluation_due, evaluate
from src.models.archcad_primitive_baseline import ArchCADPrimitiveBaseline, primitive_mask_loss
from src.data.archcad_panoptic import parse_archcad_panoptic_annotation


def test_evaluation_schedule_includes_final_partial_interval():
    assert [e for e in range(1, 24) if evaluation_due(e, 23, 10)] == [10, 20, 23]
    assert [e for e in range(1, 24) if evaluation_due(e, 23, 0)] == [23]


def test_monitoring_does_not_change_training_rng_or_update():
    torch.manual_seed(1337)
    model = ArchCADPrimitiveBaseline(dim=16, num_queries=2, encoder_layers=1, decoder_layers=1)
    truth = parse_archcad_panoptic_annotation({'entities': [
        {'type': 'LINE', 'start': [0, 0], 'end': [1, 1], 'semantic': 1, 'instance': 'a'},
        {'type': 'LINE', 'start': [2, 2], 'end': [3, 3], 'semantic': 20},
    ]}, image_id='sample')
    item = {'image_id': 'sample', 'features': torch.randn(2, 12),
            'targets': {'labels': torch.tensor([20, 1]), 'masks': torch.tensor([[False, True], [True, False]])}}
    model.train()
    before = primitive_mask_loss(model(item['features']), item['targets'])['loss']
    rng = torch.get_rng_state().clone()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    evaluate(model, [item], [truth], 'cpu')
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, weights[k]) for k, v in model.state_dict().items())
    model.train()
    after = primitive_mask_loss(model(item['features']), item['targets'])['loss']
    torch.testing.assert_close(before, after)
