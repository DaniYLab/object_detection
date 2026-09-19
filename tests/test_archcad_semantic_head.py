"""Per-primitive semantic head and the geometric grouper it is scored through.

The head exists because semantics, not grouping, carry the measured PQ gap. These tests
pin the two properties that make the comparison meaningful: the loss ignores `ignore`
primitives exactly as the metric does, and the grouper emits the instance structure the
task defines (one stuff instance per class, components for things).
"""
import numpy as np
import torch

from src.data.archcad_constants import ARCHCAD_IGNORE_LABEL, ARCHCAD_STUFF_SEMANTIC_IDS
from src.data.archcad_geometric_grouping import (fit_thresholds, group_instances, pairwise_f1,
                                                predictions_from_semantics)
from src.models.archcad_semantic_head import (ArchCADPrimitiveSemantic, pad_semantic_batch,
                                             semantic_classification_loss)


def _item(n, labels, weights=None, width=12):
    weights = torch.ones(n) if weights is None else torch.as_tensor(weights, dtype=torch.float32)
    return {'features': torch.randn(n, width),
            'targets': {'semantic_labels': torch.tensor(labels, dtype=torch.long),
                        'length_weights': weights}}


def test_semantic_head_reports_its_input_contract_and_shapes():
    model = ArchCADPrimitiveSemantic(in_features=12, dim=16, heads=4, encoder_layers=1)
    assert model.config['in_features'] == 12 and model.config['classes'] == 30
    output = model(torch.randn(2, 7, 12), None)
    assert output['semantic_logits'].shape == (2, 7, 30)
    try:
        model(torch.randn(2, 7, 18), None)
    except ValueError as error:
        assert 'features' in str(error)
    else:
        raise AssertionError('a wrong feature width must be rejected')


def test_model_config_rebuilds_the_model_so_checkpoints_stay_loadable():
    """Every consumer rebuilds with Model(**checkpoint['model_config']); keep it valid."""
    model = ArchCADPrimitiveSemantic(in_features=12, dim=16, heads=4, encoder_layers=1)
    rebuilt = ArchCADPrimitiveSemantic(**model.config)
    assert list(rebuilt.state_dict()) == list(model.state_dict())
    rebuilt.load_state_dict(model.state_dict())
    features = torch.randn(1, 5, 12)
    torch.testing.assert_close(model(features, None)['semantic_logits'],
                               rebuilt(features, None)['semantic_logits'])


def test_padding_marks_positions_as_ignore_with_zero_weight():
    batch = pad_semantic_batch([_item(3, [1, 2, 30]), _item(5, [4, 5, 6, 7, 8])])
    assert batch['sizes'] == [3, 5] and batch['features'].shape == (2, 5, 12)
    assert batch['padding_mask'][0].tolist() == [False, False, False, True, True]
    assert batch['labels'][0].tolist() == [1, 2, 30, 30, 30]
    assert batch['weights'][0].tolist() == [1., 1., 1., 0., 0.]
    assert not batch['padding_mask'][1].any()


def test_semantic_loss_ignores_ignore_label_and_respects_weights():
    torch.manual_seed(0)
    model = ArchCADPrimitiveSemantic(in_features=12, dim=16, heads=4, encoder_layers=1)
    batch = pad_semantic_batch([_item(4, [1, 2, ARCHCAD_IGNORE_LABEL, 5], weights=[1., 2., 3., 4.])])
    loss = semantic_classification_loss(model(batch['features'], batch['padding_mask']), batch)
    logits = model(batch['features'], batch['padding_mask'])['semantic_logits'][0, :4]
    # Only the three non-ignore positions can enter a cross-entropy: 30 is not a class.
    plain = torch.nn.functional.cross_entropy(logits[[0, 1, 3]], torch.tensor([1, 2, 5]))
    assert not torch.allclose(loss, plain), 'length weighting must change the value'
    # The ignore position contributes nothing, so its weight must not matter either.
    batch['weights'][0, 2] = 100.
    assert semantic_classification_loss(model(batch['features'], batch['padding_mask']), batch) == loss


def test_grouper_emits_one_stuff_instance_per_class_and_components_for_things():
    # Two well-separated pairs of overlapping points, all one class.
    points = np.array([[[0., 0.], [.001, 0.], [.002, 0.], [.003, 0.]],
                       [[0., .001], [.001, .001], [.002, .001], [.003, .001]],
                       [[.9, .9], [.901, .9], [.902, .9], [.903, .9]],
                       [[.9, .901], [.901, .901], [.902, .901], [.903, .901]]])
    types = np.zeros(4, dtype=np.int64)
    instances = group_instances(points, types, np.array([1, 1, 1, 1]), proximity=0.02, connectivity=0.0)
    assert len(instances) == 2, 'the two clusters must not merge'
    assert all(len(mask) == 4 for _, mask in instances)
    stuff = group_instances(points, types, np.full(4, ARCHCAD_STUFF_SEMANTIC_IDS[0]), proximity=0.02)
    assert len(stuff) == 1, 'stuff is one instance per class regardless of distance'


def test_pairwise_f1_is_one_for_a_perfect_grouping_and_zero_for_singletons():
    points = np.random.default_rng(0).random((4, 4, 2))
    types = np.zeros(4, dtype=np.int64)
    semantics = np.ones(4, dtype=np.int64)
    isolated = np.array([0, 1, 2, 3])
    assert pairwise_f1(points, types, semantics, isolated, proximity=1e-6, connectivity=0.0)[2] == 0.0


def test_fitted_thresholds_are_chosen_on_the_items_given():
    rng = np.random.default_rng(1)
    items = []
    for _ in range(3):
        # Two tight pairs whose centres are 0.3 apart: a small proximity separates them, a
        # large one merges them, so the fit has something to choose between.
        points = np.concatenate([rng.normal(0, 1e-4, (2, 4, 2)),
                                 rng.normal(0.3, 1e-4, (2, 4, 2))])
        items.append({'features': torch.tensor(points.reshape(4, 8), dtype=torch.float32),
                      'primitive_types': torch.zeros(4, dtype=torch.int64),
                      'targets': {'semantic_labels': torch.ones(4, dtype=torch.long),
                                  'instance_ids': torch.tensor([0, 0, 1, 1])}})
    best, scores = fit_thresholds(items, proximities=(1e-3, 0.5), connectivities=(0.0,))
    assert best[0] == 1e-3, 'the tight proximity is the one that separates the pairs'
    assert scores[(1e-3, 0.0)] > scores[(0.5, 0.0)]


def test_predictions_from_semantics_uses_only_the_given_labels():
    item = {'image_id': 'x', 'features': torch.randn(4, 12),
            'primitive_types': torch.zeros(4, dtype=torch.int64)}
    predictions = predictions_from_semantics([item], [np.array([1, 1, 2, 2])], proximity=0.02)
    assert predictions[0]['image_id'] == 'x'
    assert predictions[0]['semantic_labels'].tolist() == [1, 1, 2, 2]
    assert all(1.0 == instance['score'] for instance in predictions[0]['instances'])
