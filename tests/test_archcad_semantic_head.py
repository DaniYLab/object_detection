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
from scripts.research.train_archcad_semantic import capped_evaluation_set
from src.models.archcad_semantic_head import (ArchCADPrimitiveNoContext, ArchCADPrimitiveSemantic,
                                             pad_semantic_batch, semantic_classification_loss)


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


def _brute_force_groups(points, types, semantics, proximity, connectivity):
    """Independent reference: union-find over proximity and endpoint edges."""
    parent = list(range(len(points)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if semantics[i] != semantics[j]:
                continue
            near = float(np.linalg.norm(points[i].mean(0) - points[j].mean(0))) < proximity
            closed = {2, 3}
            if not near and types[i] not in closed and types[j] not in closed:
                near = any(float(np.linalg.norm(a - b)) < connectivity
                           for a in (points[i][0], points[i][3])
                           for b in (points[j][0], points[j][3]))
            if near:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(points)):
        groups.setdefault(find(i), []).append(i)
    return sorted(tuple(sorted(members)) for members in groups.values())


def test_capped_evaluation_set_takes_a_prefix_and_keeps_everything_when_disabled():
    """The grouper costs ~19x the model forward per image, so the diagnostic train-side
    evaluation is capped; the cap must be a stable prefix, not a random sample."""
    items = [{'i': index} for index in range(10)]
    truths = [index for index in range(10)]
    assert capped_evaluation_set(items, truths, 0) == (items, truths)
    assert capped_evaluation_set(items, truths, -1) == (items, truths)
    assert capped_evaluation_set(items, truths, 10) == (items, truths)
    capped_items, capped_truths = capped_evaluation_set(items, truths, 4)
    assert capped_items == items[:4] and capped_truths == truths[:4]


def test_grouper_matches_a_brute_force_reference_when_one_class_holds_everything():
    """The pathological case that stalled a run: an untrained model predicts one class.

    The vectorised endpoint search must still produce exactly the groupings a naive
    union-find produces, including for closed shapes that carry no endpoints.
    """
    rng = np.random.default_rng(5)
    points = rng.random((60, 4, 2))
    # Three pairs that share endpoints, so only connectivity can join them.
    for target, source in ((1, 0), (5, 4), (3, 2)):
        points[target][0], points[target][3] = points[source][3], points[source][0]
    types = np.zeros(60, dtype=np.int64)
    types[7] = 2  # a circle: no endpoints, so only proximity can join it
    points[7] = points[6].mean(0) + 1e-6
    semantics = np.full(60, 1, dtype=np.int64)
    produced = group_instances(points, types, semantics, proximity=0.02, connectivity=0.01)
    got = sorted(tuple(sorted(np.flatnonzero(mask).tolist())) for _, mask in produced)
    assert got == _brute_force_groups(points, types, semantics, 0.02, 0.01)


def test_grouper_handles_a_thousand_primitives_in_one_class_promptly():
    """Guards the quadratic-in-Python regression: it stalled a 4,096-image run entirely."""
    import time

    rng = np.random.default_rng(6)
    points = rng.random((1000, 4, 2))
    types = np.zeros(1000, dtype=np.int64)
    start = time.perf_counter()
    group_instances(points, types, np.full(1000, 1, dtype=np.int64), proximity=0.02, connectivity=0.01)
    assert time.perf_counter() - start < 5.0, 'grouping must not be quadratic in Python'


def test_no_context_model_ignores_every_other_primitive_in_the_batch():
    """The defining property: a primitive's logits depend on that primitive alone."""
    torch.manual_seed(11)
    model = ArchCADPrimitiveNoContext(in_features=12, width=32)
    alone = torch.randn(1, 5, 12)
    padded = torch.cat([alone, torch.randn(1, 7, 12)], dim=1)
    with torch.no_grad():
        single = model(alone, None)['semantic_logits']
        many = model(padded, None)['semantic_logits']
        other_image = model(torch.randn(1, 5, 12), None)['semantic_logits']
    torch.testing.assert_close(single, many[:, :5], rtol=0, atol=0), 'appending primitives must change nothing'
    assert not torch.allclose(single, other_image), 'different primitives must give different logits'


def test_no_context_model_ignores_the_padding_mask():
    torch.manual_seed(12)
    model = ArchCADPrimitiveNoContext(in_features=12, width=32)
    features = torch.randn(2, 6, 12)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, 4:] = True
    with torch.no_grad():
        torch.testing.assert_close(model(features, None)['semantic_logits'],
                                   model(features, mask)['semantic_logits'])


def test_no_context_model_config_rebuilds_the_model():
    model = ArchCADPrimitiveNoContext(in_features=12, width=16)
    rebuilt = ArchCADPrimitiveNoContext(**model.config)
    assert list(rebuilt.state_dict()) == list(model.state_dict())
    assert model(torch.randn(1, 3, 12), None)['semantic_logits'].shape == (1, 3, 30)
