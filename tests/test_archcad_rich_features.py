"""Scale-aware input features: pure augmentation, wider embedding, nothing else changed.

The point of these features is that a fixed distance threshold cannot express them, so
the tests check the properties that make them useful: invariance to where a drawing sits
in the image, responsiveness to how densely primitives are packed, and the scale-normalised
ratio behaving for a lone primitive.
"""
import math

import torch

from src.data.archcad_rich_features import (FEATURE_NAMES, RICH_WIDTH, ArchCADPrimitiveRich,
                                            augment_items, rich_features)


def _item(points, centres=None, types=None):
    points = torch.tensor(points, dtype=torch.float32)
    n = points.shape[0]
    one_hot = torch.eye(4)[torch.zeros(n, dtype=torch.long)] if types is None else torch.eye(4)[torch.tensor(types)]
    return {'features': torch.cat([points.reshape(n, 8), one_hot], dim=1),
            'centers': torch.tensor(centres, dtype=torch.float32) if centres is not None else points.mean(1)}


def test_rich_features_append_exactly_the_declared_columns():
    item = _item([[[0., 0.], [.1, 0.], [.2, 0.], [.3, 0.]], [[.5, .5], [.6, .5], [.7, .5], [.8, .5]]])
    out = rich_features(item)
    assert out.shape == (2, RICH_WIDTH) and RICH_WIDTH == 12 + len(FEATURE_NAMES)
    torch.testing.assert_close(out[:, :12], item['features'], rtol=0, atol=0)


def test_polyline_length_and_extent_measure_the_primitive():
    item = _item([[[0., 0.], [.1, 0.], [.2, 0.], [.3, 0.]]])
    out = rich_features(item)
    length = out[0, 12].item()
    extent = out[0, 13].item()
    torch.testing.assert_close(torch.tensor(length), torch.tensor(0.3), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.tensor(extent), torch.tensor(0.3), rtol=1e-5, atol=1e-6)


def test_local_scale_is_translation_invariant():
    """Moving a whole drawing must not change the packing scale it reports."""
    points = [[[0., 0.], [.1, 0.], [.2, 0.], [.3, 0.]], [[.32, 0.], [.34, 0.], [.36, 0.], [.38, 0.]]]
    shifted = [[[0.4 + x, 0.7 + y] for x, y in primitive] for primitive in points]
    first = rich_features(_item(points))
    second = rich_features(_item(shifted))
    torch.testing.assert_close(first[:, 14:17], second[:, 14:17], rtol=1e-5, atol=1e-6)


def test_packing_scale_responds_to_density():
    """The same primitive in a dense grid reports a smaller local scale than when alone."""
    lone = _item([[[0., 0.], [.01, 0.], [.02, 0.], [.03, 0.]]])
    dense = _item([[[x, 0.], [x + .01, 0.], [x + .02, 0.], [x + .03, 0.]] for x in (0., .05, .1, .15, .2)])
    torch.testing.assert_close(torch.tensor(rich_features(lone)[0, 14].item()),
                               torch.tensor(0.0), rtol=1e-6, atol=1e-6), 'a lone primitive has no neighbour'
    assert rich_features(dense)[0, 14] < 0.06, 'the dense grid reports a small nearest-neighbour distance'


def test_size_ratio_separates_small_from_large_primitives_in_the_same_packing():
    """Two primitives packed at the same spacing but of different size must differ."""
    items = _item([[[0., 0.], [.01, 0.], [.02, 0.], [.03, 0.]],
                   [[.0, .2], [.05, .2], [.10, .2], [.15, .2]]])
    out = rich_features(items)
    assert out[0, 17] < out[1, 17], 'the long primitive reports a larger size ratio'


def test_augment_items_is_in_place_and_pads_consistently():
    items = [_item([[[0., 0.], [.1, 0.], [.2, 0.], [.3, 0.]]]),
             _item([[[0., .5], [.1, .5], [.2, .5], [.3, .5]], [[.4, .5], [.5, .5], [.6, .5], [.7, .5]]])]
    returned = augment_items(items)
    assert returned is items
    assert [item['features'].shape[1] for item in items] == [RICH_WIDTH, RICH_WIDTH]


def test_rich_model_accepts_the_wider_input_and_reports_it():
    torch.manual_seed(3)
    model = ArchCADPrimitiveRich(dim=16, num_queries=4, encoder_layers=1, decoder_layers=1)
    assert model.config['input_features'] == RICH_WIDTH
    output = model(torch.randn(2, 7, RICH_WIDTH), None)
    assert output['class_logits'].shape == (2, 4, 31)
    assert output['mask_logits'].shape == (2, 4, 7)
