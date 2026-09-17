"""Batched-step contract: same model, same loss, same metrics, fewer launches.

The batched path is only worth using if it changes nothing except how many images
share one optimizer step. These tests pin that: identical parameters, a forward that
is exact at B=1, padding that cannot reach real tokens, and a per-image loss that
equals the frozen loss including when images carry different segment counts.
"""
import torch

from scripts.research.train_archcad_primitive_batched import batch_padding, training_order
from src.models.archcad_primitive_baseline import ArchCADPrimitiveBaseline, primitive_mask_loss
from src.models.archcad_primitive_batched import (ArchCADPrimitiveBatched, batch_items,
                                                  batched_primitive_mask_loss, pad_primitive_batch,
                                                  padded_fraction)


def _item(n_primitives, labels, masks):
    return {'features': torch.randn(n_primitives, 12),
            'targets': {'labels': torch.tensor(labels, dtype=torch.long),
                        'masks': torch.as_tensor(masks, dtype=torch.bool)}}


def _model(seed=1337):
    torch.manual_seed(seed)
    return ArchCADPrimitiveBaseline(dim=16, num_queries=4, encoder_layers=1, decoder_layers=1)


def test_batched_model_shares_state_dict_with_the_frozen_model():
    plain = _model()
    batched = ArchCADPrimitiveBatched(dim=16, num_queries=4, encoder_layers=1, decoder_layers=1)
    assert list(plain.state_dict()) == list(batched.state_dict())
    batched.load_state_dict(plain.state_dict())  # weights transfer without any mapping


def test_batched_forward_is_exact_for_a_single_unpadded_image():
    plain = _model()
    batched = ArchCADPrimitiveBatched(dim=16, num_queries=4, encoder_layers=1, decoder_layers=1)
    batched.load_state_dict(plain.state_dict())
    features = torch.randn(9, 12)
    with torch.no_grad():
        one = plain(features)
        many = batched(features[None], None)
    assert torch.equal(one['class_logits'], many['class_logits'][0])
    assert torch.equal(one['mask_logits'], many['mask_logits'][0])


def test_padding_does_not_reach_real_tokens():
    plain = _model()
    batched = ArchCADPrimitiveBatched(dim=16, num_queries=4, encoder_layers=1, decoder_layers=1)
    batched.load_state_dict(plain.state_dict())
    short, long = torch.randn(5, 12), torch.randn(21, 12)
    with torch.no_grad():
        alone = plain(short)
        batch = pad_primitive_batch([{'features': short, 'targets': {'labels': torch.tensor([1]),
                                                                   'masks': torch.ones(1, 5, dtype=torch.bool)}},
                                     {'features': long, 'targets': {'labels': torch.tensor([2]),
                                                                    'masks': torch.ones(1, 21, dtype=torch.bool)}}])
        together = batched(batch['features'], batch['padding_mask'])
    assert batch['sizes'] == [5, 21] and batch['padding_mask'].shape == (2, 21)
    assert batch['padding_mask'][0].sum() == 16 and not batch['padding_mask'][1].any()
    # The short image must be unaffected by 16 pad positions appended to its batch.
    torch.testing.assert_close(alone['class_logits'], together['class_logits'][0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(alone['mask_logits'], together['mask_logits'][0][:, :5], rtol=1e-4, atol=1e-5)


def test_pad_primitive_batch_places_targets_at_the_right_offsets():
    items = [_item(4, [1, 2], [[True, False, False, False], [False, True, True, False]]),
             _item(3, [7], [[False, False, True]])]
    batch = pad_primitive_batch(items)
    assert batch['sizes'] == [4, 3] and batch['features'].shape == (2, 4, 12)
    assert batch['masks'].shape == (2, 2, 4), 'segment axis is padded to the batch maximum'
    assert batch['masks'][0, 0, :4].tolist() == [True, False, False, False]
    assert batch['masks'][1, 0, :3].tolist() == [False, False, True]
    assert not batch['masks'][1, 0, 3], 'the padded primitive slot must stay clear'
    assert not batch['masks'][1, 1].any(), 'the padded segment row must stay clear'


def test_batched_loss_equals_the_frozen_loss_with_ragged_segment_counts():
    """The segment axis is padded, so the loss must slice back to each image's own rows."""
    plain = _model()
    batched = ArchCADPrimitiveBatched(dim=16, num_queries=4, encoder_layers=1, decoder_layers=1)
    batched.load_state_dict(plain.state_dict())
    items = [_item(6, [1, 2, 3], torch.eye(3, 6, dtype=torch.bool)),
             _item(4, [5], [[True, True, False, False]])]
    batch = pad_primitive_batch(items)
    with torch.no_grad():
        output = batched(batch['features'], batch['padding_mask'])
        combined = batched_primitive_mask_loss(output, batch)
        expected = torch.stack([
            primitive_mask_loss(plain(items[0]['features']), items[0]['targets'])['loss'],
            primitive_mask_loss(plain(items[1]['features']), items[1]['targets'])['loss']]).mean()
    torch.testing.assert_close(combined, expected, rtol=1e-5, atol=1e-5)


def test_batch_items_and_padding_accounting():
    items = [_item(n, [1], [[True] * n]) for n in (10, 10, 10, 10, 4)]
    batches = batch_items(items, 2)
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert padded_fraction(items[:2]) == 0.
    assert padded_fraction([items[0], items[4]]) == 1 - 14 / 20


def test_training_order_covers_every_image_once_and_can_group_by_length():
    items = [_item(n, [1], [[True] * n]) for n in (50, 5, 30, 7, 21)]
    for grouped in (False, True):
        batches = training_order(items, 2, seed=1337, epoch=1, group_by_length=grouped)
        covered = [index for batch in batches for index in batch]
        assert sorted(covered) == list(range(len(items))), 'every image exactly once'
    # Grouping by length exists to cut padding; compare the two orders directly
    # rather than asserting an arbitrary within-batch spread.
    ungrouped = training_order(items, 2, seed=1337, epoch=1, group_by_length=False)
    grouped = training_order(items, 2, seed=1337, epoch=1, group_by_length=True)
    assert batch_padding(items, grouped) < batch_padding(items, ungrouped)
    assert batch_padding(items, grouped) < 0.2
