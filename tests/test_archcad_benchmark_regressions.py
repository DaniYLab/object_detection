"""Non-identity cases that distinguish metric semantics and label leakage."""
import copy
import math

import numpy as np
import pytest

from src.data.archcad_panoptic import parse_archcad_panoptic_annotation, oracle_panoptic_prediction, primitive_length
from src.evaluation.archcad_panoptic import evaluate_archcad_panoptic


def fixture():
    sample = parse_archcad_panoptic_annotation({'entities': [
        {'type': 'LINE', 'start': [0, 0], 'end': [10, 0], 'semantic': 1, 'instance': 'door'},
        {'type': 'LINE', 'start': [0, 10], 'end': [10, 10], 'semantic': 20},
        {'type': 'LINE', 'start': [0, 20], 'end': [10, 20], 'semantic': 100},
    ]}, image_id='sample')
    return sample, oracle_panoptic_prediction(sample)


def test_ignore_predictions_do_not_change_semantic_score():
    sample, pred = fixture()
    pred['semantic_labels'][2] = 1
    pred['instances'][1]['mask'][2] = True
    report = evaluate_archcad_panoptic([sample], [pred])
    assert report['semantic']['f1_macro'] == 1
    assert report['instance']['ap50'] == 1


def test_wrong_segment_class_counts_both_fp_and_fn():
    sample, pred = fixture()
    pred['instances'][1]['semantic_id'] = 2
    pq = evaluate_archcad_panoptic([sample], [pred])['panoptic']['all']
    assert (pq['tp'], pq['fp'], pq['fn']) == (1, 1, 1)
    assert pq['pq'] == pytest.approx(.5)


def test_duplicate_predictions_cannot_match_same_target_twice():
    sample, pred = fixture()
    pred['instances'].append(copy.deepcopy(pred['instances'][1]))
    pq = evaluate_archcad_panoptic([sample], [pred])['panoptic']['all']
    assert (pq['tp'], pq['fp'], pq['fn']) == (2, 1, 0)
    assert pq['pq'] == pytest.approx(.8)


def test_high_scoring_false_instance_penalizes_ap():
    sample, pred = fixture()
    pred['instances'][1]['score'] = .9
    pred['instances'].insert(0, {'semantic_id': 1, 'score': 1., 'mask': np.array([0, 1, 0], dtype=bool)})
    report = evaluate_archcad_panoptic([sample], [pred])
    assert report['instance']['ap50'] == pytest.approx(.5)


@pytest.mark.parametrize('bad', [[1.1, 20, 30], [1, 20, 31], [-1, 20, 30]])
def test_invalid_semantic_predictions_are_rejected(bad):
    sample, pred = fixture()
    pred['semantic_labels'] = bad
    with pytest.raises(ValueError, match='integer IDs'):
        evaluate_archcad_panoptic([sample], [pred])


def test_duplicate_image_ids_are_rejected():
    sample, pred = fixture()
    with pytest.raises(ValueError, match='duplicate'):
        evaluate_archcad_panoptic([sample, sample], [pred, pred])


@pytest.mark.parametrize('start,end,direction', [(350, 10, 0), (10, 350, 1)])
def test_arc_wraparound_uses_directed_sweep(start, end, direction):
    assert primitive_length({'type': 'ARC', 'radius': 10, 'start_angle': start, 'end_angle': end, 'direction': direction}) == pytest.approx(10 * math.radians(20))


def test_partial_ellipse_length_is_not_full_circumference():
    full = {'type': 'ELLIPSE', 'major_axis': [10, 0], 'ratio': .5}
    quarter = {**full, 'start_param': 0., 'end_param': math.pi / 2, 'is_ccw': True}
    assert primitive_length(quarter) == pytest.approx(primitive_length(full) / 4)


def test_extremely_flat_ellipse_has_finite_length_without_quadrature_warning():
    # Regression from raw chunk 11ebfe47-7eee-4b3f-b174-c6793fda21c2, entity 6665.
    ellipse = {'type': 'ELLIPSE', 'major_axis': [0, -.2415], 'ratio': .0006905256071358,
               'start_param': 0., 'end_param': 2 * math.pi, 'is_ccw': True}
    assert primitive_length(ellipse) == pytest.approx(.96600188046, rel=1e-8)


def test_parser_never_silently_removes_an_entity():
    with pytest.raises(TypeError, match='primitive order'):
        parse_archcad_panoptic_annotation({'entities': [None]}, image_id='broken')


def test_zero_weight_segment_is_explicitly_reported():
    sample = parse_archcad_panoptic_annotation({'entities': [
        {'type': 'LINE', 'start': [0, 0], 'end': [0, 0], 'semantic': 1, 'instance': 'zero'},
    ]}, image_id='zero')
    result = evaluate_archcad_panoptic([sample], [oracle_panoptic_prediction(sample)])
    assert result['unscorable'] == {'gt_segments': 1, 'prediction_segments': 1}
