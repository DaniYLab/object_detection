"""Irreducible input/target ambiguity in the ArchCAD primitive representation.

A geometry-only, permutation-equivariant model cannot separate identical model
inputs. Two cases are audited: identical inputs whose target-mask signatures
differ, and identical inputs whose semantic labels differ. The latter is
irreducible for the class head even when the masks agree.
"""
import numpy as np

from scripts.research.audit_archcad_feature_ambiguity import conflicting_groups
from src.data.archcad_constants import ARCHCAD_IGNORE_LABEL


def _ambiguous_case():
    features = np.array([[0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0.]] * 3, dtype=np.float32)
    labels = np.array([1, 20, ARCHCAD_IGNORE_LABEL])
    masks = np.array([[True, False, False], [False, True, True]])
    return features, labels, masks


def test_conflicting_groups_counts_identical_inputs_with_different_targets():
    features, labels, masks = _ambiguous_case()
    mask_conflicts, label_conflicts, ambiguous = conflicting_groups(features, labels, masks)
    assert len(mask_conflicts) == 1 and len(label_conflicts) == 1 and len(ambiguous) == 1
    assert ambiguous[0].tolist() == [0, 1]
    assert mask_conflicts[0][1].tolist() == [1, 20]


def test_conflicting_groups_ignores_ignore_label_and_distinct_inputs():
    features, labels, masks = _ambiguous_case()
    # Only one non-ignore member: the ignore primitive cannot contradict anything.
    single = labels.copy()
    single[1] = ARCHCAD_IGNORE_LABEL
    assert conflicting_groups(features, single, masks)[2] == []
    # Distinct geometry: the model can separate them, so no conflict is reported.
    distinct = features.copy()
    distinct[1, 0] = 5.
    assert conflicting_groups(distinct, labels, masks)[2] == []


def test_identical_inputs_with_different_labels_conflict_even_when_masks_agree():
    """Different labels on identical inputs are irreducible class ambiguity.

    The mask head is unconstrained here (identical targets), so only the label
    branch may fire; this is the case the mask-only check used to miss.
    """
    features, labels, _ = _ambiguous_case()
    same_masks = np.array([[1, 1, 1], [1, 1, 1]])
    mask_conflicts, label_conflicts, ambiguous = conflicting_groups(features, labels, same_masks)
    assert mask_conflicts == []
    assert len(label_conflicts) == 1 and len(ambiguous) == 1
    assert label_conflicts[0][1].tolist() == [1, 20]


def test_identical_inputs_with_identical_targets_are_only_duplicates():
    features, labels, _ = _ambiguous_case()
    single_label = np.full(3, 1)
    same_masks = np.array([[1, 1, 1], [1, 1, 1]])
    assert conflicting_groups(features, single_label, same_masks)[2] == []
