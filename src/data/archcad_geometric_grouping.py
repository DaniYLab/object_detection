"""Geometry-only instance grouping over ArchCAD primitives.

Measured on the validation subset (see `docs/experiments/archcad_semantics_bottleneck_20260917.md`):
a learning-free grouper built from centre proximity and exact endpoint connectivity,
given oracle per-primitive semantics, reaches pooled PQ 0.831 where the trained query/mask
model reaches 0.119. Grouping is therefore not the binding constraint, and this module
exists so that per-primitive semantic quality can be measured against a fixed, honest
grouping instead of against a jointly learned mask head.

Construction, per image:

  stuff        one instance per stuff class, as the task defines
  things       connected components of same-class primitives linked by
               centre distance < `proximity` or shared endpoints within `connectivity`

Both constants are hyperparameters and must be fitted on the training split. Fitting them
on the evaluation split would make every number reported against this grouper optimistic;
`fit_thresholds` exists so that the choice is reproducible and attributable.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.data.archcad_constants import (ARCHCAD_IGNORE_LABEL, ARCHCAD_STUFF_SEMANTIC_IDS,
                                        ARCHCAD_THING_SEMANTIC_IDS)

LINE, ARC, CIRCLE, ELLIPSE = 0, 1, 2, 3
DEFAULT_PROXIMITY = 0.1
DEFAULT_CONNECTIVITY = 0.01
CLOSED_TYPES = (CIRCLE, ELLIPSE)


def endpoint_pairs(points, types):
    """Left and right endpoints; closed primitives have none (NaN rows)."""
    left = np.asarray(points, dtype=np.float64)[:, 0, :].copy()
    right = np.asarray(points, dtype=np.float64)[:, 3, :].copy()
    closed = np.isin(np.asarray(types), CLOSED_TYPES)
    left[closed] = np.nan
    right[closed] = np.nan
    return left, right


def _components(count, edges):
    if not edges:
        return np.arange(count)
    rows = np.fromiter((a for a, _ in edges), dtype=np.int64, count=len(edges))
    cols = np.fromiter((b for _, b in edges), dtype=np.int64, count=len(edges))
    graph = coo_matrix((np.ones(len(edges)), (rows, cols)), shape=(count, count))
    return connected_components(graph, directed=False)[1]


def group_instances(points, types, semantics, *, proximity=DEFAULT_PROXIMITY,
                    connectivity=DEFAULT_CONNECTIVITY):
    """Return a list of (semantic_id, boolean primitive mask) instance candidates.

    `semantics` may come from ground truth (to measure grouping alone) or from a model
    (to measure an end-to-end pipeline).
    """
    points = np.asarray(points, dtype=np.float64)
    types = np.asarray(types)
    semantics = np.asarray(semantics)
    count = len(semantics)
    instances = []
    left, right = endpoint_pairs(points, types)
    for semantic in sorted(set(semantics.tolist())):
        if semantic == ARCHCAD_IGNORE_LABEL:
            continue
        members = np.flatnonzero(semantics == semantic)
        if not len(members):
            continue
        mask = np.zeros(count, dtype=bool)
        mask[members] = True
        if semantic in ARCHCAD_STUFF_SEMANTIC_IDS:
            instances.append((int(semantic), mask))
            continue
        local_points = points[members]
        centres = local_points.mean(axis=1)
        distance = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
        edges = {(int(i), int(j)) for i in range(len(members))
                 for j in np.flatnonzero(distance[i] < proximity) if j > i}
        if connectivity > 0:
            local_left, local_right = left[members], right[members]
            usable = np.isfinite(local_left).all(axis=1)
            for i in range(len(members)):
                if not usable[i]:
                    continue
                for j in range(i + 1, len(members)):
                    if not usable[j]:
                        continue
                    if (np.linalg.norm(local_left[i] - local_left[j]) < connectivity
                            or np.linalg.norm(local_left[i] - local_right[j]) < connectivity
                            or np.linalg.norm(local_right[i] - local_left[j]) < connectivity
                            or np.linalg.norm(local_right[i] - local_right[j]) < connectivity):
                        edges.add((i, j))
        labels = _components(len(members), sorted(edges))
        for group in sorted(set(labels.tolist())):
            group_mask = np.zeros(count, dtype=bool)
            group_mask[members[labels == group]] = True
            instances.append((int(semantic), group_mask))
    return instances


def pairwise_f1(points, types, semantics, instance_ids, *, proximity=DEFAULT_PROXIMITY,
                connectivity=DEFAULT_CONNECTIVITY):
    """Pairwise agreement between grouped components and reference instance ids."""
    instances = group_instances(points, types, semantics, proximity=proximity,
                                connectivity=connectivity)
    predicted = np.full(len(semantics), -1, dtype=np.int64)
    for index, (semantic, mask) in enumerate(instances):
        predicted[mask] = index
    reference = np.asarray(instance_ids)
    keep = (reference >= 0) & (predicted >= 0)
    kept = int(keep.sum())
    if kept < 2:
        return 0.0, 0.0, 0.0
    predicted, reference = predicted[keep], reference[keep]
    upper = np.triu_indices(kept, k=1)
    same_predicted = (predicted[:, None] == predicted[None, :])[upper]
    same_reference = (reference[:, None] == reference[None, :])[upper]
    tp = int((same_predicted & same_reference).sum())
    fp = int((same_predicted & ~same_reference).sum())
    fn = int((~same_predicted & same_reference).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def fit_thresholds(items, *, proximities=(0.02, 0.05, 0.1, 0.2),
                   connectivities=(0.001, 0.005, 0.01, 0.02)):
    """Choose the grouping constants on the given items, by mean pairwise F1.

    Call this on the **training** split. Reporting a ceiling measured with constants
    fitted on the evaluation split would not be an honest reference.
    """
    best, scores = None, {}
    for proximity in proximities:
        for connectivity in connectivities:
            f1s = []
            for item in items:
                labels = item['targets']['semantic_labels'].numpy()
                instance_ids = item['targets']['instance_ids'].numpy()
                if len(set(instance_ids[instance_ids >= 0].tolist())) < 2:
                    continue
                points = item['features'].numpy()[:, :8].reshape(-1, 4, 2)
                f1s.append(pairwise_f1(points, item['primitive_types'].numpy(), labels,
                                       instance_ids, proximity=proximity,
                                       connectivity=connectivity)[2])
            scores[(proximity, connectivity)] = float(np.mean(f1s)) if f1s else 0.0
            if best is None or scores[(proximity, connectivity)] > scores[best]:
                best = (proximity, connectivity)
    return best, scores


def predictions_from_semantics(items, semantics_per_item, *, proximity=DEFAULT_PROXIMITY,
                               connectivity=DEFAULT_CONNECTIVITY):
    """Frozen-evaluator predictions using given per-primitive semantics and this grouper."""
    predictions = []
    for item, semantics in zip(items, semantics_per_item):
        points = item['features'].numpy()[:, :8].reshape(-1, 4, 2)
        instances = group_instances(points, item['primitive_types'].numpy(), semantics,
                                    proximity=proximity, connectivity=connectivity)
        predictions.append({'image_id': item['image_id'],
                            'semantic_labels': np.asarray(semantics, dtype=np.int64),
                            'instances': [{'semantic_id': semantic, 'score': 1.0, 'mask': mask}
                                          for semantic, mask in instances]})
    return predictions
