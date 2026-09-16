"""Check JSON/SVG primitive alignment and distinguish auxiliary point records.

This audits the release; it does not claim to reconstruct a missing upstream
loader. It reads the frozen 100-chunk oracle sample unless a limit is supplied.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.spatial import cKDTree
from svgpathtools import parse_path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.archcad_panoptic import primitive_length
from src.data.archcad_panoptic_dataset import primitive_points


def audit_pair(annotation, svg_path):
    entities = annotation['entities']
    nodes = [e for e in ET.parse(svg_path).getroot().iter()
             if e.tag.split('}')[-1] in {'path', 'circle', 'ellipse'}]
    counts = Counter(entities=len(entities), svg_elements=len(nodes))
    examples = []
    if len(entities) != len(nodes):
        return dict(counts), [{'error': 'element_count_mismatch'}]
    for i, (entity, node) in enumerate(zip(entities, nodes)):
        counts['semantic_matches'] += str(entity['semantic']) == node.get('semantic')
        counts['instance_matches'] += str(entity.get('instance', '')) == node.get('instance', '')
        kind = entity['type']
        counts[kind] += 1
        if kind not in {'LINE', 'ARC', 'CIRCLE'}:
            continue
        try:
            if node.tag.split('}')[-1] == 'path':
                svg_path = parse_path(node.attrib['d'])
                svg_length = svg_path.length()
                if kind in {'LINE', 'ARC'} and len(svg_path) == 1:
                    expected = np.array([[svg_path[0].point(t).real, svg_path[0].point(t).imag]
                                         for t in np.linspace(0, 1, 4)])
                    observed = primitive_points(entity)
                    counts['geometry_checked'] += 1
                    agrees = np.allclose(observed, expected, rtol=1e-6, atol=1e-3)
                    counts['geometry_matches'] += int(agrees)
                    if not agrees and len(examples) < 5:
                        examples.append({'index': i, 'geometry_max_error': float(np.abs(observed-expected).max())})
            elif kind == 'CIRCLE':
                svg_length = 2 * np.pi * float(node.get('r'))
            else:
                continue
            raw_length = primitive_length(entity)
            # SVG endpoint reconstruction can lose precision for nearly flat arcs.
            matched = np.isclose(svg_length, raw_length, rtol=1e-5, atol=1e-5)
            counts['length_checked'] += 1
            counts['length_matches'] += int(matched)
            if not matched and len(examples) < 5:
                examples.append({'index': i, 'type': kind, 'json_length': raw_length, 'svg_length': svg_length})
        except (ValueError, AssertionError, ZeroDivisionError) as exc:
            counts['svg_length_errors'] += 1
            if len(examples) < 5:
                examples.append({'index': i, 'error': str(exc)})
    return dict(counts), examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/ArchCAD'))
    parser.add_argument('--point-zip', type=Path, default=Path('data/ArchCAD_raw/data/point.zip'))
    parser.add_argument('--sample-count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--output', type=Path, default=Path('outputs/panoptic/representation_audit_20260916.json'))
    args = parser.parse_args()
    files = sorted((args.data_root / 'json').glob('*.json'))
    if args.sample_count <= 0 or not files:
        raise ValueError('positive sample count and source annotations required')
    selected = sorted(random.Random(args.seed).sample(files, min(len(files), args.sample_count)))
    totals, examples = Counter(), []
    source_hash = hashlib.sha256()
    with zipfile.ZipFile(args.point_zip) as archive:
        for path in selected:
            payload = path.read_bytes()
            source_hash.update(path.stem.encode() + payload)
            annotation = json.loads(payload)
            counts, errors = audit_pair(annotation, args.data_root / 'svg' / (path.stem + '.svg'))
            totals.update(counts)
            totals['samples'] += 1
            totals['equal_element_counts'] += counts['entities'] == counts['svg_elements']
            points = np.load(io.BytesIO(archive.read(f'point/{path.stem}.npy')), allow_pickle=False)
            totals['point_records'] += len(points)
            totals['point_count_differs'] += len(points) != counts['entities']
            ends = np.asarray([e[k] for e in annotation['entities'] if e['type'] == 'LINE' for k in ('start', 'end')])
            p1 = points[points[:, 0] == 1, 1:]
            if len(ends) and len(p1):
                # Diagnostic only: compare both coordinate conventions.
                tree = cKDTree(ends)
                totals['type1_points'] += len(p1)
                totals['xy_near_line_endpoint'] += int((tree.query(p1)[0] <= 1.5).sum())
                totals['rowcol_near_line_endpoint'] += int((tree.query(p1[:, ::-1])[0] <= 1.5).sum())
            if errors and len(examples) < 10:
                examples.append({'image_id': path.stem, 'differences': errors})
    report = {'schema_version': 'archcad-representation-audit-v1', 'seed': args.seed,
              'counts': dict(totals), 'sample_source_sha256': source_hash.hexdigest(),
              'sample_ids': [p.stem for p in selected], 'differences': examples,
              'interpretation': {
                  'supervision_unit': 'JSON entity; verify corresponding SVG by ordinal',
                  'point_zip_role': 'auxiliary quantized points; no verified one-to-one entity index',
                  'point_zip_required_by_released_model': 'not established; upstream loader is absent',
                  'exact_dpss_input_parity': False}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'counts': report['counts'], 'differences': examples, 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
