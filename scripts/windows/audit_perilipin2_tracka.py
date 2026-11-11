"""Explain an existing Track A comparison without changing measurements or gates.

Reads a verified reference bundle and completed run; rechecks existing tables
and TIFF arrays, but never reruns detection/segmentation. Writes ALL differing
T02 cells, exact decimal deltas, float64 ULP distances, per-column maxima and
recorded morphology-threshold comparisons into a separate new audit folder.
No tolerance, threshold, scientific file or previous receipt is changed.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import struct
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CATALOG = 'T02_brown_object_catalog.tsv'
KEYS = ('image_uid', 'rule', 'object_id')
MORPHOLOGY = {'axis_ratio': 'compact_max_axis_ratio',
              'major_axis_px': 'compact_max_major_axis_px'}
SCHEMA = 'radedge-tracka-mismatch-audit-v1'
DIFF_COLUMNS = (*KEYS, 'column', 'reference_value', 'actual_value', 'kind',
                'absolute_delta_decimal', 'relative_delta_symmetric_decimal',
                'float64_ulp_distance', 'threshold', 'reference_exceeds_threshold',
                'actual_exceeds_threshold', 'threshold_decision_changed')


def adapter():
    path = REPO / 'src/radiation_edge_ai/perilipin2_tracka.py'
    spec = importlib.util.spec_from_file_location('_tracka_audit_core', path)
    if spec is None or spec.loader is None:
        raise ValueError('Track A adapter unavailable in this worktree')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ulp_distance(left, right):
    """Diagnostic binary64 spacing only; never an acceptance criterion."""
    a, b = float(left), float(right)
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    def ordered(value):
        if value == 0:
            value = 0.0  # Treat signed zeros as the same numerical value.
        bits = struct.unpack('>Q', struct.pack('>d', value))[0]
        sign = 1 << 63
        return sign - (bits & (sign - 1)) if bits & sign else sign + bits
    return abs(ordered(a) - ordered(b))


def cell_difference(core, column, left, right, configuration):
    a, b = core.normalize_cell(left), core.normalize_cell(right)
    if a == b:
        return None
    record = {name: '' for name in DIFF_COLUMNS if name not in KEYS}
    record.update(column=column, reference_value=left, actual_value=right,
                  kind='non_numeric_or_missing_difference')
    if a[0] == b[0] == 'number':
        x, y = a[1], b[1]
        # Preserve input decimal strings; subtraction and division use enough
        # precision for the observed decimal exponents plus diagnostic guard digits.
        low = min(x.as_tuple().exponent, y.as_tuple().exponent)
        high = max(x.adjusted(), y.adjusted())
        with localcontext() as ctx:
            ctx.prec = max(80, high - low + 20)
            delta = abs(x - y)
            scale = max(abs(x), abs(y))
            record.update(kind='finite_numeric_difference',
                          absolute_delta_decimal=str(delta),
                          relative_delta_symmetric_decimal=str(delta / scale if scale else Decimal(0)),
                          float64_ulp_distance=ulp_distance(left, right))
        if column in MORPHOLOGY:
            limit = Decimal(str(configuration[MORPHOLOGY[column]]))
            if not limit.is_finite():
                raise ValueError('Nonfinite recorded morphology threshold')
            # The frozen M2 implementation uses strict >, not >=, for these gates.
            record.update(threshold=str(limit), reference_exceeds_threshold=x > limit,
                          actual_exceeds_threshold=y > limit,
                          threshold_decision_changed=(x > limit) != (y > limit))
    return record


def catalog_audit(core, reference, actual, configuration):
    columns_a, left = core.read_table(reference, KEYS)
    columns_b, right = core.read_table(actual, KEYS)
    if set(columns_a) != set(columns_b) or set(left) != set(right):
        raise ValueError('Object catalog columns/row keys differ; numerical-only audit is not applicable')
    if not {'status', 'reason'}.issubset(columns_a):
        raise ValueError('Object catalog lacks status/reason columns')
    differences, changed_rows = [], set()
    for key, old in left.items():
        for column in columns_a:
            row = cell_difference(core, column, old[column], right[key][column], configuration)
            if row is not None:
                row.update({name: old[name] for name in KEYS})
                differences.append(row)
                changed_rows.add(key)
    counts = Counter(row['column'] for row in differences)
    by_column = {}
    for name in sorted(counts):
        selected = [d for d in differences if d['column'] == name]
        numeric = [d for d in selected if d['kind'] == 'finite_numeric_difference']
        ulps = [d['float64_ulp_distance'] for d in numeric if d['float64_ulp_distance'] is not None]
        by_column[name] = {
            'differing_cells': len(selected), 'finite_numeric_cells': len(numeric),
            'max_absolute_delta_decimal': str(max(Decimal(d['absolute_delta_decimal']) for d in numeric)) if numeric else None,
            'max_relative_delta_symmetric_decimal': str(max(Decimal(d['relative_delta_symmetric_decimal']) for d in numeric)) if numeric else None,
            'max_float64_ulp_distance': max(ulps) if ulps else None,
        }
    numeric_morph = [d for d in differences if d['column'] in MORPHOLOGY
                    and d['kind'] == 'finite_numeric_difference']
    other = [d for d in differences if d['column'] not in MORPHOLOGY
             or d['kind'] != 'finite_numeric_difference']
    decision_changes = [d for d in numeric_morph if d['threshold_decision_changed']]
    summary = {
        'reference_rows': len(left), 'actual_rows': len(right),
        'columns_and_row_keys_exact': True, 'differing_cells': len(differences),
        'affected_rows': len(changed_rows), 'by_column': by_column,
        'status_exact': counts['status'] == 0, 'reason_exact': counts['reason'] == 0,
        'other_than_finite_morphology_difference_count': len(other),
        'only_finite_morphology_values_differ': bool(differences) and not other,
        'recorded_thresholds': {k: configuration[v] for k, v in MORPHOLOGY.items()},
        'threshold_operator': 'strict >; evaluated on stored table values',
        'morphology_threshold_decision_changes': len(decision_changes),
        'threshold_decision_examples': decision_changes[:10],
        'value_examples': sorted(differences, key=lambda r: (
            r['column'], r['image_uid'], r['rule'], r['object_id']))[:10],
    }
    return summary, differences


def run(args):
    core = adapter()
    bundle, manifest = core.verify_bundle(args.bundle)
    source = Path(args.run).expanduser().resolve(strict=True)
    receipt = core.verify_result(source)
    if receipt.get('bundle_sha256') != manifest['fingerprint_sha256']:
        raise ValueError('Run belongs to a different reference bundle')
    if receipt.get('configuration') != manifest['configuration']:
        raise ValueError('Run and reference configuration differ; preserve for investigation')
    if receipt.get('source_sha256') != core.SOURCE_SHA256:
        raise ValueError('Run used different scientific source')
    if receipt.get('kl720_inference') is not False or receipt.get('new_nuclear_segmentation') is not False:
        raise ValueError('This audit is for the saved-mask CPU reference workflow')
    destination = args.out or source.with_name(source.name + '_audit_' + uuid.uuid4().hex[:8])
    output, stage = core.new_output(destination, (bundle, source))
    watched = [core.bundle_manifest(args.bundle), source / 'FILE_MANIFEST.tsv',
               source / 'TRACKA_EXECUTION.json', source / 'REFERENCE_COMPARISON.json',
               bundle / 'expected/tables' / CATALOG, source / 'tables' / CATALOG]
    before = {str(p): core.digest(p) for p in watched}
    print('TRACKA_AUDIT_INPUT_INTEGRITY_VERIFIED', flush=True)
    print('Rechecking saved tables/arrays only; no analysis, segmentation or hardware run.', flush=True)
    exact = core.compare(bundle, source, manifest)
    old_comparison = core.load_json(source / 'REFERENCE_COMPARISON.json')
    if exact != old_comparison:
        raise ValueError('Exact comparison no longer reproduces the saved report; no audit result written')
    summary, rows = catalog_audit(core, bundle / 'expected/tables' / CATALOG,
                                  source / 'tables' / CATALOG, manifest['configuration'])
    # Integrity is checked again after reading; the old receipts never change.
    core.verify_bundle(bundle)
    core.verify_result(source)
    if before != {str(p): core.digest(p) for p in watched}:
        raise ValueError('Source files changed during audit')
    array_ok = len(exact['arrays']) == 14 and all(r.get('passed') is True for r in exact['arrays'].values())
    other_tables_ok = all(exact['tables'].get(n, {}).get('passed') is True
                          for n in core.TABLE_KEYS if n != CATALOG)
    portability_invariants_pass = bool(
        array_ok and other_tables_ok
        and summary['columns_and_row_keys_exact']
        and summary['status_exact'] and summary['reason_exact']
        and summary['only_finite_morphology_values_differ']
        and summary['morphology_threshold_decision_changes'] == 0
    )
    portability_status = (
        'PORTABILITY_INVARIANTS_PASS_EXACT_MISMATCH_RETAINED'
        if portability_invariants_pass else
        'PORTABILITY_INVARIANTS_FAIL_EXACT_MISMATCH_RETAINED'
    )
    identity = {
        'schema': SCHEMA, 'status': 'AUDIT_COMPLETE_EXACT_MATCH' if exact['passed'] else 'AUDIT_COMPLETE_EXACT_MISMATCH_REMAINS',
        'diagnostic_only': True, 'original_acceptance_status': receipt['status'],
        'original_receipts_modified': False, 'acceptance_policy_changed': False,
        'portability_status': portability_status,
        'portability_invariants_pass': portability_invariants_pass,
        'portability_invariants': {
            'all_14_arrays_exact': array_ok,
            'other_four_tables_exact': other_tables_ok,
            'object_catalog_columns_and_row_keys_exact': summary['columns_and_row_keys_exact'],
            'object_status_exact': summary['status_exact'],
            'object_reason_exact': summary['reason_exact'],
            'only_finite_morphology_values_differ': summary['only_finite_morphology_values_differ'],
            'morphology_threshold_decision_changes': summary['morphology_threshold_decision_changes'],
        },
        'measurement_rerun': False, 'kl720_inference': False, 'biological_validation': False,
        'reference_bundle_fingerprint': manifest['fingerprint_sha256'],
        'source_dataset_id': receipt['dataset_id'], 'audit_script_sha256': core.digest(__file__),
        'source_file_sha256': before, 'exact_difference_count': exact['difference_count'],
        'all_14_arrays_exact': array_ok, 'other_four_tables_exact': other_tables_ok,
        'catalog': summary, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'interpretation_note': 'Delta and ULP metrics describe differences; no value is accepted by tolerance. '
                               'The portability classification is decision-invariant, not magnitude-tolerant: it requires exact arrays, exact non-catalog tables, exact catalog identities/status/reason, only finite morphology-value differences, and zero recorded morphology-threshold decision changes. The original strict exact-match status is retained. No causal claim about platform math is made.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(exist_ok=False)
    with (stage / 'ALL_OBJECT_CELL_DIFFERENCES.tsv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=DIFF_COLUMNS, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    core.write_json(stage / 'EXACT_COMPARISON_RECHECK.json', exact)
    core.write_json(stage / 'TRACKA_MISMATCH_AUDIT.json', identity)
    records = {p.name: core.file_record(p) for p in stage.iterdir() if p.is_file()}
    core.write_json(stage / 'AUDIT_FILES.json', records)
    stage.rename(output)
    print(identity['status'], flush=True)
    print('All 14 arrays exact:', array_ok)
    print('Other four tables exact:', other_tables_ok)
    print('Object status/reason exact:', summary['status_exact'], summary['reason_exact'])
    print('Differing cells / affected rows:', summary['differing_cells'], summary['affected_rows'])
    print('Per-column deltas:', json.dumps(summary['by_column'], sort_keys=True))
    print('Recorded morphology-threshold crossings:', summary['morphology_threshold_decision_changes'])
    print('Portability classification:', portability_status)
    print('Audit report:', output / 'TRACKA_MISMATCH_AUDIT.json')
    print('All differences:', output / 'ALL_OBJECT_CELL_DIFFERENCES.tsv')
    print('Original acceptance status remains:', receipt['status'])
    return 0  # Audit completed, not a reference-acceptance pass.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', type=Path, help='new audit folder; defaults to a unique sibling of --run')
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print('TRACKA_AUDIT_STOP: {}: {}'.format(type(exc).__name__, exc), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
