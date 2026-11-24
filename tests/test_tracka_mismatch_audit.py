"""Read-only mismatch diagnosis tests; no real images, GPU, Docker or package install.

Numerical table tests use the existing exact comparison helpers. Orchestration
fixtures mock bundle/result integrity and array recheck; they do not establish
cross-platform scientific equivalence or biological accuracy.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/windows/audit_perilipin2_tracka.py'
SPEC = importlib.util.spec_from_file_location('_tracka_mismatch_audit_test', SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
core = audit.adapter()
CONFIG = {'compact_max_axis_ratio': 3.0, 'compact_max_major_axis_px': 24.0}
COLUMNS = ('image_uid', 'rule', 'object_id', 'status', 'reason', 'axis_ratio', 'major_axis_px')


def write_table(path, rows, columns=COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def row(index=1, **changes):
    data = dict(zip(COLUMNS, ('synthetic_only', 'working', str(index), 'compact_candidate',
                             'passes_provisional_compact_rule', '2.0', '10.0')))
    data.update(changes)
    return data


class CellTests(unittest.TestCase):
    def test_equal_decimal_spellings_are_not_mismatches(self):
        self.assertIsNone(audit.cell_difference(core, 'axis_ratio', '1', '1.0', CONFIG))
        self.assertIsNone(audit.cell_difference(core, 'axis_ratio', '', 'nan', CONFIG))

    def test_decimal_delta_not_lost_even_when_binary64_is_equal(self):
        result = audit.cell_difference(core, 'axis_ratio', '1', '1.00000000000000001', CONFIG)
        self.assertEqual(result['absolute_delta_decimal'], '1E-17')
        self.assertEqual(result['float64_ulp_distance'], 0)
        self.assertFalse(result['threshold_decision_changed'])
        self.assertEqual(result['actual_value'], '1.00000000000000001')

    def test_one_ulp_and_negative_ordering(self):
        self.assertEqual(audit.ulp_distance('1', repr(math.nextafter(1.0, math.inf))), 1)
        self.assertEqual(audit.ulp_distance('-1', repr(math.nextafter(-1.0, -math.inf))), 1)
        self.assertEqual(audit.ulp_distance('-0', '0'), 0)
        self.assertIsNone(audit.ulp_distance('1e500', '2e500'))

    def test_gate_uses_strict_greater_than_and_reports_crossing(self):
        result = audit.cell_difference(core, 'axis_ratio', '3.0', '3.0000000000000004', CONFIG)
        self.assertFalse(result['reference_exceeds_threshold'])
        self.assertTrue(result['actual_exceeds_threshold'])
        self.assertTrue(result['threshold_decision_changed'])
        major = audit.cell_difference(core, 'major_axis_px', '24.000000000000004', '24', CONFIG)
        self.assertTrue(major['reference_exceeds_threshold'])
        self.assertFalse(major['actual_exceeds_threshold'])

    def test_nonnumeric_and_missing_values_never_become_small_deltas(self):
        for left, right in (('nan', '0'), ('True', '1'), ('inf', '1'), ('a', 'b')):
            with self.subTest(pair=(left, right)):
                result = audit.cell_difference(core, 'axis_ratio', left, right, CONFIG)
                self.assertEqual(result['kind'], 'non_numeric_or_missing_difference')
                self.assertEqual(result['absolute_delta_decimal'], '')
                self.assertEqual(result['threshold_decision_changed'], '')


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.left, self.right = self.root / 'reference.tsv', self.root / 'actual.tsv'

    def measure(self, left, right):
        write_table(self.left, left)
        write_table(self.right, right)
        return audit.catalog_audit(core, self.left, self.right, CONFIG)

    def test_all_73_differences_retained_not_first40(self):
        left = [row(i) for i in range(73)]
        right = [row(i, axis_ratio='2.0000000000000004') for i in range(73)]
        summary, details = self.measure(left, list(reversed(right)))
        self.assertEqual(len(details), 73)
        self.assertEqual(summary['differing_cells'], 73)
        self.assertEqual(summary['affected_rows'], 73)
        self.assertEqual(summary['by_column']['axis_ratio']['max_absolute_delta_decimal'], '4E-16')
        self.assertEqual(summary['by_column']['axis_ratio']['max_float64_ulp_distance'], 1)
        self.assertTrue(summary['only_finite_morphology_values_differ'])
        self.assertTrue(summary['status_exact'])
        self.assertTrue(summary['reason_exact'])
        self.assertEqual(summary['morphology_threshold_decision_changes'], 0)
        self.assertEqual(len(summary['value_examples']), 10)

    def test_status_change_cannot_be_masked_by_matching_counts(self):
        summary, _ = self.measure([row(1), row(2, status='complex_review')],
                                 [row(1, status='complex_review'), row(2)])
        self.assertFalse(summary['status_exact'])
        self.assertEqual(summary['other_than_finite_morphology_difference_count'], 2)
        self.assertFalse(summary['only_finite_morphology_values_differ'])

    def test_multiple_cells_counted_separately_from_objects(self):
        summary, details = self.measure([row()], [row(axis_ratio='2.1', major_axis_px='11')])
        self.assertEqual(len(details), 2)
        self.assertEqual(summary['affected_rows'], 1)
        self.assertEqual(set(summary['by_column']), set(audit.MORPHOLOGY))

    def test_missing_and_nonfinite_morphology_not_accepted(self):
        summary, _ = self.measure([row(axis_ratio='NaN')], [row(axis_ratio='0')])
        self.assertFalse(summary['only_finite_morphology_values_differ'])
        self.assertEqual(summary['other_than_finite_morphology_difference_count'], 1)

    def test_changed_row_keys_are_not_matched_by_order(self):
        with self.assertRaisesRegex(ValueError, 'columns/row keys'):
            self.measure([row(1)], [row(2)])

    def test_duplicate_rows_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.measure([row(1)], [row(1), row(1)])

    def test_identical_tables_remain_identical(self):
        summary, details = self.measure([row()], [row()])
        self.assertEqual(details, [])
        self.assertEqual(summary['by_column'], {})
        self.assertFalse(summary['only_finite_morphology_values_differ'])


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.bundle, self.run = self.root / 'bundle', self.root / 'completed'
        self.bundle.mkdir()
        self.run.mkdir()
        self.output = self.root / 'audit'
        self.manifest = {'fingerprint_sha256': 'verified_fixture', 'configuration': CONFIG}
        self.receipt = {'bundle_sha256': 'verified_fixture', 'configuration': CONFIG,
                        'source_sha256': core.SOURCE_SHA256, 'dataset_id': 'synthetic_fixture',
                        'status': 'TRACKA_REFERENCE_MISMATCH', 'kl720_inference': False,
                        'new_nuclear_segmentation': False}
        self.comparison = {'passed': False, 'difference_count': 1,
                           'tables': {n: {'passed': n != audit.CATALOG} for n in core.TABLE_KEYS},
                           'arrays': {str(i): {'passed': True} for i in range(14)}}
        core.write_json(self.bundle / 'TRACKA_INPUT_BUNDLE.json', self.manifest)
        core.write_json(self.run / 'TRACKA_EXECUTION.json', self.receipt)
        core.write_json(self.run / 'REFERENCE_COMPARISON.json', self.comparison)
        (self.run / 'FILE_MANIFEST.tsv').write_text('fixture only\n')
        write_table(self.bundle / 'expected/tables' / audit.CATALOG, [row()])
        write_table(self.run / 'tables' / audit.CATALOG, [row(axis_ratio='2.0000000000000004')])
        self.before = {str(p): core.digest(p) for d in (self.bundle, self.run)
                       for p in d.rglob('*') if p.is_file()}
        fake = SimpleNamespace(**{name: getattr(core, name) for name in (
            'digest', 'bundle_manifest', 'new_output', 'compare', 'load_json',
            'SOURCE_SHA256', 'normalize_cell', 'read_table', 'TABLE_KEYS', 'write_json', 'file_record')})
        fake.verify_bundle = mock.Mock(return_value=(self.bundle, self.manifest))
        fake.verify_result = mock.Mock(return_value=self.receipt)
        fake.compare = mock.Mock(return_value=copy.deepcopy(self.comparison))
        self.fake = fake
        self.patch = mock.patch.object(audit, 'adapter', return_value=fake)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.args = SimpleNamespace(bundle=self.bundle, run=self.run, out=self.output)

    def test_writes_diagnostics_only_preserves_failed_gate(self):
        self.assertEqual(audit.run(self.args), 0)
        report = core.load_json(self.output / 'TRACKA_MISMATCH_AUDIT.json')
        self.assertEqual(report['status'], 'AUDIT_COMPLETE_EXACT_MISMATCH_REMAINS')
        self.assertEqual(report['original_acceptance_status'], 'TRACKA_REFERENCE_MISMATCH')
        self.assertFalse(report['acceptance_policy_changed'])
        self.assertFalse(report['measurement_rerun'])
        self.assertTrue(report['all_14_arrays_exact'])
        self.assertTrue(report['portability_invariants_pass'])
        self.assertEqual(report['portability_status'],
                         'PORTABILITY_INVARIANTS_PASS_EXACT_MISMATCH_RETAINED')
        self.assertEqual(report['catalog']['differing_cells'], 1)
        self.assertEqual(self.before, {p: core.digest(p) for p in self.before})
        self.assertEqual(self.fake.verify_bundle.call_count, 2)
        self.assertEqual(self.fake.verify_result.call_count, 2)
        with (self.output / 'ALL_OBJECT_CELL_DIFFERENCES.tsv').open(encoding='utf-8') as f:
            values = list(csv.DictReader(f, delimiter='\t'))
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0]['reference_value'], '2.0')
        self.assertEqual(values[0]['actual_value'], '2.0000000000000004')

    def test_portability_invariants_fail_if_status_changes(self):
        write_table(self.run / 'tables' / audit.CATALOG,
                    [row(axis_ratio='2.0000000000000004', status='complex_review')])
        self.fake.compare.return_value = copy.deepcopy(self.comparison)
        self.assertEqual(audit.run(self.args), 0)
        report = core.load_json(self.output / 'TRACKA_MISMATCH_AUDIT.json')
        self.assertFalse(report['portability_invariants_pass'])
        self.assertEqual(report['portability_status'],
                         'PORTABILITY_INVARIANTS_FAIL_EXACT_MISMATCH_RETAINED')
        self.assertFalse(report['catalog']['status_exact'])

    def test_missing_different_bundle_or_config_rejects(self):
        for key, value in (('bundle_sha256', 'other'), ('configuration', {}),
                           ('source_sha256', 'other'), ('new_nuclear_segmentation', True)):
            old = self.receipt[key]
            self.receipt[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.run(self.args)
            self.receipt[key] = old
        self.assertFalse(self.output.exists())
        self.fake.compare.assert_not_called()

    def test_existing_output_never_overwritten(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            audit.run(self.args)
        self.fake.compare.assert_not_called()

    def test_changed_integrity_stops_before_comparison(self):
        self.fake.verify_result.side_effect = ValueError('Modified result artifact')
        with self.assertRaisesRegex(ValueError, 'Modified'):
            audit.run(self.args)
        self.fake.compare.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_changed_saved_comparison_cannot_be_promoted(self):
        self.fake.compare.return_value['difference_count'] = 2
        with self.assertRaisesRegex(ValueError, 'no longer reproduces'):
            audit.run(self.args)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
