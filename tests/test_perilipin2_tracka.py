"""Portable M2 adapter tests. Real frozen-code tests need RADEDGE_TRACKA_HANDOFF.

No real microscopy, learned model, GPU, Docker or KL720 used by these fixtures.
"""
from __future__ import annotations
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from radiation_edge_ai import perilipin2_tracka as p

HANDOFF = os.environ.get('RADEDGE_TRACKA_HANDOFF', '')


class Guards(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name).resolve()

    def test_path_escape_rejected(self):
        for path in ('../outside', '/tmp/x', r'C:\outside', 'a/../../x', r'a\b'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                p.contained(self.root, path)

    def test_new_output_does_not_overwrite(self):
        with self.assertRaises(FileExistsError):
            p.new_output(self.root)
        with self.assertRaises(ValueError):
            p.new_output(self.root / 'child', (self.root,))
        with self.assertRaises(ValueError):
            p.new_output(self.root.parent, (self.root,))

    def test_wrong_source_archive_rejected_before_import(self):
        with self.assertRaisesRegex(ValueError, 'archive hash'):
            p.unpack_reference(b'not source', self.root / 'source')
        self.assertFalse((self.root / 'source').exists())

    def test_canonical_identity_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            p.fingerprint({'a': float('nan')})
        file = self.root / 'input.json'
        for text in ('[]', '{"x":NaN}', '{"x":Infinity}'):
            file.write_text(text)
            with self.assertRaises(ValueError):
                p.load_json(file)

    def test_exact_numeric_comparison_no_tolerance(self):
        self.assertEqual(p.normalize_cell('1'), p.normalize_cell('1.0'))
        self.assertNotEqual(p.normalize_cell('1.00000000000000001'), p.normalize_cell('1'))
        self.assertNotEqual(p.normalize_cell('True'), p.normalize_cell('1'))
        self.assertEqual(p.normalize_cell(''), p.normalize_cell('nan'))

    def test_duplicate_table_keys_rejected(self):
        tsv = self.root / 'table.tsv'
        tsv.write_text('image_uid\tvalue\nx\t1\nx\t2\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            p.read_table(tsv, ('image_uid',))

    def test_wholly_empty_legacy_table_is_supported(self):
        tsv = self.root / 'empty.tsv'
        tsv.write_text('status\n')
        columns, rows = p.read_table(tsv, ('image_uid', 'nucleus_id'))
        self.assertEqual(columns, ['status'])
        self.assertEqual(rows, {})

    def test_invalid_manifest_identity_rejected(self):
        path = self.root / 'TRACKA_INPUT_BUNDLE.json'
        p.write_json(path, {'schema': p.BUNDLE_SCHEMA, 'fingerprint_sha256': '0' * 64})
        with self.assertRaisesRegex(ValueError, 'identity'):
            p.verify_bundle(self.root)


@unittest.skipUnless(HANDOFF and Path(HANDOFF).is_file(), 'Frozen source ZIP external to Git; set RADEDGE_TRACKA_HANDOFF')
class FrozenM2Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import tifffile
        from PIL import Image
        cls.np, cls.tifffile, cls.Image = np, tifffile, Image
        cls.t = tempfile.TemporaryDirectory()
        cls.root = Path(cls.t.name).resolve()
        source, _ = p.handoff_members(Path(HANDOFF))
        cls.source = p.unpack_reference(source, cls.root / 'source')
        cls.core = p.import_file(cls.source / 'run_tracka.py', '_legacy_test_tracka')
        cls.cfg = p.load_json(cls.source / 'config.json')
        # Synthetic reference settings are recorded and replayed unchanged.
        # The adapter never modifies the user's recorded configuration.
        cls.cfg['tissue_od_sum_min'] = 0.02
        cls.cfg['patch_size_px'] = 80
        cls.raw = cls.root / 'images'
        cls.m1 = cls.root / 'm1'
        cls.m2 = cls.root / 'm2'
        rows = []
        for i, (dose, sample) in enumerate([('0', '13'), ('3.8', '27')]):
            rgb, nuclei = cls.synthetic()
            if i:
                rgb[:] = 255
                rgb[0, 0] = (254, 255, 255)
            uid = 'SYNTHETIC_ONLY_' + str(i)
            rel = 'DAB/Perilipin2-X500-' + dose + 'Gy-' + sample + ' X40.png'
            image = cls.raw / rel
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(image)
            folder = cls.m1 / '01_pilot' / 'per_image' / uid
            folder.mkdir(parents=True)
            tifffile.imwrite(folder / 'nuclei.tif', nuclei)
            valid = np.ones(nuclei.shape, dtype=np.uint8)
            valid[-12:, -12:] = 0
            tifffile.imwrite(folder / 'analysis_valid_mask.tif', valid)
            p.write_json(folder / 'summary.json', {'pixel_size_um': None, 'calibration_source': ''})
            p.write_json(folder / 'complete.json', {'output_hashes': {
                n: p.digest(folder / n) for n in ('nuclei.tif', 'analysis_valid_mask.tif', 'summary.json')}})
            rows.append({'image_uid': uid, 'relative_path': rel, 'dose_Gy': dose,
                         'sample_token': sample, 'source_sha256': p.digest(image)})
        cls.core.save_table(cls.m1 / '00_inventory/T02_selected_pilot.tsv', rows)
        args = SimpleNamespace(m1=cls.m1, root=cls.raw, out=cls.m2)
        cls.core.build(args, cls.cfg)
        cls.bundle = cls.root / 'bundle'
        p.prepare(SimpleNamespace(m1=cls.m1, m2=cls.m2, images=cls.raw,
                                  handoff=Path(HANDOFF), out=cls.bundle))
        cls.moved = cls.root / 'relocated_bundle'
        shutil.copytree(cls.bundle, cls.moved)
        cls.out = cls.root / 'rerun'
        cls.before = {str(x): p.digest(x) for r in (cls.m1, cls.m2, cls.raw, cls.moved)
                      for x in r.rglob('*') if x.is_file()}
        cls.rc = p.run(SimpleNamespace(bundle=cls.moved, out=cls.out))

    @classmethod
    def tearDownClass(cls):
        cls.t.cleanup()

    @classmethod
    def synthetic(cls):
        np = cls.np
        y, x = np.mgrid[:256, :256]
        h = np.zeros((256, 256)); d = np.full((256, 256), .04)
        n = np.zeros((256, 256), dtype=np.uint32)
        for i, (cx, cy) in enumerate(((50, 52), (195, 52)), 1):
            mask = (x-cx)**2+(y-cy)**2 < 16**2
            h[mask] = .8; n[mask] = i
        for cx in (65, 125, 188):
            d[(x-cx)**2+(y-145)**2 <= 4**2] = 1.1
        ring = ((x-120)**2+(y-215)**2 >= 12**2) & ((x-120)**2+(y-215)**2 <= 15**2)
        d[ring] = 1.1
        d[195:200, 20:73] = 1.1
        vectors = np.column_stack([cls.cfg['hematoxylin_od_vector'], cls.cfg['dab_od_vector']])
        vectors /= np.linalg.norm(vectors, axis=0)
        rgb = np.rint(np.clip(np.exp(-h[..., None]*vectors[:, 0]-d[..., None]*vectors[:, 1])*256-1, 0, 255)).astype(np.uint8)
        return rgb, n

    def test_exact_five_tables_fourteen_arrays(self):
        self.assertEqual(self.rc, 0)
        report = p.load_json(self.out / 'REFERENCE_COMPARISON.json')
        self.assertTrue(report['passed'])
        self.assertEqual(report['difference_count'], 0)
        self.assertEqual(len(report['tables']), 5)
        self.assertEqual(len(report['arrays']), 14)
        self.assertEqual(report['allowed_path_relocations'], 6)

    def test_inputs_and_reference_unchanged(self):
        for name, sha in self.before.items():
            self.assertEqual(p.digest(name), sha)

    def test_counts_negative_and_ring_semantics(self):
        import pandas as pd
        table = pd.read_csv(self.out / 'tables/T01_field_measurements.tsv', sep='\t')
        positive = table[(table.image_uid == 'SYNTHETIC_ONLY_0') & (table.rule == 'working')].iloc[0]
        negative = table[(table.image_uid == 'SYNTHETIC_ONLY_1') & (table.rule == 'working')].iloc[0]
        self.assertEqual(positive.n_compact_object_candidates, 3)
        self.assertEqual(positive.n_nuclear_proposals_in_tissue, 2)
        self.assertEqual(positive.compact_objects_per_nuclear_proposal, 1.5)
        self.assertEqual(positive.n_ring_components, 1)
        self.assertEqual(negative.n_compact_object_candidates, 0)
        self.assertEqual(negative.n_nuclear_proposals_in_tissue, 0)
        self.assertTrue(pd.isna(negative.compact_objects_per_nuclear_proposal))
        self.assertTrue(table.compact_objects_per_mm2.isna().all())

    def test_no_npu_or_new_nuclear_inference_claim(self):
        r = p.verify_result(self.out)
        self.assertTrue(r['brown_objects_recomputed_from_rgb'])
        self.assertTrue(r['saved_nuclear_masks_reused'])
        self.assertFalse(r['new_nuclear_segmentation'])
        self.assertFalse(r['kl720_inference'])
        self.assertFalse(r['biological_validation'])

    def test_frozen_qc_generated_from_original_rgb(self):
        html = (self.out / 'index.html').read_text(encoding='utf-8')
        self.assertNotIn('__DATA_JSON__', html)
        self.assertIn('No inferred whole-cell boundaries', html)
        self.assertTrue((self.out / 'per_field/SYNTHETIC_ONLY_0/QC_overlay.png').exists())

    def test_output_overwrite_rejected(self):
        with self.assertRaises(FileExistsError):
            p.run(SimpleNamespace(bundle=self.moved, out=self.out))

    def test_tampered_mask_stops_before_computation(self):
        target = self.root / 'tampered_bundle'
        shutil.copytree(self.bundle, target)
        _, manifest = p.verify_bundle(target)
        path = target / manifest['fields'][0]['assets']['nuclei']
        path.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'modified'):
            p.run(SimpleNamespace(bundle=target, out=self.root / 'should_not_run'))
        self.assertFalse((self.root / 'should_not_run.BUILDING').exists())

    def test_changed_expected_count_not_hidden(self):
        target = self.root / 'tampered_table_output'
        shutil.copytree(self.out, target)
        table = target / 'tables/T01_field_measurements.tsv'
        text = table.read_text()
        text = text.replace('1.5', '1.50000000000001', 1)
        table.write_text(text)
        with self.assertRaisesRegex(ValueError, 'Modified'):
            p.verify_result(target)
        report = p.compare(self.moved, target, p.verify_bundle(self.moved)[1])
        self.assertFalse(report['passed'])

    def test_incomplete_annotations_do_not_score_as_negatives(self):
        review = self.root / 'review_incomplete.json'
        dataset = p.load_json(self.out / 'RUN_CONFIG.json')['dataset_id']
        p.write_json(review, {'dataset_id': dataset, 'patches': {}})
        with self.assertRaisesRegex(ValueError, 'exhaustively'):
            p.score(SimpleNamespace(bundle=self.moved, run=self.out, review=review,
                                    radius=6, out=self.root / 'review_should_not_pass'))


if __name__ == '__main__':
    unittest.main()
