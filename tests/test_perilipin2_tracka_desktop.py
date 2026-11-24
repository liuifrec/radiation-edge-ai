"""Command integration tests only; numerical M2 replay is tested separately."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from radiation_edge_ai.desktop_runtime import Settings, build_job
from radiation_edge_ai.perilipin2_tracka import BUNDLE_SCHEMA


class TrackADesktopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.data = self.root / 'data'
        self.data.mkdir()
        self.source = self.data / 'TRACKA_INPUT_BUNDLE.json'
        self.settings = Settings(host_python=sys.executable, data_root=str(self.data),
                                 scientific_python=sys.executable, kl720_python='')

    def test_analysis_is_explicit_cpu_and_real_module_invocation(self):
        self.source.write_text(json.dumps({'schema': BUNDLE_SCHEMA,
            'purpose': 'frozen_two_field_reference_replay'}), encoding='utf-8')
        job = build_job(self.settings, self.repo, 'tracka-reference',
                        str(self.source), str(self.data / 'new_results'))
        self.assertFalse(job.hardware_requested)
        self.assertEqual(job.backend, 'cpu-saved-nuclear-masks')
        self.assertEqual(job.assay_id, 'perilipin2-tracka-m2-reference')
        self.assertIn('radiation_edge_ai.perilipin2_tracka', job.argv)
        self.assertIn('run', job.argv)
        self.assertIn('--bundle', job.argv)
        self.assertIn('--out', job.argv)
        self.assertNotIn('--kl720-python', job.argv)
        self.assertNotIn('--output-dir', job.argv)
        self.assertIn('utf8', job.argv)

    def test_wrong_input_receipt_is_not_accepted_as_analysis_bundle(self):
        self.source.write_text(json.dumps({'schema': 'perilipin2-kl720-synthetic-vendor-v1'}))
        with self.assertRaisesRegex(ValueError, 'TRACKA_INPUT_BUNDLE'):
            build_job(self.settings, self.repo, 'tracka-reference',
                      str(self.source), str(self.data / 'new_results'))

    def test_no_output_overwrite(self):
        self.source.write_text(json.dumps({'schema': BUNDLE_SCHEMA,
            'purpose': 'frozen_two_field_reference_replay'}))
        out = self.data / 'completed'
        out.mkdir()
        with self.assertRaisesRegex(ValueError, 'Output exists'):
            build_job(self.settings, self.repo, 'tracka-reference',
                      str(self.source), str(out))


if __name__ == '__main__':
    unittest.main()
