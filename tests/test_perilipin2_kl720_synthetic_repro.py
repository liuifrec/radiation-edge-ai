"""Safety checks for shareable synthetic KL720 ConvTranspose repro candidates.

Does not require ONNX, Docker, Kneron, real microscopy or trained models.
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] /
          "scripts" / "windows" / "perilipin2_kl720_synthetic_repro.py")
SPEC = importlib.util.spec_from_file_location("perilipin2_kl720_synthetic_repro", str(SOURCE))
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


class SyntheticKneronReproTests(unittest.TestCase):
    def test_case_control_and_grouped_models(self):
        self.assertEqual(generator.CASES, (
            ("control_c1_h8", 1, 8),
            ("group_c24_h8", 24, 8),
            ("group_c96_h8", 96, 8),
            ("stage_c96_h32", 96, 32),
        ))
        self.assertTrue(all(channels > 0 and h > 0
                            for _, channels, h in generator.CASES))

    def test_reject_existing_folder_before_any_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "already exists"):
                generator.run(directory)

    def test_reject_output_within_github_worktree(self):
        forbidden = SOURCE.parents[2] / "synthetic_onnx_do_not_create"
        self.assertFalse(forbidden.exists())
        with self.assertRaisesRegex(ValueError, "outside the Git repository"):
            generator.run(str(forbidden))

    def test_digest_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.txt"
            path.write_bytes(b"synthetic-only")
            first = generator.sha256(path)
            self.assertEqual(len(first), 64)
            self.assertEqual(first, generator.sha256(path))
            path.write_bytes(b"changed")
            self.assertNotEqual(first, generator.sha256(path))


if __name__ == "__main__":
    unittest.main()
