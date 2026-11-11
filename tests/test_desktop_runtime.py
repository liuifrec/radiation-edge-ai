"""Desktop routing/logging tests; no Docker, real images, NEFs, SDK or hardware."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from radiation_edge_ai.desktop_runtime import (
    CLI_ENTRY, Job, Settings, build_job, read_object, run_job, sha256,
)


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.data = self.root / "data"
        self.data.mkdir()
        self.config = Settings(sys.executable, str(self.data), sys.executable, sys.executable)
        self.input = self.data / "input.json"

    def obj(self, data):
        self.input.write_text(json.dumps(data), encoding="utf-8")
        return str(self.input)

    def plan(self, assay="nasa-53bp1-r1-v2", backend="kl720"):
        return self.obj({"assay": {"assay_id": assay}, "backend": backend,
                         "plan_fingerprint_sha256": "a" * 64})

    def test_doctor_does_not_scan_or_require_other_runtimes(self):
        cfg = replace(self.config, scientific_python="", kl720_python="")
        job = build_job(cfg, self.repo, "doctor")
        self.assertEqual(job.argv[-2:], ("doctor", "--json"))
        self.assertFalse(job.hardware_requested)

    def test_scan_requires_plus_python_and_explicit_probe(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            build_job(replace(self.config, kl720_python=""), self.repo, "device-scan")
        probe = self.repo / "src/radiation_edge_ai/hardware/kl720_check.py"
        probe.parent.mkdir(parents=True)
        probe.write_text("pass\n")
        job = build_job(self.config, self.repo, "device-scan")
        self.assertEqual(job.argv[0], str(Path(sys.executable).resolve()))
        self.assertTrue(job.hardware_requested)
        self.assertEqual(job.argv[-1], str(probe))

    def test_frozen_nasa_plan_routes_to_device_without_science_env(self):
        job = build_job(replace(self.config, scientific_python=""), self.repo,
                        "run-plan", self.plan())
        self.assertTrue(job.hardware_requested)
        self.assertEqual(job.backend, "kl720")
        self.assertIn("--kl720-python", job.argv)
        self.assertNotIn("--onnx-python", job.argv)
        self.assertIn(CLI_ENTRY, job.argv)

    def test_nasa_kl720_manifest_is_not_silently_run_on_cpu(self):
        path = self.obj({"assay_id": "nasa-53bp1-r1-v2", "backend": "kl720"})
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            build_job(self.config, self.repo, "run-assay", path, str(self.data / "new"))

    def test_dnai_device_manifest_keeps_both_runtimes(self):
        source = self.obj({"assay_id": "dnai-fiber-v3", "backend": "kl720"})
        job = build_job(self.config, self.repo, "run-assay", source, str(self.data / "new"))
        self.assertIn("--kl720-python", job.argv)
        self.assertIn("--onnx-python", job.argv)
        self.assertTrue(job.hardware_requested)

    def test_cpu_mode_explicit_and_requires_scientific_env(self):
        source = self.plan(backend="cpu-onnx")
        job = build_job(self.config, self.repo, "run-plan", source)
        self.assertIn("--onnx-python", job.argv)
        self.assertNotIn("--kl720-python", job.argv)
        self.assertFalse(job.hardware_requested)
        with self.assertRaisesRegex(ValueError, "scientific"):
            build_job(replace(self.config, scientific_python=""), self.repo, "run-plan", source)

    def test_unsupported_perilipin2_never_enabled_from_synthetic_success(self):
        for assay in ("perilipin2", "PNKP", "unknown"):
            with self.subTest(assay=assay):
                source = self.plan(assay=assay)
                with self.assertRaisesRegex(ValueError, "No enabled desktop"):
                    build_job(self.config, self.repo, "run-plan", source)

    def test_existing_output_and_repo_output_rejected(self):
        source = self.obj({"assay_id": "dnai-fiber-v3", "backend": "kl720"})
        for out in (self.data, self.repo / "generated"):
            with self.subTest(output=out), self.assertRaises(ValueError):
                build_job(self.config, self.repo, "run-assay", source, str(out))

    def test_frozen_run_plan_in_repository_rejected(self):
        self.plan()
        inside = self.repo / "plan.json"
        inside.write_bytes(self.input.read_bytes())
        with self.assertRaisesRegex(ValueError, "Git worktree"):
            build_job(self.config, self.repo, "run-plan", str(inside))

    def test_unknown_actions_and_invalid_device_parameters_rejected(self):
        with self.assertRaises(ValueError):
            build_job(self.config, self.repo, "shell")
        for cfg in (replace(self.config, kl720_port=-1),
                    replace(self.config, kl720_port=True),
                    replace(self.config, kl720_timeout_ms=0)):
            with self.assertRaises(ValueError):
                build_job(cfg, self.repo, "doctor")

    def test_verify_paths_with_spaces_are_single_argv_not_shell(self):
        source = self.data / "some ; command.json"
        source.write_text("{}")
        job = build_job(self.config, self.repo, "verify", str(source))
        self.assertIn(str(source), job.argv)
        self.assertFalse(job.hardware_requested)
        self.assertEqual(job.action, "verify")

    def test_settings_roundtrip(self):
        file = self.data / "config" / "desktop.json"
        self.config.save(file)
        self.assertEqual(Settings.load(file), self.config)

    def test_invalid_json_and_nonfinite_rejected(self):
        for data in ('[]', '{"x":NaN}', '{"x":Infinity}'):
            self.input.write_text(data)
            with self.assertRaises(ValueError):
                read_object(self.input)

    def test_source_change_between_preview_and_execute_stops_job(self):
        source = self.obj({"some": "data"})
        job = build_job(self.config, self.repo, "verify", source)
        self.input.write_text('{"some":"changed"}')
        with self.assertRaisesRegex(ValueError, "changed since preview"):
            run_job(job, self.config, self.repo)
        self.assertFalse((self.data / "desktop_jobs").exists())

    def test_real_child_binary_logging_and_nonzero_exit(self):
        script = "import sys; sys.stdout.buffer.write(b'valid\\n\\x93\\xff\\n'); sys.exit(7)"
        job = Job("verify", (sys.executable, "-c", script), None, None, None, None, False, None)
        lines = []
        receipt = run_job(job, self.config, self.repo, lines.append)
        record = read_object(receipt)
        self.assertEqual(record["exit_code"], 7)
        self.assertEqual(record["status"], "FAILED")
        self.assertEqual((receipt.parent / "console.raw").read_bytes(), b"valid\n\x93\xff\n")
        self.assertIn("\\x93\\xff", (receipt.parent / "console.txt").read_text())
        self.assertFalse(record["physical_inference_verified_by_launcher"])
        self.assertEqual(record["raw_console_sha256"], sha256(receipt.parent / "console.raw"))

    def test_zero_exit_is_not_hardware_or_biological_validation(self):
        job = Job("doctor", (sys.executable, "-c", "print('ready')"), None, None,
                  None, None, False, None)
        report = read_object(run_job(job, self.config, self.repo))
        self.assertEqual(report["status"], "COMMAND_RETURNED_REVIEW_CORE_RECORDS")
        self.assertFalse(report["biological_validation_performed_by_launcher"])
        self.assertFalse(report["physical_inference_verified_by_launcher"])

    @unittest.skipUnless(importlib.util.find_spec("radiation_edge_ai.cli"),
                         "Full CLI available in repository CI, not isolated source testing")
    def test_generated_run_arguments_parse_in_existing_cli(self):
        from radiation_edge_ai.cli import build_parser
        job = build_job(self.config, self.repo, "run-plan", self.plan())
        parsed = build_parser().parse_args(list(job.argv[4:]))
        self.assertEqual(parsed.command, "run")
        self.assertEqual(parsed.kl720_timeout_ms, 10000)


if __name__ == "__main__":
    unittest.main()
