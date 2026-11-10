"""Offline tests for the parity-proven KL720 Resize vendor probe.

Tests deliberately do not require ONNX, Docker, Kneron, or a KL720.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "scripts/windows/perilipin2_resize_vendor_gate.py"
SPEC = importlib.util.spec_from_file_location("perilipin2_resize_vendor_gate", str(SOURCE))
assert SPEC is not None and SPEC.loader is not None
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


class VendorResizeTest(unittest.TestCase):
    def test_vendor_zero_exit_code_cannot_override_error_134(self):
        self.assertIn("err: 134", v.failure_markers(
            "input input  Err: 134  \u2713"))
        self.assertIn("failure for model", v.failure_markers(
            'Failure for model "input/input"'))
        self.assertEqual(v.failure_markers("compilation complete Err: 0"), [])

    def test_image_pinned_by_docker_identity(self):
        def inspect(cmd, **kwargs):
            self.assertEqual(cmd[:3], ["docker", "image", "inspect"])
            self.assertTrue(kwargs["capture_output"])
            return types.SimpleNamespace(
                returncode=0, stdout=(v.PINNED_IMAGE_ID + "\n").encode(), stderr=b""
            )
        with mock.patch.object(v.subprocess, "run", side_effect=inspect):
            self.assertEqual(v.docker_image_id(v.IMAGE), v.PINNED_IMAGE_ID)
        def bad_image(cmd, **kwargs):
            return types.SimpleNamespace(returncode=0, stdout=b"sha256:other\n", stderr=b"")
        with mock.patch.object(v.subprocess, "run", side_effect=bad_image):
            with self.assertRaisesRegex(ValueError, "image ID changed"):
                v.docker_image_id(v.IMAGE)

    def test_nonpassing_proof_never_launches_docker(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            bad_proof = {
                "schema": v.PROOF_SCHEMA,
                "status": "BLOCKED",
                "original_sha256": v.FROZEN_SOURCE_SHA,
                "optimized_vs_frozen_parity_pass": False,
            }
            (directory / "M3C_RESIZE_PROOF.json").write_text(json.dumps(bad_proof))
            with mock.patch.dict(sys.modules, {"onnx": types.SimpleNamespace()}):
                with self.assertRaisesRegex(ValueError, "Not a successfully completed"):
                    v.checked_proof(directory, "static-sizes")

    def test_compiler_failure_with_binary_docker_output_creates_failure_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            proof = base / "proof"
            proof.mkdir()
            candidate = proof / "variant.onnx"
            candidate.write_bytes(b"synthetic-model")
            proof_file = proof / "M3C_RESIZE_PROOF.json"
            proof_file.write_text(json.dumps({
                "optimized_sha256": "a" * 64,
                "resize_nodes": [{"node_index": 83, "target_shape": [1, 96, 64, 64]}],
            }), encoding="utf-8")
            meta = {"sha256": v.sha256(candidate)}
            raw = b"Failure for model input/input\nErr: 134\n" + bytes([0x93, 0xff])
            process = mock.MagicMock()
            process.__enter__.return_value = process
            process.stdout = io.BytesIO(raw)
            process.wait.return_value = 0
            args = types.SimpleNamespace(proof_dir=str(proof), variant="static-sizes",
                                         out=str(base / "vendor"), image=v.IMAGE,
                                         model_id=32785, model_version="0001")
            with mock.patch.object(v, "checked_proof", return_value=(proof_file, candidate, meta)):
                with mock.patch.object(v, "docker_image_id", return_value=v.PINNED_IMAGE_ID):
                    with mock.patch.object(v.subprocess, "Popen", return_value=process) as runner:
                        self.assertEqual(v.on_windows(args), 2)
            cmd = runner.call_args[0][0]
            self.assertIn("--pull", cmd)
            self.assertIn("never", cmd)
            self.assertIn("--network", cmd)
            self.assertIn(v.PINNED_IMAGE_ID, cmd)
            self.assertIn(v.TOOLCHAIN_PYTHON, cmd)
            self.assertNotIn("onnx_optimizer", " ".join(cmd))
            out = Path(args.out)
            self.assertEqual((out / "VENDOR_DOCKER_CONSOLE.raw").read_bytes(), raw)
            report = v.read_json(out / v.HOST_REPORT)
            self.assertEqual(report["status"], "VENDOR_REJECTED")
            self.assertEqual(report["toolchain_exit_code"], 0)
            self.assertIn("err: 134", report["failure_markers"])
            self.assertFalse(report["nef_compiled"])

    def test_fresh_compiler_backtrace_overrides_vendor_summary_success(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / v.CANDIDATE).write_bytes(b"model")
            (root / v.RUNNER).write_bytes(b"frozen")
            trace = root / "backtrace.log"
            trace.write_text("BatchSizeInterpreter::InterpretAllInputBatch\n", encoding="utf-8")
            intake = {
                "schema": v.GATE_SCHEMA,
                "candidate_sha256": v.sha256(root / v.CANDIDATE),
                "runner_sha256": v.sha256(root / v.RUNNER),
                "variant": "static-sizes",
                "resize_nodes": [{"target_shape": [1, 96, 64, 64]}],
                "model_id": 32785,
                "model_version": "0001",
            }
            (root / v.INTAKE).write_text(json.dumps(intake), encoding="utf-8")
            model = object()
            fake_onnx = types.SimpleNamespace(
                load=lambda _: model,
                checker=types.SimpleNamespace(check_model=lambda _: None))
            class ModelConfig:
                def __init__(self, *args, **kwargs):
                    pass
                def evaluate(self):
                    # The vendor may return only metadata, while printing its
                    # Err: 139 flow table directly to the console.
                    return "docker_version: kneron/toolchain:v0.33.1\ncomments:"
            fake_ktc = types.SimpleNamespace(ModelConfig=ModelConfig)
            def find_fresh(filename, started):
                return trace if filename == "backtrace.log" else None
            args = types.SimpleNamespace(out=str(root))
            with mock.patch.dict(sys.modules, {"ktc": fake_ktc, "onnx": fake_onnx}):
                with mock.patch.object(v, "resize_graph_contract"):
                    with mock.patch.object(v, "newest_artifact", side_effect=find_fresh):
                        self.assertEqual(v.in_container(args), 2)
            report = v.read_json(root / v.REPORT)
            self.assertEqual(report["status"], "VENDOR_REJECTED")
            self.assertTrue(report["compiler_backtrace_present"])
            self.assertIn("compiler_backtrace", report["vendor_failure_markers"])
            self.assertFalse(report["nef_compiled"])

    def test_vendor_evaluate_no_optimizer_and_fail_closed_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / v.CANDIDATE).write_bytes(b"model")
            (root / v.RUNNER).write_bytes(b"frozen")
            intake = {
                "schema": v.GATE_SCHEMA,
                "candidate_sha256": v.sha256(root / v.CANDIDATE),
                "runner_sha256": v.sha256(root / v.RUNNER),
                "variant": "static-sizes",
                "resize_nodes": [{"target_shape": [1, 96, 64, 64]}],
                "model_id": 32785,
                "model_version": "0001",
            }
            (root / v.INTAKE).write_text(json.dumps(intake), encoding="utf-8")
            dummy = object()
            fake_onnx = types.SimpleNamespace(load=lambda _: dummy,
                                              checker=types.SimpleNamespace(
                                                  check_model=lambda _: None))
            captured = []
            class ModelConfig:
                def __init__(self, *args, **kwargs):
                    captured.append((args, kwargs))
                def evaluate(self):
                    return "kdp720/compiler frontend Err: 134"
            fake_ktc = types.SimpleNamespace(ModelConfig=ModelConfig)
            args = types.SimpleNamespace(out=str(root))
            with mock.patch.dict(sys.modules, {"ktc": fake_ktc, "onnx": fake_onnx}):
                with mock.patch.object(v, "resize_graph_contract"):
                    with mock.patch.object(v, "newest_artifact", return_value=None):
                        self.assertEqual(v.in_container(args), 2)
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0][1]["onnx_model"], dummy)
            self.assertEqual(captured[0][0][2], "720")
            receipt = v.read_json(root / v.REPORT)
            self.assertEqual(receipt["status"], "VENDOR_REJECTED")
            self.assertIn("err: 134", receipt["vendor_failure_markers"])
            self.assertFalse(receipt["optimizer_rerun"])


if __name__ == "__main__":
    unittest.main()
