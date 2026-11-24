"""Offline fail-closed tests for the single-case synthetic KL720 vendor gate."""
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

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "windows"
sys.path.insert(0, str(SCRIPTS))
SOURCE = SCRIPTS / "perilipin2_kl720_synthetic_vendor_gate.py"
SPEC = importlib.util.spec_from_file_location("perilipin2_kl720_synthetic_vendor_gate", SOURCE)
assert SPEC is not None and SPEC.loader is not None
g = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g)


def manifest():
    variants = {}
    for name, (channels, h) in g.CASES.items():
        variants[name] = {
            "filename": name + ".onnx",
            "operator": "ConvTranspose",
            "group": channels,
            "input_shape": [1, channels, h, h],
            "output_shape": [1, channels, 2 * h, 2 * h],
            "kernel_shape": [2, 2],
            "strides": [2, 2],
            "pads": [0, 0, 0, 0],
            "fixed_weight_shape": [channels, 1, 2, 2],
            "opset": 11, "ir_version": 7,
            "onnxruntime_vs_nearest_2x_max_abs_error": 0.0,
            "vendor_compilation_tested": False,
            "sha256": "a" * 64,
        }
    return {
        "schema": g.GENERATOR_SCHEMA,
        "source": "fully synthetic; no microscopy images or learned model weights",
        "vendor_evaluate_performed": False,
        "cases": variants,
    }


class SyntheticVendorGateTest(unittest.TestCase):
    def test_manifest_four_cases_exact(self):
        for case in g.CASES:
            self.assertEqual(g.metadata_contract(manifest(), case), manifest()["cases"][case])

    def test_manifest_fails_closed_for_changed_group_and_data_origin(self):
        altered = manifest()
        altered["cases"]["control_c1_h8"]["group"] = 96
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            g.metadata_contract(altered, "control_c1_h8")
        altered = manifest()
        altered["source"] = "actual Perilipin2 weights"
        with self.assertRaisesRegex(ValueError, "source classification"):
            g.metadata_contract(altered, "control_c1_h8")
        altered = manifest()
        altered["cases"]["control_c1_h8"]["vendor_compilation_tested"] = True
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            g.metadata_contract(altered, "control_c1_h8")

    def test_no_overwrite_after_previous_failed_run(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            base = p / "manifest"
            base.mkdir()
            out = p / "old_vendor"
            out.mkdir()
            args = types.SimpleNamespace(
                manifest_dir=str(base), case="control_c1_h8",
                out=str(out), image=g.v.IMAGE, model_id=32785, model_version="0001")
            with mock.patch.object(g, "preflight_source", return_value=(
                base, base / g.SOURCE_NAME, base / "control_c1_h8.onnx",
                {"sha256": "a" * 64})):
                with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                    g.on_windows(args)

    def test_outer_gate_detects_vendor_failure_with_zero_docker_exit_code(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "synthetic"
            source.mkdir()
            candidate = source / "control_c1_h8.onnx"
            candidate.write_bytes(b"fake synthetic ONNX bytes")
            man = source / g.SOURCE_NAME
            man.write_text(json.dumps(manifest()), encoding="utf-8")
            output = root / "vendor_control"
            args = types.SimpleNamespace(
                manifest_dir=str(source), case="control_c1_h8",
                out=str(output), image=g.v.IMAGE, model_id=32785,
                model_version="0001")
            meta = manifest()["cases"]["control_c1_h8"]
            meta["sha256"] = g.v.sha256(candidate)
            content = b'Failure for model "input/input"\nErr: 139\n' + b"\x93\xff"
            process = mock.MagicMock()
            process.__enter__.return_value = process
            process.stdout = io.BytesIO(content)
            process.wait.return_value = 0
            with mock.patch.object(g, "preflight_source",
                                   return_value=(source, man, candidate, meta)):
                with mock.patch.object(g.v, "docker_image_id",
                                       return_value=g.v.PINNED_IMAGE_ID):
                    with mock.patch.object(g.subprocess, "Popen",
                                           return_value=process) as popen:
                        self.assertEqual(g.on_windows(args), 2)
            command = popen.call_args.args[0]
            self.assertIn("--network", command)
            self.assertIn("none", command)
            self.assertIn("--pull", command)
            self.assertIn("never", command)
            self.assertIn(g.v.PINNED_IMAGE_ID, command)
            self.assertEqual(
                (output / "SYNTHETIC_VENDOR_DOCKER_CONSOLE.raw").read_bytes(),
                content)
            state = g.v.read_json(output / g.HOST_REPORT)
            self.assertEqual(state["status"], "VENDOR_REJECTED")
            self.assertEqual(state["toolchain_exit_code"], 0)
            self.assertEqual(state["case"], "control_c1_h8")
            self.assertFalse(state["nef_compiled"])
            self.assertIn("err: 139", state["vendor_failure_markers"])

    def test_inner_gate_classifies_invalid_input_even_without_exception(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            candidate = root / g.MODEL_NAME
            candidate.write_bytes(b"fully synthetic data")
            runner = root / g.RUNNER_NAME
            runner.write_bytes(b"runner bytes")
            helper = root / g.HELPER_NAME
            helper.write_bytes(b"helper bytes")
            intake = {
                "schema": g.SCHEMA, "case": "control_c1_h8",
                "candidate_sha256": g.v.sha256(candidate),
                "runner_sha256": g.v.sha256(runner),
                "helper_sha256": g.v.sha256(helper),
                "model_id": 32785, "model_version": "0001",
            }
            (root / g.INTAKE_NAME).write_text(json.dumps(intake), encoding="utf-8")
            obj = object()
            fake_onnx = types.SimpleNamespace(
                load=lambda _: obj,
                checker=types.SimpleNamespace(check_model=lambda _: None),
            )
            class ModelConfig:
                def __init__(self, *args, **kwargs):
                    self.parameters = (args, kwargs)
                def evaluate(self):
                    return "compiler invalid input"
            fake_ktc = types.SimpleNamespace(ModelConfig=ModelConfig)
            with mock.patch.dict(sys.modules, {"onnx": fake_onnx, "ktc": fake_ktc}):
                with mock.patch.object(g, "graph_contract"):
                    with mock.patch.object(g.v, "newest_artifact", return_value=None):
                        self.assertEqual(
                            g.in_container(types.SimpleNamespace(out=str(root))), 2)
            receipt = g.v.read_json(root / g.CONTAINER_REPORT)
            self.assertEqual(receipt["status"], "VENDOR_REJECTED")
            self.assertIn("compiler invalid input", receipt["vendor_failure_markers"])
            self.assertFalse(receipt["nef_compiled"])


if __name__ == "__main__":
    unittest.main()
