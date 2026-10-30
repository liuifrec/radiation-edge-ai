"""No Docker, Kneron SDK, torch, images or CUDA required for these tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from radiation_edge_ai.perilipin2 import windows_cli as w  # noqa: E402


class M3CWindowsTest(unittest.TestCase):
    def receipt(self, digest="0" * 64):
        return {"onnx_sha256": digest, "onnxruntime_parity_pass": True,
                "onnxruntime_parity_max_abs_error": 1e-5,
                "onnxruntime_parity_mean_abs_error": 1e-6,
                "input_shape": [1, 3, 512, 512], "output_shape": [1, 2, 512, 512],
                "opset": 11}

    def test_receipt_validates_numerical_identity(self):
        w.validate_parity_receipt(self.receipt(), "0" * 64)
        for key, replacement in (("onnx_sha256", "1" * 64),
                                 ("onnxruntime_parity_pass", False),
                                 ("opset", 17),
                                 ("onnxruntime_parity_max_abs_error", float("nan")),
                                 ("onnxruntime_parity_mean_abs_error", True),
                                 ("output_shape", [1, 1, 512, 512])):
            x = self.receipt()
            x[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                w.validate_parity_receipt(x, "0" * 64)

    def test_tensor_contract_guard(self):
        contract = {"input_name": w.INPUT_NAME, "input_shape": w.INPUT_SHAPE,
                    "output_name": w.OUTPUT_NAME, "output_shape": w.OUTPUT_SHAPE}
        w.validate_contract(contract)
        for key in contract:
            bad = contract.copy()
            bad[key] = "wrong"
            with self.subTest(key=key), self.assertRaises(ValueError):
                w.validate_contract(bad)

    def test_cpu_partitions_are_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "ioinfo.csv"
            f.write_text("N,input,0\nc,Resize,1\nN,output,2\n", encoding="utf-8")
            self.assertEqual(len(w.inspect_cpu_partitions(f)), 1)
            self.assertEqual(w.inspect_cpu_partitions(f.parent / "absent.csv"), [])

    def test_bundle_refuses_modified_model_or_runner(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / w.WORK_MODEL).write_bytes(b"model")
            (root / "perilipin2_m3c_runner.py").write_bytes(b"script")
            (root / "ONNX_PARITY.json").write_bytes(b"{}")
            data = {"schema": w.SCHEMA, "status": "M3B_PARITY_VERIFIED",
                    "model_sha256": w.sha256(root / w.WORK_MODEL),
                    "runner_sha256": w.sha256(root / "perilipin2_m3c_runner.py"),
                    "parity_sha256": w.sha256(root / "ONNX_PARITY.json")}
            (root / "M3C_HOST_READY.json").write_text(json.dumps(data), encoding="utf-8")
            w.check_bundle(root)
            (root / w.WORK_MODEL).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Original model changed"):
                w.check_bundle(root)

    def test_doctor_reports_container_python3_probe(self):
        calls = []
        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            if cmd[:2] == ["docker", "info"]:
                return types.SimpleNamespace(returncode=0, stdout="27.0", stderr="")
            if cmd[:3] == ["docker", "image", "inspect"]:
                return types.SimpleNamespace(returncode=0, stdout="sha256:pinned\n", stderr="")
            if cmd[:2] == ["docker", "run"]:
                return types.SimpleNamespace(returncode=0, stdout="/usr/bin/python3\n", stderr="")
            raise AssertionError(cmd)
        with mock.patch.object(w.shutil, "which", return_value="docker"):
            with mock.patch.object(w.subprocess, "run", side_effect=fake_run):
                with mock.patch("builtins.print") as out:
                    self.assertEqual(w.doctor(argparse.Namespace(image="kneron/toolchain:latest")), 0)
        doctor_json = json.loads(out.call_args_list[0].args[0])
        self.assertTrue(doctor_json["docker_python3"])
        self.assertTrue(any("command -v python3" in x for cmd in calls for x in cmd))

    def test_doctor_without_docker_does_not_pretend_device_pass(self):
        with mock.patch.object(w.shutil, "which", return_value=None):
            with mock.patch("builtins.print") as out:
                self.assertEqual(w.doctor(argparse.Namespace(image="kneron/toolchain:latest")), 0)
            doctor_json = json.loads(out.call_args_list[0].args[0])
            self.assertFalse(doctor_json["physical_kl720_probe_performed"])
            self.assertFalse(doctor_json["docker_cli"])
            self.assertIsNone(doctor_json["image_id"])

    def test_docker_image_rejects_bad_reference(self):
        for value in ("", "--privileged", "foo bar", "x\nmalicious"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                w.docker_image_identity(value)

    def test_cli_supports_only_explicit_stages(self):
        parser = w.build_parser()
        self.assertEqual(parser.parse_args(["doctor"]).stage, "doctor")
        self.assertEqual(parser.parse_args(["docker-evaluate"]).stage, "docker-evaluate")
        with self.assertRaises(SystemExit):
            parser.parse_args(["nef-compile"])


# Additional pipeline tests use fake ONNX module records; they do not pretend to
# run vendor optimization or physical KL720 inference.
class M3CWindowsIntegrationTest(unittest.TestCase):
    def _fake_onnx(self):
        def value(name, dims):
            ds = [types.SimpleNamespace(dim_value=k, HasField=lambda x: x == "dim_value") for k in dims]
            return types.SimpleNamespace(name=name, type=types.SimpleNamespace(
                tensor_type=types.SimpleNamespace(shape=types.SimpleNamespace(dim=ds))))
        model = types.SimpleNamespace(
            graph=types.SimpleNamespace(
                input=[value(w.INPUT_NAME, w.INPUT_SHAPE)],
                output=[value(w.OUTPUT_NAME, w.OUTPUT_SHAPE)],
                initializer=[], node=[types.SimpleNamespace(op_type="Conv")]),
            opset_import=[types.SimpleNamespace(version=11)], ir_version=9)
        return types.SimpleNamespace(load=lambda _: model,
                                     checker=types.SimpleNamespace(check_model=lambda _: None))

    def test_intake_is_frozen_and_rejects_repeated_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "source"
            src.mkdir()
            model = src / w.ONNX_NAME
            model.write_bytes(b"synthetic ONNX placeholder (fake parser)")
            parity = src / "ONNX_PARITY.json"
            parity.write_text(json.dumps(M3CWindowsTest().receipt(w.sha256(model))))
            out = root / "out" / "poc"
            with mock.patch.dict(sys.modules, {"onnx": self._fake_onnx()}):
                self.assertEqual(w.intake(argparse.Namespace(onnx=str(model), parity=str(parity), out=str(out))), 0)
                self.assertEqual(w.check_bundle(out)[1]["model_sha256"], w.sha256(model))
                with self.assertRaisesRegex(ValueError, "existing"):
                    w.intake(argparse.Namespace(onnx=str(model), parity=str(parity), out=str(out)))
            self.assertFalse((src / "M3C_HOST_READY.json").exists())
            self.assertEqual(model.read_bytes(), b"synthetic ONNX placeholder (fake parser)")

    def test_docker_error_receipt_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / w.WORK_MODEL).write_bytes(b"model")
            (root / "perilipin2_m3c_runner.py").write_bytes(b"script")
            (root / "ONNX_PARITY.json").write_bytes(b"{}")
            data = {"schema": w.SCHEMA, "status": "M3B_PARITY_VERIFIED",
                    "model_sha256": w.sha256(root / w.WORK_MODEL),
                    "runner_sha256": w.sha256(root / "perilipin2_m3c_runner.py"),
                    "parity_sha256": w.sha256(root / "ONNX_PARITY.json")}
            (root / "M3C_HOST_READY.json").write_text(json.dumps(data))
            args = argparse.Namespace(work=str(root), image="kneron/toolchain:v0.33.1", model_id=32785,
                                      model_version="0001")
            error = types.SimpleNamespace(returncode=6, stdout="compiler unsupported op")
            with mock.patch.object(w, "docker_image_identity", return_value="sha256:pinned"):
                with mock.patch.object(w.subprocess, "run", return_value=error) as runner:
                    with self.assertRaisesRegex(ValueError, "did not complete"):
                        w.evaluate(args)
            cmd = runner.call_args[0][0]
            self.assertEqual(cmd[cmd.index("-w")+1], "/data1")
            self.assertIn("sha256:pinned", cmd)
            self.assertIn("python3", cmd)
            self.assertNotIn("python", cmd[cmd.index("sha256:pinned")+1:cmd.index("sha256:pinned")+2])
            self.assertIn("--pull", cmd)
            self.assertNotIn("kneron/toolchain:v0.33.1", cmd)
            report = w.read_json(root / "WINDOWS_DOCKER_EXECUTION.json")
            self.assertEqual(report["toolchain_run_exit_code"], 6)
            self.assertFalse(report["evaluation_receipt_exists"])
            self.assertFalse((root / "M3C_KNERON_EVAL_RECEIPT.json").exists())


if __name__ == "__main__":
    unittest.main()