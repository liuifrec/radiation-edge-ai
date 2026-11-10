"""No Docker, Kneron SDK, torch, images or CUDA required for these tests."""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from radiation_edge_ai.perilipin2 import windows_cli as w  # noqa: E402


class M3CWindowsTest(unittest.TestCase):
    @staticmethod
    def doctor_result(print_mock):
        text = next(
            call.args[0] for call in print_mock.call_args_list
            if call.args and isinstance(call.args[0], str) and call.args[0].startswith("{\n")
        )
        return json.loads(text)

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

    def test_doctor_reports_verified_kneron_miniconda_probe(self):
        calls = []
        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            if cmd[:2] == ["docker", "info"]:
                return types.SimpleNamespace(returncode=0, stdout="27.0", stderr="")
            if cmd[:3] == ["docker", "image", "inspect"]:
                return types.SimpleNamespace(returncode=0, stdout="sha256:pinned\n", stderr="")
            if cmd[:2] == ["docker", "run"]:
                return types.SimpleNamespace(returncode=0, stdout="Toolchain ONNX: 1.7.0\n", stderr="")
            raise AssertionError(cmd)
        with mock.patch.object(w.shutil, "which", return_value="docker"):
            with mock.patch.object(w.subprocess, "run", side_effect=fake_run):
                with mock.patch("builtins.print") as out:
                    self.assertEqual(w.doctor(argparse.Namespace(image="kneron/toolchain:latest")), 0)
        doctor_json = self.doctor_result(out)
        self.assertTrue(doctor_json["docker_toolchain_python_ready"])
        self.assertEqual(doctor_json["docker_toolchain_python"], w.TOOLCHAIN_PYTHON)
        self.assertIn("1.7.0", doctor_json["docker_toolchain_probe"])
        self.assertTrue(any(w.TOOLCHAIN_PYTHON in cmd for cmd in calls))
        self.assertTrue(any("onnx2onnx_flow" in x for cmd in calls for x in cmd))

    def test_legacy_ktc_optimizer_supported_without_standalone_module(self):
        model = object()
        optimized = object()
        ktc = types.SimpleNamespace(onnx_optimizer=types.SimpleNamespace(
            onnx2onnx_flow=lambda value: optimized if value is model else None))
        with mock.patch.dict(sys.modules, {"kneronnxopt": None}):
            actual, backend = w.optimize_model(model, ktc)
        self.assertIs(actual, optimized)
        self.assertEqual(backend, "ktc.onnx_optimizer.onnx2onnx_flow")

    def test_toolchain_probe_timeout_is_bounded_and_targeted_container_removed(self):
        commands = []

        def fake_run(cmd, **kwargs):
            commands.append((cmd, kwargs))
            if cmd[:2] == ["docker", "run"]:
                raise subprocess.TimeoutExpired(
                    cmd, timeout=kwargs["timeout"], output=b"TensorFlow initialization..."
                )
            if cmd[:3] == ["docker", "rm", "-f"]:
                return types.SimpleNamespace(returncode=0, stdout="removed", stderr="")
            raise AssertionError(cmd)

        with mock.patch.object(w.subprocess, "run", side_effect=fake_run):
            ready, details = w.probe_toolchain_python("sha256:pinned")
        self.assertFalse(ready)
        self.assertIn("M3C_TOOLCHAIN_PROBE_TIMEOUT", details)
        self.assertIn("TensorFlow initialization", details)
        self.assertIn("Targeted probe cleanup exit code: 0", details)
        run, run_opts = commands[0]
        cleanup, cleanup_opts = commands[1]
        self.assertEqual(run_opts["timeout"], w.DOCKER_PROBE_TIMEOUT_SECONDS)
        self.assertEqual(run[run.index("--name") + 1], cleanup[3])
        self.assertEqual(cleanup_opts["timeout"], 10)
        self.assertIn("--network", run)

    def test_missing_both_optimizers_refuses_to_continue(self):
        with mock.patch.dict(sys.modules, {"kneronnxopt": None}):
            with self.assertRaisesRegex(ValueError, "Neither supported"):
                w.optimize_model(object(), types.SimpleNamespace())

    def test_doctor_flags_toolchain_import_failure(self):
        def fake_run(cmd, *args, **kwargs):
            if cmd[:2] == ["docker", "info"]:
                return types.SimpleNamespace(returncode=0, stdout="27.0", stderr="")
            if cmd[:3] == ["docker", "image", "inspect"]:
                return types.SimpleNamespace(returncode=0, stdout="sha256:pinned\n", stderr="")
            if cmd[:2] == ["docker", "run"]:
                return types.SimpleNamespace(returncode=1, stdout="", stderr="ModuleNotFoundError: onnx")
            raise AssertionError(cmd)
        with mock.patch.object(w.shutil, "which", return_value="docker"):
            with mock.patch.object(w.subprocess, "run", side_effect=fake_run):
                with mock.patch("builtins.print") as out:
                    self.assertEqual(w.doctor(argparse.Namespace(image="kneron/toolchain:latest")), 0)
        probe = self.doctor_result(out)
        self.assertFalse(probe["docker_toolchain_python_ready"])
        self.assertIn("onnx", probe["docker_toolchain_probe"])

    def test_doctor_without_docker_does_not_pretend_device_pass(self):
        with mock.patch.object(w.shutil, "which", return_value=None):
            with mock.patch("builtins.print") as out:
                self.assertEqual(w.doctor(argparse.Namespace(image="kneron/toolchain:latest")), 0)
            doctor_json = self.doctor_result(out)
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
            process = mock.MagicMock()
            process.__enter__.return_value = process
            process.stdout = io.BytesIO(b"compiler unsupported op\\n")
            process.wait.return_value = 6
            with mock.patch.object(w, "docker_image_identity", return_value="sha256:pinned"):
                with mock.patch.object(w, "probe_toolchain_python",
                                       return_value=(True, "Toolchain ONNX: 1.7.0")):
                    with mock.patch.object(w.subprocess, "Popen", return_value=process) as runner:
                        with self.assertRaisesRegex(ValueError, "did not complete"):
                            w.evaluate(args)
            cmd = runner.call_args[0][0]
            self.assertEqual(cmd[cmd.index("-w")+1], "/data1")
            self.assertIn("sha256:pinned", cmd)
            self.assertIn(w.TOOLCHAIN_PYTHON, cmd)
            self.assertNotIn("python3", cmd)
            self.assertEqual(cmd[cmd.index("sha256:pinned")+1], w.TOOLCHAIN_PYTHON)
            self.assertIn("-u", cmd)
            self.assertIn("--network", cmd)
            self.assertEqual(
                (root / "WINDOWS_DOCKER_CONSOLE.txt").read_text(encoding="utf-8"),
                "compiler unsupported op\\n",
            )
            self.assertEqual(
                (root / "WINDOWS_DOCKER_CONSOLE.raw").read_bytes(),
                b"compiler unsupported op\\n",
            )
            self.assertFalse(runner.call_args.kwargs.get("text", False))
            self.assertIn("--pull", cmd)
            self.assertNotIn("kneron/toolchain:v0.33.1", cmd)
            report = w.read_json(root / "WINDOWS_DOCKER_EXECUTION.json")
            self.assertEqual(report["toolchain_run_exit_code"], 6)
            self.assertFalse(report["evaluation_receipt_exists"])
            self.assertFalse((root / "M3C_KNERON_EVAL_RECEIPT.json").exists())


    def test_compiler_failure_non_utf8_stream_never_masks_vendor_error(self):
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
            args = argparse.Namespace(work=str(root), image="kneron/toolchain:latest",
                                      model_id=32785, model_version="0001")
            raw = (b'Failure for model "input/input" when running '
                   b'"kdp720/compiler frontend"\n' + bytes([0x93, 0xff]) + b"\n")
            process = mock.MagicMock()
            process.__enter__.return_value = process
            process.stdout = io.BytesIO(raw)
            process.wait.return_value = 0  # A vendor printout must fail closed even on code 0.
            with mock.patch.object(w, "docker_image_identity", return_value="sha256:pinned"):
                with mock.patch.object(w, "probe_toolchain_python", return_value=(True, "ready")):
                    with mock.patch.object(w.subprocess, "Popen", return_value=process):
                        with self.assertRaisesRegex(ValueError, "compiler reported a failure"):
                            w.evaluate(args)
            self.assertEqual((root / "WINDOWS_DOCKER_CONSOLE.raw").read_bytes(), raw)
            transcript = (root / "WINDOWS_DOCKER_CONSOLE.txt").read_text(encoding="utf-8")
            self.assertIn("Failure for model", transcript)
            self.assertIn("\\x93\\xff", transcript)
            report = w.read_json(root / "WINDOWS_DOCKER_EXECUTION.json")
            self.assertEqual(report["toolchain_run_exit_code"], 0)
            self.assertIn("failure for model", report["vendor_failure_markers"])
            self.assertEqual(report["raw_console_sha256"], w.sha256(root / "WINDOWS_DOCKER_CONSOLE.raw"))
            self.assertFalse((root / "M3C_KNERON_EVAL_RECEIPT.json").exists())

    def test_vendor_exception_collects_compiler_log_before_container_exits(self):
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
            fake_onnx = self._fake_onnx()
            fake_onnx.save = lambda model, path: Path(path).write_bytes(b"optimized")
            def vendor_raise():
                raise RuntimeError("compiler frontend error")
            fake_ktc = types.SimpleNamespace(
                ModelConfig=lambda *args, **kwargs: types.SimpleNamespace(evaluate=vendor_raise)
            )
            source_log = root / "batch_compile.log"
            source_log.write_text("Failure for model input/input\n", encoding="utf-8")
            def find_log(filename, started):
                return source_log if filename == "batch_compile.log" else None
            args = argparse.Namespace(work=str(root), model_id=32785, model_version="0001")
            with mock.patch.dict(sys.modules, {"onnx": fake_onnx, "ktc": fake_ktc}):
                with mock.patch.object(w, "optimize_model",
                                       return_value=(fake_onnx.load("x"), "legacy")):
                    with mock.patch.object(w, "artifact_find", side_effect=find_log):
                        with self.assertRaisesRegex(ValueError, "Kneron evaluation failed"):
                            w.docker_evaluate(args)
            failure = w.read_json(root / "M3C_KNERON_FAILURE.json")
            self.assertEqual(failure["status"], "KNERON_EVALUATION_FAILED")
            self.assertIn("RuntimeError", failure["vendor_exception"])
            self.assertIn("batch_compile.log", failure["artifact_sha256"])
            self.assertIn("failure for model", failure["vendor_failure_markers"])
            self.assertFalse((root / "M3C_KNERON_EVAL_RECEIPT.json").exists())

    def test_shape_inference_warning_is_not_compiler_rejection(self):
        self.assertEqual(w.vendor_failure_markers(
            "Cannot inference all shapes. If no other error is raised, ignore this message."), [])

if __name__ == "__main__":
    unittest.main()