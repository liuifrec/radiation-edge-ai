"""Runtime readiness is independent of any model or hardware support claim."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from radiation_edge_ai.nasa_bps import kneron_candidate_screen as screen


IMAGE_ID = "sha256:" + "a" * 64


def arguments(tmp_path):
    return SimpleNamespace(out=str(tmp_path / "runtime" / "runtime_probe.json"),
                           toolchain_image_id=IMAGE_ID)


def read_probe(args):
    return json.loads(Path(args.out).read_text(encoding="utf-8"))


def forbidden(*args, **kwargs):
    pytest.fail("A runtime probe must not load, optimize or evaluate a model")


def fake_modules(monkeypatch, optimizer="modern", missing=None):
    class Config:
        __init__ = forbidden
        evaluate = forbidden

    onnx = SimpleNamespace(__version__="1.7.0-test", __file__="/sdk/onnx/__init__.py", load=forbidden)
    ktc = SimpleNamespace(__version__="test", __file__="/sdk/ktc/__init__.py", ModelConfig=Config,
                          onnx_optimizer=SimpleNamespace(onnx2onnx_flow=forbidden))
    opt = SimpleNamespace(__version__="test", __file__="/sdk/kneronnxopt/__init__.py", optimize=forbidden)
    modules = {"onnx": onnx, "ktc": ktc, "kneronnxopt": opt}

    def import_module(name):
        if name == missing or (name == "kneronnxopt" and optimizer == "legacy"):
            raise ModuleNotFoundError("No module named '{}'".format(name), name=name)
        if name == "kneronnxopt" and optimizer == "broken":
            raise ModuleNotFoundError("Missing optimizer backend", name="optimizer_backend")
        return modules[name]

    monkeypatch.setattr(screen.importlib, "import_module", import_module)
    return modules


@pytest.mark.parametrize("optimizer,backend", [
    ("modern", "kneronnxopt.optimize"),
    ("legacy", "ktc.onnx_optimizer.onnx2onnx_flow"),
])
def test_runtime_probe_accepts_supported_apis_without_model_execution(tmp_path, monkeypatch, optimizer, backend):
    fake_modules(monkeypatch, optimizer)
    args = arguments(tmp_path)
    assert screen.probe_runtime(args) == 0
    report = read_probe(args)
    assert report["schema_version"] == 1
    assert report["purpose"] == "toolchain_runtime_preflight"
    assert report["status"] == "PASS" and report["exit_code"] == 0
    assert report["toolchain_image_id"] == IMAGE_ID
    assert report["python_executable"] == screen.sys.executable
    assert report["python_version"]
    assert report["dependency_versions"]["onnx"] == "1.7.0-test"
    assert report["dependency_paths"]["ktc"] == "/sdk/ktc/__init__.py"
    assert report["optimizer_backend"] == backend
    assert report["model_evaluation_performed"] is False
    assert report["error"] is None
    assert "support_status" not in report
    if optimizer == "legacy":
        assert report["dependency_paths"]["kneronnxopt"] is None


@pytest.mark.parametrize("missing", ["onnx", "ktc"])
def test_missing_sdk_dependency_writes_failed_receipt(tmp_path, monkeypatch, missing):
    fake_modules(monkeypatch, missing=missing)
    args = arguments(tmp_path)
    assert screen.probe_runtime(args) == 3
    report = read_probe(args)
    assert report["status"] == "FAILED" and report["exit_code"] == 3
    assert "ModuleNotFoundError" in report["error"] and missing in report["error"]
    assert report["toolchain_image_id"] == IMAGE_ID
    assert report["model_evaluation_performed"] is False


def test_broken_installed_optimizer_does_not_fall_back_to_legacy_api(tmp_path, monkeypatch):
    fake_modules(monkeypatch, optimizer="broken")
    args = arguments(tmp_path)
    assert screen.probe_runtime(args) == 3
    report = read_probe(args)
    assert report["status"] == "FAILED"
    assert "Missing optimizer backend" in report["error"]
    assert report["optimizer_backend"] is None


@pytest.mark.parametrize("missing_api", ["config", "evaluate", "modern_optimizer", "legacy_optimizer"])
def test_incomplete_sdk_api_is_rejected(tmp_path, monkeypatch, missing_api):
    optimizer = "legacy" if missing_api == "legacy_optimizer" else "modern"
    modules = fake_modules(monkeypatch, optimizer)
    if missing_api == "config":
        modules["ktc"].ModelConfig = None
    elif missing_api == "evaluate":
        modules["ktc"].ModelConfig.evaluate = None
    elif missing_api == "modern_optimizer":
        modules["kneronnxopt"].optimize = None
    else:
        modules["ktc"].onnx_optimizer.onnx2onnx_flow = None
    args = arguments(tmp_path)
    assert screen.probe_runtime(args) == 3
    report = read_probe(args)
    assert report["status"] == "FAILED"
    assert "RuntimeError" in report["error"]
    assert report["model_evaluation_performed"] is False


def test_old_interpreter_stops_before_vendor_import(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(screen, "_load_runtime", lambda: pytest.fail("SDK must not be imported"))
    with monkeypatch.context() as patch:
        patch.setattr(screen.sys, "version_info", (3, 6, 15))
        assert screen.probe_runtime(args) == 3
    report = read_probe(args)
    assert report["status"] == "FAILED"
    assert "requires Python 3.7 or later" in report["error"]


def test_existing_probe_receipt_is_preserved_before_import(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    output = Path(args.out)
    output.parent.mkdir()
    output.write_text("preserve earlier evidence", encoding="utf-8")
    monkeypatch.setattr(screen, "_load_runtime", lambda: pytest.fail("SDK must not be imported"))
    assert screen.main(["--probe-runtime", "--out", args.out]) == 3
    assert output.read_text(encoding="utf-8") == "preserve earlier evidence"


def test_probe_cli_requires_no_model_inputs(tmp_path, monkeypatch):
    fake_modules(monkeypatch)
    args = arguments(tmp_path)
    monkeypatch.setattr(screen, "run_screen", lambda args: pytest.fail("Model screening must not run"))
    assert screen.main(["--probe-runtime", "--out", args.out, "--toolchain-image-id", IMAGE_ID]) == 0
    assert read_probe(args)["status"] == "PASS"


@pytest.mark.parametrize("model_args", [[], ["--input", "model.onnx"], ["--candidate-manifest", "candidate.json"]])
def test_screen_cli_still_requires_both_model_inputs(tmp_path, monkeypatch, model_args):
    monkeypatch.setattr(screen, "run_screen", lambda args: pytest.fail("Incomplete screening must not run"))
    with pytest.raises(SystemExit) as failure:
        screen.main(["--out", str(tmp_path / "out")] + model_args)
    assert failure.value.code == 2
    assert not (tmp_path / "out").exists()


def test_probe_cli_still_requires_output():
    with pytest.raises(SystemExit) as failure:
        screen.main(["--probe-runtime"])
    assert failure.value.code == 2


def test_vendor_system_exit_still_records_failed_runtime(tmp_path, monkeypatch):
    def failing_import():
        raise SystemExit(17)

    monkeypatch.setattr(screen, "_load_runtime", failing_import)
    args = arguments(tmp_path)
    assert screen.probe_runtime(args) == 3
    assert read_probe(args)["error"] == "SystemExit: 17"
