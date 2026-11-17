"""Offline launcher-integrity tests; no vendor runtime or physical device is used."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "windows" / "run_nasa_kl720.py"


@pytest.fixture
def replay():
    spec = importlib.util.spec_from_file_location("nasa_replay_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rows():
    return [
        {
            "sample_id": f"SYNTHETIC_{index}",
            "nucleus_key": f"nucleus_{index}",
            "sample_name": "synthetic_bag",
            "source_name": "SYNTHETIC",
            "strain": "synthetic_strain",
            "sex": "Female",
            "particle_type": "X-ray",
            "dose_Gy": "0",
            "hr_post_exposure": "4",
            "fitc_filename": f"fitc_{index}.tif",
            "dapi_filename": f"dapi_{index}.tif",
        }
        for index in range(2)
    ]


def _predictions(rows):
    return [dict(row, physical_kl720_burden=str(0.5 + index)) for index, row in enumerate(rows)]


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch, replay):
    """A two-row fixture only exercises launcher logic, never frozen-data validation."""
    monkeypatch.delenv("RADEDGE_DATA_ROOT", raising=False)
    monkeypatch.delenv("RADEDGE_MODEL_ROOT", raising=False)
    root = tmp_path / "storage"
    assets = replay.resolve_assets(root)
    rows = _rows()
    predictions = _predictions(rows)
    for key in ("nef", "nef_freeze", "physical_smoke_freeze"):
        assets[key].parent.mkdir(parents=True, exist_ok=True)
        assets[key].write_bytes(("synthetic_" + key).encode())
    _write_csv(assets["holdout_manifest"], rows)
    _write_csv(assets["historical_predictions"], predictions)
    for row in rows:
        for channel in ("fitc", "dapi"):
            image = assets["image_root"] / channel / row[channel + "_filename"]
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes((row["sample_id"] + channel).encode())

    def check_sha(path, expected, label):
        actual = replay.sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{label} SHA256 mismatch")
        return actual

    def validate_manifest(manifest_rows):
        assert manifest_rows == rows

    legacy = SimpleNamespace(
        EXPECTED_NEF_SHA256=replay.sha256_file(assets["nef"]),
        EXPECTED_NEF_FREEZE_SHA256=replay.sha256_file(assets["nef_freeze"]),
        EXPECTED_PHYSICAL_SMOKE_FREEZE_SHA256=replay.sha256_file(
            assets["physical_smoke_freeze"]
        ),
        EXPECTED_QC1_MANIFEST_SHA256=replay.sha256_file(assets["holdout_manifest"]),
        EXPECTED_NUCLEI=2,
        EXPECTED_BAGS=1,
        EXPECTED_MODEL_ID=32770,
        cv2=SimpleNamespace(setNumThreads=lambda count: None),
        check_sha=check_sha,
        read_csv=replay.read_csv,
        read_json=lambda path: json.loads(path.read_text(encoding="utf-8")),
        validate_holdout_manifest=validate_manifest,
        load_tensor=lambda row, image_root: None,
        scan_descriptors=lambda kp: [SimpleNamespace(product_id=0x720, usb_port_id=81)],
    )
    monkeypatch.setattr(replay, "load_legacy_runner", lambda: legacy)
    monkeypatch.setattr(
        replay, "HISTORICAL_PREDICTIONS_SHA256", replay.sha256_file(assets["historical_predictions"])
    )
    monkeypatch.setitem(sys.modules, "kp", SimpleNamespace(offline_test_stub=True))
    return SimpleNamespace(
        replay=replay, root=root, assets=assets, rows=rows, predictions=predictions,
        legacy=legacy, output=tmp_path / "replay",
    )


def _worker_outputs(fixture, command, predictions=None, summary_changes=None):
    """Write synthetic worker artifacts without invoking any inference implementation."""
    output = Path(command[command.index("--output-dir") + 1])
    assert not output.exists(), "The launcher must never hand the worker a reused output directory"
    assert command[:4] == [sys.executable, "-X", "utf8", "-u"]
    assert Path(command[4]) == fixture.replay.LEGACY_SCRIPT
    assert command[command.index("--nef") + 1] == str(fixture.assets["nef"])
    output.mkdir()
    prediction_path = output / "physical_holdout_nucleus_predictions.csv"
    _write_csv(prediction_path, fixture.predictions if predictions is None else predictions)
    summary = {
        "status": "COMPLETE_OUTCOME_BLIND_FULL_PHYSICAL_KL720_HOLDOUT_INFERENCE",
        "platform": 720,
        "model_id": fixture.legacy.EXPECTED_MODEL_ID,
        "nef_sha256": fixture.legacy.EXPECTED_NEF_SHA256,
        "qc1_holdout_manifest_sha256": fixture.legacy.EXPECTED_QC1_MANIFEST_SHA256,
        "usb_port": 81,
        "nuclei": 2,
        "bags": 1,
        "physical_nucleus_predictions_sha256": fixture.replay.sha256_file(prediction_path),
        "mean_hardware_send_receive_ms": 1.0,
        "p95_hardware_send_receive_ms": 2.0,
    }
    summary.update(summary_changes or {})
    fixture.replay.write_json(output / "physical_holdout_inference_summary.json", summary)
    return 1


def _args(fixture):
    return ["--root", str(fixture.root), "--output-dir", str(fixture.output)]


def _receipt(fixture):
    return json.loads((fixture.output / "replay_receipt.json").read_text(encoding="utf-8"))


def test_help_works_with_only_standard_library():
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "--preflight-only" in result.stdout


def test_asset_paths_preserve_materialized_images_directory(tmp_path, monkeypatch, replay):
    monkeypatch.setenv("RADEDGE_DATA_ROOT", str(tmp_path / "data_override"))
    monkeypatch.setenv("RADEDGE_MODEL_ROOT", str(tmp_path / "model_override"))
    assets = replay.resolve_assets(tmp_path / "base")
    assert assets["image_root"] == (
        tmp_path / "data_override" / "nasa_bps_microscopy" / "r1_v2_final_holdout_blinded" / "images"
    )
    assert assets["nef"] == (
        tmp_path / "model_override" / "nasa_bps_53bp1" / "r1_v2_deployment" / "kneron_nef"
        / "r1_countnet_v2_final_kl720.nef"
    )
    explicit = replay.resolve_assets(
        tmp_path / "base", data_root=tmp_path / "data", model_root=tmp_path / "models",
        image_root=tmp_path / "custom_images",
    )
    assert explicit["image_root"] == tmp_path / "custom_images"
    assert explicit["holdout_manifest"].is_relative_to(tmp_path / "data")
    assert explicit["nef"].is_relative_to(tmp_path / "models")


def test_device_selection_requires_one_kl720_or_explicit_valid_port(replay):
    unrelated = SimpleNamespace(product_id=0x630, usb_port_id=80)
    first = SimpleNamespace(product_id=0x720, usb_port_id=81)
    second = SimpleNamespace(product_id=0x720, usb_port_id=82)
    assert replay.select_device([unrelated, first]) == 81
    assert replay.select_device([first, second], requested_port=82) == 82
    with pytest.raises(RuntimeError, match="No KL720"):
        replay.select_device([unrelated])
    with pytest.raises(RuntimeError, match="Multiple KL720"):
        replay.select_device([first, second])
    with pytest.raises(RuntimeError, match="unavailable"):
        replay.select_device([unrelated, first], requested_port=80)


def test_input_records_bind_pixel_bytes_not_only_names(synthetic_run):
    f = synthetic_run
    before = f.replay.input_file_records(f.rows, f.assets["image_root"])
    assert len(before) == 4
    assert {row["channel"] for row in before} == {"fitc", "dapi"}
    path = f.assets["image_root"] / before[0]["relative_path"]
    original = path.read_bytes()
    assert before[0]["sha256"] == f.replay.sha256_file(path)
    path.write_bytes(b"X" * len(original))
    after = f.replay.input_file_records(f.rows, f.assets["image_root"])
    assert before[0]["size_bytes"] == after[0]["size_bytes"]
    assert before[0]["sha256"] != after[0]["sha256"]


@pytest.mark.parametrize("filename", ["../outside.tif", r"..\outside.tif", "/outside.tif", ""])
def test_input_records_reject_paths_outside_channel_directory(synthetic_run, filename):
    f = synthetic_run
    rows = copy.deepcopy(f.rows)
    rows[0]["fitc_filename"] = filename
    with pytest.raises(RuntimeError, match="plain image filename"):
        f.replay.input_file_records(rows, f.assets["image_root"])


def test_missing_input_stops_before_worker(synthetic_run, monkeypatch):
    f = synthetic_run
    (f.assets["image_root"] / "dapi" / f.rows[1]["dapi_filename"]).unlink()
    monkeypatch.setattr(f.replay, "run_worker", lambda *args: pytest.fail("worker must not run"))
    with pytest.raises(FileNotFoundError, match="NASA input missing"):
        f.replay.main(_args(f))
    assert not f.output.exists()


def test_changed_frozen_nef_stops_before_worker(synthetic_run, monkeypatch):
    f = synthetic_run
    f.assets["nef"].write_bytes(b"changed synthetic model")
    monkeypatch.setattr(f.replay, "run_worker", lambda *args: pytest.fail("worker must not run"))
    with pytest.raises(RuntimeError, match="nef SHA256 mismatch"):
        f.replay.main(_args(f))
    assert not f.output.exists()


@pytest.mark.parametrize("mutation", ["partial", "duplicate", "foreign", "wrong_identity", "nan"])
def test_predictions_require_complete_unique_finite_identity(replay, mutation):
    manifest = _rows()
    predictions = _predictions(manifest)
    if mutation == "partial":
        predictions.pop()
    elif mutation == "duplicate":
        predictions.append(dict(predictions[0]))
    elif mutation == "foreign":
        predictions[0]["sample_id"] = "OTHER"
    elif mutation == "wrong_identity":
        predictions[0]["nucleus_key"] = "WRONG_NUCLEUS"
    else:
        predictions[0]["physical_kl720_burden"] = "nan"
    with pytest.raises(RuntimeError):
        replay.validate_prediction_rows(predictions, manifest)


def test_comparison_joins_by_identity_and_detects_numerical_drift(replay):
    reference = _predictions(_rows())
    fresh = copy.deepcopy(reference[::-1])
    fresh[0]["dose_Gy"] = "0.0"
    fresh[0]["hr_post_exposure"] = "4.0"
    fresh[0]["physical_kl720_burden"] = str(float(fresh[0]["physical_kl720_burden"]) + 0.00001)
    result = replay.compare_predictions(fresh, reference)
    assert result["status"] == "PASS"
    assert result["compared_nuclei"] == 2
    fresh[0]["physical_kl720_burden"] = "2.0"
    result = replay.compare_predictions(fresh, reference)
    assert result["status"] == "FAIL"
    assert result["max_abs_error"] == 0.5
    assert "not biological accuracy" in result["scope"]


def test_preflight_never_creates_outputs_or_invokes_worker(synthetic_run, monkeypatch):
    f = synthetic_run
    monkeypatch.setattr(f.replay, "run_worker", lambda *args: pytest.fail("worker must not run"))
    assert f.replay.main(_args(f) + ["--preflight-only"]) == 0
    assert not f.output.exists()


def test_invalid_image_geometry_stops_before_device_scan(synthetic_run, monkeypatch):
    f = synthetic_run

    def invalid_tensor(row, image_root):
        raise RuntimeError("synthetic crop exceeds 256")

    monkeypatch.setattr(f.legacy, "load_tensor", invalid_tensor)
    monkeypatch.setattr(f.legacy, "scan_descriptors", lambda *args: pytest.fail("no device scan"))
    monkeypatch.setattr(f.replay, "run_worker", lambda *args: pytest.fail("worker must not run"))
    with pytest.raises(RuntimeError, match="crop exceeds 256"):
        f.replay.main(_args(f) + ["--preflight-only"])
    assert not f.output.exists()


def test_success_receipt_binds_fresh_results_and_input_hashes(synthetic_run, monkeypatch):
    f = synthetic_run
    commands = []

    def worker(command, log_path):
        commands.append(command)
        return _worker_outputs(f, command)

    monkeypatch.setattr(f.replay, "run_worker", worker)
    assert f.replay.main(_args(f)) == 0
    receipt = _receipt(f)
    assert receipt["status"] == "COMPLETE"
    assert receipt["hardware_execution_completed"] is True
    assert receipt["verified_completed_inferences"] == 2
    assert receipt["completed_model_uploads"] == 1
    assert receipt["historical_comparison"]["status"] == "PASS"
    assert receipt["numerical_repeatability_verified"] is True
    assert receipt["cpu_fallback_used"] is False
    assert receipt["input_files_sha256"] == f.replay.sha256_file(f.output / "input_files.json")
    assert receipt["launcher_sha256"] == f.replay.sha256_file(SCRIPT)
    assert len(commands) == 1
    provenance = json.loads((f.output / "input_files.json").read_text(encoding="utf-8"))
    assert provenance["provenance"] == "newly_measured_not_historical_image_hash_verification"


def test_absent_historical_reference_never_claims_repeatability(synthetic_run, monkeypatch):
    f = synthetic_run
    f.assets["historical_predictions"].unlink()
    monkeypatch.setattr(
        f.replay, "run_worker", lambda command, log_path: _worker_outputs(f, command)
    )
    assert f.replay.main(_args(f)) == 0
    receipt = _receipt(f)
    assert receipt["status"] == "COMPLETE_UNCOMPARED"
    assert receipt["hardware_execution_completed"] is True
    assert receipt["numerical_repeatability_verified"] is False
    assert receipt["historical_comparison"]["status"] == "UNAVAILABLE"
    assert receipt["asset_sha256"]["historical_predictions"] is None


def test_even_empty_existing_output_is_refused(synthetic_run, monkeypatch):
    f = synthetic_run
    f.output.mkdir()
    monkeypatch.setattr(f.replay, "run_worker", lambda *args: pytest.fail("worker must not run"))
    with pytest.raises(FileExistsError):
        f.replay.main(_args(f))
    assert list(f.output.iterdir()) == []


def test_default_output_is_fresh_on_each_invocation(synthetic_run, monkeypatch):
    f = synthetic_run
    commands = []

    def worker(command, log_path):
        commands.append(command)
        return _worker_outputs(f, command)

    monkeypatch.setattr(f.replay, "run_worker", worker)
    assert f.replay.main(["--root", str(f.root)]) == 0
    assert f.replay.main(["--root", str(f.root)]) == 0
    outputs = [command[command.index("--output-dir") + 1] for command in commands]
    assert len(set(outputs)) == 2
    assert all(Path(path).is_dir() for path in outputs)


def test_failed_worker_is_recorded_and_cannot_reuse_directory(synthetic_run, monkeypatch):
    f = synthetic_run
    calls = []

    def worker(command, log_path):
        calls.append(command)
        raise RuntimeError("synthetic USB failure")

    monkeypatch.setattr(f.replay, "run_worker", worker)
    with pytest.raises(RuntimeError, match="synthetic USB failure"):
        f.replay.main(_args(f))
    receipt = _receipt(f)
    assert receipt["status"] == "FAILED"
    assert receipt["hardware_execution_completed"] is False
    assert receipt["verified_completed_inferences"] == 0
    assert "synthetic USB failure" in receipt["error"]
    with pytest.raises(FileExistsError):
        f.replay.main(_args(f))
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["partial", "wrong_port", "input_changed", "reference_changed"])
def test_postrun_integrity_failure_never_records_success(synthetic_run, monkeypatch, failure):
    f = synthetic_run

    def worker(command, log_path):
        predictions = f.predictions[:1] if failure == "partial" else None
        changes = {"usb_port": 99} if failure == "wrong_port" else None
        result = _worker_outputs(f, command, predictions, changes)
        if failure == "input_changed":
            image = f.assets["image_root"] / "fitc" / f.rows[0]["fitc_filename"]
            image.write_bytes(b"changed synthetic input")
        elif failure == "reference_changed":
            f.assets["historical_predictions"].write_bytes(b"changed synthetic reference")
        return result

    monkeypatch.setattr(f.replay, "run_worker", worker)
    with pytest.raises(RuntimeError):
        f.replay.main(_args(f))
    assert _receipt(f)["status"] == "FAILED"


def test_numerical_drift_keeps_hardware_evidence_and_returns_failure(synthetic_run, monkeypatch):
    f = synthetic_run
    changed = copy.deepcopy(f.predictions)
    changed[0]["physical_kl720_burden"] = "3.0"
    monkeypatch.setattr(
        f.replay, "run_worker", lambda command, log_path: _worker_outputs(f, command, changed)
    )
    assert f.replay.main(_args(f)) == 2
    receipt = _receipt(f)
    assert receipt["status"] == "NUMERICAL_MISMATCH"
    assert receipt["hardware_execution_completed"] is True
    assert receipt["verified_completed_inferences"] == 2
    assert receipt["historical_comparison"]["status"] == "FAIL"


def test_worker_streams_log_but_hides_obsolete_unblinding_instruction(tmp_path, replay, capsys):
    command = [
        sys.executable, "-c",
        "print('[Load NEF] PASS (synthetic test)'); print('NEXT GATE: obsolete test instruction')",
    ]
    log = tmp_path / "worker.log"
    assert replay.run_worker(command, log) == 1
    assert "NEXT GATE:" in log.read_text(encoding="utf-8")
    assert "NEXT GATE:" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "code,error",
    [
        ("print('no upload marker')", "exactly one"),
        ("print('[Load NEF] PASS'); print('[Load NEF] PASS')", "exactly one"),
        ("print('[Load NEF] PASS'); raise SystemExit(7)", "exited with code 7"),
    ],
)
def test_worker_requires_successful_exit_and_exactly_one_upload(tmp_path, replay, code, error):
    with pytest.raises(RuntimeError, match=error):
        replay.run_worker([sys.executable, "-c", code], tmp_path / "worker.log")
