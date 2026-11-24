"""Fresh hardware replay of the frozen NASA baseline, using the Windows PLUS environment.

This entry point does not train, calibrate, download, or emulate a model. The
historical runner stays unchanged; each invocation gets a new output directory.
Its old outcome-blind schema is retained as evidence, not interpreted as a new
blinded experiment. Python 3.9 compatible; heavy/vendor imports occur at runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

HISTORICAL_PREDICTIONS_SHA256 = (
    "91ea895a16cd97067af63b5c5c4434e972629bfda4c9582ad456a79e007b6d46"
)
REPLAY_MAX_ABS_ERROR = 1e-4
LEGACY_SCRIPT = Path(__file__).with_name("run_kl720_nasa_r1_v2_full_holdout.py")
IDENTITY_FIELDS = (
    "sample_id", "nucleus_key", "sample_name", "source_name", "strain", "sex",
    "particle_type", "dose_Gy", "hr_post_exposure",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_legacy_runner():
    spec = importlib.util.spec_from_file_location("_nasa_frozen_hardware_runner", LEGACY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_assets(root, data_root=None, model_root=None, image_root=None):
    root = Path(root).resolve()
    data = Path(data_root or os.environ.get("RADEDGE_DATA_ROOT", root / "data")).resolve()
    models = Path(model_root or os.environ.get("RADEDGE_MODEL_ROOT", root / "models")).resolve()
    nasa = data / "nasa_bps_microscopy"
    nef_root = models / "nasa_bps_53bp1" / "r1_v2_deployment" / "kneron_nef"
    return {
        "nef": nef_root / "r1_countnet_v2_final_kl720.nef",
        "nef_freeze": nef_root / "nef_deployment_candidate_freeze.json",
        "physical_smoke_freeze": (
            nef_root / "physical_smoke_hardware" / "physical_smoke_pass_freeze.json"
        ),
        "holdout_manifest": (
            nasa / "metadata" / "r1_v2_freeze" / "r1_v2_final_holdout_manifest_blinded_qc1.csv"
        ),
        "image_root": Path(image_root).resolve() if image_root else (
            nasa / "r1_v2_final_holdout_blinded" / "images"
        ),
        "historical_predictions": (
            nef_root / "physical_holdout_hardware" / "physical_holdout_nucleus_predictions.csv"
        ),
    }


def select_device(descriptors, requested_port=None):
    kl720 = [d for d in descriptors if int(getattr(d, "product_id", -1)) == 0x720]
    ports = sorted(int(d.usb_port_id) for d in kl720)
    if requested_port is not None:
        if requested_port not in ports:
            raise RuntimeError("Requested KL720 port {} unavailable; KL720 ports={}".format(
                requested_port, ports
            ))
        return requested_port
    if not ports:
        raise RuntimeError(
            "No KL720 found. Connect the dongle and use the existing Kneron PLUS Python runtime."
        )
    if len(ports) != 1:
        raise RuntimeError("Multiple KL720 devices found {}; choose --port explicitly.".format(ports))
    return ports[0]


def input_file_records(rows, image_root):
    """Measure this run's pixels; do not claim an unavailable historical hash match."""
    records = []
    for row in rows:
        for channel in ("fitc", "dapi"):
            filename = row[channel + "_filename"]
            if (not filename or filename in {".", ".."}
                    or Path(filename).name != filename or "\\" in filename):
                raise RuntimeError("Expected a plain image filename: {!r}".format(filename))
            path = image_root / channel / filename
            if not path.is_file():
                raise FileNotFoundError("NASA input missing: {}".format(path))
            records.append({
                "sample_id": row["sample_id"], "channel": channel,
                "relative_path": channel + "/" + filename,
                "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
            })
    return records


def validate_assets(legacy, assets):
    hashes = {}
    for key, expected in (
        ("nef", legacy.EXPECTED_NEF_SHA256),
        ("nef_freeze", legacy.EXPECTED_NEF_FREEZE_SHA256),
        ("physical_smoke_freeze", legacy.EXPECTED_PHYSICAL_SMOKE_FREEZE_SHA256),
        ("holdout_manifest", legacy.EXPECTED_QC1_MANIFEST_SHA256),
    ):
        hashes[key] = legacy.check_sha(assets[key], expected, key)
    rows = legacy.read_csv(assets["holdout_manifest"])
    legacy.validate_holdout_manifest(rows)
    rows.sort(key=lambda row: (row["sample_name"], row["sample_id"]))
    reference = assets["historical_predictions"]
    if reference.is_file():
        hashes["historical_predictions"] = legacy.check_sha(
            reference, HISTORICAL_PREDICTIONS_SHA256, "Historical physical predictions"
        )
    else:
        hashes["historical_predictions"] = None
    return rows, hashes


def validate_prediction_rows(rows, manifest_rows):
    expected = {r["sample_id"]: r for r in manifest_rows}
    actual = {r["sample_id"]: r for r in rows}
    if len(actual) != len(rows) or set(actual) != set(expected):
        raise RuntimeError("Fresh predictions do not cover the exact manifest once each")
    for sample_id, row in actual.items():
        for field in IDENTITY_FIELDS:
            left, right = row[field], expected[sample_id][field]
            equal = float(left) == float(right) if field in (
                "dose_Gy", "hr_post_exposure"
            ) else left == right
            if not equal:
                raise RuntimeError("Prediction identity mismatch: {} / {}".format(sample_id, field))
        value = float(row["physical_kl720_burden"])
        if not math.isfinite(value) or value < 0:
            raise RuntimeError("Invalid physical burden for {}".format(sample_id))
    return actual


def compare_predictions(fresh_rows, reference_rows):
    fresh = validate_prediction_rows(fresh_rows, reference_rows)
    reference = {r["sample_id"]: r for r in reference_rows}
    if len(reference) != len(reference_rows):
        raise RuntimeError("Duplicate historical prediction identity")
    differences = []
    for key, row in fresh.items():
        baseline = float(reference[key]["physical_kl720_burden"])
        if not math.isfinite(baseline) or baseline < 0:
            raise RuntimeError("Invalid historical physical burden")
        differences.append(abs(float(row["physical_kl720_burden"]) - baseline))
    maximum = max(differences)
    return {
        "status": "PASS" if maximum <= REPLAY_MAX_ABS_ERROR else "FAIL",
        "compared_nuclei": len(differences), "max_abs_error": maximum,
        "mean_abs_error": sum(differences) / len(differences),
        "max_abs_error_limit": REPLAY_MAX_ABS_ERROR,
        "scope": "numerical repeatability against historical hardware; not biological accuracy",
    }


def run_worker(command, log_path):
    model_uploads = 0
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith("[Load NEF] PASS"):
                    model_uploads += 1
                # This is a replay. The old script's instruction to unblind is obsolete.
                if not line.startswith("NEXT GATE:"):
                    print(line, end="", flush=True)
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    if code != 0:
        raise RuntimeError("Physical runner exited with code {}; inspect {}".format(code, log_path))
    if model_uploads != 1:
        raise RuntimeError("Expected exactly one completed NEF upload; observed {}".format(model_uploads))
    return model_uploads


def verify_worker_result(legacy, output_dir, rows, port):
    csv_path = output_dir / "physical_holdout_nucleus_predictions.csv"
    summary_path = output_dir / "physical_holdout_inference_summary.json"
    summary = legacy.read_json(summary_path)
    expected = {
        "status": "COMPLETE_OUTCOME_BLIND_FULL_PHYSICAL_KL720_HOLDOUT_INFERENCE",
        "platform": 720, "model_id": legacy.EXPECTED_MODEL_ID,
        "nef_sha256": legacy.EXPECTED_NEF_SHA256,
        "qc1_holdout_manifest_sha256": legacy.EXPECTED_QC1_MANIFEST_SHA256,
        "usb_port": port, "nuclei": legacy.EXPECTED_NUCLEI, "bags": legacy.EXPECTED_BAGS,
        "physical_nucleus_predictions_sha256": sha256_file(csv_path),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError("Hardware result summary mismatch: {}".format(key))
    predictions = read_csv(csv_path)
    validate_prediction_rows(predictions, rows)
    return summary, predictions


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"D:\radiation-edge-ai-data")
    parser.add_argument("--data-root")
    parser.add_argument("--model-root")
    parser.add_argument("--image-root")
    parser.add_argument("--output-dir", help="Must not already exist; defaults to a new scratch folder")
    parser.add_argument("--port", type=int, help="Auto-selects when exactly one KL720 is connected")
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--preflight-only", action="store_true", help="Check data and device; no inference")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be positive")
    print("NASA baseline: fresh KL720 hardware replay of the historical evaluation set.", flush=True)
    print("Checking frozen assets and measuring current input-file hashes...", flush=True)
    legacy = load_legacy_runner()
    assets = resolve_assets(args.root, args.data_root, args.model_root, args.image_root)
    rows, asset_hashes = validate_assets(legacy, assets)
    inputs = input_file_records(rows, assets["image_root"])
    legacy.cv2.setNumThreads(0)
    for row in rows:
        legacy.load_tensor(row, assets["image_root"])
    import kp

    port = select_device(legacy.scan_descriptors(kp), args.port)
    print("Preflight PASS: {} nuclei, {} image files, KL720 port {}.".format(
        len(rows), len(inputs), port
    ), flush=True)
    if args.preflight_only:
        print("Preflight only: no model uploaded and no inference performed.")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        Path(args.root).resolve() / "scratch" / ("nasa_kl720_" + stamp + "_" + uuid4().hex[:8])
    )
    # Refuse even an empty existing directory: a successful replay cannot reuse old outputs.
    output_dir.mkdir(parents=True, exist_ok=False)
    hardware_dir = output_dir / "hardware"
    input_path = output_dir / "input_files.json"
    write_json(input_path, {
        "provenance": "newly_measured_not_historical_image_hash_verification",
        "image_root": str(assets["image_root"]), "files": inputs,
    })
    receipt_path = output_dir / "replay_receipt.json"
    receipt = {
        "status": "STARTED", "started_utc": stamp, "hardware_execution_completed": False,
        "verified_completed_inferences": 0, "cpu_fallback_used": False,
        "scope": "fresh historical-baseline replay; not a new blinded validation",
        "python_executable": sys.executable, "usb_port": port,
        "assets": {key: str(value) for key, value in assets.items()},
        "asset_sha256": asset_hashes, "input_files_sha256": sha256_file(input_path),
        "legacy_runner_sha256": sha256_file(LEGACY_SCRIPT),
        "launcher_sha256": sha256_file(Path(__file__)),
    }
    write_json(receipt_path, receipt)
    command = [sys.executable, "-X", "utf8", "-u", str(LEGACY_SCRIPT)]
    for key in ("nef", "nef_freeze", "physical_smoke_freeze", "holdout_manifest", "image_root"):
        command.extend(["--" + key.replace("_", "-"), str(assets[key])])
    command.extend([
        "--output-dir", str(hardware_dir), "--port", str(port),
        "--timeout-ms", str(args.timeout_ms),
    ])
    print("Results: {}".format(output_dir), flush=True)
    try:
        uploads = run_worker(command, output_dir / "hardware_run.log")
        summary, predictions = verify_worker_result(legacy, hardware_dir, rows, port)
        if inputs != input_file_records(rows, assets["image_root"]):
            raise RuntimeError("Input files changed during the hardware replay")
        comparison = {"status": "UNAVAILABLE", "reason": "Historical prediction CSV is absent"}
        if asset_hashes["historical_predictions"]:
            if sha256_file(assets["historical_predictions"]) != HISTORICAL_PREDICTIONS_SHA256:
                raise RuntimeError("Historical prediction CSV changed during replay")
            comparison = compare_predictions(predictions, read_csv(assets["historical_predictions"]))
        statuses = {"PASS": "COMPLETE", "FAIL": "NUMERICAL_MISMATCH",
                    "UNAVAILABLE": "COMPLETE_UNCOMPARED"}
        receipt.update({
            "status": statuses[comparison["status"]],
            "hardware_execution_completed": True,
            "numerical_repeatability_verified": comparison["status"] == "PASS",
            "verified_completed_inferences": len(predictions), "completed_model_uploads": uploads,
            "historical_comparison": comparison,
            "mean_hardware_send_receive_ms": summary["mean_hardware_send_receive_ms"],
            "p95_hardware_send_receive_ms": summary["p95_hardware_send_receive_ms"],
            "worker_summary_sha256": sha256_file(hardware_dir / "physical_holdout_inference_summary.json"),
            "predictions_sha256": summary["physical_nucleus_predictions_sha256"],
        })
    except BaseException as exc:
        receipt.update({"status": "FAILED", "error": "{}: {}".format(type(exc).__name__, exc)})
        write_json(receipt_path, receipt)
        raise
    receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(receipt_path, receipt)
    print("\nFRESH KL720 REPLAY COMPLETE: {} real inferences; {} model upload.".format(
        len(predictions), uploads
    ))
    print("Historical numerical repeatability: {}".format(comparison["status"]))
    print("Receipt: {}".format(receipt_path))
    print("This replay does not create a new biological accuracy claim or change the frozen model.")
    return 2 if comparison["status"] == "FAIL" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print("NASA KL720 run stopped: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
