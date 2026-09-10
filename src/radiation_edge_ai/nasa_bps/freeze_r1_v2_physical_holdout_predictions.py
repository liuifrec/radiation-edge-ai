"""Freeze the complete outcome-blind NASA BPS R1 v2 physical KL720 holdout predictions.

This stage performs no inference and reads no phenotype/reference outcomes. It
pins the exact 2,058-nucleus prediction CSV produced on the physical KL720,
verifies its provenance against the frozen QC1 holdout manifest, and writes an
immutable prediction-set freeze before any biological targets or gate results
are introduced.

The next stage may evaluate the already-frozen physical predictions against the
same seven predeclared biological deployment gates. PTQ/BIE/NEF retuning remains
unauthorized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FROZEN_NEF_SHA256 = (
    "d04f02855a8a82ae7cb5ff48af91eee69c341142d1097469890385bd23ee097b"
)
FROZEN_NEF_DEPLOYMENT_FREEZE_SHA256 = (
    "f818b9d912017a1d8fdc4aafd4bf321b8bf5b84e8214e6ef0ba029b48b0b90bb"
)
FROZEN_PHYSICAL_SMOKE_PASS_FREEZE_SHA256 = (
    "224b754235ebcd22eb053ca1f279043f3b74e9e3b461ed1f8dad26beddfa679d"
)
FROZEN_QC1_HOLDOUT_MANIFEST_SHA256 = (
    "f89962f96bec05930c07149717a90bd682016a2778bbfae6b827425159aa0643"
)
FROZEN_PHYSICAL_PREDICTIONS_SHA256 = (
    "91ea895a16cd97067af63b5c5c4434e972629bfda4c9582ad456a79e007b6d46"
)
FROZEN_PHYSICAL_INFERENCE_SUMMARY_SHA256 = (
    "7147ba4de4ba44fa64ed6b414e990970cd08ac83ea5e4b993ed38e9ad0a93aba"
)

EXPECTED_PLATFORM = 720
EXPECTED_MODEL_ID = 32770
EXPECTED_MODEL_VERSION = "8b29"
EXPECTED_USB_PORT = 81
EXPECTED_NUCLEI = 2058
EXPECTED_BAGS = 22
EXPECTED_SOURCE_NUCLEI = {"BALBCF2": 858, "C57BLF2": 600, "C57BLF3": 600}
EXPECTED_OUTPUT_SEMANTICS = "scalar latent continuous 53BP1 burden; not per-nucleus focus count"
EXPECTED_ROW_LABEL_SEMANTICS = "latent_continuous_burden_not_individually_supervised"
EXPECTED_STATUS = "COMPLETE_OUTCOME_BLIND_FULL_PHYSICAL_KL720_HOLDOUT_INFERENCE"
EXPECTED_NEXT_STAGE = "FREEZE_OUTCOME_BLIND_PHYSICAL_PREDICTIONS_THEN_EVALUATE_SEVEN_BIOLOGICAL_GATES"
EXPECTED_HOLDOUT_STATUS = "FINAL_BLINDED_DO_NOT_JOIN_PHENOTYPES"

FORBIDDEN_PREDICTION_COLUMNS = (
    "avg_nfoci",
    "nfoci",
    "phenotype",
    "ground_truth",
    "target",
    "reference_output",
    "fp32_burden",
    "bie_burden",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    )
    model_root = Path(
        os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    )
    nef_root = model_root / "nasa_bps_53bp1" / "r1_v2_deployment" / "kneron_nef"
    physical_root = nef_root / "physical_holdout_hardware"
    manifest_default = (
        data_root
        / "nasa_bps_microscopy"
        / "metadata"
        / "r1_v2_freeze"
        / "r1_v2_final_holdout_manifest_blinded_qc1.csv"
    )

    parser.add_argument(
        "--predictions",
        default=str(physical_root / "physical_holdout_nucleus_predictions.csv"),
    )
    parser.add_argument(
        "--summary",
        default=str(physical_root / "physical_holdout_inference_summary.json"),
    )
    parser.add_argument("--holdout-manifest", default=str(manifest_default))
    args = parser.parse_args()

    prediction_path = Path(args.predictions).resolve()
    summary_path = Path(args.summary).resolve()
    manifest_path = Path(args.holdout_manifest).resolve()
    freeze_path = physical_root / "physical_holdout_predictions_freeze.json"

    prediction_sha = check_sha(
        prediction_path,
        FROZEN_PHYSICAL_PREDICTIONS_SHA256,
        "Physical KL720 holdout predictions",
    )
    summary_sha = check_sha(
        summary_path,
        FROZEN_PHYSICAL_INFERENCE_SUMMARY_SHA256,
        "Physical KL720 holdout inference summary",
    )
    manifest_sha = check_sha(
        manifest_path,
        FROZEN_QC1_HOLDOUT_MANIFEST_SHA256,
        "QC1 holdout manifest",
    )

    summary = read_json(summary_path)
    required_summary = {
        "status": EXPECTED_STATUS,
        "nef_sha256": FROZEN_NEF_SHA256,
        "nef_deployment_freeze_sha256": FROZEN_NEF_DEPLOYMENT_FREEZE_SHA256,
        "physical_smoke_pass_freeze_sha256": FROZEN_PHYSICAL_SMOKE_PASS_FREEZE_SHA256,
        "qc1_holdout_manifest_sha256": FROZEN_QC1_HOLDOUT_MANIFEST_SHA256,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version_frozen": EXPECTED_MODEL_VERSION,
        "usb_port": EXPECTED_USB_PORT,
        "nuclei": EXPECTED_NUCLEI,
        "bags": EXPECTED_BAGS,
        "source_nuclei": EXPECTED_SOURCE_NUCLEI,
        "mask_used_as_model_input": False,
        "raw_phenotype_table_read": False,
        "fp32_or_bie_holdout_reference_outputs_read": False,
        "biological_targets_or_gate_results_read": False,
        "ptq_bie_or_nef_changed": False,
        "post_holdout_ptq_retuning_authorized": False,
        "output_semantics": EXPECTED_OUTPUT_SEMANTICS,
        "physical_nucleus_predictions_sha256": FROZEN_PHYSICAL_PREDICTIONS_SHA256,
        "authorized_next_stage": EXPECTED_NEXT_STAGE,
    }
    for key, expected in required_summary.items():
        if summary.get(key) != expected:
            raise RuntimeError(
                f"Physical inference summary invariant mismatch for {key}: "
                f"expected {expected!r}, got {summary.get(key)!r}"
            )

    for key in (
        "physical_burden_min",
        "physical_burden_max",
        "physical_burden_mean",
        "model_load_ms",
        "mean_host_pack_ms",
        "mean_hardware_send_receive_ms",
        "median_hardware_send_receive_ms",
        "p95_hardware_send_receive_ms",
        "elapsed_seconds",
    ):
        value = float(summary.get(key))
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite summary value for {key}: {value}")

    predictions = read_csv(prediction_path)
    manifest = read_csv(manifest_path)
    if len(predictions) != EXPECTED_NUCLEI or len(manifest) != EXPECTED_NUCLEI:
        raise RuntimeError(
            f"Expected {EXPECTED_NUCLEI} rows; got predictions={len(predictions)} manifest={len(manifest)}"
        )

    columns = set(predictions[0])
    forbidden = sorted(
        column
        for column in columns
        if any(token in column.lower() for token in FORBIDDEN_PREDICTION_COLUMNS)
    )
    if forbidden:
        raise RuntimeError(
            "Physical prediction CSV contains forbidden outcome/reference columns: "
            + ", ".join(forbidden)
        )

    required_prediction_columns = {
        "sample_id",
        "nucleus_key",
        "sample_name",
        "source_name",
        "strain",
        "sex",
        "particle_type",
        "dose_Gy",
        "hr_post_exposure",
        "physical_kl720_burden",
        "host_pack_ms",
        "hardware_send_receive_ms",
        "quantized_min",
        "quantized_max",
        "label_semantics",
    }
    missing = sorted(required_prediction_columns - columns)
    if missing:
        raise RuntimeError(f"Physical prediction CSV missing columns: {missing}")

    if len({row["sample_id"] for row in predictions}) != EXPECTED_NUCLEI:
        raise RuntimeError("Duplicate sample_id values in physical prediction CSV")
    if len({row["nucleus_key"] for row in predictions}) != EXPECTED_NUCLEI:
        raise RuntimeError("Duplicate nucleus_key values in physical prediction CSV")

    manifest_by_id = {row["sample_id"]: row for row in manifest}
    if len(manifest_by_id) != EXPECTED_NUCLEI:
        raise RuntimeError("Duplicate sample_id values in QC1 holdout manifest")
    if {row.get("holdout_status") for row in manifest} != {EXPECTED_HOLDOUT_STATUS}:
        raise RuntimeError("QC1 holdout status changed")

    metadata_fields = (
        "nucleus_key",
        "sample_name",
        "source_name",
        "strain",
        "sex",
        "particle_type",
    )
    values = []
    source_counts = Counter()
    bag_names = set()
    for row in predictions:
        sample_id = row["sample_id"]
        if sample_id not in manifest_by_id:
            raise RuntimeError(f"Physical prediction sample_id absent from QC1 manifest: {sample_id}")
        ref = manifest_by_id[sample_id]
        for field in metadata_fields:
            if row[field] != ref[field]:
                raise RuntimeError(
                    f"Prediction/QC1 metadata mismatch for {sample_id} field {field}: "
                    f"{row[field]!r} != {ref[field]!r}"
                )
        if row["label_semantics"] != EXPECTED_ROW_LABEL_SEMANTICS:
            raise RuntimeError(f"Unexpected row label semantics for {sample_id}")
        burden = float(row["physical_kl720_burden"])
        pack_ms = float(row["host_pack_ms"])
        hw_ms = float(row["hardware_send_receive_ms"])
        if not all(math.isfinite(v) for v in (burden, pack_ms, hw_ms)):
            raise RuntimeError(f"Non-finite physical row values for {sample_id}")
        values.append(burden)
        source_counts[row["source_name"]] += 1
        bag_names.add(row["sample_name"])

    if dict(source_counts) != EXPECTED_SOURCE_NUCLEI:
        raise RuntimeError(
            f"Physical prediction source counts changed: {dict(source_counts)}"
        )
    if len(bag_names) != EXPECTED_BAGS:
        raise RuntimeError(f"Physical prediction bags={len(bag_names)}; expected {EXPECTED_BAGS}")

    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_OUTCOME_BLIND_FULL_PHYSICAL_KL720_HOLDOUT_PREDICTIONS_BEFORE_BIOLOGICAL_EVALUATION",
        "nef_sha256": FROZEN_NEF_SHA256,
        "nef_deployment_freeze_sha256": FROZEN_NEF_DEPLOYMENT_FREEZE_SHA256,
        "physical_smoke_pass_freeze_sha256": FROZEN_PHYSICAL_SMOKE_PASS_FREEZE_SHA256,
        "qc1_holdout_manifest_sha256": manifest_sha,
        "physical_holdout_inference_summary_sha256": summary_sha,
        "physical_holdout_nucleus_predictions_sha256": prediction_sha,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "usb_port": EXPECTED_USB_PORT,
        "nuclei": EXPECTED_NUCLEI,
        "bags": EXPECTED_BAGS,
        "source_nuclei": EXPECTED_SOURCE_NUCLEI,
        "output_semantics": EXPECTED_OUTPUT_SEMANTICS,
        "physical_burden_min_descriptive": float(min(values)),
        "physical_burden_max_descriptive": float(max(values)),
        "mean_hardware_send_receive_ms_descriptive": float(summary["mean_hardware_send_receive_ms"]),
        "median_hardware_send_receive_ms_descriptive": float(summary["median_hardware_send_receive_ms"]),
        "p95_hardware_send_receive_ms_descriptive": float(summary["p95_hardware_send_receive_ms"]),
        "raw_phenotype_table_read_by_freeze": False,
        "fp32_or_bie_holdout_reference_outputs_read_by_freeze": False,
        "biological_targets_or_gate_results_read_by_freeze": False,
        "ptq_bie_or_nef_changed": False,
        "post_holdout_ptq_retuning_authorized": False,
        "authorized_next_stage": "EVALUATE_FROZEN_PHYSICAL_PREDICTIONS_AGAINST_ORIGINAL_SEVEN_BIOLOGICAL_GATES",
    }

    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    if freeze_path.exists():
        existing = read_json(freeze_path)
        old = dict(existing)
        new = dict(freeze)
        old.pop("created_utc", None)
        new.pop("created_utc", None)
        if old != new:
            raise RuntimeError(
                f"Existing physical holdout prediction freeze differs from accepted result: {freeze_path}"
            )
        print("Radiation Edge AI - NASA BPS R1 v2 physical holdout prediction freeze")
        print("existing outcome-blind physical prediction freeze verified without overwrite: YES")
    else:
        freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
        print("Radiation Edge AI - NASA BPS R1 v2 physical holdout prediction freeze")
        print("new outcome-blind physical prediction freeze written: YES")

    print(f"NEF SHA256: {FROZEN_NEF_SHA256}")
    print(f"physical smoke PASS freeze SHA256: {FROZEN_PHYSICAL_SMOKE_PASS_FREEZE_SHA256}")
    print(f"QC1 holdout manifest SHA256: {manifest_sha}")
    print(f"physical prediction CSV SHA256: {prediction_sha}")
    print(f"physical inference summary SHA256: {summary_sha}")
    print(f"nuclei/bags: {EXPECTED_NUCLEI}/{EXPECTED_BAGS}")
    print("prediction/QC1 identity and metadata audit: PASS")
    print("raw phenotype/reference outcomes read by freeze: NO")
    print("biological targets/gates evaluated by freeze: NO")
    print("post-holdout PTQ retuning authorized: NO")
    print(f"freeze: {freeze_path}")
    print(f"freeze SHA256: {sha256_file(freeze_path)}")
    print("NASA BPS R1 V2 OUTCOME-BLIND PHYSICAL KL720 HOLDOUT PREDICTIONS FROZEN: YES")
    print("NEXT GATE: evaluate this exact frozen physical prediction set against the original seven biological gates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
