"""Freeze the successful NASA BPS R1 v2 FP32 confirmation as deployment reference.

This script does not rerun inference and does not read the raw phenotype table.
It verifies the authoritative checkpoint/manifest identities and the completed
confirmatory recovery outputs, then records SHA256 identities for the exact
JSON/CSV result set that ONNX, INT8/BIE, and physical KL720 inference must use
as their immutable FP32 deployment reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FROZEN_REPAIRED_CHECKPOINT_SHA256 = (
    "2b07c67d6e50e7a36e26e0052feebc826b518e6efae863ba312ba9761a4eb3d5"
)
FROZEN_QC1_HOLDOUT_MANIFEST_SHA256 = (
    "f89962f96bec05930c07149717a90bd682016a2778bbfae6b827425159aa0643"
)
FROZEN_RECOVERY_EVALUATOR_RUNTIME_SHA256 = (
    "62af2d2db8c47e287c49c422ce26975b34d8241875340034f4eca86e9941937f"
)
EXPECTED_EVALUATION_STATUS = (
    "COMPLETE_CONFIRMATORY_HOLDOUT_WITH_DOCUMENTED_METADATA_ALIAS_RECOVERY"
)
EXPECTED_DIRECTION_COUNTS = {
    "overall": "10/11",
    "4h": "4/4",
    "24+48h": "6/7",
    "by_source": {
        "BALBCF2": "4/5",
        "C57BLF2": "3/3",
        "C57BLF3": "3/3",
    },
}
EXPECTED_PEAK = "3/3"

REFERENCE_FILES = (
    "final_holdout_confirmation_summary.json",
    "FINAL_HOLDOUT_CONFIRMATION_COMPLETE.json",
    "FINAL_HOLDOUT_RECOVERY_COMPLETE.json",
    "final_holdout_bag_predictions.csv",
    "final_holdout_nucleus_latent_burden.csv",
    "final_holdout_matched_radiation_deltas.csv",
    "final_holdout_direction_summary.csv",
    "final_holdout_peak_time_recovery.csv",
    "final_holdout_source_residuals.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def validate_summary(summary: dict[str, Any]) -> None:
    if summary.get("evaluation_status") != EXPECTED_EVALUATION_STATUS:
        raise RuntimeError(
            "Unexpected confirmation status: "
            f"{summary.get('evaluation_status')!r}"
        )
    if summary.get("protocol_deviation") is not True:
        raise RuntimeError("Documented metadata-alias recovery provenance is missing")
    if (
        summary.get("recovery_evaluator_source_sha256")
        != FROZEN_RECOVERY_EVALUATOR_RUNTIME_SHA256
    ):
        raise RuntimeError(
            "Recovery evaluator identity mismatch: "
            f"{summary.get('recovery_evaluator_source_sha256')!r}"
        )
    if summary.get("repaired_checkpoint_sha256") != FROZEN_REPAIRED_CHECKPOINT_SHA256:
        raise RuntimeError("Summary checkpoint identity mismatch")
    if summary.get("qc1_holdout_manifest_sha256") != FROZEN_QC1_HOLDOUT_MANIFEST_SHA256:
        raise RuntimeError("Summary QC1 manifest identity mismatch")
    if summary.get("no_holdout_outcome_used_for_tuning") is not True:
        raise RuntimeError("Summary no-tuning invariant is not TRUE")
    if summary.get("model_retrained_after_unblinding") is not False:
        raise RuntimeError("Summary reports model retraining after unblinding")
    if summary.get("model_recalibrated_after_unblinding") is not False:
        raise RuntimeError("Summary reports model recalibration after unblinding")
    if summary.get("thresholds_changed_after_unblinding") is not False:
        raise RuntimeError("Summary reports threshold changes after unblinding")
    if int(summary.get("bag_count", -1)) != 22:
        raise RuntimeError(f"Unexpected bag count: {summary.get('bag_count')!r}")
    if int(summary.get("nucleus_count", -1)) != 2058:
        raise RuntimeError(f"Unexpected nucleus count: {summary.get('nucleus_count')!r}")
    if int(summary.get("matched_contrast_count", -1)) != 11:
        raise RuntimeError(
            f"Unexpected matched contrast count: {summary.get('matched_contrast_count')!r}"
        )

    gate = summary.get("gate_result")
    if not isinstance(gate, dict) or gate.get("all_pass") is not True:
        raise RuntimeError("Final FP32 confirmation did not record all seven gates PASS")
    if gate.get("direction_counts") != EXPECTED_DIRECTION_COUNTS:
        raise RuntimeError(
            "FP32 direction signature differs from frozen successful confirmation: "
            f"{gate.get('direction_counts')!r}"
        )
    if gate.get("peak_time_recovery") != EXPECTED_PEAK:
        raise RuntimeError(
            "FP32 peak-time signature differs from frozen successful confirmation: "
            f"{gate.get('peak_time_recovery')!r}"
        )
    checks = gate.get("checks")
    if not isinstance(checks, dict) or len(checks) != 7 or not all(
        value is True for value in checks.values()
    ):
        raise RuntimeError(f"Expected exactly seven PASS gate checks; got {checks!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"

    parser.add_argument(
        "--checkpoint",
        default=str(
            model_root
            / "nasa_bps_53bp1"
            / "r1_v2_final_candidate"
            / "r1_countnet_v2_final_bn_recalibrated.pt"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(
            metadata_root
            / "r1_v2_freeze"
            / "r1_v2_final_holdout_manifest_blinded_qc1.csv"
        ),
    )
    parser.add_argument(
        "--confirmation-dir",
        default=str(
            model_root
            / "nasa_bps_53bp1"
            / "r1_v2_final_holdout_confirmation"
        ),
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    confirmation_dir = Path(args.confirmation_dir).resolve()
    freeze_path = confirmation_dir / "fp32_deployment_reference_freeze.json"

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not confirmation_dir.is_dir():
        raise FileNotFoundError(confirmation_dir)

    checkpoint_sha = sha256_file(checkpoint)
    manifest_sha = sha256_file(manifest)
    if checkpoint_sha != FROZEN_REPAIRED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Authoritative FP32 SHA256 mismatch: expected {FROZEN_REPAIRED_CHECKPOINT_SHA256}, got {checkpoint_sha}"
        )
    if manifest_sha != FROZEN_QC1_HOLDOUT_MANIFEST_SHA256:
        raise RuntimeError(
            f"QC1 manifest SHA256 mismatch: expected {FROZEN_QC1_HOLDOUT_MANIFEST_SHA256}, got {manifest_sha}"
        )

    paths = {name: confirmation_dir / name for name in REFERENCE_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError("Missing completed confirmation outputs: " + ", ".join(missing))

    summary = read_json(paths["final_holdout_confirmation_summary.json"])
    validate_summary(summary)

    # Both completion markers were written from the same completed summary.
    for marker_name in (
        "FINAL_HOLDOUT_CONFIRMATION_COMPLETE.json",
        "FINAL_HOLDOUT_RECOVERY_COMPLETE.json",
    ):
        marker = read_json(paths[marker_name])
        if marker != summary:
            raise RuntimeError(f"Completion marker content differs from summary: {marker_name}")

    identities = {
        name: {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }

    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_FP32_DEPLOYMENT_REFERENCE_AFTER_CONFIRMATORY_PASS",
        "checkpoint_sha256": checkpoint_sha,
        "qc1_holdout_manifest_sha256": manifest_sha,
        "recovery_evaluator_runtime_sha256": FROZEN_RECOVERY_EVALUATOR_RUNTIME_SHA256,
        "protocol_deviation_carried_forward": (
            "documented non-outcome metadata-normalization implementation recovery"
        ),
        "biological_reference": {
            "nuclei": 2058,
            "bags": 22,
            "matched_contrasts": 11,
            "direction_counts": EXPECTED_DIRECTION_COUNTS,
            "peak_time_recovery": EXPECTED_PEAK,
            "all_seven_original_gates_pass": True,
        },
        "reference_files": identities,
        "raw_phenotype_reopened_by_freeze": False,
        "no_model_or_threshold_change": True,
        "authorized_next_stage": "FP32_ONNX_EXPORT_AND_PARITY",
    }

    if freeze_path.exists():
        existing = read_json(freeze_path)
        # Ignore timestamp only; all scientific identities must be identical.
        comparable_existing = dict(existing)
        comparable_new = dict(freeze)
        comparable_existing.pop("created_utc", None)
        comparable_new.pop("created_utc", None)
        if comparable_existing != comparable_new:
            raise RuntimeError(
                f"Existing deployment freeze differs from current authoritative outputs: {freeze_path}"
            )
        print("Radiation Edge AI - NASA BPS R1 v2 FP32 deployment reference")
        print("existing freeze verified without overwrite: YES")
    else:
        freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
        print("Radiation Edge AI - NASA BPS R1 v2 FP32 deployment reference")
        print("new deployment-reference freeze written: YES")

    print(f"repaired FP32 SHA256: {checkpoint_sha}")
    print(f"QC1 holdout manifest SHA256: {manifest_sha}")
    print(
        "confirmation output files frozen: "
        f"{len(identities)}/{len(REFERENCE_FILES)}"
    )
    print("FP32 direction signature: 10/11 overall; 4/4 at 4 h; 6/7 late")
    print("FP32 source signature: BALBCF2 4/5; C57BLF2 3/3; C57BLF3 3/3")
    print("FP32 peak-time recovery: 3/3")
    print("all seven original biological gates: PASS")
    print("raw phenotype table read by this freeze: NO")
    print(f"freeze: {freeze_path}")
    print(f"freeze SHA256: {sha256_file(freeze_path)}")
    print("NASA BPS R1 V2 FP32 DEPLOYMENT REFERENCE FROZEN: YES")
    print("NEXT GATE: export the exact checkpoint to fixed 1x3x256x256 ONNX and run parity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
