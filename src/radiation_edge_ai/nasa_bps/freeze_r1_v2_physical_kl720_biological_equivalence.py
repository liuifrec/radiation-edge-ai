"""Freeze the final NASA BPS R1 v2 physical KL720 biological-equivalence PASS.

This stage performs no inference, no preprocessing, no quantization, and does
not reopen the raw NASA phenotype table. It pins the exact terminal physical
biological-equivalence summary produced only after the complete physical KL720
prediction set had already been frozen outcome-blind.

The frozen decision is the original seven-gate biological deployment-equivalence
result. Physical-vs-BIE numerical differences are descriptive only and do not
create a post-hoc acceptance threshold. No holdout-guided model/PTQ/BIE/NEF or
threshold retuning is authorized after this result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FROZEN_FINAL_PHYSICAL_EQUIVALENCE_SUMMARY_SHA256 = (
    "8d236ac3cee264975d0bb3de2397f15063cb213070956f45ce174baa71d69786"
)
FROZEN_PHYSICAL_PREDICTIONS_FREEZE_SHA256 = (
    "e4e7d9150e2e33479e2ec8894e583f0d09a3d890a6a97fbd356a1a0fb78d2b30"
)
FROZEN_PHYSICAL_PREDICTIONS_SHA256 = (
    "91ea895a16cd97067af63b5c5c4434e972629bfda4c9582ad456a79e007b6d46"
)
FROZEN_PHYSICAL_INFERENCE_SUMMARY_SHA256 = (
    "7147ba4de4ba44fa64ed6b414e990970cd08ac83ea5e4b993ed38e9ad0a93aba"
)
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
FROZEN_FP32_REFERENCE_FREEZE_SHA256 = (
    "abb58026d4f798554ef7e44346a989499242fcbd89d8c6629a1c4209d6c16ce3"
)
FROZEN_BIE_EQUIVALENCE_FREEZE_SHA256 = (
    "e901d4be044c5ea5507b3b60643043eb8aa0af5643a0b7fba2f9bdd7c588bb0a"
)
FROZEN_BIE_EQUIVALENCE_SUMMARY_SHA256 = (
    "c15776c6820a4aa28fff8e18c16b576164ce3ed2cad61ba2057efdef79800ca4"
)

EXPECTED_STATUS = "COMPLETE_FROZEN_PHYSICAL_KL720_BIOLOGICAL_DEPLOYMENT_EQUIVALENCE"
EXPECTED_PLATFORM = 720
EXPECTED_MODEL_ID = 32770
EXPECTED_MODEL_VERSION = "8b29"
EXPECTED_USB_PORT = 81
EXPECTED_NUCLEI = 2058
EXPECTED_BAGS = 22
EXPECTED_CONTRASTS = 11
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
EXPECTED_GATE_CHECKS = {
    "bag_spearman_ge_0_50",
    "delta_spearman_ge_0_60",
    "overall_direction_ge_9_of_11",
    "four_hour_direction_eq_4_of_4",
    "late_24_48_direction_ge_5_of_7",
    "peak_time_recovery_eq_3_of_3",
    "every_source_direction_minimum",
}
EXPECTED_OUTPUT_SEMANTICS = "scalar latent continuous 53BP1 burden; not per-nucleus focus count"


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


def check_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    model_root = Path(
        os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    )
    physical_root = (
        model_root
        / "nasa_bps_53bp1"
        / "r1_v2_deployment"
        / "kneron_nef"
        / "physical_holdout_hardware"
    )
    biological_root = physical_root / "biological_equivalence"

    parser.add_argument(
        "--summary",
        default=str(biological_root / "physical_holdout_biological_equivalence_summary.json"),
    )
    args = parser.parse_args()

    summary_path = Path(args.summary).resolve()
    freeze_path = biological_root / "physical_holdout_biological_equivalence_freeze.json"

    summary_sha = check_sha(
        summary_path,
        FROZEN_FINAL_PHYSICAL_EQUIVALENCE_SUMMARY_SHA256,
        "Final physical biological-equivalence summary",
    )
    summary = read_json(summary_path)

    required = {
        "status": EXPECTED_STATUS,
        "physical_holdout_predictions_freeze_sha256": FROZEN_PHYSICAL_PREDICTIONS_FREEZE_SHA256,
        "physical_holdout_nucleus_predictions_sha256": FROZEN_PHYSICAL_PREDICTIONS_SHA256,
        "physical_holdout_inference_summary_sha256": FROZEN_PHYSICAL_INFERENCE_SUMMARY_SHA256,
        "nef_sha256": FROZEN_NEF_SHA256,
        "nef_deployment_freeze_sha256": FROZEN_NEF_DEPLOYMENT_FREEZE_SHA256,
        "physical_smoke_pass_freeze_sha256": FROZEN_PHYSICAL_SMOKE_PASS_FREEZE_SHA256,
        "qc1_holdout_manifest_sha256": FROZEN_QC1_HOLDOUT_MANIFEST_SHA256,
        "fp32_deployment_reference_freeze_sha256": FROZEN_FP32_REFERENCE_FREEZE_SHA256,
        "bie_equivalence_freeze_sha256": FROZEN_BIE_EQUIVALENCE_FREEZE_SHA256,
        "bie_equivalence_summary_sha256": FROZEN_BIE_EQUIVALENCE_SUMMARY_SHA256,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "usb_port": EXPECTED_USB_PORT,
        "nuclei": EXPECTED_NUCLEI,
        "bags": EXPECTED_BAGS,
        "matched_contrasts": EXPECTED_CONTRASTS,
        "output_semantics": EXPECTED_OUTPUT_SEMANTICS,
        "raw_phenotype_table_read": False,
        "physical_predictions_frozen_outcome_blind_before_this_evaluation": True,
        "ptq_bie_nef_or_model_changed_after_physical_predictions": False,
        "post_holdout_ptq_retuning_authorized": False,
        "all_seven_original_biological_gates_pass": True,
        "exact_fp32_direction_peak_signature_reproduced": True,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise RuntimeError(
                f"Final physical equivalence invariant mismatch for {key}: "
                f"expected {expected!r}, got {summary.get(key)!r}"
            )

    gate = summary.get("biological_gate")
    if not isinstance(gate, dict):
        raise RuntimeError("Final summary missing biological_gate")
    checks = gate.get("checks")
    if not isinstance(checks, dict):
        raise RuntimeError("Final summary missing gate checks")
    if set(checks) != EXPECTED_GATE_CHECKS or not all(value is True for value in checks.values()):
        raise RuntimeError(f"Expected exactly seven PASS biological gates; got {checks!r}")
    if gate.get("direction_counts") != EXPECTED_DIRECTION_COUNTS:
        raise RuntimeError(
            f"Physical direction signature mismatch: {gate.get('direction_counts')!r}"
        )
    if gate.get("peak_time_recovery") != EXPECTED_PEAK:
        raise RuntimeError(
            f"Physical peak-time recovery mismatch: {gate.get('peak_time_recovery')!r}"
        )
    if gate.get("all_pass") is not True:
        raise RuntimeError("Final physical biological gate is not PASS")

    bag_metrics = gate.get("bag_metrics") or {}
    delta_metrics = gate.get("delta_metrics") or {}
    if float(bag_metrics.get("spearman")) < 0.50:
        raise RuntimeError("Physical bag Spearman fails frozen >=0.50 gate")
    if float(delta_metrics.get("delta_spearman")) < 0.60:
        raise RuntimeError("Physical delta Spearman fails frozen >=0.60 gate")

    physical_vs_bie_nucleus = summary.get(
        "physical_vs_bie_nucleus_descriptive_only_no_posthoc_gate"
    )
    physical_vs_bie_bag = summary.get(
        "physical_vs_bie_bag_descriptive_only_no_posthoc_gate"
    )
    if not isinstance(physical_vs_bie_nucleus, dict) or not isinstance(physical_vs_bie_bag, dict):
        raise RuntimeError("Missing descriptive physical-vs-BIE numerical comparison")

    outputs = summary.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        "physical_holdout_bag_predictions.csv",
        "physical_holdout_matched_radiation_deltas.csv",
        "physical_holdout_peak_time_recovery.csv",
    }:
        raise RuntimeError(f"Unexpected final physical output set: {outputs!r}")

    for name, meta in outputs.items():
        if not isinstance(meta, dict):
            raise RuntimeError(f"Malformed output metadata for {name}")
        output_path = biological_root / name
        expected_sha = str(meta.get("sha256", ""))
        if len(expected_sha) != 64:
            raise RuntimeError(f"Missing output SHA256 for {name}")
        check_sha(output_path, expected_sha, name)
        if int(meta.get("size_bytes", -1)) != output_path.stat().st_size:
            raise RuntimeError(f"Output size mismatch for {name}")

    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_FINAL_PHYSICAL_KL720_BIOLOGICAL_DEPLOYMENT_EQUIVALENCE_PASS",
        "final_physical_biological_equivalence_summary_sha256": summary_sha,
        "physical_holdout_predictions_freeze_sha256": FROZEN_PHYSICAL_PREDICTIONS_FREEZE_SHA256,
        "physical_holdout_nucleus_predictions_sha256": FROZEN_PHYSICAL_PREDICTIONS_SHA256,
        "physical_holdout_inference_summary_sha256": FROZEN_PHYSICAL_INFERENCE_SUMMARY_SHA256,
        "nef_sha256": FROZEN_NEF_SHA256,
        "nef_deployment_freeze_sha256": FROZEN_NEF_DEPLOYMENT_FREEZE_SHA256,
        "physical_smoke_pass_freeze_sha256": FROZEN_PHYSICAL_SMOKE_PASS_FREEZE_SHA256,
        "qc1_holdout_manifest_sha256": FROZEN_QC1_HOLDOUT_MANIFEST_SHA256,
        "fp32_deployment_reference_freeze_sha256": FROZEN_FP32_REFERENCE_FREEZE_SHA256,
        "bie_equivalence_freeze_sha256": FROZEN_BIE_EQUIVALENCE_FREEZE_SHA256,
        "bie_equivalence_summary_sha256": FROZEN_BIE_EQUIVALENCE_SUMMARY_SHA256,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "usb_port": EXPECTED_USB_PORT,
        "nuclei": EXPECTED_NUCLEI,
        "bags": EXPECTED_BAGS,
        "matched_contrasts": EXPECTED_CONTRASTS,
        "observed_bag_metrics": bag_metrics,
        "observed_delta_metrics": delta_metrics,
        "direction_counts": EXPECTED_DIRECTION_COUNTS,
        "peak_time_recovery": EXPECTED_PEAK,
        "all_seven_original_biological_gates_pass": True,
        "exact_fp32_direction_peak_signature_reproduced": True,
        "physical_vs_bie_nucleus_descriptive_only_no_posthoc_gate": physical_vs_bie_nucleus,
        "physical_vs_bie_bag_descriptive_only_no_posthoc_gate": physical_vs_bie_bag,
        "final_output_files": outputs,
        "physical_predictions_frozen_outcome_blind_before_biological_evaluation": True,
        "raw_nasa_phenotype_table_reopened_by_final_physical_evaluation": False,
        "protocol_deviation_carried_forward": "documented non-outcome metadata-normalization implementation recovery",
        "holdout_guided_retuning_authorized": False,
        "model_ptq_bie_nef_or_threshold_changes_authorized": False,
        "deployment_conclusion": "PHYSICAL_KL720_BIOLOGICAL_DEPLOYMENT_EQUIVALENCE_PASS_UNDER_ORIGINAL_SEVEN_PREDECLARED_GATES",
        "authorized_next_stage": "PACKAGE_VALIDATION_RECORD_AND_RELEASE_CANDIDATE_WITH_NO_MODEL_OR_THRESHOLD_CHANGES",
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
                f"Existing final physical equivalence freeze differs from accepted result: {freeze_path}"
            )
        print("Radiation Edge AI - NASA BPS R1 v2 final physical KL720 biological-equivalence freeze")
        print("existing final physical equivalence freeze verified without overwrite: YES")
    else:
        freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
        print("Radiation Edge AI - NASA BPS R1 v2 final physical KL720 biological-equivalence freeze")
        print("new final physical equivalence freeze written: YES")

    print(f"final physical equivalence summary SHA256: {summary_sha}")
    print(f"physical prediction freeze SHA256: {FROZEN_PHYSICAL_PREDICTIONS_FREEZE_SHA256}")
    print(f"NEF SHA256: {FROZEN_NEF_SHA256}")
    print(
        "bag MAE/RMSE/Pearson/Spearman: "
        f"{float(bag_metrics['mae']):.4f} / {float(bag_metrics['rmse']):.4f} / "
        f"{float(bag_metrics['pearson']):.4f} / {float(bag_metrics['spearman']):.4f}"
    )
    print(
        "delta MAE/Pearson/Spearman: "
        f"{float(delta_metrics['delta_mae']):.4f} / {float(delta_metrics['delta_pearson']):.4f} / "
        f"{float(delta_metrics['delta_spearman']):.4f}"
    )
    print("direction signature: 10/11 overall; 4/4 at 4 h; 6/7 at 24+48 h")
    print("source signature: BALBCF2 4/5; C57BLF2 3/3; C57BLF3 3/3")
    print("peak-time recovery: 3/3")
    print("all seven original biological gates: PASS")
    print("exact FP32 direction/peak signature reproduced: YES")
    print(
        "physical-vs-BIE nucleus max abs error (descriptive only): "
        f"{float(physical_vs_bie_nucleus['max_abs_error']):.10g}"
    )
    print(
        "physical-vs-BIE bag max abs error (descriptive only): "
        f"{float(physical_vs_bie_bag['max_abs_error']):.10g}"
    )
    print("raw NASA phenotype table reopened by this freeze: NO")
    print("holdout-guided retuning authorized: NO")
    print(f"freeze: {freeze_path}")
    print(f"freeze SHA256: {sha256_file(freeze_path)}")
    print("NASA BPS R1 V2 FINAL PHYSICAL KL720 BIOLOGICAL EQUIVALENCE FROZEN: YES")
    print("NEXT STAGE: package the validation record/release candidate; no model/PTQ/threshold changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
