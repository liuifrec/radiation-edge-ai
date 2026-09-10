"""Package the frozen NASA BPS R1 v2 physical KL720 validation record.

This stage is packaging only. It performs no inference, preprocessing,
quantization, compilation, biological re-evaluation, or outcome-guided tuning.
It verifies the terminal physical biological-equivalence freeze and the key
upstream identities, records repository/document provenance, and writes one
release-candidate manifest for the completed NASA BPS R1 v2 KL720 track.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FINAL_PHYSICAL_FREEZE_SHA256 = (
    "0259ad745215dfb567a2edbd1593fb7dc789de6a4366dbcdfeb7234901b844bb"
)
FINAL_PHYSICAL_SUMMARY_SHA256 = (
    "8d236ac3cee264975d0bb3de2397f15063cb213070956f45ce174baa71d69786"
)
PHYSICAL_PREDICTIONS_FREEZE_SHA256 = (
    "e4e7d9150e2e33479e2ec8894e583f0d09a3d890a6a97fbd356a1a0fb78d2b30"
)
PHYSICAL_PREDICTIONS_SHA256 = (
    "91ea895a16cd97067af63b5c5c4434e972629bfda4c9582ad456a79e007b6d46"
)
PHYSICAL_INFERENCE_SUMMARY_SHA256 = (
    "7147ba4de4ba44fa64ed6b414e990970cd08ac83ea5e4b993ed38e9ad0a93aba"
)
NEF_SHA256 = (
    "d04f02855a8a82ae7cb5ff48af91eee69c341142d1097469890385bd23ee097b"
)
NEF_DEPLOYMENT_FREEZE_SHA256 = (
    "f818b9d912017a1d8fdc4aafd4bf321b8bf5b84e8214e6ef0ba029b48b0b90bb"
)
BIE_SHA256 = (
    "c52b8a78c595347667bad7a34950af92adbe734179b913646d06513a88fc1dd8"
)
BIE_EQUIVALENCE_FREEZE_SHA256 = (
    "e901d4be044c5ea5507b3b60643043eb8aa0af5643a0b7fba2f9bdd7c588bb0a"
)
FP32_REFERENCE_FREEZE_SHA256 = (
    "abb58026d4f798554ef7e44346a989499242fcbd89d8c6629a1c4209d6c16ce3"
)
QC1_MANIFEST_SHA256 = (
    "f89962f96bec05930c07149717a90bd682016a2778bbfae6b827425159aa0643"
)
ONNX_SHA256 = (
    "a65718a8f07d5bcd8707bae78d8f730d56047db21e575fa55ba5396adc49a07e"
)
OPTIMIZED_ONNX_SHA256 = (
    "c3a6aa5ed80280b0286f3b1aebf5f77f44dc60f569cc42fe2ab00ba7edde820c"
)
CALIBRATION_MANIFEST_SHA256 = (
    "78c9ed83e1992907264dfeec9ef6e2a434868bd3f299672f610a42a0018b5d4e"
)
CALIBRATION_FREEZE_SHA256 = (
    "f017f8a27563611e49fa55147920da2860b38fbb3f64a4c7db2995a36752a9df"
)
PINNED_TOOLCHAIN_IMAGE = (
    "kneron/toolchain@sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041"
)
TOOLCHAIN_VERSION = "kneron/toolchain:v0.33.1"
RELEASE_LABEL = "nasa-bps-r1-v2-kl720-rc1"

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


def git_text(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    )
    model_root = Path(
        os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    )
    repo_root = Path(__file__).resolve().parents[3]

    deployment_root = model_root / "nasa_bps_53bp1" / "r1_v2_deployment"
    nef_root = deployment_root / "kneron_nef"
    physical_root = nef_root / "physical_holdout_hardware"
    biological_root = physical_root / "biological_equivalence"
    confirmation_root = model_root / "nasa_bps_53bp1" / "r1_v2_final_holdout_confirmation"
    int8_root = deployment_root / "kneron_int8"

    parser.add_argument(
        "--final-freeze",
        default=str(biological_root / "physical_holdout_biological_equivalence_freeze.json"),
    )
    parser.add_argument(
        "--final-summary",
        default=str(biological_root / "physical_holdout_biological_equivalence_summary.json"),
    )
    parser.add_argument(
        "--physical-predictions-freeze",
        default=str(physical_root / "physical_holdout_predictions_freeze.json"),
    )
    parser.add_argument(
        "--physical-predictions",
        default=str(physical_root / "physical_holdout_nucleus_predictions.csv"),
    )
    parser.add_argument(
        "--physical-inference-summary",
        default=str(physical_root / "physical_holdout_inference_summary.json"),
    )
    parser.add_argument(
        "--nef",
        default=str(nef_root / "r1_countnet_v2_final_kl720.nef"),
    )
    parser.add_argument(
        "--nef-freeze",
        default=str(nef_root / "nef_deployment_candidate_freeze.json"),
    )
    parser.add_argument(
        "--bie",
        default=str(int8_root / "r1_countnet_v2_final_kl720_int8.bie"),
    )
    parser.add_argument(
        "--bie-equivalence-freeze",
        default=str(int8_root / "bie_holdout_equivalence" / "bie_holdout_equivalence_freeze.json"),
    )
    parser.add_argument(
        "--fp32-reference-freeze",
        default=str(confirmation_root / "fp32_deployment_reference_freeze.json"),
    )
    parser.add_argument(
        "--qc1-manifest",
        default=str(
            data_root
            / "nasa_bps_microscopy"
            / "metadata"
            / "r1_v2_freeze"
            / "r1_v2_final_holdout_manifest_blinded_qc1.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(biological_root / "release_candidate"),
    )
    args = parser.parse_args()

    paths = {
        "final_physical_equivalence_freeze": Path(args.final_freeze).resolve(),
        "final_physical_equivalence_summary": Path(args.final_summary).resolve(),
        "physical_predictions_freeze": Path(args.physical_predictions_freeze).resolve(),
        "physical_predictions": Path(args.physical_predictions).resolve(),
        "physical_inference_summary": Path(args.physical_inference_summary).resolve(),
        "nef": Path(args.nef).resolve(),
        "nef_deployment_freeze": Path(args.nef_freeze).resolve(),
        "bie": Path(args.bie).resolve(),
        "bie_equivalence_freeze": Path(args.bie_equivalence_freeze).resolve(),
        "fp32_reference_freeze": Path(args.fp32_reference_freeze).resolve(),
        "qc1_manifest": Path(args.qc1_manifest).resolve(),
    }

    expected_hashes = {
        "final_physical_equivalence_freeze": FINAL_PHYSICAL_FREEZE_SHA256,
        "final_physical_equivalence_summary": FINAL_PHYSICAL_SUMMARY_SHA256,
        "physical_predictions_freeze": PHYSICAL_PREDICTIONS_FREEZE_SHA256,
        "physical_predictions": PHYSICAL_PREDICTIONS_SHA256,
        "physical_inference_summary": PHYSICAL_INFERENCE_SUMMARY_SHA256,
        "nef": NEF_SHA256,
        "nef_deployment_freeze": NEF_DEPLOYMENT_FREEZE_SHA256,
        "bie": BIE_SHA256,
        "bie_equivalence_freeze": BIE_EQUIVALENCE_FREEZE_SHA256,
        "fp32_reference_freeze": FP32_REFERENCE_FREEZE_SHA256,
        "qc1_manifest": QC1_MANIFEST_SHA256,
    }
    verified_hashes = {
        key: check_sha(paths[key], expected, key.replace("_", " "))
        for key, expected in expected_hashes.items()
    }

    final_freeze = read_json(paths["final_physical_equivalence_freeze"])
    final_summary = read_json(paths["final_physical_equivalence_summary"])
    predictions_freeze = read_json(paths["physical_predictions_freeze"])

    if final_freeze.get("status") != "FROZEN_FINAL_PHYSICAL_KL720_BIOLOGICAL_DEPLOYMENT_EQUIVALENCE_PASS":
        raise RuntimeError("Unexpected terminal physical freeze status")
    if final_freeze.get("deployment_conclusion") != "PHYSICAL_KL720_BIOLOGICAL_DEPLOYMENT_EQUIVALENCE_PASS_UNDER_ORIGINAL_SEVEN_PREDECLARED_GATES":
        raise RuntimeError("Unexpected terminal deployment conclusion")
    if final_freeze.get("all_seven_original_biological_gates_pass") is not True:
        raise RuntimeError("Terminal freeze does not record all seven gates PASS")
    if final_freeze.get("exact_fp32_direction_peak_signature_reproduced") is not True:
        raise RuntimeError("Terminal freeze does not reproduce FP32 direction/peak signature")
    if final_freeze.get("holdout_guided_retuning_authorized") is not False:
        raise RuntimeError("Terminal freeze unexpectedly authorizes holdout-guided retuning")
    if final_freeze.get("model_ptq_bie_nef_or_threshold_changes_authorized") is not False:
        raise RuntimeError("Terminal freeze unexpectedly authorizes deployment changes")

    if final_summary.get("physical_predictions_frozen_outcome_blind_before_this_evaluation") is not True:
        raise RuntimeError("Physical predictions were not recorded as frozen outcome-blind")
    if final_summary.get("raw_phenotype_table_read") is not False:
        raise RuntimeError("Terminal physical evaluation unexpectedly reopened raw phenotype")
    if final_summary.get("post_holdout_ptq_retuning_authorized") is not False:
        raise RuntimeError("Terminal summary unexpectedly authorizes PTQ retuning")

    gate = final_summary.get("biological_gate")
    if not isinstance(gate, dict) or gate.get("all_pass") is not True:
        raise RuntimeError("Terminal biological gate missing or not PASS")
    checks = gate.get("checks")
    if not isinstance(checks, dict) or set(checks) != EXPECTED_GATE_CHECKS:
        raise RuntimeError(f"Unexpected biological gate set: {checks!r}")
    if not all(value is True for value in checks.values()):
        raise RuntimeError("One or more terminal biological gates are not PASS")
    if gate.get("direction_counts") != EXPECTED_DIRECTION_COUNTS:
        raise RuntimeError("Terminal direction signature mismatch")
    if gate.get("peak_time_recovery") != EXPECTED_PEAK:
        raise RuntimeError("Terminal peak-time signature mismatch")

    if predictions_freeze.get("physical_predictions_frozen_outcome_blind_before_biological_evaluation", True) is False:
        raise RuntimeError("Physical prediction freeze contradicts outcome-blind ordering")
    if predictions_freeze.get("raw_phenotype_table_read_by_freeze") is not False:
        raise RuntimeError("Prediction freeze unexpectedly read raw phenotype")
    if predictions_freeze.get("biological_targets_or_gate_results_read_by_freeze") is not False:
        raise RuntimeError("Prediction freeze unexpectedly read biological results")

    validation_record = repo_root / "docs" / "NASA_BPS_R1_V2_KL720_VALIDATION_RECORD.md"
    deployment_policy = repo_root / "docs" / "NASA_BPS_R1_V2_DEPLOYMENT_EQUIVALENCE_POLICY.md"
    holdout_policy = repo_root / "docs" / "NASA_BPS_R1_V2_FINAL_HOLDOUT_POLICY.md"
    recovery_amendment = repo_root / "docs" / "NASA_BPS_R1_V2_FINAL_HOLDOUT_INTERRUPTED_UNBLINDING_AMENDMENT.md"
    calibration_policy = repo_root / "docs" / "NASA_BPS_R1_V2_INT8_CALIBRATION_FREEZE.md"
    for path in (
        validation_record,
        deployment_policy,
        holdout_policy,
        recovery_amendment,
        calibration_policy,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    git_head = git_text(repo_root, "rev-parse", "HEAD")
    git_branch = git_text(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    tracked_dirty = git_text(repo_root, "status", "--porcelain", "--untracked-files=no")
    if tracked_dirty:
        raise RuntimeError(
            "Tracked Git worktree is dirty; commit/stash tracked changes before packaging:\n"
            + tracked_dirty
        )

    source_script = Path(__file__).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "nasa_bps_r1_v2_kl720_rc1_manifest.json"

    bag = gate.get("bag_metrics") or {}
    delta = gate.get("delta_metrics") or {}
    physical_vs_bie_nucleus = final_summary.get(
        "physical_vs_bie_nucleus_descriptive_only_no_posthoc_gate"
    ) or {}
    physical_vs_bie_bag = final_summary.get(
        "physical_vs_bie_bag_descriptive_only_no_posthoc_gate"
    ) or {}

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_NASA_BPS_R1_V2_KL720_RELEASE_CANDIDATE",
        "release_label": RELEASE_LABEL,
        "repository": {
            "name": "liuifrec/radiation-edge-ai",
            "branch": git_branch,
            "head_commit": git_head,
            "tracked_worktree_clean": True,
            "packager_source_sha256": sha256_file(source_script),
        },
        "scientific_identity": {
            "task": "NASA BPS OSD-366 53BP1 latent continuous burden regression",
            "input": "FITC+DAPI+zero; p1/p99.5 normalization; native scale; no resize; center-pad 256",
            "output_semantics": "latent continuous 53BP1 burden; not per-nucleus focus count",
            "final_holdout": {"nuclei": 2058, "bags": 22, "matched_contrasts": 11},
            "all_final_sources_female": True,
            "mask_used_as_model_input": False,
        },
        "terminal_decision": {
            "conclusion": "PHYSICAL_KL720_BIOLOGICAL_DEPLOYMENT_EQUIVALENCE_PASS_UNDER_ORIGINAL_SEVEN_PREDECLARED_GATES",
            "all_seven_original_biological_gates_pass": True,
            "exact_fp32_direction_peak_signature_reproduced": True,
            "bag_metrics": bag,
            "delta_metrics": delta,
            "direction_counts": EXPECTED_DIRECTION_COUNTS,
            "peak_time_recovery": EXPECTED_PEAK,
            "physical_vs_bie_nucleus_descriptive_only_no_posthoc_gate": physical_vs_bie_nucleus,
            "physical_vs_bie_bag_descriptive_only_no_posthoc_gate": physical_vs_bie_bag,
        },
        "artifact_sha256": verified_hashes,
        "additional_frozen_identities": {
            "onnx_sha256": ONNX_SHA256,
            "optimized_onnx_sha256": OPTIMIZED_ONNX_SHA256,
            "calibration_manifest_sha256": CALIBRATION_MANIFEST_SHA256,
            "calibration_freeze_sha256": CALIBRATION_FREEZE_SHA256,
            "toolchain_image": PINNED_TOOLCHAIN_IMAGE,
            "toolchain_version": TOOLCHAIN_VERSION,
        },
        "repository_validation_documents": {
            "NASA_BPS_R1_V2_KL720_VALIDATION_RECORD.md": sha256_file(validation_record),
            "NASA_BPS_R1_V2_DEPLOYMENT_EQUIVALENCE_POLICY.md": sha256_file(deployment_policy),
            "NASA_BPS_R1_V2_FINAL_HOLDOUT_POLICY.md": sha256_file(holdout_policy),
            "NASA_BPS_R1_V2_FINAL_HOLDOUT_INTERRUPTED_UNBLINDING_AMENDMENT.md": sha256_file(recovery_amendment),
            "NASA_BPS_R1_V2_INT8_CALIBRATION_FREEZE.md": sha256_file(calibration_policy),
        },
        "protocol_deviation_carried_forward": "documented non-outcome metadata-normalization implementation recovery",
        "physical_predictions_frozen_outcome_blind_before_biological_evaluation": True,
        "raw_nasa_phenotype_table_reopened_by_deployment_conversion_or_physical_freezes": False,
        "holdout_guided_model_ptq_bie_nef_or_threshold_retuning_authorized": False,
        "claim_boundaries": {
            "strict_numerical_equivalence_claimed": False,
            "per_nucleus_focus_count_accuracy_claimed": False,
            "sex_effect_claimed": False,
            "fifteen_strain_generalization_claimed": False,
            "complete_system_energy_benchmark_claimed": False,
        },
        "release_candidate_ready": True,
        "authorized_next_stage": "MERGE_OR_TAG_VALIDATION_RECORD_WITH_NO_MODEL_PTQ_OR_THRESHOLD_CHANGES",
    }

    if manifest_path.exists():
        existing = read_json(manifest_path)
        old = dict(existing)
        new = dict(manifest)
        old.pop("created_utc", None)
        new.pop("created_utc", None)
        if old != new:
            raise RuntimeError(
                f"Existing release-candidate manifest differs from current frozen package: {manifest_path}"
            )
        print("Radiation Edge AI - NASA BPS R1 v2 KL720 release-candidate package")
        print("existing release-candidate manifest verified without overwrite: YES")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("Radiation Edge AI - NASA BPS R1 v2 KL720 release-candidate package")
        print("new release-candidate manifest written: YES")

    print(f"release label: {RELEASE_LABEL}")
    print(f"repository branch/head: {git_branch} / {git_head}")
    print("tracked worktree clean: YES")
    print(f"terminal physical PASS freeze SHA256: {FINAL_PHYSICAL_FREEZE_SHA256}")
    print(f"terminal physical summary SHA256: {FINAL_PHYSICAL_SUMMARY_SHA256}")
    print(f"NEF SHA256: {NEF_SHA256}")
    print("all seven original biological gates: PASS")
    print("exact FP32 direction/peak signature reproduced: YES")
    print("strict numerical equivalence claimed: NO")
    print("holdout-guided retuning authorized: NO")
    print(f"manifest: {manifest_path}")
    print(f"manifest SHA256: {sha256_file(manifest_path)}")
    print("NASA BPS R1 V2 KL720 RELEASE CANDIDATE PACKAGED: YES")
    print("NEXT STAGE: merge/tag this validation package; scientific deployment validation is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
