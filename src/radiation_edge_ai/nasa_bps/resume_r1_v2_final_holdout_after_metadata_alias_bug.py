"""Resume the interrupted NASA BPS R1 v2 final-holdout confirmation after a
pre-existing metadata-alias validator was accidentally omitted from the frozen
one-time evaluator.

This is NOT a rewrite of the original frozen evaluator. It is a documented
recovery path that is valid only when:
- the original one-time evaluator has already written its UNBLINDING_STARTED
  marker and no confirmation-complete marker exists;
- the original evaluator source still has the exact preflight/runtime SHA256;
- checkpoint, QC1 holdout manifest, and raw phenotype hashes are unchanged;
- the interruption occurred before any confirmatory metrics were computed;
- the only correction is metadata normalization that had been validated and
  frozen before R1 v2 training/holdout evaluation:
    BALB/cByJ -> BALBC
    C57BL/6J -> C57
    Fe -> Fe 600 MeV/n branch aliases
    X-ray naming aliases
    punctuation-only plate/well normalization.

Default mode is recovery AUDIT ONLY and does not open the raw phenotype table.
Use --resume-confirmation only after freezing the recovery source identity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ORIGINAL_EVALUATOR_RUNTIME_SHA256 = (
    "0bb389434ef3d9b64ff5d5bd29761765a6a53ec905194ec88cefaef6c698a1ca"
)
FROZEN_REPAIRED_CHECKPOINT_SHA256 = (
    "2b07c67d6e50e7a36e26e0052feebc826b518e6efae863ba312ba9761a4eb3d5"
)
FROZEN_QC1_HOLDOUT_MANIFEST_SHA256 = (
    "f89962f96bec05930c07149717a90bd682016a2778bbfae6b827425159aa0643"
)
FROZEN_RAW_PHENOTYPE_SHA256 = (
    "d28dc5d9c53221c66a90074a92783fd4708ea83db0fc1a9c767e47500e643d6a"
)
ALIAS_VALIDATION_COMMIT = "6e8791061563d221915fda9567be37bbd8e29264"
REFERENCE_FREEZE_COMMIT = "3141f1e168ffe100d177fbc59b8e8c2772853b71"

STRAIN_ALIAS = {
    "BALB/cByJ": "BALBC",
    "C57BL/6J": "C57",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_original(path: Path):
    actual = sha256_file(path)
    if actual != ORIGINAL_EVALUATOR_RUNTIME_SHA256:
        raise RuntimeError(
            "Original frozen evaluator source changed: "
            f"expected {ORIGINAL_EVALUATOR_RUNTIME_SHA256}, got {actual}"
        )
    name = "nasa_bps_r1_v2_original_final_holdout_evaluator_recovery"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load original evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def norm_text(value: str) -> str:
    return "".join(str(value).strip().lower().split())


def norm_plate_or_well(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def norm_branch(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
    if (
        text in {"fe", "56fe", "fe56", "fe600mevn", "fe600mevpern"}
        or text.startswith("fe600")
    ):
        return "fe"
    if text in {"xray", "xrays", "xrayradiation"} or text.startswith("xray"):
        return "x-ray"
    return text


def expected_raw_strain(manifest_strain: str) -> str:
    return STRAIN_ALIAS.get(str(manifest_strain).strip(), str(manifest_strain).strip())


def verify_started_marker(
    start_marker: Path,
    *,
    checkpoint_sha: str,
    manifest_sha: str,
) -> dict[str, Any]:
    if not start_marker.is_file():
        raise RuntimeError(
            "No prior UNBLINDING_STARTED marker exists; this recovery path is not authorized"
        )
    obj = json.loads(start_marker.read_text(encoding="utf-8"))
    checks = {
        "evaluator_source_sha256": ORIGINAL_EVALUATOR_RUNTIME_SHA256,
        "repaired_checkpoint_sha256": checkpoint_sha,
        "qc1_holdout_manifest_sha256": manifest_sha,
        "raw_phenotype_expected_sha256": FROZEN_RAW_PHENOTYPE_SHA256,
        "status": "UNBLINDING_STARTED_NO_TUNING_PERMITTED",
    }
    for key, expected in checks.items():
        if obj.get(key) != expected:
            raise RuntimeError(
                f"Interrupted-run marker mismatch for {key}: "
                f"expected {expected!r}, got {obj.get(key)!r}"
            )
    return obj


def resolve_metadata_rows(
    original: Any,
    raw_rows: list[dict[str, str]],
    bags: list[Any],
) -> dict[str, dict[str, str]]:
    missing = [column for column in original.RAW_REQUIRED if column not in raw_rows[0]]
    if missing:
        raise RuntimeError(f"Raw phenotype table missing required columns: {missing}")

    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        by_sample[original.norm_text(row["Sample Name"])].append(row)

    resolved: dict[str, dict[str, str]] = {}
    for bag in bags:
        matches = by_sample.get(original.norm_text(bag.sample_name), [])
        if len(matches) != 1:
            raise RuntimeError(
                f"Holdout Sample Name {bag.sample_name!r} matched {len(matches)} "
                "raw phenotype rows; expected exactly 1"
            )
        nasa = matches[0]

        comparisons = (
            ("source_name", original.norm_text(bag.source_name), original.norm_text(nasa.get("UserID", ""))),
            (
                "strain",
                norm_text(expected_raw_strain(bag.strain)),
                norm_text(nasa.get("Strain", "")),
            ),
            ("sex", original.norm_sex(bag.sex), original.norm_sex(nasa.get("Gender", ""))),
            ("plate", norm_plate_or_well(bag.plate), norm_plate_or_well(nasa.get("plate", ""))),
            ("well", norm_plate_or_well(bag.well), norm_plate_or_well(nasa.get("plate_well", ""))),
            (
                "particle_type",
                norm_branch(bag.particle_type),
                norm_branch(nasa.get("radiation", "")),
            ),
            (
                "dose_Gy",
                original.norm_number(bag.dose_Gy),
                original.norm_number(nasa.get("dose..Gy.", "")),
            ),
            (
                "hr_post_exposure",
                original.norm_hour(bag.hr_post_exposure),
                original.norm_hour(nasa.get("timepoint..hr.", "")),
            ),
        )
        for field, left, right in comparisons:
            if left and right and left != right:
                raise RuntimeError(
                    "Metadata remains discordant after pre-existing alias policy for "
                    f"{bag.sample_name}: {field}={left!r} != raw={right!r}"
                )
        resolved[bag.sample_name] = nasa

    if len(resolved) != original.EXPECTED_BAGS:
        raise RuntimeError(
            f"Metadata-resolved holdout bags={len(resolved)}; expected {original.EXPECTED_BAGS}"
        )
    return resolved


def dereference_targets(
    original: Any,
    metadata_rows: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for sample_name, nasa in metadata_rows.items():
        target = float(nasa["avg_nfoci"])
        num_nuc = int(float(nasa["num_nuc"]))
        if not math.isfinite(target) or num_nuc <= 0:
            raise RuntimeError(f"Invalid NASA aggregate reference for {sample_name}")
        references[sample_name] = {
            "nasa_avg_nfoci": target,
            "nasa_num_nuc": num_nuc,
        }
    return references


def ensure_result_outputs_absent(output_dir: Path) -> None:
    protected = (
        "FINAL_HOLDOUT_CONFIRMATION_COMPLETE.json",
        "FINAL_HOLDOUT_RECOVERY_COMPLETE.json",
        "final_holdout_confirmation_summary.json",
        "final_holdout_bag_predictions.csv",
        "final_holdout_nucleus_latent_burden.csv",
        "final_holdout_matched_radiation_deltas.csv",
        "final_holdout_direction_summary.csv",
        "final_holdout_peak_time_recovery.csv",
        "final_holdout_source_residuals.csv",
    )
    existing = [name for name in protected if (output_dir / name).exists()]
    if existing:
        raise RuntimeError(
            "Confirmatory/recovery outputs already exist; refusing overwrite: "
            + ", ".join(existing)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    holdout_root = data_root / "nasa_bps_microscopy" / "r1_v2_final_holdout_blinded"
    script_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--manifest",
        default=str(
            metadata_root
            / "r1_v2_freeze"
            / "r1_v2_final_holdout_manifest_blinded_qc1.csv"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            model_root
            / "nasa_bps_53bp1"
            / "r1_v2_final_candidate"
            / "r1_countnet_v2_final_bn_recalibrated.pt"
        ),
    )
    parser.add_argument("--image-root", default=str(holdout_root / "images"))
    parser.add_argument(
        "--raw-phenotypes",
        default=str(
            metadata_root
            / "current_phenotypes"
            / "files"
            / "LSDS-111_immunostaining_Raw_pheno_V3.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            model_root
            / "nasa_bps_53bp1"
            / "r1_v2_final_holdout_confirmation"
        ),
    )
    parser.add_argument("--cache-items", type=int, default=512)
    parser.add_argument(
        "--resume-confirmation",
        action="store_true",
        help=(
            "Resume the interrupted confirmatory evaluation using only the "
            "pre-existing metadata-alias policy. Default is recovery audit-only."
        ),
    )
    args = parser.parse_args()

    original_path = script_dir / "evaluate_r1_v2_final_holdout_once.py"
    recovery_source_sha = sha256_file(Path(__file__).resolve())
    original = load_original(original_path)

    manifest_path = Path(args.manifest).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    image_root = Path(args.image_root).resolve()
    raw_path = Path(args.raw_phenotypes).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (manifest_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    checkpoint_sha = sha256_file(checkpoint_path)
    manifest_sha = sha256_file(manifest_path)
    if checkpoint_sha != FROZEN_REPAIRED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Final repaired FP32 SHA256 mismatch: {checkpoint_sha}")
    if manifest_sha != FROZEN_QC1_HOLDOUT_MANIFEST_SHA256:
        raise RuntimeError(f"QC1 holdout manifest SHA256 mismatch: {manifest_sha}")

    complete_marker = output_dir / "FINAL_HOLDOUT_CONFIRMATION_COMPLETE.json"
    recovery_complete = output_dir / "FINAL_HOLDOUT_RECOVERY_COMPLETE.json"
    if complete_marker.exists() or recovery_complete.exists():
        raise RuntimeError("Final holdout confirmation already completed; recovery is forbidden")

    start_marker = output_dir / "FINAL_HOLDOUT_UNBLINDING_STARTED.json"
    interrupted = verify_started_marker(
        start_marker,
        checkpoint_sha=checkpoint_sha,
        manifest_sha=manifest_sha,
    )

    manifest = original.read_csv(manifest_path)
    bags = original.validate_manifest(manifest)
    v1 = original.load_module(
        "nasa_bps_r1_v1_final_holdout_recovery",
        script_dir / "train_pilot_v1_r1_mil.py",
    )
    dev_qc = original.load_module(
        "nasa_bps_dev_qc_final_holdout_recovery",
        script_dir / "validate_r1_v2_development.py",
    )
    v1.seed_everything(original.SEED)
    qc = original.hard_qc(dev_qc, manifest, image_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = original.load_model(v1, checkpoint_path, device)

    print("Radiation Edge AI - NASA BPS R1 v2 FINAL HOLDOUT recovery evaluator")
    print(
        "mode: "
        + (
            "DOCUMENTED CONFIRMATION RESUME"
            if args.resume_confirmation
            else "RECOVERY AUDIT ONLY"
        )
    )
    print(f"recovery evaluator source SHA256: {recovery_source_sha}")
    print(f"original evaluator source SHA256 verified: {ORIGINAL_EVALUATOR_RUNTIME_SHA256}")
    print(f"repaired FP32 SHA256: {checkpoint_sha}")
    print(f"QC1 holdout manifest SHA256: {manifest_sha}")
    print(f"prior unblinding marker verified: {start_marker}")
    print(
        "metadata alias provenance commits: "
        f"{ALIAS_VALIDATION_COMMIT}, {REFERENCE_FREEZE_COMMIT}"
    )
    print(f"hard QC recheck: {qc['nuclei']}/{qc['nuclei']} PASS")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    blinded_bags, nucleus_rows = original.infer_blinded(
        v1,
        model,
        bags,
        image_root,
        device,
        args.cache_items,
    )
    print(
        f"frozen inference reproduced: {len(nucleus_rows)} nuclei -> "
        f"{len(blinded_bags)} bags; finite outputs YES"
    )

    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "resume" if args.resume_confirmation else "audit_only",
        "protocol_deviation": True,
        "deviation_class": "non-outcome metadata-normalization implementation omission",
        "original_evaluator_source_sha256": ORIGINAL_EVALUATOR_RUNTIME_SHA256,
        "recovery_evaluator_source_sha256": recovery_source_sha,
        "repaired_checkpoint_sha256": checkpoint_sha,
        "qc1_holdout_manifest_sha256": manifest_sha,
        "raw_phenotype_expected_sha256": FROZEN_RAW_PHENOTYPE_SHA256,
        "original_unblinding_started_utc": interrupted.get("created_utc"),
        "interruption_observed_before_confirmatory_metrics": True,
        "pre_existing_alias_validation_commit": ALIAS_VALIDATION_COMMIT,
        "pre_existing_reference_freeze_commit": REFERENCE_FREEZE_COMMIT,
        "model_changed": False,
        "checkpoint_changed": False,
        "holdout_manifest_changed": False,
        "gate_changed": False,
        "endpoint_changed": False,
        "prediction_selection_or_tuning_performed": False,
        "raw_phenotype_read_by_this_recovery_invocation": False,
        "hard_qc": qc,
        "phenotype_blind_inference_complete": True,
    }
    audit_path = output_dir / "final_holdout_recovery_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if not args.resume_confirmation:
        print("raw phenotype table read by recovery invocation: NO")
        print(f"recovery audit: {audit_path}")
        print("NASA BPS R1 V2 FINAL HOLDOUT RECOVERY AUDIT COMPLETE: YES")
        print(
            "NEXT GATE: freeze this recovery source identity; then rerun the same "
            "unchanged recovery evaluator with --resume-confirmation."
        )
        return 0

    ensure_result_outputs_absent(output_dir)

    resume_marker = output_dir / "FINAL_HOLDOUT_RECOVERY_STARTED.json"
    if resume_marker.exists():
        raise RuntimeError(
            "A recovery-resume marker already exists; do not silently rerun an interrupted recovery"
        )
    resume_marker.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "recovery_evaluator_source_sha256": recovery_source_sha,
                "original_evaluator_source_sha256": ORIGINAL_EVALUATOR_RUNTIME_SHA256,
                "repaired_checkpoint_sha256": checkpoint_sha,
                "qc1_holdout_manifest_sha256": manifest_sha,
                "raw_phenotype_expected_sha256": FROZEN_RAW_PHENOTYPE_SHA256,
                "status": "DOCUMENTED_METADATA_ALIAS_RECOVERY_STARTED_NO_TUNING_PERMITTED",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("[DOCUMENTED CONFIRMATORY RECOVERY]")
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    raw_sha = sha256_file(raw_path)
    if raw_sha != FROZEN_RAW_PHENOTYPE_SHA256:
        raise RuntimeError(
            f"Raw phenotype SHA256 mismatch: expected {FROZEN_RAW_PHENOTYPE_SHA256}, got {raw_sha}"
        )
    raw_rows = original.read_csv(raw_path)

    metadata_rows = resolve_metadata_rows(original, raw_rows, bags)
    print(
        f"exact Sample Name + pre-existing metadata aliases verified: "
        f"{len(metadata_rows)}/{original.EXPECTED_BAGS}"
    )
    print("avg_nfoci target fields dereferenced before metadata verification: NO")

    references = dereference_targets(original, metadata_rows)
    bag_rows = original.attach_references(blinded_bags, references)
    delta_rows = original.build_deltas(bag_rows)
    peaks = original.peak_rows(delta_rows)
    directions = original.direction_rows(delta_rows)
    source_residuals = original.source_residual_rows(v1, bag_rows)
    gate = original.evaluate_gate(v1, bag_rows, delta_rows, peaks)

    original.write_csv(output_dir / "final_holdout_bag_predictions.csv", bag_rows)
    original.write_csv(
        output_dir / "final_holdout_nucleus_latent_burden.csv", nucleus_rows
    )
    original.write_csv(
        output_dir / "final_holdout_matched_radiation_deltas.csv", delta_rows
    )
    original.write_csv(
        output_dir / "final_holdout_direction_summary.csv", directions
    )
    original.write_csv(
        output_dir / "final_holdout_peak_time_recovery.csv", peaks
    )
    original.write_csv(
        output_dir / "final_holdout_source_residuals.csv", source_residuals
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": (
            "COMPLETE_CONFIRMATORY_HOLDOUT_WITH_DOCUMENTED_METADATA_ALIAS_RECOVERY"
        ),
        "protocol_deviation": True,
        "deviation_class": "non-outcome metadata-normalization implementation omission",
        "deviation_interpretation": (
            "The frozen one-time evaluator omitted metadata aliases that had already "
            "been validated and frozen before R1 v2 training/holdout evaluation. "
            "Recovery changed only metadata validation; model, images, endpoint, "
            "predictions, thresholds, gates, and holdout membership were unchanged."
        ),
        "original_evaluator_source_sha256": ORIGINAL_EVALUATOR_RUNTIME_SHA256,
        "recovery_evaluator_source_sha256": recovery_source_sha,
        "pre_existing_alias_validation_commit": ALIAS_VALIDATION_COMMIT,
        "pre_existing_reference_freeze_commit": REFERENCE_FREEZE_COMMIT,
        "repaired_checkpoint_sha256": checkpoint_sha,
        "qc1_holdout_manifest_sha256": manifest_sha,
        "raw_phenotype_sha256": raw_sha,
        "raw_phenotype_read": True,
        "primary_reference_endpoint": "NASA sample/plate-well aggregate avg_nfoci",
        "per_nucleus_prediction_semantics": (
            "continuous latent 53BP1 burden; not individually supervised and not "
            "validated as discrete focus count"
        ),
        "biological_n_semantics": (
            "Source Name/sample structure; nuclei are not independent biological replicates"
        ),
        "bag_count": len(bag_rows),
        "nucleus_count": len(nucleus_rows),
        "matched_contrast_count": len(delta_rows),
        "gate_thresholds": original.GATE,
        "gate_result": gate,
        "source_residuals": source_residuals,
        "no_holdout_outcome_used_for_tuning": True,
        "model_retrained_after_unblinding": False,
        "model_recalibrated_after_unblinding": False,
        "thresholds_changed_after_unblinding": False,
        "holdout_membership_changed_after_unblinding": False,
        "advance_to_onnx_int8_kl720": bool(gate["all_pass"]),
        "interpretation": (
            "FP32 biological-fidelity gate PASSED with a documented non-outcome "
            "metadata-normalization recovery; the exact frozen FP32 model may advance "
            "to ONNX/INT8/KL720 deployment-equivalence testing, with the protocol "
            "deviation disclosed."
            if gate["all_pass"]
            else "FP32 biological-fidelity gate FAILED with a documented metadata "
            "normalization recovery; do not tune R1 v2 on this holdout."
        ),
    }

    summary_path = output_dir / "final_holdout_confirmation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    complete_marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    recovery_complete.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    bm = gate["bag_metrics"]
    dm = gate["delta_metrics"]
    print("")
    print("[Confirmatory FP32 final-holdout result - documented recovery]")
    print(
        f"bag MAE/RMSE/Pearson/Spearman: {bm['mae']:.4f} / {bm['rmse']:.4f} / "
        f"{bm['pearson']:.4f} / {bm['spearman']:.4f}"
    )
    print(
        f"delta MAE/Pearson/Spearman: {dm['delta_mae']:.4f} / "
        f"{dm['delta_pearson']:.4f} / {dm['delta_spearman']:.4f}"
    )
    print(f"direction overall: {gate['direction_counts']['overall']}")
    print(f"direction 4 h: {gate['direction_counts']['4h']}")
    print(f"direction 24+48 h: {gate['direction_counts']['24+48h']}")
    print(f"direction by Source Name: {gate['direction_counts']['by_source']}")
    print(f"peak-time recovery: {gate['peak_time_recovery']}")
    print("")
    for name, passed in gate["checks"].items():
        print(f"GATE {name}: {'PASS' if passed else 'FAIL'}")
    print(
        "ALL SEVEN PREDECLARED FINAL-HOLDOUT CRITERIA PASS: "
        + ("YES" if gate["all_pass"] else "NO")
    )
    print("DOCUMENTED PROTOCOL DEVIATION: metadata alias validator omission/recovery")
    print("NO HOLDOUT OUTCOME USED FOR TUNING: YES")
    print(f"summary: {summary_path}")
    print("FINAL HOLDOUT PHENOTYPE STATUS: UNBLINDED - CONFIRMATORY RECOVERY COMPLETE")
    if gate["all_pass"]:
        print(
            "NEXT GATE: freeze this FP32 result as deployment reference; proceed to "
            "ONNX parity, then INT8/BIE/NEF/KL720 equivalence, carrying the deviation note."
        )
    else:
        print(
            "NEXT GATE: report failed confirmatory FP32 evaluation; do not tune R1 v2 "
            "on this holdout."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
