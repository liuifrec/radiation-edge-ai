"""Freeze the NASA BPS R1 v2 KL720 INT8 BIE before holdout deployment evaluation.

This stage is deliberately outcome-blind. It reads no final-holdout image or
phenotype data. It pins the exact BIE generated from the predeclared PTQ config,
together with its optimized ONNX, Kneron floating-point report, calibration
manifest/freeze, toolchain identity, and PTQ summary.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

FROZEN_OPTIMIZED_ONNX_SHA256 = (
    "c3a6aa5ed80280b0286f3b1aebf5f77f44dc60f569cc42fe2ab00ba7edde820c"
)
FROZEN_KNERON_FP_REPORT_SHA256 = (
    "6c0355e8d0d6eb8b2345429a9f130b0a5512d9cc1c17695eaee4c0f341f6f008"
)
FROZEN_CALIBRATION_MANIFEST_SHA256 = (
    "78c9ed83e1992907264dfeec9ef6e2a434868bd3f299672f610a42a0018b5d4e"
)
FROZEN_CALIBRATION_FREEZE_SHA256 = (
    "f017f8a27563611e49fa55147920da2860b38fbb3f64a4c7db2995a36752a9df"
)
FROZEN_BIE_SHA256 = (
    "c52b8a78c595347667bad7a34950af92adbe734179b913646d06513a88fc1dd8"
)
FROZEN_PTQ_SUMMARY_SHA256 = (
    "98f0e096c059b8f7312bd9cb14c4733d5901183c0ffd712fd996f248db43fde1"
)
PINNED_TOOLCHAIN_IMAGE = (
    "kneron/toolchain@sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041"
)
EXPECTED_TOOLCHAIN_VERSION = "kneron/toolchain:v0.33.1"
EXPECTED_PLATFORM = "720"
EXPECTED_MODEL_ID = 32770
EXPECTED_MODEL_VERSION = "8b29"
EXPECTED_PTQ_CONFIG = {
    "threads": 4,
    "datapath_range_method": "percentage",
    "percentage": 0.999,
    "percentage_16b": 0.999999,
    "percentile": 0.001,
    "outlier_factor": 1.0,
    "optimize": 0,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("Expected JSON object: {}".format(path))
    return obj


def check_sha(path, expected, label):
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            "{} SHA256 mismatch: expected {}, got {}".format(label, expected, actual)
        )
    return actual


def main():
    parser = argparse.ArgumentParser()
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    deployment_root = model_root / "nasa_bps_53bp1" / "r1_v2_deployment"
    parser.add_argument(
        "--onnx",
        default=str(deployment_root / "kneron_fp" / "r1_countnet_v2_final_kl720_optimized.onnx"),
    )
    parser.add_argument(
        "--kneron-fp-report",
        default=str(deployment_root / "kneron_fp" / "kl720_fp_optimization_evaluation.json"),
    )
    parser.add_argument(
        "--calibration-manifest",
        default=str(deployment_root / "int8_calibration_288" / "manifest.json"),
    )
    parser.add_argument(
        "--calibration-freeze",
        default=str(deployment_root / "int8_calibration_288" / "int8_calibration_freeze.json"),
    )
    parser.add_argument(
        "--bie",
        default=str(deployment_root / "kneron_int8" / "r1_countnet_v2_final_kl720_int8.bie"),
    )
    parser.add_argument(
        "--ptq-summary",
        default=str(deployment_root / "kneron_int8" / "int8_ptq_summary.json"),
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx).resolve()
    fp_report_path = Path(args.kneron_fp_report).resolve()
    calibration_manifest_path = Path(args.calibration_manifest).resolve()
    calibration_freeze_path = Path(args.calibration_freeze).resolve()
    bie_path = Path(args.bie).resolve()
    ptq_summary_path = Path(args.ptq_summary).resolve()
    freeze_path = bie_path.parent / "bie_deployment_candidate_freeze.json"

    check_sha(onnx_path, FROZEN_OPTIMIZED_ONNX_SHA256, "Optimized ONNX")
    check_sha(fp_report_path, FROZEN_KNERON_FP_REPORT_SHA256, "Kneron FP report")
    check_sha(
        calibration_manifest_path,
        FROZEN_CALIBRATION_MANIFEST_SHA256,
        "Calibration manifest",
    )
    check_sha(
        calibration_freeze_path,
        FROZEN_CALIBRATION_FREEZE_SHA256,
        "Calibration freeze",
    )
    bie_sha = check_sha(bie_path, FROZEN_BIE_SHA256, "BIE")
    ptq_summary_sha = check_sha(ptq_summary_path, FROZEN_PTQ_SUMMARY_SHA256, "PTQ summary")

    ptq = read_json(ptq_summary_path)
    checks = {
        "status": "COMPLETE_FROZEN_DEVELOPMENT_ONLY_KL720_INT8_PTQ",
        "optimized_onnx_sha256": FROZEN_OPTIMIZED_ONNX_SHA256,
        "kneron_fp_report_sha256": FROZEN_KNERON_FP_REPORT_SHA256,
        "calibration_manifest_sha256": FROZEN_CALIBRATION_MANIFEST_SHA256,
        "calibration_freeze_sha256": FROZEN_CALIBRATION_FREEZE_SHA256,
        "pinned_toolchain_image": PINNED_TOOLCHAIN_IMAGE,
        "toolchain_version": EXPECTED_TOOLCHAIN_VERSION,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "ptq_config_frozen_before_bie_result": True,
        "outcome_guided_quantization_tuning": False,
        "bie_sha256": FROZEN_BIE_SHA256,
        "final_holdout_used_for_calibration": False,
        "final_holdout_read": False,
    }
    for key, expected in checks.items():
        if ptq.get(key) != expected:
            raise RuntimeError(
                "PTQ summary invariant mismatch for {}: expected {!r}, got {!r}".format(
                    key, expected, ptq.get(key)
                )
            )
    if ptq.get("ptq_config") != EXPECTED_PTQ_CONFIG:
        raise RuntimeError(
            "PTQ configuration mismatch: expected {!r}, got {!r}".format(
                EXPECTED_PTQ_CONFIG, ptq.get("ptq_config")
            )
        )
    if ptq.get("calibration_count") != 288:
        raise RuntimeError("PTQ summary calibration_count is not 288")
    if ptq.get("calibration_bags") != 72:
        raise RuntimeError("PTQ summary calibration_bags is not 72")
    if ptq.get("calibration_per_bag") != 4:
        raise RuntimeError("PTQ summary calibration_per_bag is not 4")

    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_KL720_INT8_BIE_DEPLOYMENT_CANDIDATE_BEFORE_HOLDOUT",
        "optimized_onnx_sha256": FROZEN_OPTIMIZED_ONNX_SHA256,
        "kneron_fp_report_sha256": FROZEN_KNERON_FP_REPORT_SHA256,
        "calibration_manifest_sha256": FROZEN_CALIBRATION_MANIFEST_SHA256,
        "calibration_freeze_sha256": FROZEN_CALIBRATION_FREEZE_SHA256,
        "bie_sha256": bie_sha,
        "ptq_summary_sha256": ptq_summary_sha,
        "toolchain_image": PINNED_TOOLCHAIN_IMAGE,
        "toolchain_version": EXPECTED_TOOLCHAIN_VERSION,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "ptq_config": EXPECTED_PTQ_CONFIG,
        "calibration_count": 288,
        "calibration_bags": 72,
        "calibration_per_bag": 4,
        "final_holdout_used_for_calibration": False,
        "final_holdout_read_before_this_freeze": False,
        "ptq_tuning_after_holdout_authorized": False,
        "authorized_next_stage": "FIXED_BIE_DEPLOYMENT_EQUIVALENCE_HOLDOUT",
    }

    if freeze_path.exists():
        existing = read_json(freeze_path)
        old = dict(existing)
        new = dict(freeze)
        old.pop("created_utc", None)
        new.pop("created_utc", None)
        if old != new:
            raise RuntimeError(
                "Existing BIE deployment freeze differs from the frozen PTQ result: {}".format(
                    freeze_path
                )
            )
        print("Radiation Edge AI - NASA BPS R1 v2 KL720 INT8 BIE freeze")
        print("existing BIE freeze verified without overwrite: YES")
    else:
        freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
        print("Radiation Edge AI - NASA BPS R1 v2 KL720 INT8 BIE freeze")
        print("new BIE freeze written: YES")

    print("BIE SHA256: {}".format(bie_sha))
    print("PTQ summary SHA256: {}".format(ptq_summary_sha))
    print("toolchain image: {}".format(PINNED_TOOLCHAIN_IMAGE))
    print("PTQ config frozen: YES")
    print("final holdout used for calibration: NO")
    print("final holdout read before this freeze: NO")
    print("post-holdout PTQ tuning authorized: NO")
    print("freeze: {}".format(freeze_path))
    print("freeze SHA256: {}".format(sha256_file(freeze_path)))
    print("NASA BPS R1 V2 KL720 INT8 BIE DEPLOYMENT CANDIDATE FROZEN: YES")
    print("NEXT GATE: fixed BIE deployment-equivalence inference on the frozen holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
