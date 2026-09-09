"""Freeze the successful NASA BPS R1 v2 ONNX parity artifact.

This stage performs no inference and reads no raw phenotype table. It verifies
that the completed FP32->ONNX parity report passed the predeclared numerical and
biological gates, pins the exact ONNX SHA256, and freezes all parity outputs by
SHA256 for subsequent Kneron optimization / INT8 calibration / BIE / NEF work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FROZEN_FP32_CHECKPOINT_SHA256 = (
    "2b07c67d6e50e7a36e26e0052feebc826b518e6efae863ba312ba9761a4eb3d5"
)
FROZEN_QC1_HOLDOUT_MANIFEST_SHA256 = (
    "f89962f96bec05930c07149717a90bd682016a2778bbfae6b827425159aa0643"
)
FROZEN_FP32_DEPLOYMENT_FREEZE_SHA256 = (
    "abb58026d4f798554ef7e44346a989499242fcbd89d8c6629a1c4209d6c16ce3"
)
FROZEN_ONNX_SHA256 = (
    "a65718a8f07d5bcd8707bae78d8f730d56047db21e575fa55ba5396adc49a07e"
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

PARITY_FILES = (
    "r1_countnet_v2_final_fp32_1x3x256x256_opset11.onnx",
    "r1_countnet_v2_final_fp32_opset11_parity.json",
    "r1_countnet_v2_final_fp32_opset11_nucleus_parity.csv",
    "r1_countnet_v2_final_fp32_opset11_bag_parity.csv",
    "r1_countnet_v2_final_fp32_opset11_matched_deltas.csv",
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


def validate_report(report: dict[str, Any]) -> None:
    if report.get("status") != "COMPLETE_FP32_ONNX_PARITY_EVALUATION":
        raise RuntimeError(f"Unexpected ONNX parity status: {report.get('status')!r}")
    if report.get("authoritative_checkpoint_sha256") != FROZEN_FP32_CHECKPOINT_SHA256:
        raise RuntimeError("ONNX report checkpoint identity mismatch")
    if report.get("qc1_holdout_manifest_sha256") != FROZEN_QC1_HOLDOUT_MANIFEST_SHA256:
        raise RuntimeError("ONNX report QC1 manifest identity mismatch")
    if (
        report.get("fp32_deployment_reference_freeze_sha256")
        != FROZEN_FP32_DEPLOYMENT_FREEZE_SHA256
    ):
        raise RuntimeError("ONNX report FP32 deployment-freeze identity mismatch")
    onnx = report.get("onnx")
    if not isinstance(onnx, dict) or onnx.get("sha256") != FROZEN_ONNX_SHA256:
        raise RuntimeError(f"Unexpected ONNX identity in report: {onnx!r}")
    if onnx.get("opset") != 11:
        raise RuntimeError(f"Unexpected ONNX opset: {onnx.get('opset')!r}")
    if onnx.get("fixed_input_shape") != [1, 3, 256, 256]:
        raise RuntimeError(f"Unexpected ONNX input shape: {onnx.get('fixed_input_shape')!r}")
    numerical = report.get("numerical_parity")
    if not isinstance(numerical, dict) or numerical.get("all_numerical_gates_pass") is not True:
        raise RuntimeError("Predeclared ONNX numerical parity did not pass")
    biological = report.get("biological_parity")
    if not isinstance(biological, dict):
        raise RuntimeError("Missing ONNX biological-parity result")
    if biological.get("all_biological_gates_pass") is not True:
        raise RuntimeError("ONNX biological parity did not pass")
    if biological.get("exact_signature_pass") is not True:
        raise RuntimeError("ONNX did not reproduce the exact FP32 biological signature")
    gate = biological.get("gate_result")
    if not isinstance(gate, dict) or gate.get("all_pass") is not True:
        raise RuntimeError("ONNX did not pass all seven original biological gates")
    if gate.get("direction_counts") != EXPECTED_DIRECTION_COUNTS:
        raise RuntimeError(f"Unexpected ONNX direction signature: {gate.get('direction_counts')!r}")
    if gate.get("peak_time_recovery") != EXPECTED_PEAK:
        raise RuntimeError(f"Unexpected ONNX peak signature: {gate.get('peak_time_recovery')!r}")
    if report.get("raw_phenotype_table_read") is not False:
        raise RuntimeError("ONNX stage unexpectedly reports raw phenotype-table access")
    if report.get("no_quantization_performed") is not True:
        raise RuntimeError("ONNX stage unexpectedly reports quantization")
    if report.get("overall_onnx_parity_pass") is not True:
        raise RuntimeError("Overall FP32->ONNX parity is not PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    confirmation_dir = (
        model_root / "nasa_bps_53bp1" / "r1_v2_final_holdout_confirmation"
    )
    default_onnx_dir = model_root / "nasa_bps_53bp1" / "r1_v2_deployment" / "onnx"
    parser.add_argument("--onnx-dir", default=str(default_onnx_dir))
    parser.add_argument(
        "--fp32-freeze",
        default=str(confirmation_dir / "fp32_deployment_reference_freeze.json"),
    )
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir).resolve()
    fp32_freeze = Path(args.fp32_freeze).resolve()
    freeze_path = onnx_dir / "onnx_deployment_candidate_freeze.json"
    if not onnx_dir.is_dir():
        raise FileNotFoundError(onnx_dir)
    if not fp32_freeze.is_file():
        raise FileNotFoundError(fp32_freeze)
    actual_fp32_freeze_sha = sha256_file(fp32_freeze)
    if actual_fp32_freeze_sha != FROZEN_FP32_DEPLOYMENT_FREEZE_SHA256:
        raise RuntimeError(
            "FP32 deployment freeze SHA256 mismatch: "
            f"expected {FROZEN_FP32_DEPLOYMENT_FREEZE_SHA256}, got {actual_fp32_freeze_sha}"
        )

    paths = {name: onnx_dir / name for name in PARITY_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError("Missing ONNX parity artifacts: " + ", ".join(missing))

    onnx_path = paths["r1_countnet_v2_final_fp32_1x3x256x256_opset11.onnx"]
    actual_onnx_sha = sha256_file(onnx_path)
    if actual_onnx_sha != FROZEN_ONNX_SHA256:
        raise RuntimeError(
            f"ONNX SHA256 mismatch: expected {FROZEN_ONNX_SHA256}, got {actual_onnx_sha}"
        )

    report = read_json(paths["r1_countnet_v2_final_fp32_opset11_parity.json"])
    validate_report(report)

    identities = {
        name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for name, path in paths.items()
    }
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_ONNX_DEPLOYMENT_CANDIDATE_AFTER_FP32_PARITY_PASS",
        "fp32_checkpoint_sha256": FROZEN_FP32_CHECKPOINT_SHA256,
        "qc1_holdout_manifest_sha256": FROZEN_QC1_HOLDOUT_MANIFEST_SHA256,
        "fp32_deployment_reference_freeze_sha256": FROZEN_FP32_DEPLOYMENT_FREEZE_SHA256,
        "onnx_sha256": actual_onnx_sha,
        "onnx_opset": 11,
        "onnx_input_shape": [1, 3, 256, 256],
        "onnx_output_semantics": "scalar latent continuous 53BP1 burden",
        "numerical_parity_pass": True,
        "all_seven_biological_gates_pass": True,
        "exact_fp32_direction_signature_reproduced": True,
        "exact_fp32_peak_signature_reproduced": True,
        "direction_counts": EXPECTED_DIRECTION_COUNTS,
        "peak_time_recovery": EXPECTED_PEAK,
        "parity_artifacts": identities,
        "raw_phenotype_reopened_by_freeze": False,
        "quantization_performed_by_freeze": False,
        "authorized_next_stage": "DEVELOPMENT_ONLY_INT8_CALIBRATION_AND_KL720_FLOATING_POINT_OPTIMIZATION",
    }

    if freeze_path.exists():
        existing = read_json(freeze_path)
        old = dict(existing)
        new = dict(freeze)
        old.pop("created_utc", None)
        new.pop("created_utc", None)
        if old != new:
            raise RuntimeError(f"Existing ONNX freeze differs from current artifacts: {freeze_path}")
        print("Radiation Edge AI - NASA BPS R1 v2 ONNX deployment candidate")
        print("existing ONNX freeze verified without overwrite: YES")
    else:
        freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
        print("Radiation Edge AI - NASA BPS R1 v2 ONNX deployment candidate")
        print("new ONNX deployment freeze written: YES")

    print(f"ONNX SHA256: {actual_onnx_sha}")
    print(f"FP32 deployment-reference freeze SHA256: {actual_fp32_freeze_sha}")
    print(f"parity artifacts frozen: {len(identities)}/{len(PARITY_FILES)}")
    print("predeclared numerical parity: PASS")
    print("all seven original biological gates: PASS")
    print("exact FP32 biological signature reproduced: YES")
    print("raw phenotype table read by this freeze: NO")
    print(f"freeze: {freeze_path}")
    print(f"freeze SHA256: {sha256_file(freeze_path)}")
    print("NASA BPS R1 V2 ONNX DEPLOYMENT CANDIDATE FROZEN: YES")
    print("NEXT GATE: materialize the frozen development-only INT8 calibration panel and run KL720 floating-point optimization/evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
