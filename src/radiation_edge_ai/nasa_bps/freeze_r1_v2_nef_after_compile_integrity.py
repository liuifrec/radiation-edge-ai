"""Freeze the accepted NASA BPS R1 v2 KL720 NEF after compile-integrity PASS.

This stage performs no inference, compilation, calibration, PTQ, or holdout
reading. It pins the exact compiled NEF and the one-development-tensor NEF-vs-BIE
compile-integrity result before physical KL720 inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FROZEN_NEF_SHA256 = (
    "d04f02855a8a82ae7cb5ff48af91eee69c341142d1097469890385bd23ee097b"
)
FROZEN_NEF_COMPILE_SUMMARY_SHA256 = (
    "813213f224da7ec9a9621a27c92b45ab87f8909eb0c7e203403f5a8c6e64c2ea"
)
FROZEN_NEF_BIE_INTEGRITY_SUMMARY_SHA256 = (
    "90aaac45fb84c61aaf2959dfbcf0fffd3bbb9b8f1c2afb1084596e96c3833037"
)
FROZEN_BIE_EQUIVALENCE_FREEZE_SHA256 = (
    "e901d4be044c5ea5507b3b60643043eb8aa0af5643a0b7fba2f9bdd7c588bb0a"
)
EXPECTED_PLATFORM = 720
EXPECTED_MODEL_ID = 32770
EXPECTED_MODEL_VERSION = "8b29"
EXPECTED_SELECTED_SAMPLE = "BPSR1V2D_00324"
EXPECTED_SELECTED_TENSOR_INDEX = 0
EXPECTED_MAX_ABS_GATE = 1.0e-4
EXPECTED_MAX_ABS_OBSERVED = 7.1525574e-07


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
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    nef_root = model_root / "nasa_bps_53bp1" / "r1_v2_deployment" / "kneron_nef"
    parser.add_argument(
        "--nef",
        default=str(nef_root / "r1_countnet_v2_final_kl720.nef"),
    )
    parser.add_argument(
        "--compile-summary",
        default=str(nef_root / "nef_compile_summary.json"),
    )
    parser.add_argument(
        "--integrity-summary",
        default=str(nef_root / "nef_vs_bie_compile_integrity.json"),
    )
    args = parser.parse_args()

    nef_path = Path(args.nef).resolve()
    compile_summary_path = Path(args.compile_summary).resolve()
    integrity_summary_path = Path(args.integrity_summary).resolve()
    freeze_path = nef_root / "nef_deployment_candidate_freeze.json"

    nef_sha = check_sha(nef_path, FROZEN_NEF_SHA256, "Compiled NEF")
    compile_sha = check_sha(
        compile_summary_path,
        FROZEN_NEF_COMPILE_SUMMARY_SHA256,
        "NEF compile summary",
    )
    integrity_sha = check_sha(
        integrity_summary_path,
        FROZEN_NEF_BIE_INTEGRITY_SUMMARY_SHA256,
        "NEF/BIE compile-integrity summary",
    )

    compile_summary = read_json(compile_summary_path)
    integrity = read_json(integrity_summary_path)

    compile_checks = {
        "status": "COMPLETE_KL720_NEF_COMPILE_FROM_ACCEPTED_BIE",
        "nef_sha256": FROZEN_NEF_SHA256,
        "bie_equivalence_freeze_sha256": FROZEN_BIE_EQUIVALENCE_FREEZE_SHA256,
        "platform": "720",
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "onnx_optimization_repeated": False,
        "ptq_repeated": False,
        "bie_regenerated": False,
        "holdout_inference_performed": False,
    }
    for key, expected in compile_checks.items():
        if compile_summary.get(key) != expected:
            raise RuntimeError(
                f"NEF compile invariant mismatch for {key}: "
                f"expected {expected!r}, got {compile_summary.get(key)!r}"
            )

    integrity_checks = {
        "status": "COMPLETE_ONE_SAMPLE_NEF_VS_BIE_COMPILE_INTEGRITY_GATE",
        "nef_sha256": FROZEN_NEF_SHA256,
        "nef_sha256_after_simulator": FROZEN_NEF_SHA256,
        "nef_compile_summary_sha256": FROZEN_NEF_COMPILE_SUMMARY_SHA256,
        "bie_equivalence_freeze_sha256": FROZEN_BIE_EQUIVALENCE_FREEZE_SHA256,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "selected_tensor_index": EXPECTED_SELECTED_TENSOR_INDEX,
        "selected_sample_id": EXPECTED_SELECTED_SAMPLE,
        "max_abs_error_gate": EXPECTED_MAX_ABS_GATE,
        "verification_pass": True,
        "nef_or_bie_artifact_modified_by_compatibility_shim": False,
        "final_holdout_read": False,
        "ptq_or_bie_changed": False,
        "authorized_next_stage": "FREEZE_NEF_AND_MOVE_TO_PHYSICAL_KL720",
    }
    for key, expected in integrity_checks.items():
        if integrity.get(key) != expected:
            raise RuntimeError(
                f"NEF/BIE integrity invariant mismatch for {key}: "
                f"expected {expected!r}, got {integrity.get(key)!r}"
            )

    observed = float(integrity.get("max_abs_error"))
    if abs(observed - EXPECTED_MAX_ABS_OBSERVED) > 1e-12:
        raise RuntimeError(
            f"Unexpected observed max_abs_error: expected {EXPECTED_MAX_ABS_OBSERVED}, got {observed}"
        )
    if observed > EXPECTED_MAX_ABS_GATE:
        raise RuntimeError("Compile-integrity numerical gate did not pass")

    shim = integrity.get("nef_scalar_output_parser_compatibility_shim")
    if not isinstance(shim, dict) or shim.get("applied") is not True:
        raise RuntimeError("Expected documented scalar-output parser compatibility shim")
    if shim.get("original_output_raw_shapes") != [[1]]:
        raise RuntimeError(f"Unexpected original scalar raw_shape: {shim}")
    if shim.get("patched_output_raw_shapes") != [[1, 1]]:
        raise RuntimeError(f"Unexpected temporary scalar raw_shape patch: {shim}")

    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_KL720_NEF_AFTER_COMPILE_INTEGRITY_PASS_BEFORE_PHYSICAL_HARDWARE",
        "nef_sha256": nef_sha,
        "nef_compile_summary_sha256": compile_sha,
        "nef_vs_bie_compile_integrity_summary_sha256": integrity_sha,
        "bie_equivalence_freeze_sha256": FROZEN_BIE_EQUIVALENCE_FREEZE_SHA256,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "compile_integrity_sample_id": EXPECTED_SELECTED_SAMPLE,
        "compile_integrity_tensor_index": EXPECTED_SELECTED_TENSOR_INDEX,
        "compile_integrity_exact_array_equal": False,
        "compile_integrity_max_abs_error": observed,
        "compile_integrity_max_abs_gate": EXPECTED_MAX_ABS_GATE,
        "compile_integrity_pass": True,
        "scalar_output_parser_compatibility": {
            "toolchain_v0_33_1_parser_bug": True,
            "temporary_extracted_raw_shape_before": [1],
            "temporary_extracted_raw_shape_after": [1, 1],
            "nef_artifact_modified": False,
        },
        "all_seven_bie_biological_gates_already_frozen_pass": True,
        "exact_fp32_direction_peak_signature_already_reproduced_by_bie": True,
        "final_holdout_read_by_compile_integrity_stage": False,
        "ptq_or_bie_changed_by_compile_integrity_stage": False,
        "post_holdout_ptq_retuning_authorized": False,
        "authorized_next_stage": "PHYSICAL_KL720_ONE_SAMPLE_SMOKE_THEN_FROZEN_HARDWARE_EQUIVALENCE",
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
                f"Existing NEF deployment freeze differs from accepted result: {freeze_path}"
            )
        print("Radiation Edge AI - NASA BPS R1 v2 KL720 NEF deployment freeze")
        print("existing NEF deployment freeze verified without overwrite: YES")
    else:
        freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
        print("Radiation Edge AI - NASA BPS R1 v2 KL720 NEF deployment freeze")
        print("new NEF deployment freeze written: YES")

    print(f"NEF SHA256: {nef_sha}")
    print(f"NEF compile summary SHA256: {compile_sha}")
    print(f"NEF/BIE compile-integrity summary SHA256: {integrity_sha}")
    print(f"compile-integrity max abs: {observed:.10g} <= {EXPECTED_MAX_ABS_GATE:.1e}: PASS")
    print("scalar-output parser compatibility documented: YES")
    print("compiled NEF artifact modified by compatibility shim: NO")
    print("final holdout read by this freeze: NO")
    print("post-holdout PTQ retuning authorized: NO")
    print(f"freeze: {freeze_path}")
    print(f"freeze SHA256: {sha256_file(freeze_path)}")
    print("NASA BPS R1 V2 KL720 NEF DEPLOYMENT CANDIDATE FROZEN: YES")
    print("NEXT GATE: physical KL720 one-sample smoke test with this exact NEF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
