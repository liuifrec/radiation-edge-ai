"""Freeze the successful NASA BPS R1 v2 one-sample physical KL720 smoke result.

This stage performs no inference and reads no final-holdout data. It pins the
exact physical-device result obtained from the already-frozen KL720 NEF and the
already-frozen development-only BIE smoke reference before full physical
holdout inference is authorized.
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
FROZEN_NEF_DEPLOYMENT_FREEZE_SHA256 = (
    "f818b9d912017a1d8fdc4aafd4bf321b8bf5b84e8214e6ef0ba029b48b0b90bb"
)
FROZEN_PRE_HARDWARE_REFERENCE_SUMMARY_SHA256 = (
    "18d261568361e22f7acb704b9ab15022fe4feca27d9a76776631a8f5e202286d"
)
FROZEN_PHYSICAL_SMOKE_SUMMARY_SHA256 = (
    "ec72104a5d0ed958b8dc84c4e7ae289cf1286b845e9ccd082255b1d3150be958"
)
FROZEN_SMOKE_INPUT_SHA256 = (
    "6adba2ab37a0cb2224604f120d2f855688768858d581521d90fa70656a75e1c4"
)
FROZEN_BIE_REFERENCE_SHA256 = (
    "2b253970d07ea67761631670cc4ae4a1096cdb24bed2d2353a2d0ef41f95b66a"
)
FROZEN_PHYSICAL_OUTPUT_SHA256 = (
    "2b253970d07ea67761631670cc4ae4a1096cdb24bed2d2353a2d0ef41f95b66a"
)
EXPECTED_SAMPLE_ID = "BPSR1V2D_00324"
EXPECTED_PLATFORM = 720
EXPECTED_MODEL_ID = 32770
EXPECTED_MODEL_VERSION = "8b29"
EXPECTED_USB_PORT = 81
EXPECTED_INPUT_SHAPE = [1, 3, 256, 256]
EXPECTED_OUTPUT_SHAPE = [1, 1, 1, 1]
EXPECTED_REFERENCE_SCALAR = 2.498514175415039
EXPECTED_PHYSICAL_SCALAR = 2.498514175415039
EXPECTED_ABSOLUTE_ERROR = 0.0
EXPECTED_MAX_ABS_GATE = 1.0e-4


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
    nef_root = model_root / "nasa_bps_53bp1" / "r1_v2_deployment" / "kneron_nef"
    physical_root = nef_root / "physical_smoke_hardware"
    reference_root = nef_root / "physical_smoke_reference"

    parser.add_argument(
        "--nef", default=str(nef_root / "r1_countnet_v2_final_kl720.nef")
    )
    parser.add_argument(
        "--nef-freeze", default=str(nef_root / "nef_deployment_candidate_freeze.json")
    )
    parser.add_argument(
        "--pre-hardware-reference-summary",
        default=str(reference_root / "smoke_reference_summary.json"),
    )
    parser.add_argument(
        "--physical-smoke-summary",
        default=str(physical_root / "physical_smoke_summary.json"),
    )
    parser.add_argument(
        "--physical-output", default=str(physical_root / "physical_output.npy")
    )
    args = parser.parse_args()

    nef_path = Path(args.nef).resolve()
    nef_freeze_path = Path(args.nef_freeze).resolve()
    pre_summary_path = Path(args.pre_hardware_reference_summary).resolve()
    physical_summary_path = Path(args.physical_smoke_summary).resolve()
    physical_output_path = Path(args.physical_output).resolve()
    freeze_path = physical_root / "physical_smoke_pass_freeze.json"

    nef_sha = check_sha(nef_path, FROZEN_NEF_SHA256, "Frozen NEF")
    nef_freeze_sha = check_sha(
        nef_freeze_path,
        FROZEN_NEF_DEPLOYMENT_FREEZE_SHA256,
        "NEF deployment freeze",
    )
    pre_summary_sha = check_sha(
        pre_summary_path,
        FROZEN_PRE_HARDWARE_REFERENCE_SUMMARY_SHA256,
        "Pre-hardware smoke reference summary",
    )
    physical_summary_sha = check_sha(
        physical_summary_path,
        FROZEN_PHYSICAL_SMOKE_SUMMARY_SHA256,
        "Physical smoke summary",
    )
    physical_output_sha = check_sha(
        physical_output_path,
        FROZEN_PHYSICAL_OUTPUT_SHA256,
        "Physical smoke output",
    )

    pre = read_json(pre_summary_path)
    physical = read_json(physical_summary_path)

    pre_checks = {
        "status": "FROZEN_PRE_HARDWARE_ONE_SAMPLE_BIE_REFERENCE",
        "nef_deployment_freeze_sha256": FROZEN_NEF_DEPLOYMENT_FREEZE_SHA256,
        "nef_sha256": FROZEN_NEF_SHA256,
        "selected_sample_id": EXPECTED_SAMPLE_ID,
        "smoke_input_sha256": FROZEN_SMOKE_INPUT_SHA256,
        "bie_reference_sha256": FROZEN_BIE_REFERENCE_SHA256,
        "physical_bie_max_abs_gate": EXPECTED_MAX_ABS_GATE,
        "gate_frozen_before_hardware_result": True,
        "final_holdout_read": False,
        "ptq_bie_or_nef_changed": False,
    }
    for key, expected in pre_checks.items():
        if pre.get(key) != expected:
            raise RuntimeError(
                f"Pre-hardware reference invariant mismatch for {key}: "
                f"expected {expected!r}, got {pre.get(key)!r}"
            )

    physical_checks = {
        "status": "COMPLETE_PHYSICAL_KL720_ONE_SAMPLE_NASA_R1_V2_SMOKE",
        "nef_sha256": FROZEN_NEF_SHA256,
        "nef_deployment_freeze_sha256": FROZEN_NEF_DEPLOYMENT_FREEZE_SHA256,
        "pre_hardware_smoke_reference_summary_sha256": FROZEN_PRE_HARDWARE_REFERENCE_SUMMARY_SHA256,
        "sample_id": EXPECTED_SAMPLE_ID,
        "input_sha256": FROZEN_SMOKE_INPUT_SHA256,
        "bie_reference_sha256": FROZEN_BIE_REFERENCE_SHA256,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version_frozen": EXPECTED_MODEL_VERSION,
        "usb_port": EXPECTED_USB_PORT,
        "nef_input_shape": EXPECTED_INPUT_SHAPE,
        "nef_output_shape": EXPECTED_OUTPUT_SHAPE,
        "physical_bie_max_abs_gate": EXPECTED_MAX_ABS_GATE,
        "gate_frozen_before_hardware_result": True,
        "physical_bie_numerical_smoke_pass": True,
        "physical_output_sha256": FROZEN_PHYSICAL_OUTPUT_SHA256,
        "final_holdout_read": False,
        "ptq_bie_or_nef_changed": False,
        "post_holdout_ptq_retuning_authorized": False,
        "authorized_next_stage": "FREEZE_PHYSICAL_SMOKE_THEN_RUN_FULL_HOLDOUT_HARDWARE_EQUIVALENCE",
    }
    for key, expected in physical_checks.items():
        if physical.get(key) != expected:
            raise RuntimeError(
                f"Physical smoke invariant mismatch for {key}: "
                f"expected {expected!r}, got {physical.get(key)!r}"
            )

    reference_scalar = float(physical.get("bie_reference_scalar"))
    physical_scalar = float(physical.get("physical_scalar"))
    absolute_error = float(physical.get("absolute_error"))
    if reference_scalar != EXPECTED_REFERENCE_SCALAR:
        raise RuntimeError(
            f"Unexpected BIE reference scalar: {reference_scalar}"
        )
    if physical_scalar != EXPECTED_PHYSICAL_SCALAR:
        raise RuntimeError(f"Unexpected physical scalar: {physical_scalar}")
    if absolute_error != EXPECTED_ABSOLUTE_ERROR:
        raise RuntimeError(f"Unexpected physical/BIE absolute error: {absolute_error}")
    if absolute_error > EXPECTED_MAX_ABS_GATE:
        raise RuntimeError("Physical smoke numerical gate did not pass")

    pack = physical.get("kl720_pack")
    if not isinstance(pack, dict):
        raise RuntimeError("Physical smoke summary missing KL720 packing metadata")
    pack_checks = {
        "radix": 7,
        "scale": 1.0,
        "quantization_factor": 128.0,
        "layout": "4W4C8B",
        "quantized_min": 0,
        "quantized_max": 127,
        "buffer_bytes": 262144,
    }
    for key, expected in pack_checks.items():
        if pack.get(key) != expected:
            raise RuntimeError(
                f"KL720 packing invariant mismatch for {key}: "
                f"expected {expected!r}, got {pack.get(key)!r}"
            )

    if physical_output_sha != FROZEN_BIE_REFERENCE_SHA256:
        raise RuntimeError(
            "Physical output and frozen BIE reference should be byte-identical for this smoke result"
        )

    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_PHYSICAL_KL720_ONE_SAMPLE_SMOKE_PASS_BEFORE_FULL_HOLDOUT",
        "nef_sha256": nef_sha,
        "nef_deployment_freeze_sha256": nef_freeze_sha,
        "pre_hardware_smoke_reference_summary_sha256": pre_summary_sha,
        "physical_smoke_summary_sha256": physical_summary_sha,
        "smoke_input_sha256": FROZEN_SMOKE_INPUT_SHA256,
        "bie_reference_sha256": FROZEN_BIE_REFERENCE_SHA256,
        "physical_output_sha256": physical_output_sha,
        "sample_id": EXPECTED_SAMPLE_ID,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "usb_port": EXPECTED_USB_PORT,
        "physical_bie_max_abs_gate": EXPECTED_MAX_ABS_GATE,
        "physical_bie_absolute_error": absolute_error,
        "physical_output_byte_identical_to_bie_reference": True,
        "physical_bie_numerical_smoke_pass": True,
        "kl720_pack": pack_checks,
        "hardware_send_receive_ms_descriptive": float(
            physical.get("hardware_send_receive_ms")
        ),
        "final_holdout_read_by_smoke_stage": False,
        "ptq_bie_or_nef_changed_by_smoke_stage": False,
        "post_holdout_ptq_retuning_authorized": False,
        "authorized_next_stage": "FULL_FROZEN_HOLDOUT_PHYSICAL_KL720_BIOLOGICAL_EQUIVALENCE",
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
                f"Existing physical-smoke freeze differs from accepted result: {freeze_path}"
            )
        print("Radiation Edge AI - NASA BPS R1 v2 physical KL720 smoke PASS freeze")
        print("existing physical smoke freeze verified without overwrite: YES")
    else:
        freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
        print("Radiation Edge AI - NASA BPS R1 v2 physical KL720 smoke PASS freeze")
        print("new physical smoke freeze written: YES")

    print(f"NEF SHA256: {nef_sha}")
    print(f"NEF deployment freeze SHA256: {nef_freeze_sha}")
    print(f"pre-hardware reference summary SHA256: {pre_summary_sha}")
    print(f"physical smoke summary SHA256: {physical_summary_sha}")
    print(f"physical output SHA256: {physical_output_sha}")
    print("physical output byte-identical to frozen BIE reference: YES")
    print(
        f"physical/BIE absolute error: {absolute_error:.10g} <= "
        f"{EXPECTED_MAX_ABS_GATE:.1e}: PASS"
    )
    print("final holdout read by this freeze: NO")
    print("post-holdout PTQ retuning authorized: NO")
    print(f"freeze: {freeze_path}")
    print(f"freeze SHA256: {sha256_file(freeze_path)}")
    print("NASA BPS R1 V2 PHYSICAL KL720 ONE-SAMPLE SMOKE FROZEN: YES")
    print("NEXT GATE: run the full frozen 2058-nucleus holdout on this exact KL720 NEF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
