"""Prepare a frozen one-sample BIE reference for physical KL720 smoke testing.

This stage runs only in the pinned Kneron Toolchain image. It uses record 0 from
the already-frozen 288-development-tensor calibration panel, verifies the exact
accepted BIE and the exact NEF deployment freeze, runs BIE simulation once, and
materializes the input tensor plus scalar BIE reference for the later physical
KL720 smoke test.

No final holdout image or phenotype is read. No PTQ, BIE, or NEF artifact is
changed. The physical-vs-BIE smoke-test numerical gate is frozen here before any
hardware result is observed.

Python 3.7 compatible for Kneron Toolchain v0.33.1.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import ktc
import numpy as np

FROZEN_BIE_SHA256 = "c52b8a78c595347667bad7a34950af92adbe734179b913646d06513a88fc1dd8"
FROZEN_CALIBRATION_MANIFEST_SHA256 = "78c9ed83e1992907264dfeec9ef6e2a434868bd3f299672f610a42a0018b5d4e"
FROZEN_NEF_DEPLOYMENT_FREEZE_SHA256 = "f818b9d912017a1d8fdc4aafd4bf321b8bf5b84e8214e6ef0ba029b48b0b90bb"
FROZEN_NEF_SHA256 = "d04f02855a8a82ae7cb5ff48af91eee69c341142d1097469890385bd23ee097b"
EXPECTED_PLATFORM = 720
EXPECTED_MODEL_ID = 32770
EXPECTED_MODEL_VERSION = "8b29"
EXPECTED_INPUT_SHAPE = (1, 3, 256, 256)
EXPECTED_SAMPLE_ID = "BPSR1V2D_00324"
EXPECTED_TENSOR_INDEX = 0
PHYSICAL_BIE_MAX_ABS_GATE = 1.0e-4


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


def one_scalar(result, label):
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise RuntimeError(
            "{} expected exactly one output, got {} len={}".format(
                label,
                type(result),
                len(result) if hasattr(result, "__len__") else "NA",
            )
        )
    arr = np.asarray(result[0], dtype=np.float32)
    if arr.size != 1 or not np.isfinite(arr).all():
        raise RuntimeError("{} expected one finite scalar, got shape {}".format(label, arr.shape))
    return np.ascontiguousarray(arr.reshape(1), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bie", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--nef-freeze", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    bie_path = Path(args.bie).resolve()
    manifest_path = Path(args.calibration_manifest).resolve()
    nef_freeze_path = Path(args.nef_freeze).resolve()
    output_dir = Path(args.output_dir).resolve()
    input_out = output_dir / "smoke_input.npy"
    bie_out = output_dir / "smoke_bie_reference.npy"
    summary_out = output_dir / "smoke_reference_summary.json"

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Physical smoke reference output is not empty; refusing overwrite: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    bie_sha = check_sha(bie_path, FROZEN_BIE_SHA256, "Accepted BIE")
    manifest_sha = check_sha(
        manifest_path,
        FROZEN_CALIBRATION_MANIFEST_SHA256,
        "Frozen development calibration manifest",
    )
    nef_freeze_sha = check_sha(
        nef_freeze_path,
        FROZEN_NEF_DEPLOYMENT_FREEZE_SHA256,
        "NEF deployment freeze",
    )

    nef_freeze = read_json(nef_freeze_path)
    freeze_checks = {
        "status": "FROZEN_KL720_NEF_AFTER_COMPILE_INTEGRITY_PASS_BEFORE_PHYSICAL_HARDWARE",
        "nef_sha256": FROZEN_NEF_SHA256,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "compile_integrity_pass": True,
        "post_holdout_ptq_retuning_authorized": False,
        "authorized_next_stage": "PHYSICAL_KL720_ONE_SAMPLE_SMOKE_THEN_FROZEN_HARDWARE_EQUIVALENCE",
    }
    for key, expected in freeze_checks.items():
        if nef_freeze.get(key) != expected:
            raise RuntimeError(
                "NEF freeze invariant mismatch for {}: expected {!r}, got {!r}".format(
                    key, expected, nef_freeze.get(key)
                )
            )

    manifest = read_json(manifest_path)
    if manifest.get("status") != "FROZEN_DEVELOPMENT_ONLY_INT8_CALIBRATION_PANEL":
        raise RuntimeError("Unexpected calibration manifest status")
    if tuple(manifest.get("tile", ())) != EXPECTED_INPUT_SHAPE:
        raise RuntimeError("Unexpected calibration tensor shape declaration")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 288:
        raise RuntimeError("Expected 288 frozen calibration records")

    record = records[EXPECTED_TENSOR_INDEX]
    if int(record.get("index", -1)) != EXPECTED_TENSOR_INDEX:
        raise RuntimeError("Unexpected calibration record index")
    if str(record.get("sample_id", "")) != EXPECTED_SAMPLE_ID:
        raise RuntimeError(
            "Unexpected smoke sample: expected {}, got {}".format(
                EXPECTED_SAMPLE_ID, record.get("sample_id")
            )
        )

    source_tensor = manifest_path.parent / str(record["tensor"])
    source_tensor_sha = check_sha(
        source_tensor, str(record["tensor_sha256"]), "Selected development tensor"
    )
    tensor = np.load(str(source_tensor), allow_pickle=False)
    if tensor.shape != EXPECTED_INPUT_SHAPE or tensor.dtype != np.float32:
        raise RuntimeError("Unexpected smoke tensor {} {}".format(tensor.shape, tensor.dtype))
    if not np.isfinite(tensor).all():
        raise RuntimeError("Smoke tensor contains non-finite values")
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)

    print("Radiation Edge AI - NASA BPS R1 v2 physical KL720 smoke reference")
    print("accepted BIE SHA256: {}".format(bie_sha))
    print("NEF deployment freeze SHA256: {}".format(nef_freeze_sha))
    print("frozen NEF SHA256: {}".format(FROZEN_NEF_SHA256))
    print("selected development sample: {}".format(EXPECTED_SAMPLE_ID))
    print("selected tensor index: {}".format(EXPECTED_TENSOR_INDEX))
    print("selected tensor SHA256: {}".format(source_tensor_sha))
    print("input shape/dtype: {} / {}".format(tensor.shape, tensor.dtype))
    print("final holdout read by this stage: NO")
    print("PTQ/BIE/NEF changed by this stage: NO")
    print(
        "physical smoke BIE max-abs gate frozen before hardware result: <= {:.1e}".format(
            PHYSICAL_BIE_MAX_ABS_GATE
        )
    )

    result = ktc.kneron_inference(
        [tensor],
        bie_file=str(bie_path),
        input_names=[str(manifest.get("input_name", "input"))],
        platform=EXPECTED_PLATFORM,
    )
    bie_scalar = one_scalar(result, "BIE")

    shutil.copy2(str(source_tensor), str(input_out))
    if sha256_file(input_out) != source_tensor_sha:
        raise RuntimeError("Smoke input copy SHA mismatch")
    np.save(str(bie_out), bie_scalar, allow_pickle=False)

    summary = {
        "status": "FROZEN_PRE_HARDWARE_ONE_SAMPLE_BIE_REFERENCE",
        "source_bie_sha256": bie_sha,
        "nef_deployment_freeze_sha256": nef_freeze_sha,
        "nef_sha256": FROZEN_NEF_SHA256,
        "calibration_manifest_sha256": manifest_sha,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "selected_tensor_index": EXPECTED_TENSOR_INDEX,
        "selected_sample_id": EXPECTED_SAMPLE_ID,
        "selected_tensor_sha256": source_tensor_sha,
        "smoke_input": str(input_out),
        "smoke_input_sha256": sha256_file(input_out),
        "bie_reference": str(bie_out),
        "bie_reference_sha256": sha256_file(bie_out),
        "bie_reference_shape": list(bie_scalar.shape),
        "bie_reference_dtype": str(bie_scalar.dtype),
        "bie_reference_scalar": float(bie_scalar[0]),
        "physical_bie_max_abs_gate": PHYSICAL_BIE_MAX_ABS_GATE,
        "gate_frozen_before_hardware_result": True,
        "final_holdout_read": False,
        "ptq_bie_or_nef_changed": False,
        "authorized_next_stage": "PHYSICAL_KL720_ONE_SAMPLE_SMOKE_WITH_THIS_EXACT_REFERENCE",
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("PRE-HARDWARE SMOKE REFERENCE FROZEN: YES")
    print("smoke input: {}".format(input_out))
    print("smoke input SHA256: {}".format(sha256_file(input_out)))
    print("BIE reference: {}".format(bie_out))
    print("BIE reference SHA256: {}".format(sha256_file(bie_out)))
    print("BIE reference scalar: {:.10g}".format(float(bie_scalar[0])))
    print("summary: {}".format(summary_out))
    print("summary SHA256: {}".format(sha256_file(summary_out)))
    print("final holdout read: NO")
    print("NEXT GATE: pin this reference summary SHA, then run the physical KL720 on USB port 81.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
