"""Verify compiled NASA BPS R1 v2 KL720 NEF against the accepted frozen BIE.

This is a compile-integrity check only. It uses one already-frozen DEVELOPMENT
calibration tensor, not the final holdout, and compares Kneron fixed-point BIE
simulation with compiled NEF simulation. No optimization, calibration, PTQ,
BIE generation, biological evaluation, or hardware inference occurs here.

Python 3.7 compatible for the pinned Kneron Toolchain v0.33.1 image.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import ktc
import numpy as np

FROZEN_BIE_SHA256 = (
    "c52b8a78c595347667bad7a34950af92adbe734179b913646d06513a88fc1dd8"
)
FROZEN_NEF_SHA256 = (
    "d04f02855a8a82ae7cb5ff48af91eee69c341142d1097469890385bd23ee097b"
)
FROZEN_NEF_COMPILE_SUMMARY_SHA256 = (
    "813213f224da7ec9a9621a27c92b45ab87f8909eb0c7e203403f5a8c6e64c2ea"
)
FROZEN_EQUIVALENCE_FREEZE_SHA256 = (
    "e901d4be044c5ea5507b3b60643043eb8aa0af5643a0b7fba2f9bdd7c588bb0a"
)
FROZEN_CALIBRATION_MANIFEST_SHA256 = (
    "78c9ed83e1992907264dfeec9ef6e2a434868bd3f299672f610a42a0018b5d4e"
)
EXPECTED_TOOLCHAIN_VERSION = "kneron/toolchain:v0.33.1"
EXPECTED_PLATFORM = 720
EXPECTED_MODEL_ID = 32770
EXPECTED_MODEL_VERSION = "8b29"
EXPECTED_INPUT_SHAPE = (1, 3, 256, 256)
MAX_ABS_ERROR_GATE = 1.0e-4


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


def toolchain_version():
    for candidate in (Path("/workspace/version.txt"), Path("/workspace/VERSION")):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace").strip()
    return None


def one_output(result, label):
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise RuntimeError(
            "{} expected exactly one output, got {} len={}".format(
                label,
                type(result),
                len(result) if hasattr(result, "__len__") else "NA",
            )
        )
    array = np.asarray(result[0], dtype=np.float32)
    if array.size != 1:
        raise RuntimeError("{} expected scalar output, got shape {}".format(label, array.shape))
    if not np.isfinite(array).all():
        raise RuntimeError("{} output contains non-finite values".format(label))
    return np.ascontiguousarray(array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bie", required=True)
    parser.add_argument("--nef", required=True)
    parser.add_argument("--nef-compile-summary", required=True)
    parser.add_argument("--equivalence-freeze", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bie_path = Path(args.bie).resolve()
    nef_path = Path(args.nef).resolve()
    compile_summary_path = Path(args.nef_compile_summary).resolve()
    equivalence_freeze_path = Path(args.equivalence_freeze).resolve()
    calibration_manifest_path = Path(args.calibration_manifest).resolve()
    output_path = Path(args.output).resolve()

    if output_path.exists():
        raise RuntimeError(
            "NEF/BIE compile-integrity result already exists; refusing overwrite: {}".format(
                output_path
            )
        )

    bie_sha = check_sha(bie_path, FROZEN_BIE_SHA256, "Accepted BIE")
    nef_sha = check_sha(nef_path, FROZEN_NEF_SHA256, "Compiled NEF")
    compile_sha = check_sha(
        compile_summary_path,
        FROZEN_NEF_COMPILE_SUMMARY_SHA256,
        "NEF compile summary",
    )
    equivalence_sha = check_sha(
        equivalence_freeze_path,
        FROZEN_EQUIVALENCE_FREEZE_SHA256,
        "BIE biological-equivalence freeze",
    )
    calibration_sha = check_sha(
        calibration_manifest_path,
        FROZEN_CALIBRATION_MANIFEST_SHA256,
        "Frozen development calibration manifest",
    )

    compile_summary = read_json(compile_summary_path)
    checks = {
        "status": "COMPLETE_KL720_NEF_COMPILE_FROM_ACCEPTED_BIE",
        "source_bie_sha256": FROZEN_BIE_SHA256,
        "bie_equivalence_freeze_sha256": FROZEN_EQUIVALENCE_FREEZE_SHA256,
        "nef_sha256": FROZEN_NEF_SHA256,
        "platform": "720",
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "onnx_optimization_repeated": False,
        "ptq_repeated": False,
        "bie_regenerated": False,
        "holdout_inference_performed": False,
        "authorized_next_stage": "ONE_SAMPLE_NEF_VS_BIE_COMPILE_INTEGRITY_GATE",
    }
    for key, expected in checks.items():
        if compile_summary.get(key) != expected:
            raise RuntimeError(
                "NEF compile summary mismatch for {}: expected {!r}, got {!r}".format(
                    key, expected, compile_summary.get(key)
                )
            )

    calibration = read_json(calibration_manifest_path)
    if calibration.get("status") != "FROZEN_DEVELOPMENT_ONLY_INT8_CALIBRATION_PANEL":
        raise RuntimeError("Unexpected calibration manifest status")
    if tuple(calibration.get("tile", ())) != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(
            "Unexpected calibration tile: {}".format(calibration.get("tile"))
        )
    if calibration.get("dtype") != "float32":
        raise RuntimeError("Unexpected calibration dtype: {}".format(calibration.get("dtype")))
    records = calibration.get("records")
    if not isinstance(records, list) or len(records) != 288:
        raise RuntimeError("Expected 288 calibration records")

    # Deterministically use record 0 from the already-frozen development panel.
    record = records[0]
    tensor_path = calibration_manifest_path.parent / str(record["tensor"])
    check_sha(tensor_path, str(record["tensor_sha256"]), "Selected calibration tensor")
    tensor = np.load(str(tensor_path), allow_pickle=False)
    if tensor.shape != EXPECTED_INPUT_SHAPE or tensor.dtype != np.float32:
        raise RuntimeError(
            "Unexpected selected tensor {} {}".format(tensor.shape, tensor.dtype)
        )
    if not np.isfinite(tensor).all():
        raise RuntimeError("Selected calibration tensor contains non-finite values")
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    input_name = str(calibration.get("input_name", "input"))

    version = toolchain_version()
    if version != EXPECTED_TOOLCHAIN_VERSION:
        raise RuntimeError(
            "Unexpected Kneron toolchain version: expected {}, got {}".format(
                EXPECTED_TOOLCHAIN_VERSION, version or "UNKNOWN"
            )
        )

    print("Radiation Edge AI - NASA BPS R1 v2 KL720 NEF vs BIE compile-integrity gate")
    print("accepted BIE SHA256: {}".format(bie_sha))
    print("compiled NEF SHA256: {}".format(nef_sha))
    print("NEF compile summary SHA256: {}".format(compile_sha))
    print("BIE equivalence freeze SHA256: {}".format(equivalence_sha))
    print("development calibration manifest SHA256: {}".format(calibration_sha))
    print("toolchain version: {}".format(version))
    print("platform/model: {}/{}:{}".format(EXPECTED_PLATFORM, EXPECTED_MODEL_ID, EXPECTED_MODEL_VERSION))
    print("selected development tensor: {}".format(record.get("sample_id")))
    print("selected tensor index: {}".format(record.get("index")))
    print("final holdout read by this stage: NO")
    print("compile-integrity max-abs gate frozen before result: <= {:.1e}".format(MAX_ABS_ERROR_GATE))

    started = time.perf_counter()
    bie_result = ktc.kneron_inference(
        [tensor],
        bie_file=str(bie_path),
        input_names=[input_name],
        platform=EXPECTED_PLATFORM,
    )
    bie_ms = (time.perf_counter() - started) * 1000.0
    bie = one_output(bie_result, "BIE")

    started = time.perf_counter()
    nef_result = ktc.kneron_inference(
        [tensor],
        nef_file=str(nef_path),
        input_names=[input_name],
        platform=EXPECTED_PLATFORM,
    )
    nef_ms = (time.perf_counter() - started) * 1000.0
    nef = one_output(nef_result, "NEF")

    if bie.shape != nef.shape:
        raise RuntimeError("Output shape mismatch: BIE={} NEF={}".format(bie.shape, nef.shape))

    diff = nef.astype(np.float64) - bie.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    exact = bool(np.array_equal(bie, nef))
    passed = bool(max_abs <= MAX_ABS_ERROR_GATE)

    payload = {
        "status": "COMPLETE_ONE_SAMPLE_NEF_VS_BIE_COMPILE_INTEGRITY_GATE",
        "source_bie_sha256": bie_sha,
        "nef_sha256": nef_sha,
        "nef_compile_summary_sha256": compile_sha,
        "bie_equivalence_freeze_sha256": equivalence_sha,
        "calibration_manifest_sha256": calibration_sha,
        "toolchain_version": version,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "input_name": input_name,
        "selected_tensor_index": int(record.get("index", 0)),
        "selected_sample_id": str(record.get("sample_id", "")),
        "selected_tensor_sha256": str(record["tensor_sha256"]),
        "output_shape": list(bie.shape),
        "exact_array_equal": exact,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rmse": rmse,
        "max_abs_error_gate": MAX_ABS_ERROR_GATE,
        "verification_pass": passed,
        "bie_simulator_ms": bie_ms,
        "nef_simulator_ms": nef_ms,
        "final_holdout_read": False,
        "ptq_or_bie_changed": False,
        "authorized_next_stage": (
            "FREEZE_NEF_AND_MOVE_TO_PHYSICAL_KL720"
            if passed
            else "BLOCKED_COMPILE_INTEGRITY_FAILURE"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("")
    print("BIE output shape: {}".format(tuple(bie.shape)))
    print("NEF output shape: {}".format(tuple(nef.shape)))
    print("exact array equality: {}".format("YES" if exact else "NO"))
    print("max absolute error: {:.8g}".format(max_abs))
    print("mean absolute error: {:.8g}".format(mean_abs))
    print("RMSE: {:.8g}".format(rmse))
    print("BIE simulator: {:.1f} ms".format(bie_ms))
    print("NEF simulator: {:.1f} ms".format(nef_ms))
    print("NASA BPS R1 V2 KL720 NEF/BIE COMPILE INTEGRITY PASS: {}".format("YES" if passed else "NO"))
    print("Summary: {}".format(output_path))
    print("Summary SHA256: {}".format(sha256_file(output_path)))
    print("final holdout read: NO")
    print("PTQ/BIE changed: NO")
    if passed:
        print("NEXT GATE: freeze this exact NEF identity, then run physical KL720 inference.")
        return 0
    print("NEXT GATE: STOP; diagnose compile-integrity failure without PTQ retuning.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
