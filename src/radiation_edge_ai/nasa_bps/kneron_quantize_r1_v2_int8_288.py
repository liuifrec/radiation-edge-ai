"""Quantize the frozen NASA BPS R1 v2 KL720 ONNX with the frozen 288-dev panel.

Run inside the pinned Kneron Toolchain image used by this project. This stage is
PTQ/BIE generation only. It refuses all outcome-guided tuning: the optimized
ONNX, Kneron floating-point report, calibration manifest, calibration freeze,
model identity, toolchain version, and PTQ settings are pinned in source.

No final-holdout image or phenotype table is read here. BIE biological fidelity
is evaluated only in the next fixed deployment-equivalence stage.
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import ktc
import numpy as np
import onnx

EXPECTED_SHAPE = (1, 3, 256, 256)
EXPECTED_TENSORS = 288
EXPECTED_BAGS = 72
EXPECTED_PER_BAG = 4
EXPECTED_PLATFORM = "720"
EXPECTED_MODEL_ID = 32770
EXPECTED_MODEL_VERSION = "8b29"
EXPECTED_TOOLCHAIN_VERSION = "kneron/toolchain:v0.33.1"
EXPECTED_OPTIMIZER_BACKEND = "ktc.onnx_optimizer.onnx2onnx_flow"

FROZEN_SOURCE_ONNX_SHA256 = (
    "a65718a8f07d5bcd8707bae78d8f730d56047db21e575fa55ba5396adc49a07e"
)
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
PINNED_TOOLCHAIN_IMAGE = (
    "kneron/toolchain@sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041"
)

# Predeclared PTQ configuration. These values are intentionally not CLI-tunable.
PTQ_CONFIG = {
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


def graph_input_names(model):
    initializer_names = {item.name for item in model.graph.initializer}
    return [item.name for item in model.graph.input if item.name not in initializer_names]


def toolchain_version():
    for candidate in (Path("/workspace/version.txt"), Path("/workspace/VERSION")):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace").strip()
    return None


def find_newest(filename, started):
    roots = (Path("/data1/kneron_flow"), Path("/workspace/.tmp"), Path("/tmp"))
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        try:
            direct = root / filename
            if direct.is_file():
                candidates.append(direct)
            for path in root.rglob(filename):
                if path.is_file() and path not in candidates:
                    candidates.append(path)
        except OSError:
            continue
    if not candidates:
        return None
    fresh = [path for path in candidates if path.stat().st_mtime >= started - 5.0]
    pool = fresh if fresh else candidates
    return max(pool, key=lambda path: path.stat().st_mtime)


def copy_if_present(filename, output_dir, started):
    source = find_newest(filename, started)
    if source is None:
        return None
    destination = output_dir / filename
    if source.resolve() != destination.resolve():
        shutil.copy2(str(source), str(destination))
    return str(destination)


def verify_fp_report(report_path):
    actual_sha = sha256_file(report_path)
    if actual_sha != FROZEN_KNERON_FP_REPORT_SHA256:
        raise RuntimeError(
            "Kneron FP report SHA256 mismatch: expected {}, got {}".format(
                FROZEN_KNERON_FP_REPORT_SHA256, actual_sha
            )
        )
    report = read_json(report_path)
    checks = {
        "status": "COMPLETE_KL720_FLOATING_POINT_OPTIMIZATION_EVALUATION",
        "frozen_onnx_sha256": FROZEN_SOURCE_ONNX_SHA256,
        "optimized_onnx_sha256": FROZEN_OPTIMIZED_ONNX_SHA256,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "toolchain_version": EXPECTED_TOOLCHAIN_VERSION,
        "optimizer_backend": EXPECTED_OPTIMIZER_BACKEND,
        "hardware_supported": True,
        "ptq_performed": False,
        "bie_generated": False,
        "nef_compiled": False,
        "final_holdout_read": False,
    }
    for key, expected in checks.items():
        if report.get(key) != expected:
            raise RuntimeError(
                "Kneron FP report invariant mismatch for {}: expected {!r}, got {!r}".format(
                    key, expected, report.get(key)
                )
            )
    return report


def verify_calibration(manifest_path, freeze_path):
    manifest_sha = sha256_file(manifest_path)
    freeze_sha = sha256_file(freeze_path)
    if manifest_sha != FROZEN_CALIBRATION_MANIFEST_SHA256:
        raise RuntimeError(
            "Calibration manifest SHA256 mismatch: expected {}, got {}".format(
                FROZEN_CALIBRATION_MANIFEST_SHA256, manifest_sha
            )
        )
    if freeze_sha != FROZEN_CALIBRATION_FREEZE_SHA256:
        raise RuntimeError(
            "Calibration freeze SHA256 mismatch: expected {}, got {}".format(
                FROZEN_CALIBRATION_FREEZE_SHA256, freeze_sha
            )
        )

    manifest = read_json(manifest_path)
    freeze = read_json(freeze_path)
    if manifest.get("status") != "FROZEN_DEVELOPMENT_ONLY_INT8_CALIBRATION_PANEL":
        raise RuntimeError("Unexpected calibration manifest status")
    if freeze.get("status") != "FROZEN_VERIFIED_DEVELOPMENT_ONLY_INT8_CALIBRATION_PANEL":
        raise RuntimeError("Unexpected calibration freeze status")
    if manifest.get("n_tensors") != EXPECTED_TENSORS or freeze.get("n_tensors") != EXPECTED_TENSORS:
        raise RuntimeError("Calibration tensor count is not frozen at 288")
    if tuple(manifest.get("tile", ())) != EXPECTED_SHAPE:
        raise RuntimeError("Unexpected calibration manifest tensor shape")
    if tuple(freeze.get("shape", ())) != EXPECTED_SHAPE:
        raise RuntimeError("Unexpected calibration freeze tensor shape")
    if manifest.get("dtype") != "float32" or freeze.get("dtype") != "float32":
        raise RuntimeError("Calibration dtype is not float32")
    if freeze.get("bags") != EXPECTED_BAGS or freeze.get("selected_per_bag") != EXPECTED_PER_BAG:
        raise RuntimeError("Calibration freeze does not represent 72 bags x 4 tensors")
    if freeze.get("all_tensor_hashes_verified") is not True:
        raise RuntimeError("Calibration freeze did not verify all tensor hashes")
    if freeze.get("final_holdout_used_for_calibration") is not False:
        raise RuntimeError("Final holdout is not permitted for calibration")
    if freeze.get("phenotype_or_prediction_guided_selection") is not False:
        raise RuntimeError("Outcome-guided calibration selection is not permitted")

    authorized = freeze.get("authorized_ptq_config") or {}
    expected_authorized = {
        "platform": EXPECTED_PLATFORM,
        "datapath_range_method": PTQ_CONFIG["datapath_range_method"],
        "percentage": PTQ_CONFIG["percentage"],
        "percentage_16b": PTQ_CONFIG["percentage_16b"],
        "optimize": PTQ_CONFIG["optimize"],
    }
    if authorized != expected_authorized:
        raise RuntimeError(
            "Calibration freeze PTQ authorization mismatch: expected {!r}, got {!r}".format(
                expected_authorized, authorized
            )
        )
    return manifest, freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--kneron-fp-report", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--calibration-freeze", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    onnx_path = Path(args.onnx).resolve()
    fp_report_path = Path(args.kneron_fp_report).resolve()
    manifest_path = Path(args.calibration_manifest).resolve()
    freeze_path = Path(args.calibration_freeze).resolve()
    output_dir = Path(args.output_dir).resolve()

    for path in (onnx_path, fp_report_path, manifest_path, freeze_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    bie_destination = output_dir / "r1_countnet_v2_final_kl720_int8.bie"
    summary_path = output_dir / "int8_ptq_summary.json"
    if summary_path.exists() or bie_destination.exists():
        raise RuntimeError(
            "Frozen PTQ output already exists; refusing overwrite: {}".format(output_dir)
        )

    version = toolchain_version()
    if version != EXPECTED_TOOLCHAIN_VERSION:
        raise RuntimeError(
            "Toolchain version mismatch: expected {!r}, got {!r}".format(
                EXPECTED_TOOLCHAIN_VERSION, version
            )
        )

    onnx_sha = sha256_file(onnx_path)
    if onnx_sha != FROZEN_OPTIMIZED_ONNX_SHA256:
        raise RuntimeError(
            "Optimized ONNX SHA256 mismatch: expected {}, got {}".format(
                FROZEN_OPTIMIZED_ONNX_SHA256, onnx_sha
            )
        )
    fp_report = verify_fp_report(fp_report_path)
    manifest, calibration_freeze = verify_calibration(manifest_path, freeze_path)

    print("Radiation Edge AI - NASA BPS R1 v2 frozen KL720 INT8 PTQ")
    print("optimized ONNX SHA256: {}".format(onnx_sha))
    print("Kneron FP report SHA256: {}".format(FROZEN_KNERON_FP_REPORT_SHA256))
    print("calibration manifest SHA256: {}".format(FROZEN_CALIBRATION_MANIFEST_SHA256))
    print("calibration freeze SHA256: {}".format(FROZEN_CALIBRATION_FREEZE_SHA256))
    print("toolchain image: {}".format(PINNED_TOOLCHAIN_IMAGE))
    print("toolchain version: {}".format(version))
    print("platform/model: {}/{}:{}".format(EXPECTED_PLATFORM, EXPECTED_MODEL_ID, EXPECTED_MODEL_VERSION))
    print("PTQ config frozen before BIE/holdout result: YES")
    print("PTQ config: {}".format(json.dumps(PTQ_CONFIG, sort_keys=True)))
    print("final holdout used for calibration: NO")
    print("final holdout read by this stage: NO")

    print("")
    print("[Load optimized ONNX]")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    input_names = graph_input_names(model)
    if len(input_names) != 1:
        raise RuntimeError("Expected exactly one model input, got {!r}".format(input_names))
    input_name = input_names[0]
    if manifest.get("input_name") and manifest.get("input_name") != input_name:
        raise RuntimeError(
            "Calibration input name {!r} does not match optimized ONNX {!r}".format(
                manifest.get("input_name"), input_name
            )
        )
    print("ONNX checker: PASS")
    print("input: {}".format(input_name))

    print("")
    print("[Load and re-verify frozen 288 calibration tensors]")
    calibration_root = manifest_path.parent
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_TENSORS:
        raise RuntimeError("Calibration manifest records are not exactly 288")

    frozen_tensor_sha = {}
    for item in calibration_freeze.get("tensor_identities", []):
        frozen_tensor_sha[str(item.get("tensor"))] = str(item.get("sha256"))
    if len(frozen_tensor_sha) != EXPECTED_TENSORS:
        raise RuntimeError("Calibration freeze tensor identity table is not exactly 288")

    arrays = []
    bag_counts = {}
    global_min = float("inf")
    global_max = float("-inf")
    for index, record in enumerate(records):
        tensor_rel = str(record.get("tensor", ""))
        tensor_path = calibration_root / tensor_rel
        if not tensor_path.is_file():
            raise FileNotFoundError(tensor_path)
        tensor_sha = sha256_file(tensor_path)
        if tensor_sha != record.get("tensor_sha256"):
            raise RuntimeError("Manifest tensor hash mismatch: {}".format(tensor_path))
        if tensor_sha != frozen_tensor_sha.get(tensor_rel):
            raise RuntimeError("Calibration-freeze tensor hash mismatch: {}".format(tensor_path))
        array = np.load(str(tensor_path), allow_pickle=False)
        if array.shape != EXPECTED_SHAPE:
            raise RuntimeError("Unexpected calibration shape {}: {}".format(array.shape, tensor_path))
        if array.dtype != np.float32:
            raise RuntimeError("Unexpected calibration dtype {}: {}".format(array.dtype, tensor_path))
        if not np.isfinite(array).all():
            raise RuntimeError("Non-finite calibration values: {}".format(tensor_path))
        if not np.all(array[:, 2, :, :] == 0.0):
            raise RuntimeError("Third calibration channel is not zero: {}".format(tensor_path))
        array = np.ascontiguousarray(array, dtype=np.float32)
        arrays.append(array)
        global_min = min(global_min, float(array.min()))
        global_max = max(global_max, float(array.max()))
        sample_name = str(record.get("sample_name", ""))
        bag_counts[sample_name] = bag_counts.get(sample_name, 0) + 1
        if (index + 1) % 36 == 0 or index + 1 == EXPECTED_TENSORS:
            print("calibration tensors verified: {}/{}".format(index + 1, EXPECTED_TENSORS))

    if len(bag_counts) != EXPECTED_BAGS or set(bag_counts.values()) != {EXPECTED_PER_BAG}:
        raise RuntimeError("Calibration tensors are not exactly 72 bags x 4")
    print("calibration range: [{:.6f}, {:.6f}]".format(global_min, global_max))
    print("bags represented: {}/{}; per bag={}".format(len(bag_counts), EXPECTED_BAGS, EXPECTED_PER_BAG))

    print("")
    print("[Create KL720 ModelConfig]")
    km = ktc.ModelConfig(
        EXPECTED_MODEL_ID,
        EXPECTED_MODEL_VERSION,
        EXPECTED_PLATFORM,
        onnx_model=model,
    )
    print("ModelConfig: PASS")

    print("")
    print("[Frozen fixed-point analysis / PTQ]")
    input_mapping = {input_name: arrays}
    old_cwd = Path.cwd()
    started = time.time()
    try:
        os.chdir(str(output_dir))
        bie_returned = Path(
            km.analysis(
                input_mapping,
                threads=PTQ_CONFIG["threads"],
                datapath_range_method=PTQ_CONFIG["datapath_range_method"],
                percentage=PTQ_CONFIG["percentage"],
                percentage_16b=PTQ_CONFIG["percentage_16b"],
                percentile=PTQ_CONFIG["percentile"],
                outlier_factor=PTQ_CONFIG["outlier_factor"],
                optimize=PTQ_CONFIG["optimize"],
            )
        ).resolve()
    finally:
        os.chdir(str(old_cwd))

    if not bie_returned.is_file():
        raise RuntimeError("Kneron analysis returned missing BIE path: {}".format(bie_returned))
    if bie_returned.resolve() != bie_destination.resolve():
        shutil.copy2(str(bie_returned), str(bie_destination))

    artifacts = {}
    for filename in (
        "model_fx_report.html",
        "model_fx_report.json",
        "analysis.log",
        "quantize.log",
        "batch_compile.log",
        "backtrace.log",
        "ioinfo.csv",
    ):
        artifacts[filename] = copy_if_present(filename, output_dir, started)

    elapsed = time.time() - started
    bie_sha = sha256_file(bie_destination)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_FROZEN_DEVELOPMENT_ONLY_KL720_INT8_PTQ",
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "source_fp32_onnx_sha256": FROZEN_SOURCE_ONNX_SHA256,
        "optimized_onnx_sha256": onnx_sha,
        "kneron_fp_report_sha256": FROZEN_KNERON_FP_REPORT_SHA256,
        "kneron_fp_hardware_supported": fp_report.get("hardware_supported"),
        "calibration_manifest_sha256": FROZEN_CALIBRATION_MANIFEST_SHA256,
        "calibration_freeze_sha256": FROZEN_CALIBRATION_FREEZE_SHA256,
        "calibration_count": len(arrays),
        "calibration_bags": len(bag_counts),
        "calibration_per_bag": EXPECTED_PER_BAG,
        "calibration_shape": list(EXPECTED_SHAPE),
        "calibration_dtype": "float32",
        "calibration_global_min": global_min,
        "calibration_global_max": global_max,
        "input_name": input_name,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "pinned_toolchain_image": PINNED_TOOLCHAIN_IMAGE,
        "toolchain_version": version,
        "ptq_config": PTQ_CONFIG,
        "ptq_config_frozen_before_bie_result": True,
        "outcome_guided_quantization_tuning": False,
        "final_holdout_used_for_calibration": False,
        "final_holdout_read": False,
        "bie_returned": str(bie_returned),
        "bie_persistent": str(bie_destination),
        "bie_sha256": bie_sha,
        "bie_size_bytes": bie_destination.stat().st_size,
        "elapsed_seconds": elapsed,
        "artifacts": artifacts,
        "authorized_next_stage": "FIXED_BIE_DEPLOYMENT_EQUIVALENCE_EVALUATION",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("NASA BPS R1 V2 KL720 FROZEN INT8 PTQ COMPLETE: YES")
    print("BIE GENERATED: YES")
    print("BIE: {}".format(bie_destination))
    print("BIE SHA256: {}".format(bie_sha))
    print("BIE size: {:.2f} MiB".format(bie_destination.stat().st_size / 1024.0 ** 2))
    print("Elapsed: {:.1f} s".format(elapsed))
    print("Summary: {}".format(summary_path))
    print("Summary SHA256: {}".format(sha256_file(summary_path)))
    print("final holdout used for calibration: NO")
    print("final holdout read by PTQ stage: NO")
    print("NEXT GATE: fixed BIE inference on the frozen deployment-equivalence holdout; do not tune PTQ settings after seeing that result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
