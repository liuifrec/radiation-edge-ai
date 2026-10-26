"""Read-only numerical decomposition of four cached DNAi 512-window logits.

The Kneron floating/BIE cache for the previously developed field has no
contemporaneous per-output source hashes; all results are descriptive, not
causal quantization evidence or a biological-fidelity assessment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_ONNX = "a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0"
EXPECTED_BIE = "01f902e283b565b712cbecbf22c71f19ae6f9ba826d2c6be87e8976cd1bc9863"
EXPECTED_NEF = "4b3dfec9a61c99e186dd4b8482fa5b06e6a4958f325ed4a0db0546f1dcab2bfc"
SAMPLE = "1__tile_4"
WINDOWS = 9
TENSOR_SHAPE = (1, 3, 512, 512)
STAGES = ("fp512_cpu", "kneron_floating_legacy", "v3_bie_legacy", "physical_kl720")
PAIRS = (
    ("cpu_to_kneron_floating", 0, 1),
    ("kneron_floating_to_bie", 1, 2),
    ("bie_to_physical", 2, 3),
    ("cpu_to_physical", 0, 3),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return obj


def required_path(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required source does not exist: {path}")
    return path


def load_logits(path: Path) -> np.ndarray:
    array = np.load(required_path(path), allow_pickle=False)
    if array.shape != TENSOR_SHAPE or array.dtype != np.float32:
        raise ValueError(f"Unexpected float32 BCHW logits shape: {path}: {array.shape} {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite saved logits: {path}")
    return array


def softmax(logits: np.ndarray) -> np.ndarray:
    """Float64 intermediate, float32 return, matching frozen field workers."""
    value = logits.astype(np.float64, copy=False)
    exponent = np.exp(value - value.max(axis=1, keepdims=True))
    return (exponent / exponent.sum(axis=1, keepdims=True)).astype(np.float32)


def centered_rmse(a: np.ndarray, b: np.ndarray) -> float:
    left = a.astype(np.float64)
    right = b.astype(np.float64)
    left -= left.mean(axis=1, keepdims=True)
    right -= right.mean(axis=1, keepdims=True)
    return float(np.sqrt(np.mean((right - left) ** 2)))


def margin(prob: np.ndarray, target_class: np.ndarray) -> np.ndarray:
    """Per-pixel class-minus-background signed softmax margin (float64)."""
    selected = np.take_along_axis(prob[0], target_class[None, :, :], axis=0)[0]
    return selected.astype(np.float64) - prob[0, 0].astype(np.float64)


def pair_metrics(
    left_logits: np.ndarray, right_logits: np.ndarray,
    left_prob: np.ndarray, right_prob: np.ndarray,
    left_pred: np.ndarray, right_pred: np.ndarray,
    endpoint_changed: np.ndarray, target_class: np.ndarray,
) -> dict[str, Any]:
    """Avoid background-dominated scores by exposing the fixed endpoint-change set."""
    probability_abs = np.abs(right_prob.astype(np.float64) - left_prob.astype(np.float64))[0]
    result: dict[str, Any] = {
        "raw_array_identical": bool(np.array_equal(left_logits, right_logits)),
        "centered_logit_rmse_all": centered_rmse(left_logits, right_logits),
        "probability_mae_all": float(probability_abs.mean()),
        "argmax_changed_pixels_all": int(np.count_nonzero(left_pred != right_pred)),
    }
    if np.any(endpoint_changed):
        delta = margin(right_prob, target_class) - margin(left_prob, target_class)
        result.update({
            "probability_mae_on_cpu_physical_changed_pixels": float(probability_abs[:, endpoint_changed].mean()),
            "margin_delta_mean_on_cpu_physical_changed_pixels": float(delta[endpoint_changed].mean()),
            "margin_delta_abs_mean_on_cpu_physical_changed_pixels": float(np.abs(delta[endpoint_changed]).mean()),
        })
    else:
        result.update({
            "probability_mae_on_cpu_physical_changed_pixels": None,
            "margin_delta_mean_on_cpu_physical_changed_pixels": None,
            "margin_delta_abs_mean_on_cpu_physical_changed_pixels": None,
        })
    return result


def analyze_window(arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> dict[str, Any]:
    for name, arr in zip(STAGES, arrays):
        if arr.shape != TENSOR_SHAPE or arr.dtype != np.float32 or not np.isfinite(arr).all():
            raise ValueError(f"Invalid {name} logits: {arr.shape} {arr.dtype}")
    probs = tuple(softmax(a) for a in arrays)
    preds = tuple(a.argmax(axis=1)[0] for a in arrays)
    endpoint_changed = preds[0] != preds[3]
    # Deterministic, predeclared class on endpoint-disagreement pixels:
    # FP512 foreground class when present, else physical foreground class.
    target_class = np.where(preds[0] > 0, preds[0], preds[3]).astype(np.intp)
    if np.any(endpoint_changed & (target_class == 0)):
        raise ValueError("Endpoint-changed pixel lacks a foreground target class")
    pairs: dict[str, Any] = {}
    for name, left, right in PAIRS:
        pairs[name] = pair_metrics(
            arrays[left], arrays[right], probs[left], probs[right], preds[left], preds[right],
            endpoint_changed, target_class,
        )
    if not np.array_equal(arrays[2], arrays[3]):
        raise ValueError("Saved v3 BIE and physical KL720 logits are not bitwise identical")
    if not np.array_equal(probs[2], probs[3]):
        raise ValueError("Saved v3 BIE and physical KL720 probabilities are not identical")

    max_telescoping_error = 0.0
    if np.any(endpoint_changed):
        changes = [margin(p, target_class) for p in probs]
        residual = (changes[1] - changes[0]) + (changes[2] - changes[1]) + (
            changes[3] - changes[2]
        ) - (changes[3] - changes[0])
        max_telescoping_error = float(np.abs(residual[endpoint_changed]).max())
        if max_telescoping_error > 1e-12:
            raise ValueError(f"Stagewise margin delta telescope failed: {max_telescoping_error}")
    return {
        "cpu_physical_argmax_disagreement_pixels": int(endpoint_changed.sum()),
        "cpu_physical_argmax_agreement": float(np.mean(~endpoint_changed)),
        "stage_pairs": pairs,
        "max_stagewise_margin_telescope_error": max_telescoping_error,
    }


def validate_model_chain(v3_root: Path) -> dict[str, str | bool]:
    analysis = load_json(required_path(v3_root / "int8_analysis_summary.json"))
    simulator = load_json(required_path(v3_root / "bie_validation" / "summary.json"))
    compile_paths = list(v3_root.rglob("nef_compile_summary.json"))
    if len(compile_paths) != 1:
        raise ValueError(f"Expected exactly one saved NEF compile record; found {len(compile_paths)}")
    compile_record = load_json(required_path(compile_paths[0]))
    bie_path = required_path(v3_root / "unet_mobileone_s1_512x512_int8.bie")
    nef_path = required_path(v3_root / "nef" / "unet_mobileone_s1_512x512_v3.nef")
    bie_sha = sha256_file(bie_path)
    nef_sha = sha256_file(nef_path)
    if not all([
        bie_sha == EXPECTED_BIE,
        nef_sha == EXPECTED_NEF,
        analysis.get("bie_sha256", "").lower() == bie_sha,
        analysis.get("validation_panel_used_for_calibration") is False,
        analysis.get("calibration_count") == 192,
        analysis.get("ptq_config") == {
            "datapath_range_method": "percentage", "percentage": 1.0,
            "percentage_16b": 1.0, "percentile": 0.001,
            "outlier_factor": 1.0, "optimize": 0,
        },
        analysis.get("onnx_sha256", "").lower() == EXPECTED_ONNX,
        simulator.get("bie_sha256", "").lower() == bie_sha,
        simulator.get("onnx_sha256", "").lower() == EXPECTED_ONNX,
        compile_record.get("source_bie_sha256", "").lower() == bie_sha,
        compile_record.get("nef_sha256", "").lower() == nef_sha,
        compile_record.get("source_onnx_sha256", "").lower() == EXPECTED_ONNX,
    ]):
        raise ValueError("Selected v3 BIE/NEF/recorded ONNX lineage integrity check failed")
    return {
        "selected_bie_sha256_now": bie_sha,
        "selected_nef_sha256_now": nef_sha,
        "optimized_onnx_sha256_recorded_only": EXPECTED_ONNX,
        "bie_nef_lineage_verified": True,
        "target_legacy_simulator_outputs_contemporaneously_hash_bound": False,
    }


def _sorted_windows(field: dict[str, Any], key: str) -> list[dict[str, Any]]:
    windows = field.get(key)
    if not isinstance(windows, list) or len(windows) != WINDOWS:
        raise ValueError(f"Field {key} must have exactly {WINDOWS} entries")
    values = sorted(windows, key=lambda v: int(v["window_index"]))
    if [int(v["window_index"]) for v in values] != list(range(WINDOWS)):
        raise ValueError(f"Field {key} indices must be 0..8 exactly once")
    return values


def verified_window_output(window: dict[str, Any]) -> tuple[Path, str]:
    binding = window.get("output")
    if not isinstance(binding, dict):
        raise ValueError("Missing authoritative output binding")
    path = required_path(Path(binding["path"]))
    hash_now = sha256_file(path)
    if hash_now.lower() != str(binding["sha256"]).lower():
        raise ValueError(f"Frozen field window output SHA256 mismatch: {path}")
    if "size_bytes" in binding and path.stat().st_size != int(binding["size_bytes"]):
        raise ValueError(f"Frozen field window output size mismatch: {path}")
    return path, hash_now


def _manifest_map(manifest: dict[str, Any], sample: str) -> list[dict[str, Any]]:
    if list(manifest.get("tile", [])) != list(TENSOR_SHAPE):
        raise ValueError("Unexpected frozen validation tile shape")
    if manifest.get("stride") != 256 or manifest.get("overlap") != 0.5:
        raise ValueError("Unexpected frozen validation overlap/stride")
    selected = [v for v in manifest.get("records", []) if v.get("sample_id") == sample]
    if len(selected) != WINDOWS:
        raise ValueError(f"Expected {WINDOWS} selected validation inputs; got {len(selected)}")
    selected.sort(key=lambda v: int(v["window_index"]))
    if [int(v["window_index"]) for v in selected] != list(range(WINDOWS)):
        raise ValueError("Incomplete or duplicate manifest window indices")
    return selected


def audit(fp_json: Path, kl_json: Path, v3_root: Path, input_manifest_path: Path) -> dict[str, Any]:
    for path in (fp_json, kl_json, input_manifest_path):
        required_path(path)
    before_fp, before_kl = sha256_file(fp_json), sha256_file(kl_json)
    fp_field, kl_field = load_json(fp_json), load_json(kl_json)
    if fp_field.get("identity", {}).get("sample_id") != SAMPLE or kl_field.get("identity", {}).get("sample_id") != SAMPLE:
        raise ValueError("Unexpected sample identity in frozen field")
    model_chain = validate_model_chain(v3_root)
    manifest = load_json(input_manifest_path)
    manifest_windows = _manifest_map(manifest, SAMPLE)
    fp_inputs = _sorted_windows(fp_field["identity"], "windows")
    kl_inputs = _sorted_windows(kl_field["identity"], "windows")
    fp_outputs = _sorted_windows(fp_field, "window_outputs")
    kl_outputs = _sorted_windows(kl_field, "window_outputs")
    rows: list[dict[str, Any]] = []
    manifest_root = input_manifest_path.parent
    for i in range(WINDOWS):
        a, f, k = manifest_windows[i], fp_inputs[i], kl_inputs[i]
        coord = (int(a["window_index"]), int(a["x"]), int(a["y"]))
        if any((int(v["window_index"]), int(v["x"]), int(v["y"])) != coord for v in (f, k)):
            raise ValueError(f"Field/manifest coordinates changed for window {i}")
        input_file = required_path(manifest_root / str(a["tensor"]))
        input_sha = sha256_file(input_file)
        if input_sha.lower() != str(f["tensor_sha256"]).lower() or input_sha.lower() != str(k["tensor_sha256"]).lower():
            raise ValueError(f"Frozen input tensor SHA256 mismatch for window {i}")
        cpu_file, cpu_sha = verified_window_output(fp_outputs[i])
        physical_file, physical_sha = verified_window_output(kl_outputs[i])
        stem = f"00_{i:02d}_{SAMPLE}.npy"
        float_file = required_path(v3_root / "bie_validation" / "floating_outputs" / stem)
        bie_file = required_path(v3_root / "bie_validation" / "bie_outputs" / stem)
        float_sha, bie_sha = sha256_file(float_file), sha256_file(bie_file)
        if bie_sha != physical_sha:
            raise ValueError(f"Expected BIE/physical cache byte identity failed on window {i}")
        arrays = tuple(load_logits(p) for p in (cpu_file, float_file, bie_file, physical_file))
        metrics = analyze_window(arrays)  # Includes a second, array-level BIE==physical check.
        rows.append({
            "window_index": i, "x": coord[1], "y": coord[2],
            "frozen_input_sha256": input_sha,
            "current_output_sha256": {
                STAGES[0]: cpu_sha, STAGES[1]: float_sha,
                STAGES[2]: bie_sha, STAGES[3]: physical_sha,
            },
            **metrics,
        })
    if sha256_file(fp_json) != before_fp or sha256_file(kl_json) != before_kl:
        raise ValueError("Frozen field JSON bytes changed during read-only audit")
    totals: dict[str, Any] = {
        "cpu_physical_argmax_disagreement_pixels": sum(row["cpu_physical_argmax_disagreement_pixels"] for row in rows),
        "bie_and_physical_byte_identical_windows": sum(
            row["current_output_sha256"][STAGES[2]] == row["current_output_sha256"][STAGES[3]]
            for row in rows
        ),
    }
    for name, _, _ in PAIRS:
        totals[name] = {
            "mean_full_window_probability_mae": float(np.mean([row["stage_pairs"][name]["probability_mae_all"] for row in rows])),
            "total_argmax_changed_pixels": sum(row["stage_pairs"][name]["argmax_changed_pixels_all"] for row in rows),
            "mean_centered_logit_rmse": float(np.mean([row["stage_pairs"][name]["centered_logit_rmse_all"] for row in rows])),
        }
    return {
        "report_type": "dnai_saved_v3_stagewise_descriptive_v014",
        "status": "descriptive_only_legacy_simulator_outputs",
        "sample_id": SAMPLE,
        "model_lineage": model_chain,
        "fp512_field_json_sha256_now": before_fp,
        "kl720_field_json_sha256_now": before_kl,
        "frozen_input_sha256s_verified": WINDOWS,
        "frozen_source_output_sha256s_verified": 2 * WINDOWS,
        "legacy_simulator_output_sha256s_computed_now_not_recorded_at_production": 2 * WINDOWS,
        "contemporaneous_legacy_simulator_output_provenance_verified": False,
        "summary": totals,
        "windows": rows,
        "interpretation_limit": (
            "Cached legacy simulator outputs lack contemporaneous per-output binding: "
            "differences and decompositions are descriptive only, not causal "
            "evidence of quantization, model conversion, or biological fidelity."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp512", type=Path, required=True)
    parser.add_argument("--kl720", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.fp512, args.kl720, args.v3_root, args.input_manifest)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
