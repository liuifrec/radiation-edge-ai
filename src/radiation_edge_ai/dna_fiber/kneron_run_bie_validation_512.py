"""Run KL720 fixed-point BIE inference on frozen DNAi validation windows.

Run this script inside the pinned Kneron Toolchain Docker image after
``prepare_bie_validation_windows_512.py`` has created the 512x512, 50%-overlap
validation windows. Each input window is evaluated with both the Kneron
floating-point ONNX simulator and the fixed-point BIE simulator. Raw outputs
are persisted for downstream stitching and biological validation in the DNAi
Windows environment.

Raw-logit errors can look large even when a common-mode shift leaves class
probabilities and decisions nearly unchanged. Therefore this script reports
raw logits, per-pixel centered logits, softmax probabilities, and argmax
agreement separately.

This stage does not compile an NEF and does not touch calibration data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import ktc
import numpy as np

EXPECTED_INPUT_SHAPE = (1, 3, 512, 512)
EXPECTED_OUTPUT_SHAPE = (1, 3, 512, 512)

V015_SELECTED_SAMPLES = frozenset((
    "1__tile_4", "2__tile_9", "16__tile_5",
))
V015_ONNX_SHA256 = (
    "a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0"
)
V015_BIE_SHA256 = (
    "a4986b71e48305b19f21f576c25cd6fc02e66f5aa5cf35c6e076dba252b53e9c"
)
V015_OUTPUT_DIRECTORY = "bie_validation_v015_3dev"
V015_EXPECTED_WINDOWS = 27


def validate_v015_development(
    args: argparse.Namespace,
    onnx_path: Path, bie_path: Path, output_dir: Path,
    manifest: dict, records: list[dict],
) -> dict:
    """Exact three-sample development screen; reject before any output creation."""
    if not args.v015_strict_development:
        raise ValueError("Strict gate called without explicit opt-in")
    if args.sample_ids is None or parse_sample_ids(args.sample_ids) != V015_SELECTED_SAMPLES:
        raise SystemExit("v0.15 requires exactly the three predeclared samples")
    if args.image_indices is not None or args.limit is not None:
        raise SystemExit("v0.15 forbids ad hoc index or window limit selection")
    if args.platform != 720:
        raise SystemExit("v0.15 model is for KL720")
    if output_dir.name != V015_OUTPUT_DIRECTORY or output_dir.exists():
        raise SystemExit("v0.15 output must be a new bie_validation_v015_3dev directory")
    if bie_path.name != "unet_mobileone_s1_512x512_int16out.bie":
        raise SystemExit("v0.15 requires the named INT16-output experimental BIE")
    if bie_path.parent.name != "int8_512_v5_int16out":
        raise SystemExit("v0.15 BIE must be inside the isolated v5 candidate directory")
    bie_hash, onnx_hash = sha256_file(bie_path), sha256_file(onnx_path)
    if bie_hash != V015_BIE_SHA256 or onnx_hash != V015_ONNX_SHA256:
        raise SystemExit("v0.15 BIE/ONNX immutable SHA256 mismatch")
    if manifest.get("dnai_commit") != "fcf20c7d6eb385675ff7d07da4fdf471589ce0cf":
        raise SystemExit("v0.15 frozen DNAi upstream revision mismatch")
    if (manifest.get("windows_per_image") != 9 or manifest.get("stride") != 256
        or manifest.get("blend") != "gaussian" or manifest.get("input_name") != "input"):
        raise SystemExit("v0.15 frozen preprocessing/window manifest mismatch")
    if len(records) != V015_EXPECTED_WINDOWS:
        raise SystemExit("v0.15 requires 27 windows, nine from each of three samples")
    by_sample = defaultdict(list)
    for record in records:
        by_sample[str(record["sample_id"])].append(record)
        tensor = (Path(args.validation_manifest).parent / record["tensor"]).resolve()
        root = Path(args.validation_manifest).parent.resolve()
        if os.path.commonpath([str(tensor), str(root)]) != str(root):
            raise SystemExit("v0.15 frozen input path escapes manifest root")
    if set(by_sample) != V015_SELECTED_SAMPLES:
        raise SystemExit("v0.15 selected sample set mismatch")
    for sample, sample_records in by_sample.items():
        if len(sample_records) != 9 or {
            int(record["window_index"]) for record in sample_records
        } != set(range(9)):
            raise SystemExit("v0.15 nine unique windows missing for " + sample)
        if len({str(record["image_index"]) for record in sample_records}) != 1:
            raise SystemExit("v0.15 sample has multiple image indices: " + sample)
        expected_xy = {
            (x, y) for x in (0, 256, 512) for y in (0, 256, 512)
        }
        xy = {(int(record["x"]), int(record["y"])) for record in sample_records}
        if xy != expected_xy:
            raise SystemExit("v0.15 coordinates mismatch for " + sample)
    return {
        "status": "development_only_not_holdout",
        "v015_bie_sha256_verified": bie_hash,
        "v015_onnx_sha256_verified": onnx_hash,
        "frozen_input_manifest_sha256": sha256_file(
            Path(args.validation_manifest)
        ),
        "frozen_expected_samples": sorted(V015_SELECTED_SAMPLES),
        "frozen_expected_windows": V015_EXPECTED_WINDOWS,
        "source_model_output_precision": "int16",
    }



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_array(result, label: str) -> np.ndarray:
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise RuntimeError(
            f"{label} expected exactly one output tensor, got {type(result)} "
            f"len={len(result) if hasattr(result, '__len__') else 'NA'}"
        )
    array = np.asarray(result[0], dtype=np.float32)
    if array.shape != EXPECTED_OUTPUT_SHAPE:
        raise RuntimeError(f"{label} output shape {array.shape}; expected {EXPECTED_OUTPUT_SHAPE}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"{label} output contains non-finite values")
    return np.ascontiguousarray(array)


def error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref64 = reference.astype(np.float64, copy=False)
    cand64 = candidate.astype(np.float64, copy=False)
    diff = cand64 - ref64
    abs_diff = np.abs(diff)
    return {
        "max_abs_error": float(abs_diff.max()),
        "mean_abs_error": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
    }


def softmax(array: np.ndarray) -> np.ndarray:
    shifted = array.astype(np.float64, copy=False) - np.max(array, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    return probs.astype(np.float32)


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    raw = error_metrics(reference, candidate)

    # Remove each pixel's class-common offset. Softmax and argmax are invariant
    # to such an offset, so this better reflects decision-relevant logit drift.
    ref_centered = reference - reference.mean(axis=1, keepdims=True)
    cand_centered = candidate - candidate.mean(axis=1, keepdims=True)
    centered = error_metrics(ref_centered, cand_centered)

    ref_prob = softmax(reference)
    cand_prob = softmax(candidate)
    probability = error_metrics(ref_prob, cand_prob)

    ref_class = reference.argmax(axis=1)
    cand_class = candidate.argmax(axis=1)
    disagreement_pixels = int(np.count_nonzero(ref_class != cand_class))
    total_pixels = int(ref_class.size)

    return {
        **raw,
        "centered_logit_max_abs_error": centered["max_abs_error"],
        "centered_logit_mean_abs_error": centered["mean_abs_error"],
        "centered_logit_rmse": centered["rmse"],
        "probability_max_abs_error": probability["max_abs_error"],
        "probability_mean_abs_error": probability["mean_abs_error"],
        "probability_rmse": probability["rmse"],
        "argmax_agreement": float(np.mean(ref_class == cand_class)),
        "disagreement_pixels": disagreement_pixels,
        "total_pixels": total_pixels,
        "floating_logit_min": float(reference.min()),
        "floating_logit_max": float(reference.max()),
        "fixed_logit_min": float(candidate.min()),
        "fixed_logit_max": float(candidate.max()),
    }


def parse_image_indices(value: str | None) -> set[int] | None:
    if value is None:
        return None
    result: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            index = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid image index {token!r}; expected comma-separated integers"
            ) from exc
        if index < 0:
            raise argparse.ArgumentTypeError("Image indices must be >= 0")
        result.add(index)
    if not result:
        raise argparse.ArgumentTypeError("--image-indices selected no indices")
    return result


def parse_sample_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    result = {token.strip() for token in value.split(",") if token.strip()}
    if not result:
        raise argparse.ArgumentTypeError("--sample-ids selected no sample IDs")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--bie", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--platform", type=int, default=720)
    parser.add_argument(
        "--image-indices",
        default=None,
        help=(
            "Optional comma-separated 0-based image_index values from the validation "
            "manifest. All nine windows of each selected image are run. Prefer "
            "--sample-ids for human-readable selection."
        ),
    )
    parser.add_argument(
        "--sample-ids",
        default=None,
        help=(
            "Optional comma-separated frozen sample IDs, e.g. "
            "11__tile_17,7__tile_6,9__tile_16. All nine windows of each selected "
            "sample are run."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of windows after image filtering")
    parser.add_argument(
        "--v015-strict-development", action="store_true",
        help="Record fresh input and output SHA256 for exactly three frozen v0.15 development fields",
    )
    args = parser.parse_args()

    if args.image_indices is not None and args.sample_ids is not None:
        raise SystemExit("Use only one of --image-indices or --sample-ids")

    onnx_path = Path(args.onnx).resolve()
    bie_path = Path(args.bie).resolve()
    manifest_path = Path(args.validation_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    fp_dir = output_dir / "floating_outputs"
    bie_dir = output_dir / "bie_outputs"
    if not args.v015_strict_development:
        # Preserve the historical execution and output-directory behavior.
        output_dir.mkdir(parents=True, exist_ok=True)
        fp_dir.mkdir(parents=True, exist_ok=True)
        bie_dir.mkdir(parents=True, exist_ok=True)

    for path in (onnx_path, bie_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest.get("tile", ())) != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(f"Validation manifest tile mismatch: {manifest.get('tile')}")
    if manifest.get("overlap") != 0.5:
        raise RuntimeError(f"Expected frozen overlap 0.5, got {manifest.get('overlap')}")
    input_name = manifest.get("input_name", "input")
    all_records = manifest.get("records", [])

    selected_indices = parse_image_indices(args.image_indices)
    selected_sample_ids = parse_sample_ids(args.sample_ids)
    records = all_records

    if selected_indices is not None:
        available_indices = {int(record["image_index"]) for record in all_records}
        missing = sorted(selected_indices - available_indices)
        if missing:
            raise RuntimeError(
                f"Requested image indices not found in validation manifest: {missing}; "
                f"available={sorted(available_indices)}"
            )
        records = [
            record for record in all_records
            if int(record["image_index"]) in selected_indices
        ]
    elif selected_sample_ids is not None:
        available_sample_ids = {str(record["sample_id"]) for record in all_records}
        missing = sorted(selected_sample_ids - available_sample_ids)
        if missing:
            raise RuntimeError(
                f"Requested sample IDs not found in validation manifest: {missing}; "
                f"available={sorted(available_sample_ids)}"
            )
        records = [
            record for record in all_records
            if str(record["sample_id"]) in selected_sample_ids
        ]

    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No validation windows found")

    v015_provenance = None
    if args.v015_strict_development:
        v015_provenance = validate_v015_development(
            args, onnx_path, bie_path, output_dir, manifest, records
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        fp_dir.mkdir()
        bie_dir.mkdir()

    selected_samples = []
    seen_samples = set()
    for record in records:
        sample = (int(record["image_index"]), record["sample_id"])
        if sample not in seen_samples:
            selected_samples.append(sample)
            seen_samples.add(sample)

    print("Radiation Edge AI - KL720 DNAi BIE fixed-point validation")
    print(f"ONNX: {onnx_path}")
    print(f"BIE: {bie_path}")
    print(f"Validation windows: {len(records)}")
    if selected_indices is not None or selected_sample_ids is not None:
        print(
            "Selected images: "
            + ", ".join(f"{index}:{sample_id}" for index, sample_id in selected_samples)
        )
    print(f"Input name: {input_name}")
    print(f"Platform: {args.platform}")
    print("Comparison: Kneron floating-point ONNX simulator vs fixed-point BIE simulator")
    print("")

    rows = []
    started_all = time.perf_counter()
    validation_root = manifest_path.parent

    for position, record in enumerate(records, start=1):
        tensor_path = validation_root / record["tensor"]
        input_hash = sha256_file(tensor_path) if v015_provenance else None
        array = np.load(tensor_path, allow_pickle=False)
        if array.shape != EXPECTED_INPUT_SHAPE or array.dtype != np.float32:
            raise RuntimeError(f"Invalid validation tensor {array.shape} {array.dtype}: {tensor_path}")
        array = np.ascontiguousarray(array, dtype=np.float32)

        t0 = time.perf_counter()
        floating_result = ktc.kneron_inference(
            [array],
            onnx_file=str(onnx_path),
            input_names=[input_name],
        )
        floating_ms = (time.perf_counter() - t0) * 1000.0
        floating = output_array(floating_result, "floating ONNX")

        t0 = time.perf_counter()
        fixed_result = ktc.kneron_inference(
            [array],
            bie_file=str(bie_path),
            input_names=[input_name],
            platform=args.platform,
        )
        fixed_ms = (time.perf_counter() - t0) * 1000.0
        fixed = output_array(fixed_result, "fixed BIE")

        m = metrics(floating, fixed)
        stem = f"{record['image_index']:02d}_{record['window_index']:02d}_{record['sample_id']}"
        fp_path = fp_dir / f"{stem}.npy"
        bie_out_path = bie_dir / f"{stem}.npy"
        np.save(fp_path, floating, allow_pickle=False)
        np.save(bie_out_path, fixed, allow_pickle=False)

        row = {
            "position": position,
            "image_index": record["image_index"],
            "window_index": record["window_index"],
            "sample_id": record["sample_id"],
            "key": record["key"],
            "y": record["y"],
            "x": record["x"],
            "input_tensor": str(tensor_path),
            "floating_output": str(fp_path),
            "bie_output": str(bie_out_path),
            **m,
            "floating_ms": floating_ms,
            "bie_ms": fixed_ms,
        }
        if v015_provenance:
            # Bind bytes at the point of production, not in a later replay.
            row["input_tensor_sha256"] = input_hash
            row["floating_output_sha256"] = sha256_file(fp_path)
            row["bie_output_sha256"] = sha256_file(bie_out_path)
        rows.append(row)
        print(
            f"[{position:03d}/{len(records):03d}] {record['sample_id']} w{record['window_index']} "
            f"argmax={m['argmax_agreement']:.8f} disagree={m['disagreement_pixels']} "
            f"raw_max={m['max_abs_error']:.5g} prob_max={m['probability_max_abs_error']:.5g} "
            f"fp={floating_ms:.1f}ms bie={fixed_ms:.1f}ms"
        )

    csv_path = output_dir / "per_window.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_argmax = float(np.mean([r["argmax_agreement"] for r in rows]))
    min_argmax = float(np.min([r["argmax_agreement"] for r in rows]))
    max_abs = float(np.max([r["max_abs_error"] for r in rows]))
    max_prob_abs = float(np.max([r["probability_max_abs_error"] for r in rows]))
    mean_prob_rmse = float(np.mean([r["probability_rmse"] for r in rows]))
    mean_centered_rmse = float(np.mean([r["centered_logit_rmse"] for r in rows]))
    mean_bie_ms = float(np.mean([r["bie_ms"] for r in rows]))
    disagreement_pixels = int(sum(r["disagreement_pixels"] for r in rows))
    total_pixels = int(sum(r["total_pixels"] for r in rows))

    summary = {
        "onnx": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "bie": str(bie_path),
        "bie_sha256": sha256_file(bie_path),
        "validation_manifest": str(manifest_path),
        "platform": args.platform,
        "selected_image_indices": sorted(selected_indices) if selected_indices is not None else None,
        "selected_sample_ids": sorted(selected_sample_ids) if selected_sample_ids is not None else None,
        "selected_samples": [
            {"image_index": index, "sample_id": sample_id}
            for index, sample_id in selected_samples
        ],
        "n_windows": len(rows),
        "metrics": {
            "mean_argmax_agreement": mean_argmax,
            "min_argmax_agreement": min_argmax,
            "disagreement_pixels": disagreement_pixels,
            "total_pixels": total_pixels,
            "max_raw_logit_abs_error": max_abs,
            "mean_centered_logit_rmse": mean_centered_rmse,
            "max_softmax_probability_abs_error": max_prob_abs,
            "mean_softmax_probability_rmse": mean_prob_rmse,
            "mean_bie_simulator_ms_per_window": mean_bie_ms,
        },
        "elapsed_seconds": time.perf_counter() - started_all,
        "per_window_csv": str(csv_path),
        "floating_output_dir": str(fp_dir),
        "bie_output_dir": str(bie_dir),
    }
    if v015_provenance:
        v015_provenance["n_window_receipts"] = len(rows)
        v015_provenance["per_window_receipt_columns"] = [
            "input_tensor_sha256", "floating_output_sha256",
            "bie_output_sha256",
        ]
        summary["v015_strict_provenance"] = v015_provenance
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("KL720 BIE WINDOW VALIDATION COMPLETE: YES")
    print(f"Windows: {len(rows)}")
    print(f"mean argmax agreement: {mean_argmax:.8f}")
    print(f"min argmax agreement: {min_argmax:.8f}")
    print(f"disagreement pixels: {disagreement_pixels}/{total_pixels}")
    print(f"max raw-logit absolute error: {max_abs:.8g}")
    print(f"mean centered-logit RMSE: {mean_centered_rmse:.8g}")
    print(f"max softmax-probability absolute error: {max_prob_abs:.8g}")
    print(f"mean softmax-probability RMSE: {mean_prob_rmse:.8g}")
    print(f"mean BIE simulator time/window: {mean_bie_ms:.1f} ms")
    print(f"Summary: {summary_path}")
    print(f"Per-window: {csv_path}")
    print("")
    print("NEXT: stitch BIE outputs with the frozen 50%-overlap Gaussian policy and evaluate DNA-fiber biology.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
