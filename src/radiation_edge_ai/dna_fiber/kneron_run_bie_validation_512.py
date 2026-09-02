"""Run KL720 fixed-point BIE inference on frozen DNAi validation windows.

Run this script inside the pinned Kneron Toolchain Docker image after
``prepare_bie_validation_windows_512.py`` has created the 512x512, 50%-overlap
validation windows. Each input window is evaluated with both the Kneron
floating-point ONNX simulator and the fixed-point BIE simulator. Raw outputs
are persisted for downstream stitching and biological validation in the DNAi
Windows environment.

This stage does not compile an NEF and does not touch calibration data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import ktc
import numpy as np

EXPECTED_INPUT_SHAPE = (1, 3, 512, 512)
EXPECTED_OUTPUT_SHAPE = (1, 3, 512, 512)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_array(result, label: str) -> np.ndarray:
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise RuntimeError(f"{label} expected exactly one output tensor, got {type(result)} len={len(result) if hasattr(result, '__len__') else 'NA'}")
    array = np.asarray(result[0], dtype=np.float32)
    if array.shape != EXPECTED_OUTPUT_SHAPE:
        raise RuntimeError(f"{label} output shape {array.shape}; expected {EXPECTED_OUTPUT_SHAPE}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"{label} output contains non-finite values")
    return np.ascontiguousarray(array)


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref64 = reference.astype(np.float64, copy=False)
    cand64 = candidate.astype(np.float64, copy=False)
    diff = cand64 - ref64
    abs_diff = np.abs(diff)
    return {
        "max_abs_error": float(abs_diff.max()),
        "mean_abs_error": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "argmax_agreement": float(np.mean(reference.argmax(axis=1) == candidate.argmax(axis=1))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--bie", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--platform", type=int, default=720)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of windows for smoke testing")
    args = parser.parse_args()

    onnx_path = Path(args.onnx).resolve()
    bie_path = Path(args.bie).resolve()
    manifest_path = Path(args.validation_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    fp_dir = output_dir / "floating_outputs"
    bie_dir = output_dir / "bie_outputs"
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
    records = manifest.get("records", [])
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No validation windows found")

    print("Radiation Edge AI - KL720 DNAi BIE fixed-point validation")
    print(f"ONNX: {onnx_path}")
    print(f"BIE: {bie_path}")
    print(f"Validation windows: {len(records)}")
    print(f"Input name: {input_name}")
    print(f"Platform: {args.platform}")
    print("Comparison: Kneron floating-point ONNX simulator vs fixed-point BIE simulator")
    print("")

    rows = []
    started_all = time.perf_counter()
    validation_root = manifest_path.parent

    for position, record in enumerate(records, start=1):
        tensor_path = validation_root / record["tensor"]
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
        rows.append(row)
        print(
            f"[{position:03d}/{len(records):03d}] {record['sample_id']} w{record['window_index']} "
            f"argmax={m['argmax_agreement']:.8f} max_abs={m['max_abs_error']:.6g} "
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
    mean_abs = float(np.mean([r["mean_abs_error"] for r in rows]))
    rmse = float(np.mean([r["rmse"] for r in rows]))
    mean_bie_ms = float(np.mean([r["bie_ms"] for r in rows]))

    summary = {
        "onnx": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "bie": str(bie_path),
        "bie_sha256": sha256_file(bie_path),
        "validation_manifest": str(manifest_path),
        "platform": args.platform,
        "n_windows": len(rows),
        "metrics": {
            "mean_argmax_agreement": mean_argmax,
            "min_argmax_agreement": min_argmax,
            "max_abs_error": max_abs,
            "mean_of_window_mean_abs_error": mean_abs,
            "mean_window_rmse": rmse,
            "mean_bie_simulator_ms_per_window": mean_bie_ms,
        },
        "elapsed_seconds": time.perf_counter() - started_all,
        "per_window_csv": str(csv_path),
        "floating_output_dir": str(fp_dir),
        "bie_output_dir": str(bie_dir),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("KL720 BIE WINDOW VALIDATION COMPLETE: YES")
    print(f"Windows: {len(rows)}")
    print(f"mean argmax agreement: {mean_argmax:.8f}")
    print(f"min argmax agreement: {min_argmax:.8f}")
    print(f"max absolute error: {max_abs:.8g}")
    print(f"mean window RMSE: {rmse:.8g}")
    print(f"mean BIE simulator time/window: {mean_bie_ms:.1f} ms")
    print(f"Summary: {summary_path}")
    print(f"Per-window: {csv_path}")
    print("")
    print("NEXT: stitch BIE outputs with the frozen 50%-overlap Gaussian policy and evaluate DNA-fiber biology.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
