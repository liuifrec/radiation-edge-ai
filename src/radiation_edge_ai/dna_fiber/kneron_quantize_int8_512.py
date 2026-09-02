"""Quantize the KL720-compatible DNAi MobileOne-S1 512 model to INT8.

Run this script inside the pinned Kneron Toolchain Docker image. It consumes the
public-training calibration manifest produced by ``prepare_int8_calibration_512``
and the already Kneron-optimized 512x512 ONNX model, then runs
``ktc.ModelConfig.analysis`` to generate the fixed-point BIE model.

This is deliberately a quantization-only gate. It does NOT compile an NEF and
does NOT use the frozen 20-image inter-grader validation panel. BIE biological
fidelity is checked in the next stage before NEF compilation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import ktc
import numpy as np
import onnx

EXPECTED_SHAPE = (1, 3, 512, 512)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_input_names(model: onnx.ModelProto) -> list[str]:
    initializer_names = {item.name for item in model.graph.initializer}
    return [item.name for item in model.graph.input if item.name not in initializer_names]


def find_newest(filename: str, started: float) -> Path | None:
    roots = (Path("/data1/kneron_flow"), Path("/workspace/.tmp"), Path("/tmp"))
    candidates: list[Path] = []
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
    fresh = [p for p in candidates if p.stat().st_mtime >= started - 5.0]
    pool = fresh if fresh else candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def copy_if_present(filename: str, output_dir: Path, started: float) -> str | None:
    source = find_newest(filename, started)
    if source is None:
        return None
    destination = output_dir / filename
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return str(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Kneron-optimized fixed 512x512 ONNX")
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--platform", default="720")
    parser.add_argument("--model-id", type=int, default=32769)
    parser.add_argument("--model-version", default="8b28")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    onnx_path = Path(args.onnx).resolve()
    manifest_path = Path(args.calibration_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")

    print("Radiation Edge AI - KL720 DNAi INT8 calibration")
    print(f"ONNX: {onnx_path}")
    print(f"Calibration manifest: {manifest_path}")
    print(f"Platform: {args.platform}")
    print(f"Model ID/version: {args.model_id}/{args.model_version}")
    print(f"Threads: {args.threads}")
    print("")

    print("[Load optimized ONNX]")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    input_names = graph_input_names(model)
    if len(input_names) != 1:
        raise RuntimeError(f"Expected exactly one model input, got {input_names}")
    input_name = input_names[0]
    print(f" - checker: PASS")
    print(f" - input: {input_name}")
    print(f" - ONNX SHA256: {sha256_file(onnx_path)}")

    print("[Load calibration tensors]")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_input_name = manifest.get("input_name")
    if declared_input_name and declared_input_name != input_name:
        raise RuntimeError(
            f"Calibration manifest input name {declared_input_name!r} does not match ONNX {input_name!r}"
        )
    if tuple(manifest.get("tile", ())) != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Calibration manifest tile {manifest.get('tile')} does not match {EXPECTED_SHAPE}"
        )

    calibration_root = manifest_path.parent
    arrays: list[np.ndarray] = []
    tensor_records = manifest.get("records", [])
    if not tensor_records:
        raise RuntimeError("Calibration manifest contains no tensor records")

    global_min = float("inf")
    global_max = float("-inf")
    for record in tensor_records:
        tensor_path = calibration_root / record["tensor"]
        if not tensor_path.is_file():
            raise FileNotFoundError(tensor_path)
        array = np.load(tensor_path, allow_pickle=False)
        if array.shape != EXPECTED_SHAPE:
            raise RuntimeError(f"Unexpected calibration shape {array.shape}: {tensor_path}")
        if array.dtype != np.float32:
            raise RuntimeError(f"Unexpected calibration dtype {array.dtype}: {tensor_path}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite calibration values: {tensor_path}")
        array = np.ascontiguousarray(array, dtype=np.float32)
        arrays.append(array)
        global_min = min(global_min, float(array.min()))
        global_max = max(global_max, float(array.max()))

    print(f" - tensors: {len(arrays)}")
    print(f" - shape: {EXPECTED_SHAPE}")
    print(f" - dtype: float32")
    print(f" - global range: [{global_min:.6f}, {global_max:.6f}]")
    print(f" - validation panel used for calibration: NO")

    print("[Create KL720 ModelConfig]")
    km = ktc.ModelConfig(
        args.model_id,
        args.model_version,
        str(args.platform),
        onnx_model=model,
    )
    print(" - Success")

    print("[INT8 fixed-point analysis]")
    input_mapping = {input_name: arrays}
    old_cwd = Path.cwd()
    started = time.time()
    try:
        os.chdir(output_dir)
        bie_returned = Path(km.analysis(input_mapping, threads=args.threads)).resolve()
    finally:
        os.chdir(old_cwd)

    if not bie_returned.is_file():
        raise RuntimeError(f"Kneron analysis returned missing BIE path: {bie_returned}")

    bie_destination = output_dir / "unet_mobileone_s1_512x512_int8.bie"
    if bie_returned.resolve() != bie_destination.resolve():
        shutil.copy2(bie_returned, bie_destination)

    artifacts = {}
    for filename in (
        "model_fx_report.html",
        "model_fx_report.json",
        "analysis.log",
        "quantize.log",
    ):
        artifacts[filename] = copy_if_present(filename, output_dir, started)

    elapsed = time.time() - started
    summary = {
        "onnx": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "calibration_manifest": str(manifest_path),
        "calibration_count": len(arrays),
        "calibration_shape": list(EXPECTED_SHAPE),
        "calibration_dtype": "float32",
        "calibration_global_min": global_min,
        "calibration_global_max": global_max,
        "input_name": input_name,
        "platform": str(args.platform),
        "model_id": args.model_id,
        "model_version": args.model_version,
        "threads": args.threads,
        "bie_returned": str(bie_returned),
        "bie_persistent": str(bie_destination),
        "bie_sha256": sha256_file(bie_destination),
        "bie_size_bytes": bie_destination.stat().st_size,
        "elapsed_seconds": elapsed,
        "artifacts": artifacts,
        "validation_panel_used_for_calibration": False,
    }
    summary_path = output_dir / "int8_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("KL720 INT8 ANALYSIS COMPLETE: YES")
    print("BIE GENERATED: YES")
    print(f"Calibration tensors: {len(arrays)}")
    print(f"BIE: {bie_destination}")
    print(f"BIE SHA256: {summary['bie_sha256']}")
    print(f"BIE size: {summary['bie_size_bytes'] / 1024**2:.2f} MiB")
    print(f"Elapsed: {elapsed:.1f} s")
    print(f"Summary: {summary_path}")
    if artifacts.get("model_fx_report.html"):
        print(f"FX report: {artifacts['model_fx_report.html']}")
    print("")
    print("NEXT GATE: BIE fixed-point inference on frozen validation images; do not compile NEF yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
