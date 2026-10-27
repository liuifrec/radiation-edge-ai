"""Quantize the KL720-compatible DNAi MobileOne-S1 512 model to fixed point.

Run this script inside the pinned Kneron Toolchain Docker image. It consumes a
public-training calibration manifest and the already Kneron-optimized 512x512
ONNX model, then runs ``ktc.ModelConfig.analysis`` to generate a BIE model.

The CLI exposes Kneron's documented PTQ range-tuning controls so quantization
experiments can be reproduced without editing source code. The default values
match the standard Kneron PTQ settings and keep the existing v1/v2 behavior.

This is deliberately a quantization-only gate. It does NOT compile an NEF and
does NOT use the frozen 20-image inter-grader validation panel. BIE biological
fidelity is checked in the next stage before NEF compilation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import ktc
import numpy as np
import onnx

EXPECTED_SHAPE = (1, 3, 512, 512)

# Frozen historical models are immutable; v0.15 alters ONLY model-output precision.
V015_ONNX_SHA256 = "a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0"
V015_TOOLCHAIN_IMAGE_ID = (
    "sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041"
)
V015_OUTPUT_DIRECTORY = "int8_512_v5_int16out"
HISTORICAL_PTQ_DIRECTORIES = frozenset((
    "int8_512", "int8_512_v2", "int8_512_v3_pct100", "int8_512_v4_bias",
))
V015_V3_RECIPE = {
    "datapath_range_method": "percentage",
    "percentage": 1.0,
    "percentage_16b": 1.0,
    "percentile": 0.001,
    "outlier_factor": 1.0,
    "optimize": 0,
}


def build_analysis_options(args: argparse.Namespace) -> dict:
    """Keep the historical default analysis kwargs unchanged, including order."""
    options = {
        "threads": args.threads,
        "datapath_range_method": args.datapath_range_method,
        "percentage": args.percentage,
        "percentage_16b": args.percentage_16b,
        "percentile": args.percentile,
        "outlier_factor": args.outlier_factor,
        "optimize": args.optimize,
    }
    if args.model_out_bitwidth_mode is not None:
        options["model_out_bitwidth_mode"] = args.model_out_bitwidth_mode
    return options


def validate_int16_candidate(
    args: argparse.Namespace, onnx_path: Path, manifest_path: Path, output_dir: Path
) -> dict | None:
    """Fail closed before creating an experimental output directory."""
    if args.model_out_bitwidth_mode is None:
        if args.candidate_toolchain_image_id or args.candidate_manifest_sha256 or (
            args.candidate_preflight_only
        ):
            raise SystemExit("Candidate-only arguments require --model-out-bitwidth-mode int16")
        return None
    if args.model_out_bitwidth_mode != "int16":
        raise SystemExit("Only the isolated INT16-output candidate is allowed")
    if str(args.platform) != "720" or args.model_id != 32769 or (
        args.model_version != "8b28"
    ):
        raise SystemExit("v0.15 candidate must use frozen KL720 model ID/version")
    if args.candidate_toolchain_image_id != V015_TOOLCHAIN_IMAGE_ID:
        raise SystemExit("Candidate requires the verified immutable Docker image ID")
    if not args.candidate_manifest_sha256 or not re.fullmatch(
        r"[0-9a-fA-F]{64}", args.candidate_manifest_sha256
    ):
        raise SystemExit("Candidate requires 64-hex --candidate-manifest-sha256")
    if output_dir.name != V015_OUTPUT_DIRECTORY or (
        HISTORICAL_PTQ_DIRECTORIES.intersection(output_dir.parts)
    ):
        raise SystemExit("Candidate must use a new isolated int8_512_v5_int16out directory")
    if output_dir.exists():
        raise SystemExit("Candidate output directory must not exist (no overwrites)")
    observed_recipe = {
        "datapath_range_method": args.datapath_range_method,
        "percentage": args.percentage,
        "percentage_16b": args.percentage_16b,
        "percentile": args.percentile,
        "outlier_factor": args.outlier_factor,
        "optimize": args.optimize,
    }
    if observed_recipe != V015_V3_RECIPE:
        raise SystemExit("Candidate may change ONLY output bitwidth; exact v3 PTQ required")
    onnx_digest = sha256_file(onnx_path)
    if onnx_digest != V015_ONNX_SHA256:
        raise SystemExit("Frozen optimized ONNX SHA256 mismatch")
    manifest_digest = sha256_file(manifest_path)
    if manifest_digest.lower() != args.candidate_manifest_sha256.lower():
        raise SystemExit("Calibration manifest SHA256 mismatch")
    return {
        "candidate": "v0.15_INT16_model_output_only",
        "model_out_bitwidth_mode": "int16",
        "toolchain_image_id_declared": V015_TOOLCHAIN_IMAGE_ID,
        "optimized_onnx_sha256_verified": onnx_digest,
        "calibration_manifest_sha256_verified": manifest_digest,
        "experimental_not_deployed": True,
    }


def validate_candidate_calibration(manifest: dict, records: list) -> None:
    """Require the exact historical 192-source calibration scope."""
    if manifest.get("validation_panel_used_for_calibration") is not False:
        raise SystemExit("Candidate calibration must explicitly exclude validation panel")
    if len(records) != 192 or manifest.get("n_tensors") != 192:
        raise SystemExit("Candidate requires exactly 192 v2 calibration tensors")
    if len({record["tensor"] for record in records}) != 192:
        raise SystemExit("Duplicate candidate calibration tensor paths")
    if len({record["source_sha256"] for record in records}) != 192:
        raise SystemExit("Candidate requires 192 unique training-source hashes")



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
    parser.add_argument(
        "--datapath-range-method",
        choices=("percentage", "mmse"),
        default="percentage",
        help="Kneron datapath dynamic-range estimator",
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=0.999,
        help="Retained datapath fraction in percentage mode (Kneron default 0.999)",
    )
    parser.add_argument(
        "--percentage-16b",
        type=float,
        default=0.999999,
        help="Retained 16-bit datapath fraction in percentage mode",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.001,
        help="MMSE search-range parameter (Kneron default 0.001)",
    )
    parser.add_argument(
        "--outlier-factor",
        type=float,
        default=1.0,
        help="MMSE outlier factor; larger values reduce outlier removal",
    )
    parser.add_argument(
        "--optimize",
        type=int,
        choices=(0, 1, 2, 3, 4),
        default=0,
        help="Kneron bias/feature-map optimization level",
    )
    parser.add_argument(
        "--model-out-bitwidth-mode", choices=("int16",), default=None,
        help="Opt-in v0.15; omit to preserve unchanged historical analysis options",
    )
    parser.add_argument(
        "--candidate-toolchain-image-id", default=None,
        help="Required: independently verified immutable Docker image ID",
    )
    parser.add_argument(
        "--candidate-manifest-sha256", default=None,
        help="Required: 64-character frozen calibration manifest SHA256",
    )
    parser.add_argument(
        "--candidate-preflight-only", action="store_true",
        help="Check all v0.15 bytes and exit WITHOUT analysis or creating output dir",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx).resolve()
    manifest_path = Path(args.calibration_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.model_out_bitwidth_mode is None:
        # Preserve the historical v1-v4 behavior, including directory creation.
        output_dir.mkdir(parents=True, exist_ok=True)

    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")
    if not 0.0 < args.percentage <= 1.0:
        raise SystemExit("--percentage must be in (0, 1]")
    if not 0.0 < args.percentage_16b <= 1.0:
        raise SystemExit("--percentage-16b must be in (0, 1]")
    if args.percentage > args.percentage_16b:
        raise SystemExit("--percentage must be <= --percentage-16b")
    if args.percentile <= 0.0:
        raise SystemExit("--percentile must be positive")
    if args.outlier_factor <= 0.0:
        raise SystemExit("--outlier-factor must be positive")

    candidate = validate_int16_candidate(args, onnx_path, manifest_path, output_dir)

    print("Radiation Edge AI - KL720 DNAi fixed-point calibration")
    print(f"ONNX: {onnx_path}")
    print(f"Calibration manifest: {manifest_path}")
    print(f"Platform: {args.platform}")
    print(f"Model ID/version: {args.model_id}/{args.model_version}")
    print(f"Threads: {args.threads}")
    print(
        "PTQ: "
        f"range={args.datapath_range_method} "
        f"percentage={args.percentage} "
        f"percentage_16b={args.percentage_16b} "
        f"percentile={args.percentile} "
        f"outlier_factor={args.outlier_factor} "
        f"optimize={args.optimize}"
    )
    print("")

    print("[Load optimized ONNX]")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    input_names = graph_input_names(model)
    if len(input_names) != 1:
        raise RuntimeError(f"Expected exactly one model input, got {input_names}")
    input_name = input_names[0]
    print(" - checker: PASS")
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
    if candidate:
        validate_candidate_calibration(manifest, tensor_records)

    candidate_tensor_hashes = []
    global_min = float("inf")
    global_max = float("-inf")
    for record in tensor_records:
        tensor_path = calibration_root / record["tensor"]
        if candidate and os.path.commonpath([
            str(tensor_path.resolve()), str(calibration_root.resolve())
        ]) != str(calibration_root.resolve()):
            raise SystemExit("Candidate tensor escapes frozen calibration root")
        if not tensor_path.is_file():
            raise FileNotFoundError(tensor_path)
        array = np.load(tensor_path, allow_pickle=False)
        if array.shape != EXPECTED_SHAPE:
            raise RuntimeError(f"Unexpected calibration shape {array.shape}: {tensor_path}")
        if array.dtype != np.float32:
            raise RuntimeError(f"Unexpected calibration dtype {array.dtype}: {tensor_path}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite calibration values: {tensor_path}")
        if candidate:
            candidate_tensor_hashes.append({
                "tensor": record["tensor"],
                "sha256": sha256_file(tensor_path),
            })
        array = np.ascontiguousarray(array, dtype=np.float32)
        arrays.append(array)
        global_min = min(global_min, float(array.min()))
        global_max = max(global_max, float(array.max()))

    print(f" - tensors: {len(arrays)}")
    print(f" - shape: {EXPECTED_SHAPE}")
    print(" - dtype: float32")
    print(f" - global range: [{global_min:.6f}, {global_max:.6f}]")
    print(" - validation panel used for calibration: NO")

    if candidate:
        candidate["calibration_tensors_sha256_now"] = candidate_tensor_hashes
        candidate["calibration_tensor_count_verified"] = len(candidate_tensor_hashes)
        print(" - v0.15 ONNX/manifest/tensor SHA256 preflight: PASS")
        if args.candidate_preflight_only:
            print(json.dumps({
                "status": "V015_PREFLIGHT_PASS_NO_ANALYSIS_NO_OUTPUT_DIRECTORY",
                "candidate": candidate["candidate"],
                "onnx_sha256": candidate["optimized_onnx_sha256_verified"],
                "calibration_manifest_sha256": candidate[
                    "calibration_manifest_sha256_verified"
                ],
                "calibration_tensor_count": len(candidate_tensor_hashes),
                "toolchain_image_id_declared": V015_TOOLCHAIN_IMAGE_ID,
            }, sort_keys=True))
            return 0
        output_dir.mkdir(parents=True, exist_ok=False)

    print("[Create KL720 ModelConfig]")
    km = ktc.ModelConfig(
        args.model_id,
        args.model_version,
        str(args.platform),
        onnx_model=model,
    )
    print(" - Success")

    print("[Fixed-point analysis]")
    input_mapping = {input_name: arrays}
    old_cwd = Path.cwd()
    started = time.time()
    try:
        os.chdir(output_dir)
        bie_returned = Path(km.analysis(
            input_mapping, **build_analysis_options(args)
        )).resolve()
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
    ptq_config = {
        "datapath_range_method": args.datapath_range_method,
        "percentage": args.percentage,
        "percentage_16b": args.percentage_16b,
        "percentile": args.percentile,
        "outlier_factor": args.outlier_factor,
        "optimize": args.optimize,
    }
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
        "ptq_config": ptq_config,
        "bie_returned": str(bie_returned),
        "bie_persistent": str(bie_destination),
        "bie_sha256": sha256_file(bie_destination),
        "bie_size_bytes": bie_destination.stat().st_size,
        "elapsed_seconds": elapsed,
        "artifacts": artifacts,
        "validation_panel_used_for_calibration": False,
    }
    if candidate:
        ptq_config["model_out_bitwidth_mode"] = "int16"
        summary["ptq_config"] = ptq_config
        summary["v015_candidate_provenance"] = candidate
    summary_path = output_dir / "int8_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("KL720 FIXED-POINT ANALYSIS COMPLETE: YES")
    print("BIE GENERATED: YES")
    print(f"Calibration tensors: {len(arrays)}")
    print(f"PTQ config: {json.dumps(ptq_config, sort_keys=True)}")
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
