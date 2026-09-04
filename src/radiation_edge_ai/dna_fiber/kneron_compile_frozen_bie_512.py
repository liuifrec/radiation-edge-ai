"""Compile an already-frozen KL720 DNAi BIE into NEF without re-quantizing.

This script is intentionally downstream of biological-fidelity selection. It
loads the exact persistent BIE artifact, verifies its SHA256 and provenance from
the saved analysis summary, initializes ``ktc.ModelConfig`` directly from that
BIE, and batch-compiles a single-model NEF.

No ONNX optimization, calibration, or fixed-point analysis is repeated here.
That separation prevents an accepted PTQ artifact from silently changing during
NEF compilation.

Run inside the pinned Kneron Toolchain Docker image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import ktc

EXPECTED_V3_BIE_SHA256 = (
    "01f902e283b565b712cbecbf22c71f19ae6f9ba826d2c6be87e8976cd1bc9863"
)
EXPECTED_V3_ONNX_SHA256 = (
    "a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0"
)
EXPECTED_V3_PTQ = {
    "datapath_range_method": "percentage",
    "percentage": 1.0,
    "percentage_16b": 1.0,
    "percentile": 0.001,
    "outlier_factor": 1.0,
    "optimize": 0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_summary_path(value: str, summary_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (summary_path.parent / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-bie-sha256",
        default=EXPECTED_V3_BIE_SHA256,
        help="Accepted frozen BIE SHA256; defaults to selected PTQ v3 artifact",
    )
    parser.add_argument(
        "--output-name",
        default="unet_mobileone_s1_512x512_v3.nef",
    )
    args = parser.parse_args()

    summary_path = Path(args.analysis_summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    bie_value = summary.get("bie_persistent") or summary.get("bie_returned")
    if not bie_value:
        raise RuntimeError("Analysis summary does not contain a BIE path")
    bie_path = resolve_summary_path(str(bie_value), summary_path)
    if not bie_path.is_file():
        raise FileNotFoundError(bie_path)

    actual_bie_sha = sha256_file(bie_path)
    summary_bie_sha = str(summary.get("bie_sha256", "")).lower()
    expected_bie_sha = args.expected_bie_sha256.lower()

    print("Radiation Edge AI - frozen DNAi BIE -> KL720 NEF")
    print(f"Analysis summary: {summary_path}")
    print(f"BIE: {bie_path}")
    print(f"BIE SHA256: {actual_bie_sha}")

    if summary_bie_sha and actual_bie_sha != summary_bie_sha:
        raise RuntimeError(
            "Persistent BIE SHA256 does not match the saved analysis summary: "
            f"actual={actual_bie_sha} summary={summary_bie_sha}"
        )
    if actual_bie_sha != expected_bie_sha:
        raise RuntimeError(
            "Refusing to compile an unapproved BIE: "
            f"actual={actual_bie_sha} expected={expected_bie_sha}"
        )

    onnx_sha = str(summary.get("onnx_sha256", "")).lower()
    if onnx_sha != EXPECTED_V3_ONNX_SHA256:
        raise RuntimeError(
            "Unexpected source ONNX SHA256 for frozen v3: "
            f"{onnx_sha or 'missing'}"
        )

    ptq_config = summary.get("ptq_config")
    if ptq_config != EXPECTED_V3_PTQ:
        raise RuntimeError(
            "Unexpected PTQ recipe for frozen v3: "
            f"observed={ptq_config} expected={EXPECTED_V3_PTQ}"
        )

    if summary.get("validation_panel_used_for_calibration") is not False:
        raise RuntimeError("Calibration provenance is not explicitly validation-free")
    if int(summary.get("calibration_count", -1)) != 192:
        raise RuntimeError(
            f"Expected 192 frozen calibration tensors, got {summary.get('calibration_count')}"
        )
    if list(summary.get("calibration_shape", [])) != [1, 3, 512, 512]:
        raise RuntimeError(
            f"Unexpected calibration shape: {summary.get('calibration_shape')}"
        )

    platform = str(summary.get("platform", ""))
    model_id = int(summary.get("model_id"))
    model_version = str(summary.get("model_version"))
    input_name = str(summary.get("input_name", "input"))
    if platform != "720":
        raise RuntimeError(f"Expected KL720 platform, got {platform!r}")

    print("[Frozen provenance gate]")
    print(" - selected artifact: PTQ v3 percentage=1.0 optimize=0")
    print(f" - source ONNX SHA256: {onnx_sha}")
    print(f" - calibration tensors: {summary['calibration_count']}")
    print(f" - model ID/version: {model_id}/{model_version}")
    print(f" - input name: {input_name}")
    print(" - validation used for calibration: NO")
    print(" - provenance gate: PASS")

    print("[Create ModelConfig from frozen BIE]")
    km = ktc.ModelConfig(
        model_id,
        model_version,
        platform,
        bie_path=str(bie_path),
    )
    print(" - Success")

    print("[Compile NEF]")
    started = time.time()
    returned = Path(ktc.compile([km])).resolve()
    elapsed = time.time() - started
    if not returned.is_file():
        raise RuntimeError(f"Kneron compile returned missing NEF path: {returned}")

    destination = output_dir / args.output_name
    if returned.resolve() != destination.resolve():
        shutil.copy2(returned, destination)

    nef_sha = sha256_file(destination)
    result = {
        "analysis_summary": str(summary_path),
        "source_bie": str(bie_path),
        "source_bie_sha256": actual_bie_sha,
        "source_onnx_sha256": onnx_sha,
        "ptq_config": ptq_config,
        "calibration_count": int(summary["calibration_count"]),
        "validation_panel_used_for_calibration": False,
        "platform": platform,
        "model_id": model_id,
        "model_version": model_version,
        "input_name": input_name,
        "nef_returned": str(returned),
        "nef_persistent": str(destination),
        "nef_sha256": nef_sha,
        "nef_size_bytes": destination.stat().st_size,
        "compile_elapsed_seconds": elapsed,
    }
    out_summary = output_dir / "nef_compile_summary.json"
    out_summary.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("")
    print("KL720 NEF COMPILE COMPLETE: YES")
    print("FROZEN V3 BIE PRESERVED: YES")
    print(f"NEF: {destination}")
    print(f"NEF SHA256: {nef_sha}")
    print(f"NEF size: {destination.stat().st_size / 1024**2:.2f} MiB")
    print(f"Elapsed: {elapsed:.1f} s")
    print(f"Summary: {out_summary}")
    print("")
    print(
        "NEXT GATE: compare NEF vs frozen BIE on one validation window, then move "
        "to physical KL720 inference."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
