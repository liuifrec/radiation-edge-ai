"""Verify a compiled KL720 NEF against its frozen DNAi BIE on one window.

This is a compile-integrity gate, not a biological-validation run. It evaluates
one already-prepared 512x512 validation tensor with both Kneron fixed-point BIE
simulation and compiled NEF simulation, then compares the raw output tensors and
argmax decisions.

Run inside the pinned Kneron Toolchain Docker image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import ktc
import numpy as np

EXPECTED_INPUT_SHAPE = (1, 3, 512, 512)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_output(result, label: str) -> np.ndarray:
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise RuntimeError(
            f"{label} expected exactly one output, got {type(result)} "
            f"len={len(result) if hasattr(result, '__len__') else 'NA'}"
        )
    array = np.asarray(result[0], dtype=np.float32)
    if not np.isfinite(array).all():
        raise RuntimeError(f"{label} contains non-finite values")
    return np.ascontiguousarray(array)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bie", required=True)
    parser.add_argument("--nef", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--sample-id", default="1__tile_4")
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--platform", type=int, default=720)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    bie_path = Path(args.bie).resolve()
    nef_path = Path(args.nef).resolve()
    manifest_path = Path(args.validation_manifest).resolve()
    for path in (bie_path, nef_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest.get("tile", ())) != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(f"Unexpected validation tile: {manifest.get('tile')}")
    input_name = manifest.get("input_name", "input")

    selected = [
        record
        for record in manifest.get("records", [])
        if record.get("sample_id") == args.sample_id
        and int(record.get("window_index", -1)) == args.window_index
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one validation record for {args.sample_id} window "
            f"{args.window_index}, got {len(selected)}"
        )
    record = selected[0]
    tensor_path = manifest_path.parent / record["tensor"]
    array = np.load(tensor_path, allow_pickle=False)
    if array.shape != EXPECTED_INPUT_SHAPE or array.dtype != np.float32:
        raise RuntimeError(
            f"Unexpected validation tensor {array.shape} {array.dtype}: {tensor_path}"
        )
    array = np.ascontiguousarray(array, dtype=np.float32)

    print("Radiation Edge AI - KL720 NEF vs frozen BIE verification")
    print(f"Sample/window: {args.sample_id} / {args.window_index}")
    print(f"BIE SHA256: {sha256_file(bie_path)}")
    print(f"NEF SHA256: {sha256_file(nef_path)}")
    print(f"Input: {tensor_path}")

    t0 = time.perf_counter()
    bie_result = ktc.kneron_inference(
        [array],
        bie_file=str(bie_path),
        input_names=[input_name],
        platform=args.platform,
    )
    bie_ms = (time.perf_counter() - t0) * 1000.0
    bie = one_output(bie_result, "BIE")

    t0 = time.perf_counter()
    nef_result = ktc.kneron_inference(
        [array],
        nef_file=str(nef_path),
        input_names=[input_name],
        platform=args.platform,
    )
    nef_ms = (time.perf_counter() - t0) * 1000.0
    nef = one_output(nef_result, "NEF")

    if bie.shape != nef.shape:
        raise RuntimeError(f"Output shape mismatch: BIE={bie.shape} NEF={nef.shape}")

    diff = nef.astype(np.float64) - bie.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    exact = bool(np.array_equal(bie, nef))
    bie_class = bie.argmax(axis=1)
    nef_class = nef.argmax(axis=1)
    argmax_agreement = float(np.mean(bie_class == nef_class))
    disagree = int(np.count_nonzero(bie_class != nef_class))
    total = int(bie_class.size)

    # Kneron documents NEF and BIE as expected to match. We retain a tiny
    # numerical tolerance while requiring identical class decisions.
    passed = bool(argmax_agreement == 1.0 and max_abs <= 1e-4)

    payload = {
        "bie": str(bie_path),
        "bie_sha256": sha256_file(bie_path),
        "nef": str(nef_path),
        "nef_sha256": sha256_file(nef_path),
        "validation_manifest": str(manifest_path),
        "sample_id": args.sample_id,
        "window_index": args.window_index,
        "tensor": str(tensor_path),
        "input_name": input_name,
        "platform": args.platform,
        "output_shape": list(bie.shape),
        "exact_array_equal": exact,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rmse": rmse,
        "argmax_agreement": argmax_agreement,
        "disagreement_pixels": disagree,
        "total_pixels": total,
        "bie_ms": bie_ms,
        "nef_ms": nef_ms,
        "verification_pass": passed,
    }

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        output_path = None

    print("")
    print(f"exact array equality: {exact}")
    print(f"max absolute error: {max_abs:.8g}")
    print(f"mean absolute error: {mean_abs:.8g}")
    print(f"RMSE: {rmse:.8g}")
    print(f"argmax agreement: {argmax_agreement:.8f}")
    print(f"disagreement pixels: {disagree}/{total}")
    print(f"BIE simulator: {bie_ms:.1f} ms")
    print(f"NEF simulator: {nef_ms:.1f} ms")
    print(f"NEF/BIE VERIFICATION PASS: {'YES' if passed else 'NO'}")
    if output_path:
        print(f"Summary: {output_path}")
    print("")
    print("NEXT GATE: physical KL720 inference with the verified NEF.")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
