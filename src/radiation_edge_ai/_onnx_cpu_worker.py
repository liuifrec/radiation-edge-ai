"""Isolated CPU ONNX Runtime worker used by ``radedge run``.

This file is executed by a Python interpreter that already contains NumPy and
ONNX Runtime.  ONNX Runtime is imported only inside ``main`` so importing the
package from the lightweight control environment does not require it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temp.replace(path)


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _shape_matches(expected: list[object], observed: tuple[int, ...]) -> bool:
    if len(expected) != len(observed):
        return False
    for expected_dim, observed_dim in zip(expected, observed):
        if isinstance(expected_dim, int) and expected_dim != observed_dim:
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    import onnxruntime as ort

    model_path = args.model.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()

    value = np.load(input_path, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise RuntimeError("Input .npy did not contain an ndarray")
    if value.dtype != np.float32:
        raise RuntimeError(f"Input dtype must be float32, got {value.dtype}")
    if not np.isfinite(value).all():
        raise RuntimeError("Input tensor contains non-finite values")

    options = ort.SessionOptions()
    session_start = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    session_seconds = time.perf_counter() - session_start

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        counts = f"{len(inputs)}/{len(outputs)}"
        raise RuntimeError(
            "v0.1b worker requires exactly one input and one output, "
            f"got {counts}"
        )

    input_node = inputs[0]
    output_node = outputs[0]
    if input_node.type != "tensor(float)":
        raise RuntimeError(f"Expected float32 ONNX input, got {input_node.type}")
    if not _shape_matches(list(input_node.shape), value.shape):
        raise RuntimeError(
            f"Input shape mismatch: model expects {input_node.shape}, got {list(value.shape)}"
        )

    infer_start = time.perf_counter()
    results = session.run([output_node.name], {input_node.name: value})
    inference_seconds = time.perf_counter() - infer_start
    output = np.asarray(results[0])
    if not np.issubdtype(output.dtype, np.floating):
        raise RuntimeError(f"Expected floating ONNX output, got {output.dtype}")
    if not np.isfinite(output).all():
        raise RuntimeError("ONNX output contains non-finite values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_npy(output_path, output)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "backend": "cpu-onnx",
        "input": {
            "path": str(input_path),
            "sha256": _sha256_file(input_path),
            "dtype": str(value.dtype),
            "shape": [int(x) for x in value.shape],
        },
        "model": {
            "path": str(model_path),
            "sha256": _sha256_file(model_path),
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
            "dtype": str(output.dtype),
            "shape": [int(x) for x in output.shape],
        },
        "model_io": {
            "input_name": input_node.name,
            "input_type": input_node.type,
            "input_shape": list(input_node.shape),
            "output_name": output_node.name,
            "output_type": output_node.type,
            "output_shape": list(output_node.shape),
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "onnxruntime_version": ort.__version__,
            "requested_provider": "CPUExecutionProvider",
            "active_providers": session.get_providers(),
        },
        "timing": {
            "session_create_seconds": session_seconds,
            "inference_seconds": inference_seconds,
        },
    }

    for key in ("session_create_seconds", "inference_seconds"):
        if not math.isfinite(float(manifest["timing"][key])):
            raise RuntimeError(f"Non-finite timing value: {key}")

    _atomic_write_json(manifest_path, manifest)
    print(json.dumps({"status": "complete", "output": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
