"""Export DNAi segmentation models to fixed-shape ONNX and verify parity.

The exporter is generic across DNAi model-zoo entries. It exports the neural
segmentation core (3-channel logits) at a fixed NCHW shape, checks the ONNX
graph, runs ONNX Runtime on CPU, and compares ONNX logits against strict-FP32
PyTorch logits using the same deterministic input.

Softmax, DNAi segmentation morphology, and fiber reconstruction deliberately
stay outside the ONNX graph so the accelerator boundary is explicit and the
same CPU post-processing can be reused for PyTorch, ONNX, and KL720 outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from dnafiber.model.models_zoo import Models
from dnafiber.model.utils import _get_model

DNAI_COMMIT = "fcf20c7d6eb385675ff7d07da4fdf471589ce0cf"
DEFAULT_SHAPE = (1, 3, 1024, 1024)
DEFAULT_OPSET = 11


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state_dict(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key in sorted(model.state_dict()):
        tensor = model.state_dict()[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def available_model_names() -> list[str]:
    return [item.value for item in Models if item != Models.ENSEMBLE]


def parse_model(value: str) -> Models:
    try:
        model = Models(value)
    except ValueError as exc:
        valid = ", ".join(available_model_names())
        raise argparse.ArgumentTypeError(f"Unknown DNAi model '{value}'. Valid: {valid}") from exc
    if model == Models.ENSEMBLE:
        raise argparse.ArgumentTypeError("ENSEMBLE is not a single exportable model")
    return model


def graph_summary(model_proto: onnx.ModelProto) -> dict:
    ops = Counter(node.op_type for node in model_proto.graph.node)
    return {
        "ir_version": model_proto.ir_version,
        "opset_imports": {item.domain or "ai.onnx": item.version for item in model_proto.opset_import},
        "n_nodes": len(model_proto.graph.node),
        "n_initializers": len(model_proto.graph.initializer),
        "operator_counts": dict(sorted(ops.items())),
        "inputs": [value.name for value in model_proto.graph.input],
        "outputs": [value.name for value in model_proto.graph.output],
    }


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict:
    diff = candidate.astype(np.float64) - reference.astype(np.float64)
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(reference.astype(np.float64)), 1e-8)
    rel = abs_diff / denom
    ref_argmax = reference.argmax(axis=1)
    cand_argmax = candidate.argmax(axis=1)
    return {
        "max_abs_error": float(abs_diff.max()),
        "mean_abs_error": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_rel_error": float(rel.max()),
        "mean_rel_error": float(rel.mean()),
        "argmax_agreement": float(np.mean(ref_argmax == cand_argmax)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    default_model_root = os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    parser.add_argument("--model", type=parse_model, default=Models.UNET_MOBILEONE_S1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--model-root", default=default_model_root)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    shape = (1, 3, args.height, args.width)
    if args.height <= 0 or args.width <= 0:
        raise SystemExit("Height and width must be positive")

    output_dir = Path(args.model_root) / "dnai" / args.model.value / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{args.model.value}_{args.height}x{args.width}_opset{args.opset}.onnx"
    report_path = output_dir / f"{args.model.value}_{args.height}x{args.width}_opset{args.opset}.json"

    print("Radiation Edge AI - DNAi ONNX export")
    print(f"DNAi commit: {DNAI_COMMIT}")
    print(f"Model: {args.model.value}")
    print(f"Input shape: {shape}")
    print(f"Opset: {args.opset}")
    print(f"Output: {onnx_path}")
    print("")

    print("[Load model]")
    t0 = time.perf_counter()
    model = _get_model(args.model).to("cpu", dtype=torch.float32).eval()
    model_hash = sha256_state_dict(model)
    print(f" - Success ({time.perf_counter() - t0:.2f}s)")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"state_dict SHA256: {model_hash}")

    torch.manual_seed(args.seed)
    dummy = torch.rand(shape, dtype=torch.float32)

    print("[PyTorch reference]")
    with torch.inference_mode():
        ref = model(dummy).detach().cpu().numpy().astype(np.float32, copy=False)
    print(f" - output shape: {tuple(ref.shape)}")
    print(f" - output dtype: {ref.dtype}")
    if ref.shape != (1, 3, args.height, args.width):
        raise RuntimeError(f"Unexpected PyTorch output shape: {ref.shape}")

    print("[Export ONNX]")
    export_start = time.perf_counter()
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        training=torch.onnx.TrainingMode.EVAL,
    )
    print(f" - Success ({time.perf_counter() - export_start:.2f}s)")

    print("[Check ONNX graph]")
    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)
    summary = graph_summary(model_proto)
    print(" - ONNX checker: PASS")
    print(f" - nodes: {summary['n_nodes']}")
    print(" - operators: " + ", ".join(f"{k}={v}" for k, v in summary["operator_counts"].items()))

    print("[ONNX Runtime parity]")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_start = time.perf_counter()
    output = session.run(["logits"], {"input": dummy.numpy()})[0]
    ort_ms = (time.perf_counter() - ort_start) * 1000.0
    parity = compare_arrays(ref, output)
    print(f" - ORT latency: {ort_ms:.2f} ms")
    print(f" - max abs error: {parity['max_abs_error']:.8g}")
    print(f" - mean abs error: {parity['mean_abs_error']:.8g}")
    print(f" - RMSE: {parity['rmse']:.8g}")
    print(f" - argmax agreement: {parity['argmax_agreement']:.8f}")

    # Numerical tolerances are intentionally strict enough to catch export
    # mistakes while allowing normal CPU kernel floating-point differences.
    parity_pass = (
        parity["max_abs_error"] <= 1e-3
        and parity["mean_abs_error"] <= 1e-5
        and parity["argmax_agreement"] >= 0.99999
    )

    report = {
        "dnai_commit": DNAI_COMMIT,
        "model": args.model.value,
        "model_state_dict_sha256": model_hash,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "input_shape": list(shape),
        "output_shape": list(output.shape),
        "opset": args.opset,
        "fixed_shape": True,
        "onnx_path": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "graph": summary,
        "onnxruntime": {
            "version": ort.__version__,
            "provider": "CPUExecutionProvider",
            "latency_ms": ort_ms,
            "parity": parity,
            "pass": parity_pass,
        },
        "accelerator_boundary": {
            "inside_onnx": "DNAi segmentation network producing 3-class logits",
            "outside_onnx": [
                "DNAi microscopy image preprocessing",
                "ImageNet normalization",
                "softmax",
                "3-class argmax/morphological dilation",
                "fiber reconstruction and tract measurements",
            ],
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("")
    print(f"ONNX PARITY READY: {'YES' if parity_pass else 'NO'}")
    print(f"ONNX:   {onnx_path}")
    print(f"SHA256: {report['onnx_sha256']}")
    print(f"Size:   {onnx_path.stat().st_size / 1024**2:.2f} MiB")
    print(f"Report: {report_path}")
    if not parity_pass:
        raise SystemExit("ONNX export completed but parity thresholds did not pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
