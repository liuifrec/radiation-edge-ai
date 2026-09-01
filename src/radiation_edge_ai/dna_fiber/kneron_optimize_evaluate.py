"""Optimize a DNAi ONNX model with Kneron Toolchain and evaluate KL720 support.

This script is intended to run *inside* the official kneron/toolchain Docker
image. It performs only the floating-point preparation gate:

1. load ONNX
2. optimize with kneronnxopt
3. save optimized ONNX
4. create ktc.ModelConfig for platform 720
5. run IP evaluation to expose unsupported/CPU operators and estimated NPU cost

No quantization or compilation is performed here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx
import kneronnxopt
import ktc


def graph_summary(model: onnx.ModelProto) -> dict:
    ops = Counter(node.op_type for node in model.graph.node)
    return {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "operators": dict(sorted(ops.items())),
        "inputs": [x.name for x in model.graph.input],
        "outputs": [x.name for x in model.graph.output],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--platform", default="720")
    parser.add_argument("--model-id", type=int, default=32769)
    parser.add_argument("--model-version", default="8b28")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    report = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    print("Radiation Edge AI - Kneron ONNX optimize/evaluate")
    print(f"Input: {source}")
    print(f"Platform: {args.platform}")
    print("")

    if not source.is_file():
        raise FileNotFoundError(source)

    print("[Load ONNX]")
    original = onnx.load(str(source))
    onnx.checker.check_model(original)
    before = graph_summary(original)
    print(f" - checker: PASS")
    print(f" - nodes: {before['nodes']}")
    print(" - operators: " + ", ".join(f"{k}={v}" for k, v in before["operators"].items()))

    print("[Kneron optimize]")
    optimized = kneronnxopt.optimize(original)
    onnx.checker.check_model(optimized)
    onnx.save(optimized, str(output))
    after = graph_summary(optimized)
    print(" - optimizer: PASS")
    print(f" - optimized ONNX: {output}")
    print(f" - nodes: {after['nodes']}")
    print(" - operators: " + ", ".join(f"{k}={v}" for k, v in after["operators"].items()))

    print("[KL720 IP evaluation]")
    km = ktc.ModelConfig(
        args.model_id,
        args.model_version,
        str(args.platform),
        onnx_model=optimized,
    )
    evaluation = km.evaluate()
    print(evaluation)

    payload = {
        "input": str(source),
        "optimized_onnx": str(output),
        "platform": str(args.platform),
        "model_id": args.model_id,
        "model_version": args.model_version,
        "before": before,
        "after": after,
        "evaluation": str(evaluation),
    }
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("")
    print("KNERON KL720 FLOATING-POINT GATE COMPLETE: YES")
    print(f"Optimized ONNX: {output}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
