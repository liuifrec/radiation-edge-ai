"""Optimize a DNAi ONNX model with Kneron Toolchain and evaluate KL720 support.

This script is intended to run *inside* the official kneron/toolchain Docker
image. It performs only the floating-point preparation gate:

1. load ONNX
2. optimize with kneronnxopt
3. save optimized ONNX
4. create ktc.ModelConfig for platform 720
5. run IP evaluation to expose unsupported/CPU operators and estimated NPU cost
6. harvest Kneron's reports and compiler logs into the persistent report folder

No quantization or compilation-to-NEF is performed here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import kneronnxopt
import ktc
import onnx


def graph_summary(model: onnx.ModelProto) -> dict:
    ops = Counter(node.op_type for node in model.graph.node)
    return {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "operators": dict(sorted(ops.items())),
        "inputs": [x.name for x in model.graph.input],
        "outputs": [x.name for x in model.graph.output],
    }


def read_text_if_present(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def find_newest_artifact(filename: str, evaluation_started: float) -> Path | None:
    """Find a KTC-generated artifact in known internal work roots."""

    roots = (Path("/workspace/.tmp"), Path("/data1/kneron_flow"), Path("/tmp"))
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

    fresh = [p for p in candidates if p.stat().st_mtime >= evaluation_started - 5.0]
    pool = fresh if fresh else candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def harvest_artifact(
    filename: str,
    destination_dir: Path,
    evaluation_started: float,
    destination_name: str | None = None,
) -> tuple[Path | None, Path | None]:
    source = find_newest_artifact(filename, evaluation_started)
    if source is None:
        return None, None
    destination = destination_dir / (destination_name or filename)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return source, destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--platform", default="720")
    parser.add_argument("--model-id", type=int, default=32769)
    parser.add_argument("--model-version", default="8b28")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    report = Path(args.report).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    print("Radiation Edge AI - Kneron ONNX optimize/evaluate")
    print(f"Input: {source}")
    print(f"Platform: {args.platform}")
    print(f"Persistent diagnostic directory: {report.parent}")
    print("")

    if not source.is_file():
        raise FileNotFoundError(source)

    print("[Load ONNX]")
    original = onnx.load(str(source))
    onnx.checker.check_model(original)
    before = graph_summary(original)
    print(" - checker: PASS")
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

    old_cwd = Path.cwd()
    evaluation_started = time.time()
    try:
        os.chdir(report.parent)
        evaluation = km.evaluate()
    finally:
        os.chdir(old_cwd)

    evaluation_text = str(evaluation)
    print(evaluation_text)

    artifacts: dict[str, tuple[Path | None, Path | None]] = {}
    for filename in (
        "model_fx_report.json",
        "model_fx_report.html",
        "batch_compile.log",
        "backtrace.log",
        "ioinfo.csv",
    ):
        artifacts[filename] = harvest_artifact(
            filename, report.parent, evaluation_started
        )
        source_path, destination_path = artifacts[filename]
        if destination_path is not None:
            print(f" - harvested {filename}: {source_path} -> {destination_path}")

    fx_json = artifacts["model_fx_report.json"][1]
    fx_html = artifacts["model_fx_report.html"][1]
    batch_log = artifacts["batch_compile.log"][1]
    backtrace_log = artifacts["backtrace.log"][1]
    ioinfo_csv = artifacts["ioinfo.csv"][1]

    status_corpus = "\n".join(
        (
            evaluation_text,
            read_text_if_present(fx_json),
            read_text_if_present(fx_html),
            read_text_if_present(batch_log),
            read_text_if_present(backtrace_log),
        )
    ).lower()
    failure_markers = (
        "hw not support",
        "hardware not support",
        "hardwarenotsupport",
        "not supported",
        "unsupported",
        "failure for model",
        "failed to compile",
        "err: 4",
    )
    hardware_supported = not any(marker in status_corpus for marker in failure_markers)

    payload = {
        "input": str(source),
        "optimized_onnx": str(output),
        "platform": str(args.platform),
        "model_id": args.model_id,
        "model_version": args.model_version,
        "before": before,
        "after": after,
        "evaluation": evaluation_text,
        "hardware_supported": hardware_supported,
        "artifacts": {
            name: {
                "source": str(paths[0]) if paths[0] else None,
                "persistent": str(paths[1]) if paths[1] else None,
            }
            for name, paths in artifacts.items()
        },
    }
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("")
    print("KNERON KL720 FLOATING-POINT GATE COMPLETE: YES")
    print(f"KL720 HW SUPPORT: {'YES' if hardware_supported else 'NO'}")
    print(f"Optimized ONNX: {output}")
    print(f"Report: {report}")
    if batch_log:
        print(f"Compiler log: {batch_log}")
    if fx_html:
        print(f"FX HTML: {fx_html}")
    if fx_json:
        print(f"FX JSON: {fx_json}")
    if ioinfo_csv:
        print(f"IO info: {ioinfo_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
