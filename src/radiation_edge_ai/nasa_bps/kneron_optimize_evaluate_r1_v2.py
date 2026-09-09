"""Run the frozen NASA BPS R1 v2 ONNX through Kneron optimize/IP evaluation.

This script is intentionally compatible with the pinned historical Kneron
Toolchain image already used by this project (Python 3.7). Older Kneron images
expose ONNX optimization through ``ktc.onnx_optimizer.onnx2onnx_flow`` whereas
newer images may provide the standalone ``kneronnxopt`` package. The script
supports both without changing model weights, preprocessing, biological gates,
or any holdout data.

Run inside the Kneron Toolchain Docker image. This stage performs floating-point
deployment preparation only: verify frozen identities, optimize the ONNX graph,
create a KL720 ModelConfig, and run IP evaluation for unsupported operators /
estimated NPU support. No PTQ, BIE generation, NEF compilation, or final-holdout
inference occurs here.
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import ktc
import onnx

try:
    import kneronnxopt
except ImportError:
    kneronnxopt = None

FROZEN_ONNX_SHA256 = (
    "a65718a8f07d5bcd8707bae78d8f730d56047db21e575fa55ba5396adc49a07e"
)
FROZEN_ONNX_FREEZE_SHA256 = (
    "e31c64050244ea93cbb3ea57433e0f281218bbbd352ee66d058e98bb3e36fe06"
)
EXPECTED_INPUT_SHAPE = [1, 3, 256, 256]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("Expected JSON object: {}".format(path))
    return obj


def dims(value_info):
    result = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(str(dim.dim_param))
        else:
            result.append(None)
    return result


def graph_summary(model):
    ops = Counter(node.op_type for node in model.graph.node)
    initializer_names = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializer_names]
    return {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "operators": dict(sorted(ops.items())),
        "inputs": [{"name": value.name, "shape": dims(value)} for value in inputs],
        "outputs": [{"name": value.name, "shape": dims(value)} for value in model.graph.output],
    }


def optimize_model(model):
    """Use the optimizer API exposed by the installed Kneron toolchain."""
    if kneronnxopt is not None:
        return kneronnxopt.optimize(model), "kneronnxopt.optimize", getattr(
            kneronnxopt, "__version__", None
        )

    onnx_optimizer = getattr(ktc, "onnx_optimizer", None)
    if onnx_optimizer is None:
        raise RuntimeError(
            "Pinned Kneron toolchain has neither standalone kneronnxopt nor "
            "ktc.onnx_optimizer"
        )
    flow = getattr(onnx_optimizer, "onnx2onnx_flow", None)
    if flow is None:
        raise RuntimeError(
            "Pinned Kneron toolchain exposes ktc.onnx_optimizer but not "
            "onnx2onnx_flow"
        )
    return flow(model), "ktc.onnx_optimizer.onnx2onnx_flow", None


def read_text_if_present(path):
    if path is None or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def find_newest(filename, started):
    roots = (Path("/workspace/.tmp"), Path("/data1/kneron_flow"), Path("/tmp"))
    candidates = []
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
    fresh = [path for path in candidates if path.stat().st_mtime >= started - 5.0]
    pool = fresh if fresh else candidates
    return max(pool, key=lambda path: path.stat().st_mtime)


def harvest(filename, output_dir, started):
    source = find_newest(filename, started)
    if source is None:
        return None
    destination = output_dir / filename
    if source.resolve() != destination.resolve():
        shutil.copy2(str(source), str(destination))
    return str(destination)


def toolchain_version():
    for candidate in (Path("/workspace/version.txt"), Path("/workspace/VERSION")):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace").strip()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--onnx-freeze", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--platform", default="720")
    parser.add_argument("--model-id", type=int, default=32770)
    parser.add_argument("--model-version", default="8b29")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    freeze_path = Path(args.onnx_freeze).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    optimized_path = output_dir / "r1_countnet_v2_final_kl720_optimized.onnx"
    report_path = output_dir / "kl720_fp_optimization_evaluation.json"

    if report_path.exists():
        raise RuntimeError(
            "KL720 floating-point gate already evaluated; refusing overwrite: {}".format(
                report_path
            )
        )
    if not source.is_file():
        raise FileNotFoundError(source)
    if not freeze_path.is_file():
        raise FileNotFoundError(freeze_path)

    source_sha = sha256_file(source)
    freeze_sha = sha256_file(freeze_path)
    if source_sha != FROZEN_ONNX_SHA256:
        raise RuntimeError(
            "Frozen ONNX SHA256 mismatch: expected {}, got {}".format(
                FROZEN_ONNX_SHA256, source_sha
            )
        )
    if freeze_sha != FROZEN_ONNX_FREEZE_SHA256:
        raise RuntimeError(
            "ONNX freeze SHA256 mismatch: expected {}, got {}".format(
                FROZEN_ONNX_FREEZE_SHA256, freeze_sha
            )
        )
    freeze = read_json(freeze_path)
    if freeze.get("onnx_sha256") != FROZEN_ONNX_SHA256:
        raise RuntimeError("ONNX freeze JSON points to a different model")
    if freeze.get("all_seven_biological_gates_pass") is not True:
        raise RuntimeError("ONNX freeze does not authorize Kneron deployment")

    version = toolchain_version()
    print("Radiation Edge AI - NASA BPS R1 v2 Kneron KL720 floating-point gate")
    print("frozen ONNX SHA256: {}".format(source_sha))
    print("ONNX deployment-freeze SHA256: {}".format(freeze_sha))
    print("platform: {}".format(args.platform))
    print("model ID/version: {}/{}".format(args.model_id, args.model_version))
    print("Kneron toolchain version: {}".format(version or "UNKNOWN"))
    print("Python-compatible toolchain adapter: ENABLED")
    print("PTQ performed: NO")
    print("final holdout read: NO")

    print("")
    print("[Load frozen ONNX]")
    original = onnx.load(str(source))
    onnx.checker.check_model(original)
    before = graph_summary(original)
    if len(before["inputs"]) != 1 or before["inputs"][0]["shape"] != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(
            "Unexpected frozen ONNX input signature: {}".format(before["inputs"])
        )
    print("ONNX checker: PASS")
    print("nodes: {}".format(before["nodes"]))
    print(
        "operators: "
        + ", ".join("{}={}".format(k, v) for k, v in before["operators"].items())
    )

    print("")
    print("[Kneron optimize]")
    optimized, optimizer_backend, optimizer_version = optimize_model(original)
    onnx.checker.check_model(optimized)
    onnx.save(optimized, str(optimized_path))
    after = graph_summary(optimized)
    optimized_sha = sha256_file(optimized_path)
    print("optimizer backend: {}".format(optimizer_backend))
    print("optimizer: PASS")
    print("optimized ONNX SHA256: {}".format(optimized_sha))
    print("nodes: {}".format(after["nodes"]))
    print(
        "operators: "
        + ", ".join("{}={}".format(k, v) for k, v in after["operators"].items())
    )

    print("")
    print("[KL720 IP evaluation]")
    km = ktc.ModelConfig(
        args.model_id,
        args.model_version,
        str(args.platform),
        onnx_model=optimized,
    )
    old_cwd = Path.cwd()
    started = time.time()
    try:
        os.chdir(str(output_dir))
        evaluation = km.evaluate()
    finally:
        os.chdir(str(old_cwd))
    evaluation_text = str(evaluation)
    print(evaluation_text)

    artifacts = {}
    for filename in (
        "model_fx_report.json",
        "model_fx_report.html",
        "batch_compile.log",
        "backtrace.log",
        "ioinfo.csv",
    ):
        artifacts[filename] = harvest(filename, output_dir, started)

    status_parts = [evaluation_text]
    for path in artifacts.values():
        if path:
            status_parts.append(read_text_if_present(Path(path)))
    status_corpus = "\n".join(status_parts).lower()
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

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_KL720_FLOATING_POINT_OPTIMIZATION_EVALUATION",
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_onnx_sha256": source_sha,
        "onnx_deployment_freeze_sha256": freeze_sha,
        "optimized_onnx": str(optimized_path),
        "optimized_onnx_sha256": optimized_sha,
        "platform": str(args.platform),
        "model_id": args.model_id,
        "model_version": args.model_version,
        "toolchain_version": version,
        "optimizer_backend": optimizer_backend,
        "optimizer_version": optimizer_version,
        "before": before,
        "after": after,
        "evaluation": evaluation_text,
        "hardware_supported": hardware_supported,
        "artifacts": artifacts,
        "ptq_performed": False,
        "bie_generated": False,
        "nef_compiled": False,
        "final_holdout_read": False,
        "compatibility_recovery": (
            "Python 3.7 syntax + historical ktc.onnx_optimizer adapter; "
            "no model/preprocessing/endpoint/gate change"
        ),
        "authorized_next_stage": (
            "FROZEN_DEVELOPMENT_ONLY_PTQ"
            if hardware_supported
            else "BLOCKED_PENDING_IMPLEMENTATION_COMPATIBILITY_DIAGNOSIS"
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("")
    print("KNERON KL720 FLOATING-POINT GATE COMPLETE: YES")
    print("KL720 HW SUPPORT: {}".format("YES" if hardware_supported else "NO"))
    print("optimized ONNX: {}".format(optimized_path))
    print("optimized ONNX SHA256: {}".format(optimized_sha))
    print("report: {}".format(report_path))
    print("report SHA256: {}".format(sha256_file(report_path)))
    print("PTQ performed: NO")
    print("final holdout read: NO")
    if not hardware_supported:
        raise SystemExit(
            "KL720 floating-point IP-support gate did not pass; do not quantize yet."
        )
    print(
        "NEXT GATE: quantize this exact optimized ONNX using the frozen "
        "288-development-nucleus calibration panel."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
