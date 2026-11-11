"""Isolated, single-case KL720 compiler feasibility for synthetic ConvTranspose ONNX.

Only synthetic models from perilipin2_kl720_synthetic_repro.py are accepted.
No trained model, microscopy image, calibration set, NEF, BIE, PTQ, or physical
KL720 is used. Each trial requires a new D: scratch folder. Docker uses
the previously pinned local toolchain image with networking disabled.

This is a compiler DIAGNOSTIC, not proof of accelerator support.
Compatible with the pinned Kneron toolchain's historical Python 3.7.
"""
from __future__ import print_function

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import perilipin2_resize_vendor_gate as v

SCHEMA = "perilipin2-kl720-synthetic-vendor-v1"
GENERATOR_SCHEMA = "perilipin2-kneron-synthetic-deconv-repro-v1"
CASES = {
    "control_c1_h8": (1, 8),
    "group_c24_h8": (24, 8),
    "group_c96_h8": (96, 8),
    "stage_c96_h32": (96, 32),
}
SOURCE_NAME = "SYNTHETIC_REPRO_MANIFEST.json"
RUNNER_NAME = "synthetic_vendor_runner.py"
HELPER_NAME = "perilipin2_resize_vendor_gate.py"
MODEL_NAME = "synthetic_candidate.onnx"
INTAKE_NAME = "SYNTHETIC_VENDOR_INTAKE.json"
CONTAINER_REPORT = "SYNTHETIC_VENDOR_REPORT.json"
HOST_REPORT = "SYNTHETIC_VENDOR_EXECUTION.json"
ARTIFACT_NAMES = (
    "ioinfo.csv", "model_fx_report.json", "model_fx_report.html",
    "batch_compile.log", "backtrace.log",
)


def metadata_contract(manifest, case):
    if manifest.get("schema") != GENERATOR_SCHEMA:
        raise ValueError("Unexpected synthetic generator schema")
    if manifest.get("source") != "fully synthetic; no microscopy images or learned model weights":
        raise ValueError("Unrecognized source classification")
    if manifest.get("vendor_evaluate_performed") is not False:
        raise ValueError("Expected untested synthetic source manifest")
    if case not in CASES:
        raise ValueError("Unknown synthetic model case")
    meta = manifest.get("cases", {}).get(case)
    if not isinstance(meta, dict):
        raise ValueError("Synthetic case missing in manifest: " + case)
    channels, height = CASES[case]
    expected = {
        "filename": case + ".onnx", "operator": "ConvTranspose",
        "group": channels, "input_shape": [1, channels, height, height],
        "output_shape": [1, channels, 2 * height, 2 * height],
        "kernel_shape": [2, 2], "strides": [2, 2],
        "pads": [0, 0, 0, 0], "fixed_weight_shape": [channels, 1, 2, 2],
        "opset": 11, "ir_version": 7,
        "onnxruntime_vs_nearest_2x_max_abs_error": 0.0,
        "vendor_compilation_tested": False,
    }
    for key, wanted in expected.items():
        if meta.get(key) != wanted:
            raise ValueError("Synthetic manifest mismatch at {}.{}".format(case, key))
    digest = meta.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Missing synthetically generated ONNX SHA256")
    if not all(c in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid synthetic candidate digest")
    return meta


def graph_contract(onnx, model, case):
    import numpy as np

    channels, height = CASES[case]
    if model.ir_version != 7:
        raise ValueError("Synthetic IR version changed")
    if [(x.domain, x.version) for x in model.opset_import] != [("", 11)]:
        raise ValueError("Synthetic ONNX opset changed")
    graph = model.graph
    if len(graph.node) != 1 or len(graph.initializer) != 1:
        raise ValueError("Expected a single ConvTranspose node and fixed kernel")
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("Expected exactly one synthetic input/output")
    if len(graph.value_info) != 0:
        raise ValueError("Expected no synthetic intermediate value_info")
    def verify_tensor(value, name, dimensions):
        if value.name != name or value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError("Unexpected synthetic IO tensor name/dtype")
        shape = [int(x.dim_value) for x in value.type.tensor_type.shape.dim]
        if shape != dimensions:
            raise ValueError("Unexpected synthetic IO tensor dimensions: " + repr(shape))
    verify_tensor(graph.input[0], "input", [1, channels, height, height])
    verify_tensor(graph.output[0], "output", [1, channels, 2 * height, 2 * height])
    node = graph.node[0]
    if (node.op_type != "ConvTranspose" or len(node.input) != 2
            or list(node.input) != ["input", "fixed_all_ones_kernel"]
            or list(node.output) != ["output"]):
        raise ValueError("Synthetic graph is not the expected single ConvTranspose")
    expected_attrs = {
        "group": channels, "strides": [2, 2],
        "kernel_shape": [2, 2], "pads": [0, 0, 0, 0],
    }
    attrs = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.INT:
            attrs[attr.name] = int(attr.i)
        elif attr.type == onnx.AttributeProto.INTS:
            attrs[attr.name] = [int(x) for x in attr.ints]
        else:
            raise ValueError("Unexpected ConvTranspose attribute type")
    if attrs != expected_attrs:
        raise ValueError("Unexpected synthetic ConvTranspose attributes: " + repr(attrs))
    weight = graph.initializer[0]
    if weight.name != "fixed_all_ones_kernel":
        raise ValueError("Unexpected synthetic weight name")
    kernel = onnx.numpy_helper.to_array(weight)
    if (list(kernel.shape) != [channels, 1, 2, 2]
            or kernel.dtype != np.float32 or not np.all(kernel == 1)):
        raise ValueError("Unexpected synthetic kernel: not all float32 ones")


def preflight_source(manifest_dir, case):
    import onnx

    root = Path(manifest_dir).expanduser().resolve(strict=True)
    manifest_file = root / SOURCE_NAME
    manifest = v.read_json(manifest_file)
    meta = metadata_contract(manifest, case)
    path = root / meta["filename"]
    if not path.is_file() or path.is_symlink():
        raise ValueError("Synthetic model is missing or symlinked")
    if v.sha256(path) != meta["sha256"]:
        raise ValueError("Synthetic ONNX SHA256 changed since generator proof")
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    graph_contract(onnx, model, case)
    return root, manifest_file, path, meta


def in_container(args):
    import onnx
    import ktc

    root = Path(args.out).resolve(strict=True)
    intake = v.read_json(root / INTAKE_NAME)
    if intake.get("schema") != SCHEMA:
        raise ValueError("Unexpected synthetic trial intake schema")
    for filename, digest_key in (
        (MODEL_NAME, "candidate_sha256"),
        (RUNNER_NAME, "runner_sha256"),
        (HELPER_NAME, "helper_sha256"),
    ):
        if v.sha256(root / filename) != intake[digest_key]:
            raise ValueError("Staged synthetic trial artifact was modified: " + filename)
    case = intake["case"]
    if case not in CASES:
        raise ValueError("Unexpected synthetic case in container")
    model = onnx.load(str(root / MODEL_NAME))
    onnx.checker.check_model(model)
    graph_contract(onnx, model, case)

    print("SYNTHETIC_KL720_VENDOR_CASE: " + case, flush=True)
    print("MODEL_SHA256: " + intake["candidate_sha256"], flush=True)
    print("NO ONNX OPTIMIZER: sending exact single-op synthetic model", flush=True)
    config = ktc.ModelConfig(
        intake["model_id"], intake["model_version"], "720", onnx_model=model
    )
    started = time.time()
    error = None
    returned = ""
    try:
        returned = str(config.evaluate())
    except Exception as exc:
        error = "{}: {}".format(type(exc).__name__, exc)
        print("SYNTHETIC_VENDOR_EXCEPTION: " + error, flush=True)
    (root / "SYNTHETIC_VENDOR_EVALUATION_TEXT.txt").write_text(
        returned + "\n", encoding="utf-8")
    if returned:
        print(returned, flush=True)

    collected = {}
    for filename in ARTIFACT_NAMES:
        found = v.newest_artifact(filename, started)
        if found is None:
            continue
        target = root / filename
        if found.resolve() != target.resolve():
            if target.exists():
                raise ValueError("Refusing to replace synthetic vendor report " + filename)
            shutil.copyfile(str(found), str(target))
        collected[filename] = v.sha256(target)
    artifact_text = "\n".join(
        (root / name).read_text(encoding="utf-8", errors="replace")
        for name in collected if name.endswith((".log", ".csv", ".json"))
    )
    markers = v.failure_markers(returned + "\n" + artifact_text)
    # KTC can print the failure table directly to C stdout while returning
    # only a metadata report from evaluate(), even with a 0 exit code.
    if "backtrace.log" in collected and "compiler_backtrace" not in markers:
        markers.append("compiler_backtrace")
    state = (
        "VENDOR_REJECTED" if markers or error
        else "VENDOR_RETURNED_REQUIRES_REPORT_REVIEW"
    )
    receipt = {
        "schema": SCHEMA, "status": state, "case": case,
        "candidate_sha256": intake["candidate_sha256"],
        "compiler_backtrace_present": "backtrace.log" in collected,
        "vendor_exception": error, "vendor_failure_markers": markers,
        "artifact_sha256": collected, "optimizer_rerun": False,
        "compiler_report_present": "model_fx_report.json" in collected,
        "ioinfo_present": "ioinfo.csv" in collected,
        "npu_only_verified": False, "bie_generated": False,
        "nef_compiled": False, "physical_kl720_inference": False,
    }
    v.write_new_json(root / CONTAINER_REPORT, receipt)
    print("SYNTHETIC_KL720_INNER_STATUS: " + state, flush=True)
    return 2 if state == "VENDOR_REJECTED" else 0


def on_windows(args):
    source_dir, manifest, source, meta = preflight_source(
        args.manifest_dir, args.case)
    repo = Path(__file__).resolve().parents[2]
    out = Path(args.out).expanduser().resolve()
    if out == repo or repo in out.parents:
        raise ValueError("Output must stay outside the Git repository")
    if out == source_dir or source_dir in out.parents or out in source_dir.parents:
        raise ValueError("Output must be a NEW sibling of the source manifest directory")
    if out.exists() or out.is_symlink():
        raise ValueError("Refusing to overwrite any existing vendor output folder")
    staging = out.with_name(out.name + ".BUILDING")
    if staging.exists() or staging.is_symlink():
        raise ValueError("Existing staging folder: " + str(staging))
    image_id = v.docker_image_id(args.image)
    runner = Path(__file__).resolve(strict=True)
    helper = runner.with_name(HELPER_NAME)
    if not helper.is_file():
        raise ValueError("Missing pinned vendor helper script")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    shutil.copyfile(str(source), str(staging / MODEL_NAME))
    shutil.copyfile(str(runner), str(staging / RUNNER_NAME))
    shutil.copyfile(str(helper), str(staging / HELPER_NAME))
    if v.sha256(staging / MODEL_NAME) != meta["sha256"]:
        raise ValueError("Synthetic trial copy mismatch")
    intake = {
        "schema": SCHEMA, "case": args.case,
        "generator_manifest_sha256": v.sha256(manifest),
        "candidate_sha256": meta["sha256"],
        "runner_sha256": v.sha256(staging / RUNNER_NAME),
        "helper_sha256": v.sha256(staging / HELPER_NAME),
        "docker_image_id": image_id,
        "model_id": args.model_id, "model_version": args.model_version,
        "optimizer_rerun": False, "bie_generated": False,
        "nef_compiled": False, "physical_kl720_inference": False,
    }
    v.write_new_json(staging / INTAKE_NAME, intake)
    staging.rename(out)

    command = [
        "docker", "run", "--rm", "--pull", "never", "--network", "none",
        "-v", str(out) + ":/data1", "-w", "/data1", image_id,
        v.TOOLCHAIN_PYTHON, "-u", "/data1/" + RUNNER_NAME,
        "--container", "--out", "/data1",
    ]
    print("SYNTHETIC KL720 compiler feasibility ONLY; no reoptimization", flush=True)
    print("Case: {} | saved to {}".format(args.case, out), flush=True)
    started = time.time()
    raw_path = out / "SYNTHETIC_VENDOR_DOCKER_CONSOLE.raw"
    txt_path = out / "SYNTHETIC_VENDOR_DOCKER_CONSOLE.txt"
    with raw_path.open("xb") as raw, txt_path.open("x", encoding="utf-8") as text:
        with subprocess.Popen(command, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, bufsize=0) as process:
            for line in process.stdout:
                raw.write(line)
                raw.flush()
                decoded = line.decode("utf-8", errors="backslashreplace")
                text.write(decoded)
                text.flush()
                print(decoded.encode("ascii", errors="backslashreplace").decode("ascii"),
                      end="", flush=True)
            exit_code = process.wait()
    markers = v.failure_markers(txt_path.read_text(encoding="utf-8"))
    inner_file = out / CONTAINER_REPORT
    inner = v.read_json(inner_file) if inner_file.is_file() else None
    status = (
        "VENDOR_REJECTED" if exit_code or markers or not inner
        or inner.get("status") != "VENDOR_RETURNED_REQUIRES_REPORT_REVIEW"
        else "VENDOR_RETURNED_REQUIRES_REPORT_REVIEW"
    )
    receipt = {
        "schema": SCHEMA, "case": args.case, "status": status,
        "candidate_sha256": meta["sha256"],
        "docker_image_id": image_id, "toolchain_exit_code": exit_code,
        "elapsed_seconds": time.time() - started,
        "vendor_failure_markers": markers,
        "vendor_report_exists": inner is not None,
        "compiler_backtrace_present": bool(
            inner and inner.get("compiler_backtrace_present")),
        "raw_console_sha256": v.sha256(raw_path),
        "npu_only_verified": False, "bie_generated": False,
        "nef_compiled": False, "physical_kl720_inference": False,
    }
    v.write_new_json(out / HOST_REPORT, receipt)
    print("SYNTHETIC_KL720_HOST_STATUS: " + status, flush=True)
    return 2 if status == "VENDOR_REJECTED" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", action="store_true")
    parser.add_argument("--manifest-dir", help="directory with synthetic manifest and ONNX")
    parser.add_argument("--case", choices=tuple(CASES), default="control_c1_h8")
    parser.add_argument("--out", required=True, help="new D: trial output directory")
    parser.add_argument("--image", default=v.IMAGE)
    parser.add_argument("--model-id", type=int, default=32785)
    parser.add_argument("--model-version", default="0001")
    args = parser.parse_args()
    try:
        if args.container:
            return in_container(args)
        if not args.manifest_dir:
            raise ValueError("Windows host requires --manifest-dir")
        return on_windows(args)
    except Exception as exc:
        print("SYNTHETIC_KL720_GATE_STOP: {}: {}".format(
            type(exc).__name__, exc), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
