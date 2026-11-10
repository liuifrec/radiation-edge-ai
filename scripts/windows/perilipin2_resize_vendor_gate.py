"""Isolated Perilipin2 KL720 Resize candidate vendor feasibility gate.

Windows host: use the existing FP32 resize proof, stage one accepted candidate
in a NEW D: work folder, run the pinned local Kneron image once, preserve
binary console output and compiler reports.

Docker: call ModelConfig.evaluate() on the *pre-optimized, parity-checked*
candidate directly. NEVER call onnx2onnx_flow/kneronnxopt again here.
Neither PTQ, BIE, NEF, nor a physical KL720 is performed.
Compatible with the historical vendor Python 3.7 environment.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROOF_SCHEMA = "perilipin2-m3c-resize-proof-v1"
GATE_SCHEMA = "perilipin2-m3c-resize-vendor-v1"
FROZEN_SOURCE_SHA = "8c256cd69249e01d6066f6dd3a377e1ef0c57e6c4c1c3799ef8aeeb0e204b797"
PINNED_IMAGE_ID = "sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041"
TOOLCHAIN_PYTHON = "/workspace/miniconda/bin/python"
IMAGE = "kneron/toolchain:latest"
INPUT_NAME = "rgb_chw_normalized"
OUTPUT_NAME = "fg_boundary_logits"
INPUT_SHAPE = [1, 3, 512, 512]
OUTPUT_SHAPE = [1, 2, 512, 512]
CANDIDATE = "resize_candidate.onnx"
RUNNER = "resize_vendor_runner.py"
INTAKE = "M3C_RESIZE_VENDOR_INTAKE.json"
REPORT = "M3C_RESIZE_VENDOR_REPORT.json"
HOST_REPORT = "M3C_RESIZE_VENDOR_EXECUTION.json"
ARTIFACT_NAMES = ("ioinfo.csv", "model_fx_report.json", "model_fx_report.html",
                  "batch_compile.log", "backtrace.log")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return result


def write_new_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def resize_graph_contract(model, variant, sizes):
    import onnx
    from onnx import numpy_helper
    initializer_names = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializer_names]
    outputs = list(model.graph.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Expected exactly one model input and output")
    for value, name, shape in ((inputs[0], INPUT_NAME, INPUT_SHAPE),
                               (outputs[0], OUTPUT_NAME, OUTPUT_SHAPE)):
        dims = [int(x.dim_value) for x in value.type.tensor_type.shape.dim]
        if value.name != name or dims != shape:
            raise ValueError("ONNX IO contract mismatch: " + str((value.name, dims)))
    resize = [node for node in model.graph.node if node.op_type == "Resize"]
    if len(resize) != 4 or len(sizes) != 4:
        raise ValueError("Four proven Resize nodes required")
    initializers = {obj.name: obj for obj in model.graph.initializer}
    for node, expected in zip(resize, sizes):
        attributes = {}
        for a in node.attribute:
            if a.type == onnx.AttributeProto.STRING:
                attributes[a.name] = a.s
        if (attributes.get("mode"), attributes.get("coordinate_transformation_mode"),
                attributes.get("nearest_mode")) != (b"nearest", b"asymmetric", b"floor"):
            raise ValueError("Resize semantics differ from the proof")
        if variant == "static-sizes":
            name = node.input[3] if len(node.input) > 3 else ""
            want = expected["target_shape"]
        else:
            name = node.input[2] if len(node.input) > 2 else ""
            want = expected["integer_scales"]
        if name not in initializers:
            raise ValueError("Resize axis parameter is not a frozen initializer")
        got = numpy_helper.to_array(initializers[name]).tolist()
        if got != want:
            raise ValueError("Resize values disagree with measured proof: {} vs {}".format(
                got, want))


def checked_proof(proof_dir, variant):
    import onnx
    proof_dir = Path(proof_dir).expanduser().resolve(strict=True)
    proof_path = proof_dir / "M3C_RESIZE_PROOF.json"
    proof = read_json(proof_path)
    if proof.get("schema") != PROOF_SCHEMA or proof.get("status") != "FP32_COMPATIBILITY_CANDIDATES":
        raise ValueError("Not a successfully completed Resize compatibility proof")
    if proof.get("original_sha256") != FROZEN_SOURCE_SHA:
        raise ValueError("Unexpected source model identity")
    if proof.get("optimized_vs_frozen_parity_pass") is not True:
        raise ValueError("Optimized model did not pass frozen-model FP32 parity")
    candidate = proof.get("variants", {}).get(variant)
    if not isinstance(candidate, dict) or candidate.get("parity_pass") is not True:
        raise ValueError("Selected Resize variant has no successful FP32 parity receipt")
    for row in candidate.get("parity_vs_optimized", []):
        if (row.get("max_abs_error", 1) > 1e-6 or
                row.get("mean_abs_error", 1) > 1e-7 or
                row.get("foreground_decision_mismatches", 1) != 0 or
                row.get("boundary_decision_mismatches", 1) != 0):
            raise ValueError("Individual parity measurements reject candidate")
    if len(candidate.get("parity_vs_optimized", [])) != 2:
        raise ValueError("Expected both independent synthetic FP32 parity probes")
    name = candidate.get("filename")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".onnx"):
        raise ValueError("Invalid candidate filename")
    path = proof_dir / name
    if sha256(path) != candidate.get("sha256"):
        raise ValueError("Candidate ONNX differs from frozen proof SHA256")
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    resize_graph_contract(model, variant, proof.get("resize_nodes", []))
    return proof_path, path, candidate


def failure_markers(text):
    lowered = text.lower()
    words = ("failure for model", "failed to compile", "hw not support",
             "hardware not support", "not supported", "unsupported")
    found = [term for term in words if term in lowered]
    for code in re.findall(r"\berr:\s*(\d+)\b", lowered):
        if int(code) and "err: {}".format(int(code)) not in found:
            found.append("err: {}".format(int(code)))
    return found


def newest_artifact(filename, started):
    found = []
    for directory in (Path("/workspace/.tmp"), Path("/data1/kneron_flow"),
                      Path("/tmp"), Path("/data1")):
        if not directory.is_dir():
            continue
        try:
            for path in directory.rglob(filename):
                if path.is_file() and path.stat().st_mtime >= started - 5:
                    found.append(path)
        except OSError:
            continue
    return max(found, key=lambda x: x.stat().st_mtime) if found else None


def in_container(args):
    import ktc
    import onnx
    root = Path(args.out).resolve(strict=True)
    intake = read_json(root / INTAKE)
    if intake.get("schema") != GATE_SCHEMA:
        raise ValueError("Not a Resize vendor trial")
    if sha256(root / CANDIDATE) != intake["candidate_sha256"]:
        raise ValueError("Candidate ONNX modified after intake")
    if sha256(root / RUNNER) != intake["runner_sha256"]:
        raise ValueError("Vendor runner modified after intake")
    model = onnx.load(str(root / CANDIDATE))
    onnx.checker.check_model(model)
    resize_graph_contract(model, intake["variant"], intake["resize_nodes"])
    print("Kneron KL720 Resize variant:", intake["variant"], flush=True)
    print("Candidate SHA256:", intake["candidate_sha256"], flush=True)
    print("Kneron ONNX optimizer: SKIPPED (already parity-proven)", flush=True)
    km = ktc.ModelConfig(intake["model_id"], intake["model_version"],
                         "720", onnx_model=model)
    started = time.time()
    exception = None
    evaluation = ""
    try:
        evaluation = str(km.evaluate())
    except Exception as exc:
        exception = "{}: {}".format(type(exc).__name__, exc)
        print("KNERON_VENDOR_EXCEPTION:", exception, flush=True)
    (root / "VENDOR_EVALUATION_TEXT.txt").write_text(evaluation + "\n", encoding="utf-8")
    print(evaluation, flush=True)
    artifacts = {}
    for filename in ARTIFACT_NAMES:
        found = newest_artifact(filename, started)
        destination = root / filename
        if found is not None:
            if found.resolve() != destination.resolve():
                if destination.exists():
                    raise ValueError("Refusing to overwrite report: " + filename)
                shutil.copyfile(str(found), str(destination))
            artifacts[filename] = sha256(destination)
    artifact_text = "\n".join(
        (root / name).read_text(encoding="utf-8", errors="replace")
        for name in artifacts if name.endswith((".log", ".csv", ".json"))
    )
    markers = failure_markers(evaluation + "\n" + artifact_text)
    status = "VENDOR_REJECTED" if markers or exception else "VENDOR_RETURNED_REQUIRES_REPORT_REVIEW"
    result = {
        "schema": GATE_SCHEMA, "status": status, "variant": intake["variant"],
        "original_sha256": FROZEN_SOURCE_SHA,
        "candidate_sha256": intake["candidate_sha256"],
        "optimizer_rerun": False, "vendor_exception": exception,
        "vendor_failure_markers": markers, "artifact_sha256": artifacts,
        "compiler_report_present": "model_fx_report.json" in artifacts,
        "ioinfo_present": "ioinfo.csv" in artifacts,
        "npu_only_verified": False, "bie_generated": False,
        "nef_compiled": False, "physical_kl720_inference": False,
    }
    write_new_json(root / REPORT, result)
    print("M3C_RESIZE_VENDOR_GATE:", status, flush=True)
    return 2 if status == "VENDOR_REJECTED" else 0


def docker_image_id(image):
    result = subprocess.run(["docker", "image", "inspect", image,
                             "--format", "{{.Id}}"], capture_output=True,
                            timeout=20, check=False)
    if result.returncode:
        raise ValueError("Docker image unavailable: " +
                         result.stderr.decode("utf-8", errors="backslashreplace"))
    identity = result.stdout.decode("ascii", errors="replace").strip()
    if identity != PINNED_IMAGE_ID:
        raise ValueError("Kneron toolchain image ID changed: " + identity)
    return identity


def on_windows(args):
    proof_dir = Path(args.proof_dir).expanduser().resolve(strict=True)
    proof_path, candidate_path, candidate = checked_proof(proof_dir, args.variant)
    out = Path(args.out).expanduser().resolve()
    if out.exists() or out.is_symlink() or out == proof_dir or proof_dir in out.parents:
        raise ValueError("Output must be a NEW directory outside proof directory")
    if out in proof_dir.parents:
        raise ValueError("Output cannot be an ancestor of the proof directory")
    staged = out.with_name(out.name + ".BUILDING")
    if staged.exists() or staged.is_symlink():
        raise ValueError("Existing staging folder; preserve and inspect")
    image_id = docker_image_id(args.image)
    runner = Path(__file__).resolve(strict=True)
    proof = read_json(proof_path)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.mkdir(exist_ok=False)
    shutil.copyfile(str(candidate_path), str(staged / CANDIDATE))
    shutil.copyfile(str(runner), str(staged / RUNNER))
    if sha256(staged / CANDIDATE) != candidate["sha256"]:
        raise ValueError("Candidate copy integrity failure")
    intake = {
        "schema": GATE_SCHEMA, "variant": args.variant,
        "proof_sha256": sha256(proof_path),
        "source_original_sha256": FROZEN_SOURCE_SHA,
        "source_optimized_sha256": proof["optimized_sha256"],
        "candidate_sha256": candidate["sha256"],
        "runner_sha256": sha256(staged / RUNNER),
        "resize_nodes": proof["resize_nodes"],
        "docker_image_id": image_id,
        "model_id": args.model_id, "model_version": args.model_version,
        "optimizer_rerun": False, "parity_proven": True,
        "bie_generated": False, "nef_compiled": False,
        "physical_kl720_inference": False,
    }
    write_new_json(staged / INTAKE, intake)
    staged.rename(out)
    cmd = ["docker", "run", "--rm", "--pull", "never", "--network", "none",
           "-v", str(out) + ":/data1", "-w", "/data1", image_id,
           TOOLCHAIN_PYTHON, "-u", "/data1/" + RUNNER, "--container",
           "--out", "/data1"]
    print("KL720 vendor feasibility only; re-optimization: NO", flush=True)
    print("Variant:", args.variant, "output:", out, flush=True)
    started = time.time()
    raw_path = out / "VENDOR_DOCKER_CONSOLE.raw"
    text_path = out / "VENDOR_DOCKER_CONSOLE.txt"
    with raw_path.open("xb") as raw_log, text_path.open("x", encoding="utf-8") as log:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, bufsize=0) as process:
            for line in process.stdout:
                raw_log.write(line)
                raw_log.flush()
                decoded = line.decode("utf-8", errors="backslashreplace")
                log.write(decoded)
                log.flush()
                print(decoded.encode("ascii", errors="backslashreplace").decode("ascii"),
                      end="", flush=True)
            code = process.wait()
    markers = failure_markers(text_path.read_text(encoding="utf-8"))
    inner_report = out / REPORT
    inner = read_json(inner_report) if inner_report.is_file() else None
    status = (
        "VENDOR_REJECTED" if code != 0 or markers or
        inner is None or inner["status"] == "VENDOR_REJECTED"
        else "VENDOR_RETURNED_REQUIRES_REPORT_REVIEW"
    )
    final = {
        "schema": GATE_SCHEMA, "status": status, "variant": args.variant,
        "docker_image_id": image_id, "toolchain_exit_code": code,
        "elapsed_seconds": time.time() - started,
        "candidate_sha256": candidate["sha256"],
        "raw_console_sha256": sha256(raw_path),
        "failure_markers": markers,
        "vendor_report_exists": inner is not None,
        "npu_only_verified": False, "bie_generated": False,
        "nef_compiled": False, "physical_kl720_inference": False,
    }
    write_new_json(out / HOST_REPORT, final)
    print("M3C_RESIZE_VENDOR_HOST_GATE:", status, flush=True)
    return 2 if status == "VENDOR_REJECTED" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", action="store_true",
                        help="internal Docker entry point")
    parser.add_argument("--proof-dir", help="existing static Resize proof folder")
    parser.add_argument("--variant", choices=("static-sizes", "integer-scales"),
                        default="static-sizes")
    parser.add_argument("--out", required=True, help="new D: output directory")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--model-id", type=int, default=32785)
    parser.add_argument("--model-version", default="0001")
    args = parser.parse_args()
    try:
        if args.container:
            return in_container(args)
        if not args.proof_dir:
            raise ValueError("--proof-dir required on the Windows host")
        return on_windows(args)
    except Exception as exc:
        print("M3C_RESIZE_VENDOR_GATE_STOP: {}: {}".format(
            type(exc).__name__, exc), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
