"""Vendor-test ONE parity-gated Perilipin2 decoder prefix with pinned KL720 toolchain.

Model files contain real learned weights: keep under private D: scratch only.
No re-optimization, PTQ, BIE, NEF, USB device inference or measured throughput.
Docker code is Python 3.7-compatible; host must have the existing ONNX tools.
"""
from __future__ import print_function

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import perilipin2_resize_vendor_gate as v

SCHEMA = "perilipin2-kl720-stagewise-vendor-v1"
PROOF_SCHEMA = "perilipin2-kl720-stagewise-proof-v1"
STAGES = ("encoder_before_first_upsample", "upsample_1", "upsample_2",
          "upsample_3", "upsample_4")
CANDIDATE = "compiler_candidate.onnx"
RUNNER = "stagewise_vendor_runner.py"
HELPER = "perilipin2_resize_vendor_gate.py"
INTAKE = "STAGEWISE_VENDOR_INTAKE.json"
INNER = "STAGEWISE_VENDOR_REPORT.json"
HOST = "STAGEWISE_VENDOR_EXECUTION.json"
ARTIFACTS = ("ioinfo.csv", "model_fx_report.json", "model_fx_report.html",
             "batch_compile.log", "backtrace.log")


def checked_stage(proof, name):
    if (proof.get("schema") != PROOF_SCHEMA or
            proof.get("status") != "STAGED_FP32_PARITY_EXACT" or
            proof.get("source_frozen_sha256") != v.FROZEN_SOURCE_SHA or
            proof.get("nef_compiled") is not False or
            proof.get("physical_kl720_inference") is not False):
        raise ValueError("Not a valid source-frozen stagewise proof")
    rows = proof.get("stages", [])
    if len(rows) != 5 or tuple(r.get("stage") for r in rows) != STAGES:
        raise ValueError("Five stagewise cut points are required")
    row = rows[STAGES.index(name)]
    if (row.get("filename") != name + ".onnx" or
            row.get("convtranspose_count") != STAGES.index(name) or
            not isinstance(row.get("node_count"), int)):
        raise ValueError("Unexpected stage node-count contract")
    h = row.get("sha256", "")
    if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
        raise ValueError("Invalid stage SHA256")
    if len(row.get("parity", [])) != 2 or any(
            p.get("exact_values") is not True or p.get("max_abs_error") != 0.0
            for p in row["parity"]):
        raise ValueError("Missing exact-parity evidence")
    return row


def graph_contract(onnx, graph_model, row):
    onnx.checker.check_model(graph_model)
    g = graph_model.graph
    initializers = {x.name for x in g.initializer}
    inputs = [x for x in g.input if x.name not in initializers]
    outputs = list(g.output)
    if (len(inputs) != 1 or len(outputs) != 1 or
            len(g.node) != row["node_count"] or
            len(g.initializer) != row["initializers"]):
        raise ValueError("Probe node or IO counts changed")
    actual_input = inputs[0]
    actual_output = outputs[0]
    ins = [int(x.dim_value) for x in actual_input.type.tensor_type.shape.dim]
    outs = [int(x.dim_value) for x in actual_output.type.tensor_type.shape.dim]
    if actual_input.name != v.INPUT_NAME or ins != v.INPUT_SHAPE:
        raise ValueError("Input is not the frozen original RGB tensor")
    if actual_output.name != row["tensor"] or outs != row["shape"]:
        raise ValueError("Output differs from the original decoder cut point")
    if sum(x.op_type == "ConvTranspose" for x in g.node) != row["convtranspose_count"]:
        raise ValueError("Unexpected upsampling operators")


def in_container(args):
    import onnx
    import ktc
    root = Path(args.out).resolve(strict=True)
    intake = v.read_json(root / INTAKE)
    if intake.get("schema") != SCHEMA or intake.get("docker_image_id") != v.PINNED_IMAGE_ID:
        raise ValueError("Invalid pinned vendor intake")
    for file, key in ((CANDIDATE, "candidate_sha256"),
                      (RUNNER, "runner_sha256"), (HELPER, "helper_sha256")):
        if v.sha256(root / file) != intake[key]:
            raise ValueError("Staged artifact hash mismatch: " + file)
    model = onnx.load(str(root / CANDIDATE))
    graph_contract(onnx, model, intake["stage_record"])
    print("STAGEWISE_KL720_COMPILER:", intake["stage"], flush=True)
    print("Kneron optimizer rerun: NO; device inference: NO", flush=True)
    configured = ktc.ModelConfig(intake["model_id"], intake["model_version"],
                                 "720", onnx_model=model)
    started = time.time()
    exception, returned = None, ""
    try:
        returned = str(configured.evaluate())
    except Exception as error:
        exception = "{}: {}".format(type(error).__name__, error)
        print("VENDOR_EXCEPTION:", exception, flush=True)
    (root / "VENDOR_EVALUATION.txt").write_text(returned + "\n", encoding="utf-8")
    print(returned, flush=True)
    artifacts = {}
    for filename in ARTIFACTS:
        found = v.newest_artifact(filename, started)
        if found is not None:
            destination = root / filename
            if found.resolve() != destination.resolve():
                if destination.exists():
                    raise ValueError("Refusing overwritten vendor artifact")
                shutil.copyfile(str(found), str(destination))
            artifacts[filename] = v.sha256(destination)
    evidence = "\n".join(
        (root / filename).read_text(encoding="utf-8", errors="replace")
        for filename in artifacts if filename.endswith((".csv", ".log", ".json")))
    markers = v.failure_markers(returned + "\n" + evidence)
    if "backtrace.log" in artifacts:
        markers.append("compiler_backtrace")
    status = ("VENDOR_REJECTED" if markers or exception
              else "VENDOR_RETURNED_REQUIRES_REPORT_REVIEW")
    v.write_new_json(root / INNER, {
        "schema": SCHEMA, "stage": intake["stage"], "status": status,
        "candidate_sha256": intake["candidate_sha256"],
        "vendor_exception": exception, "failure_markers": markers,
        "artifacts": artifacts, "compiler_backtrace": "backtrace.log" in artifacts,
        "nef_compiled": False, "bie_generated": False,
        "physical_kl720_inference": False, "npu_only_verified": False})
    print("STAGEWISE_VENDOR_INNER_STATUS:", status, flush=True)
    return 2 if status == "VENDOR_REJECTED" else 0


def on_windows(args):
    import onnx
    root = Path(args.proof_dir).expanduser().resolve(strict=True)
    proof = v.read_json(root / "STAGEWISE_PROOF.json")
    row = checked_stage(proof, args.stage)
    candidate = root / row["filename"]
    if v.sha256(candidate) != row["sha256"]:
        raise ValueError("Candidate source does not match its parity proof")
    graph_contract(onnx, onnx.load(str(candidate)), row)
    output = Path(args.out).expanduser().resolve()
    if (output.exists() or output.is_symlink() or output == root or
            root in output.parents or output in root.parents):
        raise ValueError("Vendor output must be NEW and separate")
    staging = output.with_name(output.name + ".BUILDING")
    if staging.exists() or staging.is_symlink():
        raise ValueError("Preserve existing staging folder")
    image_id = v.docker_image_id(args.image)
    source_runner = Path(__file__).resolve(strict=True)
    source_helper = source_runner.with_name(HELPER)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(exist_ok=False)
    shutil.copyfile(str(candidate), str(staging / CANDIDATE))
    shutil.copyfile(str(source_runner), str(staging / RUNNER))
    shutil.copyfile(str(source_helper), str(staging / HELPER))
    intake = {
        "schema": SCHEMA, "stage": args.stage, "stage_record": row,
        "reference_proof_sha256": v.sha256(root / "STAGEWISE_PROOF.json"),
        "candidate_sha256": v.sha256(staging / CANDIDATE),
        "runner_sha256": v.sha256(staging / RUNNER),
        "helper_sha256": v.sha256(staging / HELPER),
        "docker_image_id": image_id, "model_id": args.model_id,
        "model_version": args.model_version, "optimizer_rerun": False,
        "nef_compiled": False, "physical_kl720_inference": False}
    v.write_new_json(staging / INTAKE, intake)
    staging.rename(output)
    command = ["docker", "run", "--rm", "--pull", "never", "--network", "none",
               "-v", str(output) + ":/data1", "-w", "/data1", image_id,
               v.TOOLCHAIN_PYTHON, "-u", "/data1/" + RUNNER,
               "--container", "--out", "/data1"]
    print("Vendor compiler stage:", args.stage, flush=True)
    started = time.time()
    raw = output / "VENDOR_DOCKER_CONSOLE.raw"
    log = output / "VENDOR_DOCKER_CONSOLE.txt"
    with raw.open("xb") as binary, log.open("x", encoding="utf-8") as human:
        with subprocess.Popen(command, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, shell=False, bufsize=0) as process:
            for line in process.stdout:
                binary.write(line)
                binary.flush()
                text = line.decode("utf-8", errors="backslashreplace")
                human.write(text)
                human.flush()
                print(text.encode("ascii", errors="backslashreplace").decode("ascii"),
                      end="", flush=True)
            code = process.wait()
    markers = v.failure_markers(log.read_text(encoding="utf-8"))
    inner_path = output / INNER
    inner = v.read_json(inner_path) if inner_path.is_file() else None
    if inner:
        markers.extend(inner.get("failure_markers", []))
    status = ("VENDOR_REJECTED" if code != 0 or markers or inner is None or
              inner.get("status") == "VENDOR_REJECTED" else
              "VENDOR_RETURNED_REQUIRES_REPORT_REVIEW")
    v.write_new_json(output / HOST, {
        "schema": SCHEMA, "stage": args.stage, "status": status,
        "candidate_sha256": intake["candidate_sha256"],
        "toolchain_exit_code": code, "docker_image_id": image_id,
        "elapsed_seconds": time.time() - started,
        "raw_console_sha256": v.sha256(raw), "failure_markers": markers,
        "compiler_backtrace": bool(inner and inner.get("compiler_backtrace")),
        "nef_compiled": False, "bie_generated": False,
        "physical_kl720_inference": False, "npu_only_verified": False})
    print("STAGEWISE_VENDOR_HOST_STATUS:", status, flush=True)
    print("Report:", output / HOST)
    return 2 if status == "VENDOR_REJECTED" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", action="store_true")
    parser.add_argument("--proof-dir", help="frozen stagewise proof directory")
    parser.add_argument("--stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--out", required=True, help="NEW D: output directory")
    parser.add_argument("--image", default=v.IMAGE)
    parser.add_argument("--model-id", type=int, default=32785)
    parser.add_argument("--model-version", default="0001")
    args = parser.parse_args()
    try:
        if args.container:
            return in_container(args)
        if not args.proof_dir:
            raise ValueError("--proof-dir required on Windows")
        return on_windows(args)
    except Exception as error:
        print("STAGEWISE_VENDOR_STOP: {}: {}".format(type(error).__name__, error),
              file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
