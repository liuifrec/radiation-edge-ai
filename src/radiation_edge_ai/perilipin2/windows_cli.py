"""Perilipin2 M3C Windows engineering gates; no data or model files in Git.

The same file is copied to an immutable work directory and used inside the
pinned Kneron Docker image. No PyTorch, CUDA, Cellpose, or microscopy data are
required on the Windows host for these *engineering* checks.

Stages: doctor -> import -> evaluate (Docker) -> parity (ONNX Runtime).
None creates BIE, NEF, or physically runs a KL720.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

INPUT_NAME = "rgb_chw_normalized"
OUTPUT_NAME = "fg_boundary_logits"
INPUT_SHAPE = [1, 3, 512, 512]
OUTPUT_SHAPE = [1, 2, 512, 512]
ONNX_NAME = "nuclear_foreground_boundary_512_opset11.onnx"
WORK_MODEL = "nuclear_candidate.onnx"
WORK_OPT = "nuclear_candidate.kneron.opt.onnx"
DEFAULT_IMAGE = "kneron/toolchain:latest"  # local-only, never auto-pull
SCHEMA = "perilipin2-m3c-windows-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Expected JSON object: {}".format(path))
    return result


def dump_json_new(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def tensor_info(value):
    dim = value.type.tensor_type.shape.dim
    sizes = []
    for entry in dim:
        if not entry.HasField("dim_value"):
            raise ValueError("Dynamic/unresolved ONNX dimension: {}".format(value.name))
        sizes.append(int(entry.dim_value))
    return value.name, sizes


def graph_contract(model):
    initializer_names = set(x.name for x in model.graph.initializer)
    inputs = [x for x in model.graph.input if x.name not in initializer_names]
    outputs = list(model.graph.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Expected exactly one ONNX input/output")
    a_name, a_shape = tensor_info(inputs[0])
    b_name, b_shape = tensor_info(outputs[0])
    return {"input_name": a_name, "input_shape": a_shape,
            "output_name": b_name, "output_shape": b_shape,
            "operators": dict(sorted(Counter(x.op_type for x in model.graph.node).items()))}


def validate_contract(contract):
    expected = {"input_name": INPUT_NAME, "input_shape": INPUT_SHAPE,
                "output_name": OUTPUT_NAME, "output_shape": OUTPUT_SHAPE}
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError("Model {}: expected {!r}, observed {!r}".format(key, value, contract.get(key)))


def validate_parity_receipt(parity, digest):
    expected = {"onnx_sha256": digest, "onnxruntime_parity_pass": True,
                "input_shape": INPUT_SHAPE, "output_shape": OUTPUT_SHAPE,
                "opset": 11}
    for key, value in expected.items():
        if parity.get(key) != value:
            raise ValueError("M3B parity receipt mismatch for {}".format(key))
    for key, limit in (("onnxruntime_parity_max_abs_error", 1e-3),
                       ("onnxruntime_parity_mean_abs_error", 1e-5)):
        value = parity.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Missing or invalid M3B numeric parity field: " + key)
        if not math.isfinite(value) or value > limit or value < 0:
            raise ValueError("M3B numerical parity exceeds gate: " + key)


def check_bundle(root, need_optimized=False):
    root = Path(root).expanduser().resolve(strict=True)
    receipt = read_json(root / "M3C_HOST_READY.json")
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "M3B_PARITY_VERIFIED":
        raise ValueError("Not a frozen Windows M3C intake")
    if receipt.get("model_sha256") != sha256(root / WORK_MODEL):
        raise ValueError("Original model changed since M3C intake")
    if receipt.get("runner_sha256") != sha256(root / "perilipin2_m3c_runner.py"):
        raise ValueError("Docker runner changed since intake")
    if receipt.get("parity_sha256") != sha256(root / "ONNX_PARITY.json"):
        raise ValueError("M3B parity receipt changed since intake")
    if need_optimized:
        ev = read_json(root / "M3C_KNERON_EVAL_RECEIPT.json")
        if ev.get("original_sha256") != receipt["model_sha256"]:
            raise ValueError("Evaluator used an unexpected original model")
        if ev.get("optimized_sha256") != sha256(root / WORK_OPT):
            raise ValueError("Optimized model changed after evaluator")
        return root, receipt, ev
    return root, receipt


def docker_image_identity(image):
    if not image or any(c.isspace() for c in image) or image.startswith("-"):
        raise ValueError("Invalid Docker image reference")
    result = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("Docker image not installed locally: {}. No pull performed. {}".format(
            image, result.stderr.strip()))
    return result.stdout.strip()


def doctor(args):
    libs = {name: importlib.util.find_spec(name) is not None for name in ("onnx", "onnxruntime", "numpy")}
    probe = {"schema": SCHEMA, "stage": "doctor", "python": sys.executable,
             "host_libraries": libs, "docker_cli": bool(shutil.which("docker")),
             "image_reference": args.image, "image_id": None, "docker_engine": False,
             "docker_python3": False, "physical_kl720_probe_performed": False}
    if probe["docker_cli"]:
        engine = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                                capture_output=True, text=True, check=False)
        probe["docker_engine"] = engine.returncode == 0
        if probe["docker_engine"]:
            try:
                probe["image_id"] = docker_image_identity(args.image)
                py = subprocess.run(
                    ["docker", "run", "--rm", "--pull", "never", "--entrypoint", "/bin/sh",
                     probe["image_id"], "-c", "command -v python3"],
                    capture_output=True, text=True, check=False,
                )
                probe["docker_python3"] = py.returncode == 0 and bool(py.stdout.strip())
            except ValueError:
                pass
    print(json.dumps(probe, indent=2, sort_keys=True))
    if not all(libs.values()):
        print("Install ONNX/ONNX Runtime in the separate Windows engineering environment.")
    if not probe["image_id"]:
        print("No accessible local Kneron toolchain image; check Docker Desktop and the installed tag.")
    elif not probe["docker_python3"]:
        print("Kneron toolchain image is present but /usr/bin/python3 was not found.")
    return 0


def intake(args):
    import onnx
    source = Path(args.onnx).expanduser().resolve(strict=True)
    receipt_path = Path(args.parity).expanduser().resolve(strict=True)
    out = Path(args.out).expanduser().resolve()
    if out.exists() or out.is_symlink() or out == source.parent or out == receipt_path.parent:
        raise ValueError("Refusing an existing or input output path: {}".format(out))
    if source.suffix.lower() != ".onnx":
        raise ValueError("Expected an ONNX input")
    model_hash = sha256(source)
    parity = read_json(receipt_path)
    validate_parity_receipt(parity, model_hash)
    model = onnx.load(str(source))
    onnx.checker.check_model(model)
    if len(model.opset_import) != 1 or model.opset_import[0].version != 11:
        raise ValueError("M3B candidate must have exactly ONNX opset 11")
    contract = graph_contract(model)
    validate_contract(contract)
    stage = out.with_name(out.name + ".BUILDING")
    if stage.exists() or stage.is_symlink():
        raise ValueError("Existing staging directory: {}".format(stage))
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=False, exist_ok=False)
    runner = Path(__file__).resolve(strict=True)
    try:
        shutil.copyfile(source, stage / WORK_MODEL)
        shutil.copyfile(receipt_path, stage / "ONNX_PARITY.json")
        shutil.copyfile(runner, stage / "perilipin2_m3c_runner.py")
        if sha256(stage / WORK_MODEL) != model_hash:
            raise ValueError("Copy-to-Windows staging changed ONNX bytes")
        data = {"schema": SCHEMA, "status": "M3B_PARITY_VERIFIED",
                "model_sha256": model_hash, "parity_sha256": sha256(stage / "ONNX_PARITY.json"),
                "runner_sha256": sha256(stage / "perilipin2_m3c_runner.py"),
                "contract": contract, "onnx_ir_version": model.ir_version,
                "source_path": str(source), "training_reviewed_calibration": "EXTERNAL_NOT_VERIFIED_HERE",
                "heldout_instance_validation": False, "biological_accuracy_validated": False,
                "bie_generated": False, "nef_compiled": False, "physical_kl720_inference": False}
        dump_json_new(stage / "M3C_HOST_READY.json", data)
        stage.rename(out)
    except Exception:
        (stage / "IMPORT_FAILED.txt").write_text("Unsafe or partial staging; investigate before reuse.\n")
        raise
    print("M3C_WINDOWS_INTAKE_PASS")
    print("Work folder:", out)
    print("ONNX SHA256:", model_hash)
    return 0


def artifact_find(filename, started):
    # Kneron's internal compiler logs may land outside the mounted work folder.
    candidates = []
    for root in (Path("/workspace/.tmp"), Path("/data1/kneron_flow"), Path("/tmp")):
        if not root.is_dir():
            continue
        try:
            candidates.extend(x for x in root.rglob(filename) if x.is_file())
        except OSError:
            pass
    fresh = [x for x in candidates if x.stat().st_mtime >= started - 5]
    pool = fresh or candidates
    return max(pool, key=lambda x: x.stat().st_mtime) if pool else None


def inspect_cpu_partitions(path):
    rows = []
    if path is None or not Path(path).is_file():
        return rows
    for row in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [x.strip().lower() for x in row.split(",")]
        if parts and parts[0] == "c":
            rows.append(row)
    return rows


def docker_evaluate(args):
    import onnx
    import kneronnxopt
    import ktc
    root, receipt = check_bundle(args.work)
    model_file = root / WORK_MODEL
    model = onnx.load(str(model_file))
    onnx.checker.check_model(model)
    validate_contract(graph_contract(model))
    opt = root / WORK_OPT
    ev_file = root / "M3C_KNERON_EVAL_RECEIPT.json"
    if opt.exists() or ev_file.exists():
        raise ValueError("Refusing to overwrite an earlier Kneron evaluation")
    print("Kneron optimizer starting for KL720", flush=True)
    optimized = kneronnxopt.optimize(model)
    onnx.checker.check_model(optimized)
    opt_contract = graph_contract(optimized)
    validate_contract(opt_contract)
    onnx.save(optimized, str(opt))
    print("Kneron evaluate() starting", flush=True)
    km = ktc.ModelConfig(args.model_id, args.model_version, "720", onnx_model=optimized)
    started = time.time()
    evaluation = km.evaluate()
    text = str(evaluation)
    dump = root / "KNERON_EVALUATOR_OUTPUT.txt"
    dump.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    collected = {}
    for filename in ("ioinfo.csv", "model_fx_report.json", "model_fx_report.html",
                     "batch_compile.log", "backtrace.log"):
        found = artifact_find(filename, started)
        if found is not None:
            target = root / filename
            if target.exists() and found.resolve() != target.resolve():
                raise ValueError("Refusing to overwrite existing toolchain artifact {}".format(target))
            if found.resolve() != target.resolve():
                shutil.copyfile(found, target)
            collected[filename] = sha256(target)
    ioinfo = root / "ioinfo.csv"
    cpu = inspect_cpu_partitions(ioinfo if ioinfo.exists() else None)
    status = {"schema": SCHEMA, "status": "EVALUATOR_COMPLETED_REQUIRES_REPORT_REVIEW",
              "original_sha256": receipt["model_sha256"], "optimized_sha256": sha256(opt),
              "runner_sha256": receipt["runner_sha256"], "model_id": args.model_id,
              "model_version": args.model_version, "platform": "720",
              "before": graph_contract(model), "after": opt_contract,
              "artifact_sha256": collected, "n_cpu_partition_rows": len(cpu),
              "cpu_partition_rows": cpu, "ioinfo_found": ioinfo.is_file(),
              "npu_only_verified": False,  # needs independent review of full compiler output
              "float_parity_checked": False, "bie_generated": False,
              "nef_compiled": False, "physical_kl720_inference": False}
    dump_json_new(ev_file, status)
    print("M3C_KNERON_EVALUATOR_COMPLETED_REQUIRES_REPORT_REVIEW", flush=True)
    print("CPU partition rows:", len(cpu), "IO info found:", ioinfo.is_file(), flush=True)
    return 0


def evaluate(args):
    root, receipt = check_bundle(args.work)
    if (root / "M3C_KNERON_EVAL_RECEIPT.json").exists() or (root / WORK_OPT).exists():
        raise ValueError("Refusing to reuse an evaluation work directory")
    image_id = docker_image_identity(args.image)
    if (root / "WINDOWS_DOCKER_EXECUTION.json").exists():
        raise ValueError("Existing Windows execution receipt; use a fresh work directory")
    cmd = ["docker", "run", "--rm", "--pull", "never", "-v", str(root) + ":/data1",
           "-w", "/data1", image_id, "python3", "/data1/perilipin2_m3c_runner.py",
           "docker-evaluate", "--work", "/data1", "--model-id", str(args.model_id),
           "--model-version", args.model_version]
    start = time.time()
    run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (root / "WINDOWS_DOCKER_CONSOLE.txt").write_text(run.stdout or "", encoding="utf-8")
    if run.stdout:
        print(run.stdout)
    status = {"schema": SCHEMA, "stage": "evaluate", "image_reference": args.image,
              "docker_image_id": image_id, "model_sha256": receipt["model_sha256"],
              "toolchain_run_exit_code": run.returncode, "elapsed_seconds": time.time() - start,
              "evaluation_receipt_exists": (root / "M3C_KNERON_EVAL_RECEIPT.json").is_file(),
              "bie_generated": False, "nef_compiled": False, "physical_kl720_inference": False}
    dump_json_new(root / "WINDOWS_DOCKER_EXECUTION.json", status)
    if run.returncode != 0 or not status["evaluation_receipt_exists"]:
        raise ValueError("Kneron evaluate() did not complete; see WINDOWS_DOCKER_CONSOLE.txt")
    check_bundle(root, need_optimized=True)
    print("M3C_WINDOWS_DOCKER_EVALUATE_COMPLETE_NOT_QUANTIZED")
    return 0


def parity(args):
    import numpy as np
    import onnxruntime as ort
    root, _, ev = check_bundle(args.work, need_optimized=True)
    if (root / "M3C_OPTIMIZED_FP32_PARITY.json").exists():
        raise ValueError("Existing FP32 parity result: use fresh work directory")
    rng = np.random.default_rng(20261030)
    x = rng.uniform(-0.5, 0.5, size=INPUT_SHAPE).astype(np.float32)
    out = []
    for name in (WORK_MODEL, WORK_OPT):
        sess = ort.InferenceSession(str(root / name), providers=["CPUExecutionProvider"])
        a, b = sess.get_inputs(), sess.get_outputs()
        if len(a) != 1 or len(b) != 1 or a[0].shape != INPUT_SHAPE or b[0].shape != OUTPUT_SHAPE:
            raise ValueError("ONNX Runtime tensor contract mismatch for " + name)
        value = sess.run([b[0].name], {a[0].name: x})[0]
        if value.shape != tuple(OUTPUT_SHAPE) or not np.isfinite(value).all():
            raise ValueError("Nonfinite or invalid ONNX output: " + name)
        out.append(value.astype(np.float64))
    delta = np.abs(out[0] - out[1])
    boundary_logit = float(np.log(.45 / .55))
    data = {"schema": SCHEMA, "stage": "optimized_float_parity",
            "original_sha256": ev["original_sha256"], "optimized_sha256": ev["optimized_sha256"],
            "seed": 20261030, "max_abs_error": float(delta.max()),
            "mean_abs_error": float(delta.mean()),
            "foreground_decision_mismatches": int(np.count_nonzero((out[0][:, 0] >= 0) != (out[1][:, 0] >= 0))),
            "boundary_decision_mismatches": int(np.count_nonzero((out[0][:, 1] >= boundary_logit) != (out[1][:, 1] >= boundary_logit))),
            "numeric_pass": bool(float(delta.max()) <= 1e-3 and float(delta.mean()) <= 1e-5),
            "test_input": "deterministic pseudo-random 512x512 tensor, not real microscopy",
            "biological_accuracy_validated": False, "npu_only_verified": False,
            "bie_generated": False, "nef_compiled": False, "physical_kl720_inference": False}
    dump_json_new(root / "M3C_OPTIMIZED_FP32_PARITY.json", data)
    print("M3C_ORIGINAL_OPTIMIZED_FP32_PARITY_" + ("PASS" if data["numeric_pass"] else "FAIL"))
    print("Max absolute difference:", data["max_abs_error"])
    print("Mean absolute difference:", data["mean_abs_error"])
    print("Foreground decision disagreements:", data["foreground_decision_mismatches"])
    print("Boundary decision disagreements:", data["boundary_decision_mismatches"])
    return 0 if data["numeric_pass"] else 2


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    d = sub.add_parser("doctor", help="check the Windows engineering environment (no model or inference)")
    d.add_argument("--image", default=DEFAULT_IMAGE)
    i = sub.add_parser("import", help="freeze parity-verified Ubuntu ONNX into a new Windows work directory")
    i.add_argument("--onnx", required=True)
    i.add_argument("--parity", required=True)
    i.add_argument("--out", required=True)
    e = sub.add_parser("evaluate", help="run pinned local Kneron Docker optimizer/evaluator")
    e.add_argument("--work", required=True)
    e.add_argument("--image", default=DEFAULT_IMAGE)
    e.add_argument("--model-id", type=int, default=32785)
    e.add_argument("--model-version", default="0001")
    p = sub.add_parser("parity", help="check optimized ONNX numerical parity on Windows CPU")
    p.add_argument("--work", required=True)
    k = sub.add_parser("docker-evaluate", help="internal entrypoint: run only inside Kneron toolchain Docker")
    k.add_argument("--work", default="/data1")
    k.add_argument("--model-id", type=int, default=32785)
    k.add_argument("--model-version", default="0001")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    funcs = {"doctor": doctor, "import": intake, "evaluate": evaluate,
             "parity": parity, "docker-evaluate": docker_evaluate}
    try:
        return funcs[args.stage](args)
    except (ValueError, OSError, KeyError, ImportError, subprocess.SubprocessError) as exc:
        print("M3C_WINDOWS_GATE_STOP: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())