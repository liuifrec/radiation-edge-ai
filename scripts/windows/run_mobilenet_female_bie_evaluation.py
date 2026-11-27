"""Bind the MobileNetV2 FP32 replay, run pinned KL720 INT8 inference and score all eight gates."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("RADEDGE_DATA_ROOT", "D:/radiation-edge-ai-data"))
POLICY = REPO / "docs/NASA_MOBILENET_V2_FEMALE_INT8_V1.json"
MODULES = REPO / "src/radiation_edge_ai/nasa_bps"
sys.path.insert(0, str(REPO / "src"))
from radiation_edge_ai.nasa_bps.bind_mobilenet_bie_evaluation import bind, staged_path
from radiation_edge_ai.nasa_bps.multimodel_ptq import sha256 as digest, IMAGE_ID as IMAGE
from radiation_edge_ai.nasa_bps.multimodel_qualification import POLICY_SHA


def utc():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def mount(source, target, readonly=True):
    if "," in str(source):
        raise ValueError(f"Unrepresentable Docker bind path: {source}")
    return ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if readonly else "")]


def launch(prepared, jobs, document, data_root):
    prepared = prepared.resolve(strict=True)
    scratch = (data_root / "scratch").resolve(strict=True)
    if scratch not in prepared.parents:
        raise ValueError("Use a completed preparation in the external scratch directory")
    policy = load(POLICY)
    git = ["git", "-c", "safe.directory=" + REPO.as_posix(), "-C", str(REPO)]
    relative = POLICY.relative_to(REPO).as_posix()
    freeze = subprocess.check_output(git + ["log", "-1", "--format=%H", "--", relative],
                                     text=True).strip()
    if not freeze:
        raise ValueError("Acceptance policy must be committed before evaluation")
    if hashlib.sha256(subprocess.check_output(git + ["show", freeze + ":" + relative])).hexdigest() != POLICY_SHA:
        raise ValueError("Committed acceptance policy differs from the frozen identity")
    prep = bind(prepared, scratch, POLICY, document)
    preparation = load(prep)
    artifacts = preparation["artifacts"]
    bie = Path(artifacts["bie"]["path"])
    def archived(key):
        return staged_path(key, artifacts[key]["path"], scratch.parent)
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker CLI is unavailable")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = scratch / f"nasa_mobilenet_female_bie_readout_{stamp}_{uuid4().hex[:8]}"
    out.mkdir()
    print(f"Readout directory: {out}", flush=True)
    record = dict(status="STARTED", created_utc=utc(), policy_commit=freeze, policy_sha256=POLICY_SHA,
                  acceptance_document_sha256=digest(document), prepared=str(prepared),
                  preparation_sha256=digest(prep), bie_sha256=digest(bie), toolchain_image_id=IMAGE,
                  fp32_preparation_sha256=policy["fp32_preparation_sha256"],
                  simulator_jobs=jobs, chunk_size=24, omp_num_threads=1, openblas_num_threads=1,
                  stages=[], artifact_sha256={}, host_script_sha256=digest(Path(__file__)),
                  nef_compiled=False, physical_execution=False)
    receipt = out / "evaluation_receipt.json"

    def save():
        partial = receipt.with_suffix(".partial")
        partial.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        partial.replace(receipt)

    def run(name, argv):
        argv = list(map(str, argv))
        log = out / (name + ".log")
        stage = dict(name=name, arguments=argv, log=log.name, started_utc=utc())
        record["stages"].append(stage)
        save()
        print(f"Starting {name}", flush=True)
        start = time.monotonic()
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1")
        with log.open("xb") as stream:
            process = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=str(REPO))
        stage.update(exit_code=process.returncode, completed_utc=utc(), elapsed_seconds=time.monotonic()-start,
                     log_sha256=digest(log))
        save()
        return process.returncode

    save()
    try:
        if run("engine", [docker, "info", "--format", "{{.NCPU}} {{.MemTotal}} {{.ServerVersion}}"]):
            raise RuntimeError("Docker engine information unavailable")
        if run("image", [docker, "image", "inspect", IMAGE, "--format", "{{.Id}}"]) or (
                (out / "image.log").read_text().strip() != IMAGE):
            raise ValueError("Pinned Docker image unavailable")
        snapshot = out / "worker"
        snapshot.mkdir()
        record["worker_files_sha256"] = {}
        for name in ("multimodel_bie_inference.py", "multimodel_ptq.py",
                     "multimodel_qualification.py", "deployment_cohort.py"):
            shutil.copyfile(MODULES / name, snapshot / name)
            record["worker_files_sha256"][name] = digest(snapshot / name)
        record["worker_sha256"] = record["worker_files_sha256"]["multimodel_bie_inference.py"]
        sdk_python = "/workspace/miniconda/bin/python"
        options = [docker, "run", "--rm", "--pull", "never", "--network", "none",
                   "--env", "OMP_NUM_THREADS=1", "--env", "OPENBLAS_NUM_THREADS=1"]
        if run("sdk", options + ["--entrypoint", sdk_python, IMAGE, "-B", "-c",
                "import ktc,inspect,sys; from sys_flow.inference import inference_dynasty_fx; "
                "print(sys.version); print(ktc.__file__); print(inspect.signature(ktc.kneron_inference)); "
                "print(inspect.getsource(ktc.kneron_inference)); print(inspect.getsource(inference_dynasty_fx))"]):
            raise RuntimeError("Kneron SDK inspection failed")
        cmd = options + ["--shm-size", "1g"] + mount(snapshot, "/worker") + mount(prepared, "/inputs")
        cmd += mount(bie, "/model/quantized_model.bie") + mount(POLICY, "/policy.json")
        cmd += mount(Path(artifacts["training_pilot"]["path"]).parent, "/training-pilot")
        cmd += mount(archived("calibration_manifest").parent, "/calibration")
        cmd += mount(out, "/out", readonly=False)
        cmd += ["--workdir", "/tmp", "--entrypoint", sdk_python, IMAGE, "-B", "-u",
                "/worker/multimodel_bie_inference.py", "--inputs", "/inputs", "--bie",
                "/model/quantized_model.bie", "--out", "/out/vendor", "--policy-sha256",
                POLICY_SHA, "--preparation-sha256", record["preparation_sha256"]]
        cmd += ["--jobs", str(jobs), "--policy", "/policy.json",
                "--training-pilot", "/training-pilot", "--calibration", "/calibration"]
        if run("simulator", cmd):
            raise RuntimeError("BIE simulator did not complete; see simulator.log and vendor receipt")
        vendor = out / "vendor"
        code = run("score", [sys.executable, "-B", "-m", "radiation_edge_ai.nasa_bps.multimodel_bie_comparison",
                    "--policy", POLICY, "--policy-sha256", POLICY_SHA, "--acceptance-document", document,
                    "--development-manifest", archived("manifest"), "--nasa-reference", archived("reference"),
                    "--fp32-nuclei", archived("fp32_nuclei"), "--fp32-bags", archived("fp32_bags"),
                    "--bie-receipt", vendor / "bie_receipt.json",
                    "--bie-nuclei", vendor / "bie_nuclei.csv", "--prepared-inputs", prepared / "inputs.csv",
                    "--preparation-receipt", prep, "--out", out / "comparison"])
        result = load(out / "comparison/comparison.json")
        if result.get("status") not in ("COMPARISON_COMPLETE", "INVALID_PANEL") or set(result["decisions"]) != set(policy["maximums"]):
            raise ValueError("Incomplete scientific comparison")
        passed = result["all_declared_limits_pass"]
        if code != (0 if passed else 1) or result["eligible_for_nef_compilation"] != passed:
            raise ValueError("Scorer exit and decision disagree")
        if digest(prep) != record["preparation_sha256"] or digest(bie) != record["bie_sha256"]:
            raise ValueError("Preparation or BIE changed during evaluation")
        if any(digest(snapshot / k) != v for k, v in record["worker_files_sha256"].items()):
            raise ValueError("Simulator code changed during evaluation")
        for path in [vendor / "bie_receipt.json", vendor / "bie_nuclei.csv", *(out / "comparison").iterdir()]:
            if path.is_file():
                record["artifact_sha256"][path.relative_to(out).as_posix()] = digest(path)
        record.update(status="ACCEPTANCE_PASS" if passed else "ACCEPTANCE_FAIL",
                      comparison_status=result["status"], comparison_error=result.get("error"),
                      eligible_for_nef_compilation=passed, decisions=result["decisions"])
        return 0 if passed else 1
    except Exception as exc:
        record.update(status="EXECUTION_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["completed_utc"] = utc()
        save()
        print(f"{record['status']}: {receipt}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3, choices=range(1, 7))
    parser.add_argument("--acceptance-document", type=Path, default=POLICY.with_suffix(".md"))
    parser.add_argument("--data-root", type=Path, default=DATA)
    args = parser.parse_args()
    sys.exit(launch(args.prepared, args.jobs, args.acceptance_document, args.data_root))
