"""Authenticate and run the frozen ResNet18 female-fold full-INT16 development BIE."""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
MODULE = "radiation_edge_ai.nasa_bps.neural16_native"
HELPERS = (
    "__init__", "multimodel_bie_inference", "multimodel_ptq", "multimodel_qualification",
    "deployment_cohort", "training_precision", "training_precision_readout",
    "neural16_regression", "neural16_native",
)


def launch(data, out, jobs, resume):
    sys.path.insert(0, str(SRC))
    from radiation_edge_ai.nasa_bps import multimodel_ptq as ptq
    from radiation_edge_ai.nasa_bps import neural16_regression as q
    from radiation_edge_ai.nasa_bps.multimodel_bie_inference import utc

    data = data.resolve(strict=True)
    scratch = (data / "scratch").resolve(strict=True)
    out = out.resolve()
    ptq.require(out.parent == scratch, "Use a dedicated directory directly under external scratch")
    out.mkdir(exist_ok=True)
    docs = REPO / "docs"
    frozen_bytes = subprocess.check_output(["git", "-c", "safe.directory=" + REPO.as_posix(),
                                           "-C", str(REPO), "show", q.FREEZE + ":docs/" + q.POLICY_NAME])
    ptq.require(hashlib.sha256(frozen_bytes).hexdigest() == q.POLICY_SHA, "Frozen Git policy mismatch")
    binding = q.authenticate(docs, data)
    previous = out / "preparation.json"
    snapshot = out / "regression_source"
    worker_dir = snapshot / "radiation_edge_ai/nasa_bps"
    code = [Path("radiation_edge_ai/__init__.py")] + [
        Path("radiation_edge_ai/nasa_bps") / (name + ".py") for name in HELPERS]
    if resume:
        ptq.require(previous.is_file() and ptq.load_json(previous) == binding, "Readout binding changed")
        for relative in code:
            ptq.require(ptq.sha256(SRC / relative) == ptq.sha256(snapshot / relative),
                        "Frozen worker changed: " + str(relative))
        for path in (docs / q.POLICY_NAME, docs / "NASA_PRECISION_QUALIFICATION_HANDOFF.md"):
            ptq.require(ptq.sha256(path) == ptq.sha256(snapshot / path.name), "Frozen instructions changed")
    else:
        ptq.require(not any(path.exists() for path in (previous, snapshot, out / "vendor")),
                    "Prior regression evidence exists; inspect before resuming")
        worker_dir.mkdir(parents=True)
        for relative in code:
            shutil.copyfile(SRC / relative, snapshot / relative)
        for path in (docs / q.POLICY_NAME, docs / "NASA_PRECISION_QUALIFICATION_HANDOFF.md", Path(__file__)):
            shutil.copyfile(path, snapshot / path.name)
        ptq.save_json(previous, binding)

    docker = shutil.which("docker", mode=os.F_OK | os.X_OK)
    ptq.require(docker, "Docker CLI is unavailable")
    image = subprocess.check_output([docker, "image", "inspect", ptq.IMAGE_ID, "--format", "{{.Id}}"],
                                    text=True).strip()
    ptq.require(image == ptq.IMAGE_ID, "Wrong Docker image")
    root = binding["artifacts"]
    mounts = [
        (snapshot, "/code", True), (previous, "/binding.json", True),
        (Path(binding["historical_preparation"]), "/inputs", True),
        (Path(root["bie_sha256"]["path"]), "/model/model.bie", True),
        (Path(root["training_predictions_sha256"]["path"]).parent, "/training", True),
        (Path(root["calibration_manifest_sha256"]["path"]).parent, "/calibration", True),
        (out / "vendor", "/out", False),
    ]
    (out / "vendor").mkdir(exist_ok=True)
    cmd = [docker, "run", "--rm", "--network", "none", "--pull", "never",
           "--env", "OMP_NUM_THREADS=1", "--env", "OPENBLAS_NUM_THREADS=1",
           "--env", "PYTHONPATH=/code", "--shm-size", "1g", "--workdir", "/tmp"]
    for host, target, readonly in mounts:
        ptq.require("," not in str(host), "Invalid Docker mount path")
        cmd.extend(["--mount", "type=bind,source={},target={}{}".format(host, target, ",readonly" if readonly else "")])
    cmd.extend(["--entrypoint", "/workspace/miniconda/bin/python", image, "-B", "-u", "-m", MODULE,
                "--policy", "/code/" + q.POLICY_NAME, "--binding", "/binding.json",
                "--binding-sha", ptq.sha256(previous), "--bie", "/model/model.bie", "--inputs", "/inputs",
                "--calibration", "/calibration", "--training", "/training", "--out", "/out",
                "--jobs", str(jobs), "--chunk", "24"])
    attempt = out / ("regression_invocation_" + uuid4().hex[:12])
    attempt.mkdir()
    log = attempt / "simulator.log"
    receipt = attempt / "invocation.json"
    record = dict(started_utc=utc(), argv=cmd, image_id=image, scope=binding["scope"],
                  policy_sha256=q.POLICY_SHA, policy_commit=q.FREEZE, preparation_sha256=ptq.sha256(previous),
                  sources={p.relative_to(snapshot).as_posix(): ptq.sha256(p) for p in snapshot.rglob("*") if p.is_file()},
                  launcher_sha256=ptq.sha256(__file__), jobs=jobs, status="STARTED", error=None)
    ptq.save_json(receipt, record)
    tick = time.monotonic()
    print("Authenticated development inputs. Native simulator log: " + str(log), flush=True)
    try:
        with log.open("xb") as stream:
            record["return_code"] = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT).returncode
        native = out / "vendor/bie_receipt.json"
        if native.is_file():
            record["vendor_receipt_sha256"] = ptq.sha256(native)
        ptq.require(record["return_code"] == 0 and ptq.load_json(native)["status"] == "BIE_SIMULATION_COMPLETE",
                    "Native regression failed. See the invocation log and vendor receipt.")
        record["status"] = "BIE_SIMULATION_COMPLETE"
    except Exception as exc:
        record.update(status="FAILED", error=repr(exc))
        raise
    finally:
        record.update(ended_utc=utc(), seconds=time.monotonic()-tick)
        if log.exists():
            record["log_sha256"] = ptq.sha256(log)
        ptq.save_json(receipt, record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("RADEDGE_DATA_ROOT", "D:/radiation-edge-ai-data")))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--resume", action="store_true", help="Honor previously sealed chunks of an interrupted invocation")
    args = parser.parse_args()
    launch(args.data_root, args.out, args.jobs, args.resume)

