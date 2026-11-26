"""Replay the female-fold MobileNetV2 training-only parity, INT8 PTQ and fixed pilot."""
import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from radiation_edge_ai.nasa_bps import kneron_candidate_screen as screen
from radiation_edge_ai.nasa_bps import multimodel_ptq as ptq
from radiation_edge_ai.nasa_bps.prepare_multimodel_bie_evaluation import recorded_data_path


CAMPAIGN = "nasa_trained_graph_screen_20261118T020501Z_06255b35"
CAMPAIGN_SHA = "7d5c042e2d6f929a3454fc3890f2228c13670a3a0577b674bae9b170b6585687"
ARCH = ptq.MOBILENET_FEMALE.architecture


def mount(source, target, writable=False):
    ptq.require("," not in str(source), "Docker mount path contains a comma")
    return ["--mount", "type=bind,source={},target={}{}".format(
        source, target, "" if writable else ",readonly")]


def docker_run(log, mounts, args, workdir):
    image = subprocess.check_output(["docker", "image", "inspect", ptq.IMAGE_ID,
                                     "--format", "{{.Id}}"], text=True).strip()
    ptq.require(image == ptq.IMAGE_ID, "Pinned local toolchain image is unavailable")
    command = ["docker", "run", "--rm", "--pull", "never", "--network", "none", "--shm-size", "1g"]
    for binding in mounts:
        command.extend(mount(*binding))
    command.extend(["--workdir", workdir, "--entrypoint", "/workspace/miniconda/bin/python",
                    image, "-B"] + args)
    with log.open("xb") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def inputs(root):
    campaign_path = root / "scratch" / CAMPAIGN / "screen_receipt.json"
    ptq.require(ptq.sha256(campaign_path) == CAMPAIGN_SHA, "Changed trained screen campaign")
    case, = [c for c in ptq.load_json(campaign_path)["cases"]
             if (c["architecture"], c["fold"]) == (ARCH, ptq.FOLD)]
    directory = recorded_data_path(root, case["host_export_dir"])
    vendor = recorded_data_path(root, case["screen_report"])
    graph = directory / case["onnx_name"]
    export = directory / "export.json"
    frozen = screen._validate_trained_export(export, graph, ptq.sha256(export), ptq.sha256(graph))
    data = ptq.load_json(export)
    for name in ("checkpoint", "parent_checkpoint", "training_run"):
        path = recorded_data_path(root, data[name]["path"])
        ptq.require(ptq.sha256(path) == data[name]["sha256"], "Changed " + name)
    return directory, vendor, graph, frozen


def quantize(root, out):
    out.mkdir(exist_ok=False)
    policy = dict(schema_version=1, purpose="training_only_mobilenet_v2_standard_int8",
                  status="STARTED", created_utc=datetime.now(timezone.utc).isoformat(),
                  case=ARCH + "/" + ptq.FOLD, architecture=ARCH, fold=ptq.FOLD,
                  model_id=ptq.MOBILENET_FEMALE.model_id, image_id=ptq.IMAGE_ID,
                  screen_campaign_sha256=CAMPAIGN_SHA, worker_report="pilot/ptq_receipt.json",
                  optimized_fp32_parity_receipt="fp32_optimized_parity/fp32_optimized_parity.json",
                  optimized_fp32_parity_pass=False, bie_generated=False,
                  nef_compiled=False, physical_execution=False, error=None)
    receipt = out / "host_ptq_receipt.json"
    ptq.save_json(receipt, policy)
    try:
        directory, vendor, graph, sources = inputs(root)
        implementation = out / "workers"
        implementation.mkdir()
        for name in ("multimodel_ptq.py", "multimodel_bie_inference.py",
                     "multimodel_training_bie_pilot.py"):
            shutil.copyfile(Path(ptq.__file__).with_name(name), implementation / name)
        policy.update(sources, host_script_sha256=ptq.sha256(__file__),
                      source_onnx_sha256=ptq.sha256(graph), export_json_sha256=ptq.sha256(directory / "export.json"),
                      screen_json_sha256=ptq.sha256(vendor),
                      worker_sha256=ptq.sha256(implementation / "multimodel_ptq.py"),
                      pilot_worker_sha256=ptq.sha256(implementation / "multimodel_training_bie_pilot.py"),
                      fixed_inference_worker_sha256=ptq.sha256(implementation / "multimodel_bie_inference.py"))
        check = SimpleNamespace(architecture=ARCH, onnx=graph, optimized=vendor.with_name("optimized.onnx"),
                                manifest=directory / "calibration_manifest.csv", calibration_root=directory,
                                export=directory / "export.json", screen=vendor, image_id=ptq.IMAGE_ID,
                                out=out / "fp32_optimized_parity")
        ptq.require(ptq.parity(check) == 0, "Original to optimized FP32 parity failed")
        parity_file = out / policy["optimized_fp32_parity_receipt"]
        ptq.verify_parity_receipt(parity_file, check)
        policy.update(optimized_fp32_parity_pass=True, calibration_arrays=192,
                      optimized_fp32_parity_receipt_sha256=ptq.sha256(parity_file),
                      calibration_dataset_role="development_train_ptq_calibration")
        ptq.save_json(receipt, policy)
        docker_run(out / "docker_ptq.log", [(implementation, "/worker"), (directory, "/inputs"),
                   (vendor.parent, "/screen"), (check.out, "/parity"), (out, "/data1", True)],
                   ["/worker/multimodel_ptq.py", "quantize", "--architecture", ARCH,
                    "--onnx", "/inputs/" + graph.name, "--optimized", "/screen/optimized.onnx",
                    "--export", "/inputs/export.json", "--screen", "/screen/screen.json",
                    "--manifest", "/inputs/calibration_manifest.csv", "--calibration-root", "/inputs",
                    "--out", "/data1/pilot", "--image-id", ptq.IMAGE_ID,
                    "--parity-receipt", "/parity/fp32_optimized_parity.json"], "/data1")
        result_path = out / policy["worker_report"]
        record = ptq.load_json(result_path)
        bie_hash = ptq.sha256(out / "pilot" / "quantized_model.bie")
        ptq.require(record["status"] == "PTQ_GENERATED_UNVALIDATED" and record["architecture"] == ARCH
                    and record["bie"]["sha256"] == bie_hash and record["bie_generated"] is True
                    and record["model_id"] == policy["model_id"] and record["fold"] == ptq.FOLD
                    and record["optimized_fp32_parity_receipt_sha256"] == ptq.sha256(parity_file),
                    "PTQ output identity mismatch")
        policy.update(status=record["status"], bie_sha256=bie_hash, bie_generated=True,
                      worker_report_sha256=ptq.sha256(result_path), ptq_worker_exit_code=0)
    except Exception as exc:
        policy.update(status="FAILED", error=str(exc))
        raise
    finally:
        policy["completed_utc"] = datetime.now(timezone.utc).isoformat()
        ptq.save_json(receipt, policy)


def characterize(root, out):
    directory, _, _, _ = inputs(root)
    receipt = ptq.load_json(out / "host_ptq_receipt.json")
    ptq.require(receipt["status"] == "PTQ_GENERATED_UNVALIDATED" and receipt["case"] == ARCH + "/" + ptq.FOLD,
                "Successful MobileNet PTQ is required")
    for name, key in (("multimodel_ptq.py", "worker_sha256"),
                      ("multimodel_training_bie_pilot.py", "pilot_worker_sha256"),
                      ("multimodel_bie_inference.py", "fixed_inference_worker_sha256")):
        ptq.require(ptq.sha256(out / "workers" / name) == receipt[key], "Changed training pilot implementation")
    pilot = out / "training_pilot"
    pilot.mkdir(exist_ok=False)
    docker_run(pilot / "docker.log", [(out / "workers", "/worker"), (directory, "/inputs"),
               (out, "/ptq"), (pilot, "/pilot", True)], ["/worker/multimodel_training_bie_pilot.py",
               "--inputs", "/inputs", "--ptq", "/ptq", "--out", "/pilot"], "/pilot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("quantize", "characterize"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.data_root.resolve(), args.out.resolve()
    ptq.require(out.parent == root / "scratch" and Path(__file__).resolve().parents[2] not in out.parents,
                "Use a new external scratch child for PTQ")
    (quantize if args.stage == "quantize" else characterize)(root, out)


if __name__ == "__main__":
    sys.exit(main())
