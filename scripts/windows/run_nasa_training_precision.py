"""Run or score the sealed two-recipe MobileNet training screen in fresh external directories."""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

from run_mobilenet_training_ptq import docker_run, inputs

from radiation_edge_ai.nasa_bps import multimodel_ptq as ptq
from radiation_edge_ai.nasa_bps import training_precision as precision
from radiation_edge_ai.nasa_bps.training_precision_readout import summarize, training_decision

BASE = "nasa_mobilenet_female_int8_20261126T234932Z_a7c472a7"
PANEL = "mobilenet_feature_training_20261127T015528Z_795ec600/selection.json"
ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "docs/NASA_MOBILENET_TRAINING_PRECISION_V1.json"


def score(out, panel):
    policy = ptq.load_json(POLICY)
    ptq.require(
        ptq.sha256(POLICY) == precision.POLICY_SHA256
        and ptq.sha256(panel) == policy["panel_sha256"]
        and not (out / "selection.json").exists(),
        "Changed selection inputs or existing selection",
    )
    candidates, receipts = {}, {}
    for key in ("baseline", "datapath16", "neural16"):
        directory = out / key
        q = None if key == "baseline" else ptq.load_json(directory / "quantization.json")
        if q is not None:
            ptq.require(
                q["policy_sha256"] == precision.POLICY_SHA256 and q["recipe"] == key,
                "Mismatched conversion receipt",
            )
        if q and q["status"] == "BLOCKED":
            candidates[key] = q
            receipts[key] = ptq.sha256(directory / "quantization.json")
            continue
        receipt = directory / "evaluation/readout.json"
        record = ptq.load_json(receipt)
        expected = q["bie"]["sha256"] if q else policy["standard_bie_sha256"]
        ptq.require(
            record["status"] == "TRAINING_READOUT_COMPLETE"
            and record["recipe"] == key
            and record["policy_sha256"] == precision.POLICY_SHA256
            and record["panel_sha256"] == ptq.sha256(panel)
            and record["bie"]["sha256"] == expected
            and record["paired_sha256"] == ptq.sha256(receipt.with_name("paired.csv")),
            "Mismatched completed training readout",
        )
        if q:
            ptq.require(
                ptq.sha256(directory / q["bie"]["path"]) == expected
                and record["quantization_sha256"] == ptq.sha256(directory / "quantization.json"),
                "Conversion binary or readout provenance mismatch",
            )
        paired = ptq.checked_rows(receipt.with_name("paired.csv"))
        panel_rows = ptq.load_json(panel)["rows"]
        keys = ("sample_id", "array_sha256", "source_name", "sample_name", "panel", "stratum")
        ptq.require(
            len(paired) == len(panel_rows) == 40
            and all(
                all(left[k] == right[k] for k in keys)
                and float(left["source_fp32"]) == right["source_fp32"]
                and float(left["error"]) == float(left["bie_burden"]) - right["source_fp32"]
                for left, right in zip(paired, panel_rows)
            )
            and summarize(paired) == record["summary"],
            "Paired results disagree with the sealed panel or summary",
        )
        for chunk in record["chunks"]:
            ptq.require(
                ptq.sha256(receipt.parent / chunk["archive"]) == chunk["archive_sha256"],
                "Changed native logs",
            )
        receipts[key] = ptq.sha256(receipt)
        candidates[key] = record
    decision = training_decision(candidates.pop("baseline"), candidates)
    decision.update(
        policy_sha256=precision.POLICY_SHA256,
        panel_sha256=ptq.sha256(panel),
        receipt_sha256=receipts,
    )
    ptq.save_json(out / "selection.json", decision)


def verified_trigger(receipt, panel):
    evidence = ptq.load_json(receipt)
    results = receipt.with_name("nuclear_results.json")
    ptq.require(
        evidence["study"] == "mobilenet_v2_training_only"
        and evidence["held_out"] is False
        and evidence["selection_sha256"] == ptq.sha256(panel)
        and evidence["nuclear_results_sha256"] == ptq.sha256(results),
        "Changed baseline training attribution",
    )
    confirmation = {
        r["sample_id"]: r for r in ptq.load_json(panel)["rows"] if r["panel"] == "confirmation"
    }
    for row in json.loads(results.read_bytes()):
        if row["sample_id"] not in confirmation:
            continue
        case = confirmation[row["sample_id"]]
        contribution = row["floating_head_on_native_pool"] - row["original_head"]
        capture = panel.parent / case["sample_id"] / "receipt.json"
        ptq.require(
            row["panel"] == "confirmation"
            and row["input_sha256"] == case["array_sha256"]
            and row["capture_receipt_sha256"] == ptq.sha256(capture)
            and row["bie"] == ptq.load_json(capture)["burden"]
            and row["layers"]["burden"]["fixed_range"] == [row["bie"], row["bie"]],
            "Nonpilot terminal identity mismatch",
        )
        if math.isfinite(contribution) and contribution != 0:
            return {
                "sample_id": case["sample_id"],
                "upstream_head_units": contribution,
                "receipt_sha256": ptq.sha256(receipt),
                "capture_sha256": ptq.sha256(capture),
            }
    raise ValueError("No qualifying original BIE upstream observation")


def run(root, out, stage, trigger=None):
    baseline = root / "scratch" / BASE
    panel = root / "scratch" / PANEL
    ptq.require(out.parent == root / "scratch" and ROOT not in out.parents, "Use external scratch")
    out.mkdir(exist_ok=True)
    if stage == "score":
        return score(out, panel)
    directory = out / stage
    directory.mkdir(exist_ok=False)
    policy = ptq.load_json(POLICY)
    ptq.require(
        ptq.sha256(POLICY) == precision.POLICY_SHA256
        and ptq.sha256(panel) == policy["panel_sha256"],
        "Training policy or panel changed",
    )
    model_dir, vendor, graph, _ = inputs(root)
    workers = directory / "workers"
    workers.mkdir()
    for name in (
        "training_precision",
        "training_precision_readout",
        "multimodel_ptq",
        "multimodel_bie_inference",
        "multimodel_qualification",
        "deployment_cohort",
    ):
        shutil.copyfile(Path(ptq.__file__).with_name(name + ".py"), workers / (name + ".py"))
    parity = baseline / "fp32_optimized_parity"
    mounts = [
        (workers, "/worker"),
        (model_dir, "/inputs"),
        (parity, "/parity"),
        (panel, "/panel.json"),
        (POLICY, "/policy.json"),
        (directory, "/data1", True),
    ]
    common = ["--policy", "/policy.json"]
    started = time.monotonic()
    host = {
        "recipe": stage,
        "policy_sha256": precision.POLICY_SHA256,
        "panel_sha256": ptq.sha256(panel),
        "workers": {p.name: ptq.sha256(p) for p in workers.iterdir()},
        "launcher_sha256": ptq.sha256(__file__),
        "status": "STARTED",
        "physical_execution": False,
        "nef_compiled": False,
    }
    if stage != "baseline":
        ptq.require(
            trigger is not None and trigger.is_file(),
            "External verified baseline attribution required",
        )
        host["baseline_trigger"] = verified_trigger(trigger, panel)
    host_file = directory / "host.json"
    ptq.save_json(host_file, host)
    try:
        if stage == "baseline":
            bie = baseline / "pilot/quantized_model.bie"
            mounts.append((bie, "/baseline.bie"))
            options = ["--bie", "/baseline.bie"]
        else:
            docker_run(
                directory / "docker_ptq.log",
                mounts + [(vendor.parent, "/screen")],
                [
                    "/worker/training_precision.py",
                    "--architecture",
                    "mobilenet_v2",
                    "--recipe",
                    stage,
                    "--onnx",
                    "/inputs/" + graph.name,
                    "--optimized",
                    "/screen/optimized.onnx",
                    "--screen",
                    "/screen/screen.json",
                    "--export",
                    "/inputs/export.json",
                    "--manifest",
                    "/inputs/calibration_manifest.csv",
                    "--calibration-root",
                    "/inputs",
                    "--image-id",
                    ptq.IMAGE_ID,
                    "--out",
                    "/data1",
                    "--parity-receipt",
                    "/parity/fp32_optimized_parity.json",
                ]
                + common,
                "/data1",
            )
            result = ptq.load_json(directory / "quantization.json")
            ptq.require(result["status"] == "TRAINING_BIE_GENERATED", "Recipe blocked")
            options = [
                "--bie",
                "/data1/" + result["bie"]["path"],
                "--quantization",
                "/data1/quantization.json",
            ]
        docker_run(
            directory / "docker_readout.log",
            mounts,
            [
                "/worker/training_precision_readout.py",
                "--inputs",
                "/inputs",
                "--parity",
                "/parity",
                "--panel",
                "/panel.json",
                "--out",
                "/data1/evaluation",
                "--recipe",
                stage,
            ]
            + options
            + common,
            "/data1",
        )
        host["status"] = "COMPLETE"
    except (Exception, SystemExit) as exc:
        host.update(status="FAILED", error=repr(exc))
        raise
    finally:
        host["wall_seconds"] = time.monotonic() - started
        ptq.save_json(host_file, host)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("baseline", "datapath16", "neural16", "score"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--trigger", type=Path, help="Baseline upstream attribution evidence for corrective runs"
    )
    args = parser.parse_args()
    run(args.data_root.resolve(), args.out.resolve(), args.stage, args.trigger)
