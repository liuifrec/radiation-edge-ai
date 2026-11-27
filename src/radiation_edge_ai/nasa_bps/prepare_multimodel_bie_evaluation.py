"""Stage held-out development inputs and replay both archived FP32 graphs.

Run with the campaign's scientific Python. The vendor worker consumes the staged
.npy files and their hashes in inputs.csv. Each output directory is single-use.
"""
from __future__ import annotations

import argparse
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

from . import multimodel as mm
from . import multimodel_ptq as ptq
from . import multimodel_qualification as mobile
from .deployment_cohort import cohort_identity


DATA_ROOT = PureWindowsPath(r"D:\radiation-edge-ai-data")
SCREEN = "nasa_trained_graph_screen_20261118T020501Z_06255b35"
PILOT = "nasa_resnet18_female_ptq_20261118T052424Z_61d42817"
HOST_SHA = "e2238de53c4fc7f70f3ee3e59b239b93b5d314c31cc7e23da063ee135e3a5c51"
WORKER_SHA = "a8d1189e3f4e886352c5ce28c67dbfecf8f3872876479c6d552b94e02d3a7641"
BIE_SHA = "2428c60742e862f3bc3b6a6831284a3afa2b9e4db652fa760f331e3500bbea2f"
SOURCE_FIELDS = ("sample_name", "source_name", "strain", "sex", "particle_type")


def utc():
    return datetime.now(timezone.utc).isoformat()


def recorded_data_path(data_root, path):
    """Map the historical Windows data anchor to the caller's local data directory."""
    root = Path(data_root).resolve()
    relative = PureWindowsPath(path).relative_to(DATA_ROOT)
    for part in relative.parts:
        ptq.require(not part.endswith((".", " ")) and not PureWindowsPath(part).is_reserved()
                    and not any(c in '<>:"|?*' or ord(c) < 32 for c in part),
                    "Invalid recorded data path")
    result = root.joinpath(*relative.parts).resolve()
    ptq.require(result == root or root in result.parents, "Recorded data path escaped data root")
    return result


def frozen_inputs(data_root, policy_path, policy_sha256, artifacts):
    def record(name, path, digest):
        artifacts[name] = mm._check_sha(path, digest)
        return path

    ptq.require(policy_sha256 == ptq.V1_POLICY_SHA, "Unrecognized prospective V1 policy identity")
    record("policy", policy_path, policy_sha256)
    policy = ptq.load_json(policy_path)
    record("acceptance_document", policy_path.with_suffix(".md"), policy["acceptance_document_sha256"])
    ptq.require((policy["architecture"], policy["fold"]) == (ptq.ARCH, ptq.FOLD),
                "Policy addresses another architecture or fold")
    pilot = data_root / "scratch" / PILOT
    host = ptq.load_json(record("ptq_host", pilot / "host_ptq_receipt.json", HOST_SHA))
    worker = ptq.load_json(record("ptq_worker", pilot / host["worker_report"], WORKER_SHA))
    ptq.require(host["status"] == worker["status"] == "PTQ_GENERATED_UNVALIDATED"
                and host["optimized_fp32_parity_pass"] is True
                and host["image_id"] == worker["toolchain_image_id"] == ptq.IMAGE_ID
                and worker["bie"]["sha256"] == host["bie_sha256"] == policy["bie_sha256"] == BIE_SHA,
                "Incomplete or mismatched PTQ provenance")
    record("bie", pilot / "pilot/quantized_model.bie", BIE_SHA)
    parity = ptq.load_json(record("calibration_fp32_parity", pilot / host["optimized_fp32_parity_receipt"],
                                  worker["optimized_fp32_parity_receipt_sha256"]))
    record("calibration_fp32_pairs", pilot / "fp32_optimized_parity/fp32_optimized_parity.csv",
           parity["predictions_sha256"])
    run, image_root = campaign_inputs(
        data_root, policy, artifacts, ptq.EXPECTED_SCREEN_SHA, ptq.EXPECTED_EXPORT_SHA,
        host["screen_campaign_sha256"])
    for key in ("source_onnx", "optimized_onnx", "calibration_manifest"):
        ptq.require(worker["input"][key]["sha256"] == artifacts[key]["sha256"],
                    f"PTQ consumed different {key}")
    return run, image_root


def campaign_inputs(data_root, identities, artifacts, screen_sha, export_sha, campaign_sha):
    def local(path):
        return recorded_data_path(data_root, path)

    def record(name, path, digest):
        artifacts[name] = mm._check_sha(path, digest)
        return path

    campaign = ptq.load_json(record("campaign", data_root / "scratch" / SCREEN / "screen_receipt.json",
                                    campaign_sha))
    case, = (c for c in campaign["cases"] if (c["architecture"], c["fold"]) ==
             (identities["architecture"], identities["fold"]))
    screen_path = record("screen", local(case["screen_report"]), screen_sha)
    screen = ptq.load_json(screen_path)
    export_dir = local(case["host_export_dir"])
    export = ptq.load_json(record("export", export_dir / "export.json", export_sha))
    record("source_onnx", export_dir / case["onnx_name"], identities["source_onnx_sha256"])
    record("optimized_onnx", screen_path.with_name("optimized.onnx"), screen["optimized_onnx_sha256"])
    record("calibration_manifest", export_dir / "calibration_manifest.csv", export["calibration"]["sha256"])
    run_path = record("training_run", local(export["training_run"]["path"]), identities["training_run_sha256"])
    run = ptq.load_json(run_path)
    record("checkpoint", local(run["checkpoint"]["path"]), run["checkpoint"]["sha256"])
    for key in ("source_onnx", "optimized_onnx", "training_run"):
        ptq.require(artifacts[key]["sha256"] == identities[key + "_sha256"], f"Policy mismatch for {key}")
    for key in ("manifest", "reference", "reference_freeze", "verified_image_inventory"):
        item = run["dataset"][key]
        record(key, local(item["path"]), item["sha256"])
    ptq.require(artifacts["manifest"]["sha256"] == identities["development_manifest_sha256"] == mm.MANIFEST_SHA256
                and artifacts["reference"]["sha256"] == identities["nasa_reference_sha256"] == mm.REFERENCE_SHA256,
                "Changed cohort or reference")
    record("fp32_nuclei", run_path.with_name("nucleus_predictions.csv"), identities["fp32_nuclei_sha256"])
    record("fp32_bags", run_path.with_name("bag_predictions.csv"), identities["fp32_bags_sha256"])
    script = "train_pilot_v1_r1_mil.py"
    record("preprocessing_code", Path(__file__).with_name(script), run["implementation"][script]["sha256"])
    return run, local(run["dataset"]["image_root"])


def select_held_out(manifest, calibration, nuclei, bags, reference, split):
    grouping = mm.validate_grouping(manifest)
    ptq.require(grouping["folds"][ptq.FOLD] == split and grouping["nuclei"] == 7200
                and split["test_sources"] == ["BALBCF1", "C57BLF1"], "Development split changed")
    selected = {r["sample_id"]: r for r in manifest if r[ptq.FOLD] == "test"}
    selected_bags = Counter(r["sample_name"] for r in selected.values())
    ptq.require(len(selected) == 2400 and len(selected_bags) == 24 and set(selected_bags.values()) == {100}
                and all(r["cohort"] == "development" for r in selected.values()),
                "Expected 24 held-out development bags of 100 nuclei")
    ptq.require(len(calibration) == 192 and not selected.keys() & {r["sample_id"] for r in calibration}
                and all(r[ptq.FOLD] == "train" and r["source_name"] in split["train_sources"]
                        for r in calibration), "PTQ calibration overlaps held-out evaluation")
    ptq.require(len(nuclei) == len(selected) and {r["sample_id"] for r in nuclei} == selected.keys(),
                "Frozen nucleus predictions differ from held-out membership")
    ptq.require(len(bags) == 24 and {b["sample_name"] for b in bags} == selected_bags.keys(),
                "Frozen bag predictions differ from held-out membership")
    targets = {r["sample_name"]: r for r in reference}
    for bag in bags:
        target = targets[bag["sample_name"]]
        ptq.require(bag["fold"] == ptq.FOLD and int(bag["n_development_nuclei"]) == 100
                    and float(bag["nasa_avg_nfoci"]) == float(target["nasa_avg_nfoci"])
                    and all(bag[k] == target[k] for k in SOURCE_FIELDS),
                    "Frozen bag and NASA target mismatch")
    order = []
    for original in nuclei:
        row = selected[original["sample_id"]]
        ptq.require(original["fold"] == ptq.FOLD and original["nucleus_key"] == row["nucleus_key"]
                    and all(original[k] == row[k] for k in SOURCE_FIELDS)
                    and all(float(original[k]) == float(row[k]) for k in ("dose_Gy", "hr_post_exposure")),
                    "Frozen FP32 nucleus identity or exposure mismatch")
        order.append(row)
    return order


def graph_session(path, ort):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for nodes, expected in ((session.get_inputs(), ("input", [1, 3, 256, 256])),
                            (session.get_outputs(), ("burden", [1]))):
        ptq.require(len(nodes) == 1 and (nodes[0].name, nodes[0].shape) == expected
                    and nodes[0].type == "tensor(float)", "FP32 ONNX interface changed")
    return session


def stage_row(output, i, row, native, source, optimized, np, preprocess):
    array = np.ascontiguousarray(preprocess.pad_and_pack(native)[None])
    ptq.require(array.shape == (1, 3, 256, 256) and array.dtype == np.float32
                and np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1
                and not array[:, 2].any(), f"Invalid staged tensor for {row['sample_id']}")
    relative = f"tensors/{i:04d}.npy"
    path = output / relative
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    saved = np.load(path, allow_pickle=False)
    ptq.require(np.array_equal(saved, array), f"Staged tensor differs for {row['sample_id']}")
    result = {k: row[k] for k in ("sample_id", "nucleus_key", *SOURCE_FIELDS, "dose_Gy", "hr_post_exposure")}
    result.update(fold=ptq.FOLD, array_file=relative, array_sha256=ptq.sha256(path),
                  native_height=native.shape[1], native_width=native.shape[2])
    for key, session in (("original_fp32", source), ("optimized_fp32", optimized)):
        value, = session.run(["burden"], {"input": saved})
        ptq.require(value.dtype == np.float32 and value.shape == (1,) and np.isfinite(value).all(),
                    f"Invalid {key} prediction at {row['sample_id']}")
        result[key] = float(value[0])
    return result


def prepare(args):
    import cv2
    import numpy as np
    import onnxruntime as ort
    from . import train_pilot_v1_r1_mil as preprocess

    output = args.out.resolve()
    repo = Path(__file__).resolve().parents[3]
    ptq.require(output != repo and repo not in output.parents, "Use external storage for evaluation")
    # A nested run under scratch keeps staged tensors separate from the frozen artifacts.
    pilot = args.data_root.resolve() / "scratch" / PILOT
    ptq.require(pilot not in output.parents and output != pilot, "Preserve the existing PTQ run")
    output.mkdir(parents=True, exist_ok=False)
    mobile_stage = getattr(args, "stage_mobilenet_fp32", False)
    receipt = output / ("fp32_preparation.json" if mobile_stage else "preparation.json")
    artifacts = {}
    report = {"architecture": mobile.ARCHITECTURE if mobile_stage else ptq.ARCH,
              "fold": ptq.FOLD, "schema_version": 1,
              "created_utc": utc(), "status": "STARTED", "error": None, "evaluated_nuclei": 0,
              "policy_sha256": args.policy_sha256, "preparation_code_sha256": ptq.sha256(__file__),
              "artifacts": artifacts, "python": platform.python_version(), "numpy": np.__version__,
              "opencv": cv2.__version__, "onnxruntime": ort.__version__, "runtime": "CPUExecutionProvider"}
    ptq.save_json(receipt, report)
    try:
        if mobile_stage:
            identities = dict(mobile.MOBILE, architecture=mobile.ARCHITECTURE, fold=mobile.FOLD)
            run, image_root = campaign_inputs(args.data_root.resolve(), identities, artifacts,
                                              mobile.SCREEN_SHA, mobile.EXPORT_SHA, mobile.CAMPAIGN_SHA)
        else:
            run, image_root = frozen_inputs(args.data_root.resolve(), args.policy, args.policy_sha256, artifacts)
        def table(key):
            return mm._read_csv(Path(artifacts[key]["path"]))
        rows = select_held_out(table("manifest"), table("calibration_manifest"), table("fp32_nuclei"),
                               table("fp32_bags"), table("reference"), run["source_split"])
        if mobile_stage:
            report.update(cohort_identity(rows))
        verified = mm.verify_images(rows, image_root, table("verified_image_inventory"))
        mm._write_csv(output / "evaluation_image_hashes.csv", verified)
        source, optimized = (graph_session(artifacts[key]["path"], ort)
                             for key in ("source_onnx", "optimized_onnx"))
        cache = preprocess.NativeImageCache(image_root, max_items=0)
        (output / "tensors").mkdir()
        staged = []
        for i, row in enumerate(rows):
            staged.append(stage_row(output, i, row, cache.get(row), source, optimized, np, preprocess))
            report["evaluated_nuclei"] = i + 1
            if (i + 1) % 100 == 0:
                print(f"Prepared {i + 1}/{len(rows)} tensors and paired FP32 predictions", flush=True)
        mm.verify_images(rows, image_root, verified)
        mm._write_csv(output / "inputs.csv", staged)
        for item in artifacts.values():
            mm._check_sha(Path(item["path"]), item["sha256"])
        report.update({key + "_sha256": artifacts[key]["sha256"] for key in
                       ("source_onnx", "optimized_onnx", "training_run", "fp32_nuclei", "fp32_bags")})
        if not mobile_stage:
            report["bie_sha256"] = artifacts["bie"]["sha256"]
        report.update(input_manifest_sha256=ptq.sha256(output / "inputs.csv"),
                      development_manifest_sha256=artifacts["manifest"]["sha256"],
                      evaluation_image_hashes_sha256=ptq.sha256(output / "evaluation_image_hashes.csv"),
                      evaluated_bags=24, evaluated_sources=run["source_split"]["test_sources"],
                      input_contract=run["contract"]["input"],
                      status="FP32_PREPARATION_COMPLETE" if mobile_stage else "PREPARATION_COMPLETE")
    except Exception as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["completed_utc"] = utc()
        ptq.save_json(receipt, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--policy-sha256")
    parser.add_argument("--stage-mobilenet-fp32", action="store_true",
                        help="Stage original and optimized MobileNetV2 ONNX inputs before fixed readout.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(str(DATA_ROOT)))
    args = parser.parse_args()
    if not args.stage_mobilenet_fp32 and not (args.policy and args.policy_sha256):
        parser.error("--policy and --policy-sha256 are required for a bound evaluation")
    prepare(args)


if __name__ == "__main__":
    main()
