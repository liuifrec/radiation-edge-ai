"""Bind a sealed MobileNetV2 FP32 replay to its prospectively frozen INT8 candidate."""

import json
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

from . import multimodel_qualification as mobile
from .deployment_cohort import require_cohort
from .multimodel_ptq import FIXED_SIM_SHA, IMAGE_ID, checked_rows, load_json, require, sha256
from .multimodel_ptq import standard_int8_report


def staged_path(key, recorded, data_root):
    if key == "preprocessing_code":
        return Path(__file__).with_name("train_pilot_v1_r1_mil.py")
    original = Path(recorded)
    if data_root in original.resolve().parents:
        return original
    relative = PureWindowsPath(recorded).relative_to(PureWindowsPath("D:/radiation-edge-ai-data"))
    candidate = data_root.joinpath(*relative.parts).resolve()
    require(data_root in candidate.parents, "Invalid recorded archive location")
    return candidate


def bind(prepared, scratch, policy_path, document):
    """Check original FP32 replay, exact training-only PTQ and candidate bytes before simulation."""
    policy = load_json(policy_path)
    policy_sha = sha256(policy_path)
    mobile.verify_policy(policy, policy_sha)
    scratch = scratch.resolve(strict=True)
    prepared = prepared.resolve(strict=True)
    require(scratch in prepared.parents, "Expected preparation under external scratch")
    pilot = (scratch / policy["ptq_pilot"]).resolve(strict=True)
    require(pilot.parent == scratch and pilot not in prepared.parents, "Invalid PTQ pilot location")

    def verify(path, expected):
        require(sha256(path) == expected, "SHA256 mismatch for " + str(path))
        return {"path": str(path), "sha256": expected, "bytes": path.stat().st_size}

    fp32_receipt = prepared / "fp32_preparation.json"
    verify(fp32_receipt, policy["fp32_preparation_sha256"])
    stage = load_json(fp32_receipt)
    require(stage["status"] == "FP32_PREPARATION_COMPLETE" and stage["error"] is None and
            stage["evaluated_nuclei"] == 2400 and stage["evaluated_bags"] == 24 and
            stage["architecture"] == policy["architecture"] and stage["fold"] == policy["fold"],
            "Expected the completed MobileNetV2 female-fold original/optimized FP32 replay")
    artifacts = stage["artifacts"]
    for key, value in artifacts.items():
        verify(staged_path(key, value["path"], scratch.parent), value["sha256"])
    artifacts["policy"] = verify(policy_path, policy_sha)
    artifacts["acceptance_document"] = verify(document, policy["acceptance_document_sha256"])
    verify(prepared / "inputs.csv", stage["input_manifest_sha256"])
    verify(prepared / "evaluation_image_hashes.csv", stage["evaluation_image_hashes_sha256"])
    for k, v in require_cohort(checked_rows(prepared / "inputs.csv"), policy).items():
        require(stage[k] == v, "FP32 stage membership differs from policy: " + k)
    for key in ("training_run_sha256", "source_onnx_sha256", "optimized_onnx_sha256",
                "development_manifest_sha256", "fp32_nuclei_sha256", "fp32_bags_sha256"):
        require(stage[key] == policy[key], "FP32 stage model or data differs from policy: " + key)
    host_path = pilot / "host_ptq_receipt.json"
    worker_path = pilot / "pilot" / "ptq_receipt.json"
    artifacts["ptq_host"] = verify(host_path, policy["ptq_host_sha256"])
    artifacts["ptq_worker"] = verify(worker_path, policy["ptq_worker_sha256"])
    host, worker = load_json(host_path), load_json(worker_path)
    require(host["status"] == worker["status"] == "PTQ_GENERATED_UNVALIDATED"
            and host["error"] is None and worker["error"] is None
            and host["optimized_fp32_parity_pass"] is True and worker["sources"] == "development_train_only"
            and worker["input"]["calibration_arrays"] == 192 and worker["platform"] == "720"
            and worker["model_id"] == 32781 and worker["architecture"] == policy["architecture"]
            and worker["fold"] == policy["fold"] and host["ptq_worker_exit_code"] == 0
            and host["image_id"] == worker["toolchain_image_id"] == IMAGE_ID
            and host["bie_sha256"] == worker["bie"]["sha256"] == policy["bie_sha256"],
            "Incomplete or mismatched training-only PTQ pilot")
    for key in ("source_onnx", "optimized_onnx", "calibration_manifest", "export", "screen"):
        require(worker["input"][key]["sha256"] == artifacts[key]["sha256"],
                "Candidate was generated from different " + key)
    path = pilot / "fp32_optimized_parity" / "fp32_optimized_parity.json"
    artifacts["calibration_fp32_parity"] = verify(path, worker["optimized_fp32_parity_receipt_sha256"])
    parity = load_json(path)
    require(parity["status"] == "PASS" and parity["inputs"] == parity["pass_count"] == 192
            and parity["manifest_sha256"] == artifacts["calibration_manifest"]["sha256"]
            and parity["source_sha256"] == policy["source_onnx_sha256"]
            and parity["optimized_sha256"] == policy["optimized_onnx_sha256"],
            "Training-only optimized replay differs from the candidate")
    artifacts["calibration_fp32_pairs"] = verify(path.with_suffix(".csv"), parity["predictions_sha256"])
    artifacts["bie"] = verify(pilot / "pilot/quantized_model.bie", policy["bie_sha256"])
    directory = pilot / "training_pilot"
    precision_path = directory / "precision_receipt.json"
    artifacts["precision"] = verify(precision_path, policy["precision_receipt_sha256"])
    precision = standard_int8_report(pilot / "kneron_flow", policy["bie_sha256"])
    require(precision == load_json(precision_path), "Vendor INT8 precision report changed")
    artifacts["vendor_fx_report"] = verify(pilot / "kneron_flow/model_fx_report.json",
                                           precision["vendor_fx_report_sha256"])
    training_path = directory / "training_pilot.json"
    artifacts["training_pilot"] = verify(training_path, policy["training_pilot_sha256"])
    training = load_json(training_path)
    require(training["status"] == "TRAINING_CHARACTERIZATION_COMPLETE"
            and training["host_ptq_sha256"] == policy["ptq_host_sha256"]
            and training["worker_ptq_sha256"] == policy["ptq_worker_sha256"]
            and training["bie_sha256"] == policy["bie_sha256"] and training["sdk_module_sha256"] == FIXED_SIM_SHA
            and training["qualification_bags"] == 0 and training["summary"]["n"] == 16
            and {k: v["n"] for k, v in training["per_source"].items()} ==
            {k: 4 for k in worker["input"]["train_sources"]}, "Training-only numerical pilot mismatch")
    for key, filename in (("selection", "training_inputs.csv"), ("predictions", "training_predictions.csv"),
                          ("native_logs", "native_logs.zip")):
        artifacts["training_pilot_" + key] = verify(directory / filename, training[key + "_sha256"])
    stage.update(status="PREPARATION_COMPLETE", policy_sha256=policy_sha, bie_sha256=policy["bie_sha256"],
                 fp32_preparation_sha256=policy["fp32_preparation_sha256"], binding_code_sha256=sha256(__file__),
                 bound_utc=datetime.now(timezone.utc).isoformat())
    target = prepared / "preparation.json"
    if target.is_file():
        existing = load_json(target)
        bound_utc = existing.get("bound_utc")
        require(isinstance(bound_utc, str) and existing == dict(stage, bound_utc=bound_utc),
                "Existing preparation belongs to a different binding")
    else:
        with target.open("x", encoding="utf-8") as stream:
            json.dump(stage, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return target
