"""Bind one ResNet18 full-neural INT16 development regression to the frozen campaign."""

from pathlib import Path, PureWindowsPath

from . import multimodel_bie_inference as sim
from . import multimodel_ptq as ptq
from .deployment_cohort import require_cohort

FREEZE = "a6c3da20273a9c43b5a47bc7501724f3981dc206"
POLICY_NAME = "NASA_RESNET18_NEURAL16_REGRESSION_V1.json"
POLICY_SHA = "94a7ba514a32676c32b30f745faab5c037b729de89e3c8d29d5e854ffe656cc8"
CAMPAIGN = "nasa_cross_architecture_20261127T0158Z"
HISTORICAL_PREP = "nasa_resnet18_female_eval_20261118T060157Z_5ffa44b2/preparation"
SELECTION = "nasa_multisource_bie_20261118T232421Z_3a8998bc/selection.json"
BIE_SHA = "3f654f267ae280f27d6aaf6715b32af4285c5de6f4eb13c844e02e70f2f69c9b"

HISTORICAL = {
    "checkpoint": "model_training_checkpoint_sha256", "training_run": "training_run_sha256",
    "source_onnx": "source_onnx_sha256", "optimized_onnx": "optimized_onnx_sha256",
    "calibration_manifest": "calibration_manifest_sha256", "manifest": "development_manifest_sha256",
    "reference": "nasa_reference_sha256", "fp32_nuclei": "fp32_nuclei_sha256",
    "fp32_bags": "fp32_bags_sha256", "bie": "standard_bie_sha256",
}


def external_path(data, historical_path):
    relative = PureWindowsPath(historical_path).relative_to("D:/radiation-edge-ai-data")
    path = (data / Path(*relative.parts)).resolve(strict=True)
    ptq.require(data.resolve() in path.parents, "Artifact escapes the external data root")
    return path


def verify_policy(path):
    ptq.require(ptq.sha256(path) == POLICY_SHA, "Unrecognized frozen neural16 policy")
    policy = ptq.load_json(path)
    ptq.require(policy["bie_sha256"] == BIE_SHA and policy["scope"] == "known_female_development_regression"
                and policy["image_id"] == ptq.IMAGE_ID, "Unexpected candidate or evaluation scope")
    return policy


def verify_descriptor(descriptor):
    public = descriptor["transport"]
    ptq.require(descriptor["node"] == "/Gather" and descriptor["width"] == 15
                and descriptor["radix"] == 12 and descriptor["scale"] == 1.3206359148025513
                and public["name"] == "burden" and public["data_format"] == "8W1C16B"
                and public["bitw"] == 16 and public["radix"] == [13]
                and public["scale"] == [descriptor["scale"]] and public["onnx_shape"] == [1]
                and public["shape"] == [1, 1, 1, 1], "Unexpected native or transport scalar contract")


def authenticate(docs, data):
    """Authenticate prior evidence and sealed input identities without running either network."""
    data = data.resolve(strict=True)
    policy = verify_policy(docs / POLICY_NAME)
    scratch = data / "scratch"
    prepared = scratch / HISTORICAL_PREP
    historic = prepared / "preparation.json"
    rows, previous = sim.load_inputs(prepared, policy["historical_fp32_preparation_sha256"],
                                     policy["historical_policy_sha256"])
    members = require_cohort(rows, policy)
    prior = previous["artifacts"]
    paths = {}
    for name, key in HISTORICAL.items():
        ptq.require(prior[name]["sha256"] == policy[key], "Historical binding mismatch: " + name)
        paths[key] = external_path(data, prior[name]["path"])
    study = scratch / CAMPAIGN
    paths.update({
        "historical_fp32_preparation_sha256": historic,
        "historical_policy_sha256": docs / "NASA_PTQ_ACCEPTANCE_V1.json",
        "acceptance_document_sha256": docs / policy["protocol"],
        "prospective_training_policy_sha256": docs / "NASA_RESNET18_FULL_PRECISION_TRAINING_V1.json",
        "bie_sha256": study / "resnet_neural16/kneron_flow/input.kdp720.scaled.bie",
        "quantization_receipt_sha256": study / "resnet_neural16/quantization.json",
        "effective_width_report_sha256": study / "resnet_neural16/kneron_flow/model_fx_report.json",
        "training_predictions_sha256": study / "resnet_readout_v2/paired.csv",
        "training_readout_sha256": study / "resnet_readout_v2/readout.json",
        "training_panel_sha256": scratch / SELECTION,
    })
    artifacts = {}
    for key, path in paths.items():
        ptq.require(path.is_file() and ptq.sha256(path) == policy[key], "Artifact hash mismatch: " + key)
        artifacts[key] = {"path": str(path), "sha256": policy[key], "bytes": path.stat().st_size}
    conversion = ptq.load_json(paths["quantization_receipt_sha256"])
    training = ptq.load_json(paths["training_readout_sha256"])
    verify_descriptor(training["output_descriptor"])
    ptq.require(training["status"] == "TRAINING_READOUT_COMPLETE" and training["bie"]["sha256"] == BIE_SHA
                and training["paired_sha256"] == policy["training_predictions_sha256"]
                and training["panel_sha256"] == policy["training_panel_sha256"]
                and conversion["status"] == "TRAINING_BIE_GENERATED" and conversion["bie"]["sha256"] == BIE_SHA
                and conversion["image_id"] == policy["image_id"]
                and conversion["precision"]["report_sha256"] == policy["effective_width_report_sha256"]
                and conversion["input"]["calibration_manifest"]["sha256"] == policy["calibration_manifest_sha256"],
                "Training/conversion receipts disagree with frozen policy")
    manifest = ptq.checked_rows(paths["development_manifest_sha256"])
    require_cohort([r for r in manifest if r[policy["fold"]] == "test"], policy)
    require_cohort(ptq.checked_rows(paths["fp32_nuclei_sha256"]), policy)
    calibration = ptq.checked_rows(paths["calibration_manifest_sha256"])
    ptq.require(len(calibration) == 192 and all(r[policy["fold"]] == "train" for r in calibration)
                and not {r["nucleus_key"] for r in rows}.intersection(r["nucleus_key"] for r in calibration),
                "Calibration/development membership overlaps or is incomplete")
    for row in rows:
        sim.tensor(prepared, row)
    return dict(status="PREPARATION_COMPLETE", scope=policy["scope"], policy_commit=FREEZE,
                policy_sha256=POLICY_SHA, evaluated_nuclei=len(rows), error=None,
                historical_preparation=str(prepared), input_manifest_sha256=previous["input_manifest_sha256"],
                input_contract=previous["input_contract"], artifacts=artifacts,
                output_descriptor=training["output_descriptor"], **members,
                **{key: policy[key] for key in (
                    "architecture", "fold", "bie_sha256", "historical_fp32_preparation_sha256",
                    "source_onnx_sha256", "optimized_onnx_sha256", "training_run_sha256",
                    "development_manifest_sha256", "fp32_nuclei_sha256", "fp32_bags_sha256")})

