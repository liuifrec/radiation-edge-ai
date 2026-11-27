"""Seal a training-selected MobileNet binary for the next known development regression."""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from . import multimodel_qualification as mobile
from .multimodel_bie_comparison import V1_LIMITS
from .multimodel_ptq import load_json, require, sha256

TRAINING_POLICY = "9b50fefcb86e6b01df766c0506fc545b807ac2fc4f9303ebdf029d18c3744dd9"
TRAINING_PANEL = "534fffa2f104bf395952fe25ecc8708ca4b91bf032dc521b19298cf5fe35ebfe"
SOURCES = {"BALBCM1", "BALBCM2", "C57BLM1", "C57BLM3"}
RECIPES = {"datapath16", "neural16"}
FROZEN_FIELDS = tuple(mobile.MOBILE) + (
    "architecture", "fold", "model_id", "model_version", "image_id", "checkpoint_sha256",
    "calibration_manifest_sha256", "screen_sha256", "export_sha256", "nasa_reference_sha256",
    "cohort_membership_sha256", "contrast_membership_sha256", "fp32_preparation_sha256",
    "input_manifest_sha256", "evaluation_image_hashes_sha256", "sources", "nuclei", "bags",
    "contrasts", "training_evaluation_overlap", "maximums", "strong_effect_min_abs_fp32_delta",
)


def check_summary(summary):
    sources = summary["by_source"]
    require(summary["n"] == 24 and set(sources) == SOURCES and all(r["n"] == 6 for r in sources.values()),
            "Expected the complete confirmation panel")
    for s in [summary, *sources.values()]:
        require(math.isfinite(s["bias"]) and math.isfinite(s["mae"]) and s["mae"] >= 0,
                "Invalid confirmation errors")


def select_binary(selection):
    require(selection["status"] == "TRAINING_SELECTION_COMPLETE" and selection["nef_eligible"] is False
            and selection["physical_execution"] is False and selection["policy_sha256"] == TRAINING_POLICY
            and selection["panel_sha256"] == TRAINING_PANEL and set(selection["candidates"]) == RECIPES,
            "Unexpected training screen")
    base = selection["baseline"]["confirmation"]
    check_summary(base)
    passing = []
    for recipe, row in selection["candidates"].items():
        if row["status"] == "VENDOR_BLOCKED":
            require(bool(row["error"]), "Missing vendor failure")
            continue
        metrics = row["summary"]["confirmation"]
        check_summary(metrics)
        checks = dict(lower_mae=metrics["mae"] < base["mae"],
                      lower_absolute_bias=abs(metrics["bias"]) < abs(base["bias"]),
                      source_mae_nonincrease=all(metrics["by_source"][s]["mae"] <= base["by_source"][s]["mae"]
                                                 for s in SOURCES))
        passed = all(checks.values())
        require(checks == row["checks"] and row["status"] == ("SCREEN_PASS" if passed else "SCREEN_FAIL")
                and row["bie"]["bytes"] > 0, "Inconsistent candidate decision")
        if passed:
            passing.append((metrics["mae"], row["bie"]["bytes"], recipe))
    selected = min(passing)[2] if passing else None
    require(selected == selection["selected_configuration"], "Incorrect training selection")
    digest = selection["candidates"][selected]["bie"]["sha256"] if selected else None
    require(digest == selection["selected_bie_sha256"], "Selected BIE identity mismatch")
    return selected


def enrollment(original_policy, screen, screen_sha256, protocol, out):
    """Consume only training evidence and prior committed policy identities."""
    original = load_json(original_policy)
    mobile.verify_policy(original, sha256(original_policy))
    require(original["maximums"] == V1_LIMITS and original["strong_effect_min_abs_fp32_delta"] == .20,
            "Previous acceptance limits changed")
    require(sha256(screen) == screen_sha256, "Unexpected selection receipt")
    selection = load_json(screen)
    recipe = select_binary(selection)
    require(recipe is not None, "No training candidate warrants an engineering regression")
    directory = screen.parent / recipe
    readout = directory / "evaluation/readout.json"
    build = directory / "quantization.json"
    q, measured = load_json(build), load_json(readout)
    require(selection["receipt_sha256"][recipe] == sha256(readout)
            and measured["quantization_sha256"] == sha256(build)
            and q["status"] == "TRAINING_BIE_GENERATED"
            and measured["status"] == "TRAINING_READOUT_COMPLETE"
            and q["policy_sha256"] == measured["policy_sha256"] == TRAINING_POLICY
            and q["recipe"] == measured["recipe"] == recipe
            and measured["panel_sha256"] == TRAINING_PANEL
            and q["checkpoint_sha256"] == original["checkpoint_sha256"]
            and q["input"]["source_onnx"]["sha256"] == original["source_onnx_sha256"]
            and q["input"]["optimized_onnx"]["sha256"] == original["optimized_onnx_sha256"]
            and q["input"]["calibration_manifest"]["sha256"] == original["calibration_manifest_sha256"]
            and measured["summary"] == selection["candidates"][recipe]["summary"],
            "Training receipt chain does not identify the selected conversion")
    binary = (directory / q["bie"]["path"]).resolve()
    require(directory.resolve() in binary.parents and binary.suffix == ".bie"
            and sha256(binary) == q["bie"]["sha256"] == measured["bie"]["sha256"] == selection["selected_bie_sha256"]
            and binary.stat().st_size == q["bie"]["bytes"] == selection["candidates"][recipe]["bie"]["bytes"],
            "Training winner differs from the supplied BIE")
    precision = binary.parent / "model_fx_report.json"
    require(sha256(precision) == q["precision"]["report_sha256"]
            and sha256(readout.with_name("paired.csv")) == measured["paired_sha256"],
            "Changed vendor precision or training predictions")
    result = {k: original[k] for k in FROZEN_FIELDS}
    result.update(schema="nasa_precision_development_regression_v1", candidate=recipe,
                  stage="prospective_policy_before_development_readout",
                  scope="known_female_development_regression", historical_policy_sha256=mobile.POLICY_SHA,
                  acceptance_document_sha256=sha256(protocol), bie_sha256=sha256(binary),
                  prospective_training_policy_sha256=TRAINING_POLICY, training_panel_sha256=TRAINING_PANEL,
                  selection_receipt_sha256=screen_sha256, quantization_receipt_sha256=sha256(build),
                  effective_width_report_sha256=sha256(precision), training_readout_sha256=sha256(readout),
                  training_predictions_sha256=measured["paired_sha256"],
                  declared_utc=datetime.now(timezone.utc).isoformat(), nef_eligible_from_training=False)
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-policy", "screen", "protocol", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--screen-sha256", required=True)
    args = parser.parse_args()
    enrollment(args.original_policy, args.screen, args.screen_sha256, args.protocol, args.out)


if __name__ == "__main__":
    main()
