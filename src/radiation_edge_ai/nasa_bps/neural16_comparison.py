"""Score the single policy-bound ResNet18 neural16 known development regression."""
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from radiation_edge_ai.audit_utils import write_csv_rows

from . import multimodel_bie_comparison as scorer
from .deployment_cohort import require_cohort
from .multimodel_ptq import FIXED_SIM_API, FIXED_SIM_SHA, checked_rows, load_json, require, save_json, sha256
from .neural16_regression import POLICY_SHA, verify_descriptor
from .prepare_multimodel_bie_evaluation import recorded_data_path


TABLES = ("nucleus_pairs.csv", "bag_pairs.csv", "matched_contrasts.csv")


def frozen_inputs(data_root, prepared, policy):
    preparation = prepared / "preparation.json"
    require(sha256(preparation) == policy["historical_fp32_preparation_sha256"], "Changed historical preparation")
    initial = load_json(preparation)
    require(initial["status"] == "PREPARATION_COMPLETE" and initial["bie_sha256"] == policy["standard_bie_sha256"],
            "Unexpected source of historical preparation")
    path = prepared / "inputs.csv"
    require(sha256(path) == initial["input_manifest_sha256"], "Changed sealed input manifest")
    tables = {"inputs": checked_rows(path)}
    for name, field in (("fp32_nuclei", "fp32_nuclei_sha256"), ("fp32_bags", "fp32_bags_sha256"),
                        ("manifest", "development_manifest_sha256"), ("reference", "nasa_reference_sha256")):
        entry = initial["artifacts"][name]
        path = recorded_data_path(data_root, entry["path"])
        require(sha256(path) == entry["sha256"] == policy[field], "Changed reference table " + name)
        tables[name] = checked_rows(path)
        if name == "reference":
            scorer.verify_targets(path, tables["fp32_bags"], policy["fold"])
    selected = [r for r in tables["manifest"] if r[policy["fold"]] == "test"]
    for panel in (tables["inputs"], tables["fp32_nuclei"], selected):
        require_cohort(panel, policy)
    return initial, tables


def verify_simulation(run, policy):
    vendor = run / "vendor"
    receipt = load_json(vendor / "bie_receipt.json")
    require(receipt["status"] == "BIE_SIMULATION_COMPLETE" and receipt["error"] is None
            and receipt["evaluated_nuclei"] == policy["nuclei"], "Incomplete model execution")
    require(receipt["policy_sha256"] == POLICY_SHA and receipt["bie_sha256"] == policy["bie_sha256"]
            and receipt["architecture"] == policy["architecture"] and receipt["fold"] == policy["fold"]
            and receipt["image_id"] == policy["image_id"] and receipt["platform"] == 720
            and receipt["sdk_module_sha256"] == FIXED_SIM_SHA and receipt["sdk_api"] == FIXED_SIM_API
            and all(receipt[k] is False for k in ("physical_execution", "nef_compiled", "calibrated")),
            "Unexpected candidate or simulator")
    for key in ("cohort_membership_sha256", "contrast_membership_sha256"):
        require(receipt[key] == policy[key], "Readout membership mismatch: " + key)
    require(sha256(vendor / "bie_nuclei.csv") == receipt["predictions_sha256"]
            and sha256(run / "preparation.json") == receipt["preparation_sha256"],
            "Changed readout or campaign binding")
    verify_descriptor(receipt["output_descriptor"])
    require(receipt["public_transport_words_observed"] is False, "Unexpected transport claim in native BIE receipt")
    return receipt


def score(run, prepared, data_root, policy_path):
    out = run / "comparison"
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = dict(status="INVALID_PANEL", eligible_for_nef_compilation=False,
                  all_declared_limits_pass=False, scope="known_female_development_regression",
                  created_utc=datetime.now(timezone.utc).isoformat(), evaluator_sha256=sha256(__file__),
                  scorer_sha256=sha256(scorer.__file__), physical_execution=False)
    try:
        require(sha256(policy_path) == POLICY_SHA, "Unrecognized prospective neural16 policy")
        policy = load_json(policy_path)
        require(policy["maximums"] == scorer.V1_LIMITS and policy["strong_effect_min_abs_fp32_delta"] == 0.20,
                "Frozen acceptance limits differ from scorer")
        report.update(policy=policy, policy_sha256=POLICY_SHA,
                      decisions={key: {"actual": None, "maximum": limit, "pass": None, "status": "UNEVALUABLE"}
                                 for key, limit in policy["maximums"].items()})
        receipt = verify_simulation(run, policy)
        initial, tables = frozen_inputs(data_root, prepared, policy)
        require(initial["input_manifest_sha256"] == receipt["input_manifest_sha256"], "Different FP32/BIE inputs")
        pairs = checked_rows(run / "vendor/bie_nuclei.csv")
        nuclei, bags, contrasts, summary, metrics = scorer.score_rows(
            tables["fp32_nuclei"], pairs, tables["fp32_bags"], tables["manifest"], policy["fold"],
            tables["inputs"], policy["strong_effect_min_abs_fp32_delta"])
        for name, rows in zip(TABLES, (nuclei, bags, contrasts)):
            write_csv_rows(out / name, rows)
        decisions = {key: {"actual": metrics[key], "maximum": limit, "pass": metrics[key] <= limit}
                     for key, limit in policy["maximums"].items()}
        passed = all(row["pass"] for row in decisions.values())
        report.update(status="COMPARISON_COMPLETE", **{k: policy[k] for k in
                      ("architecture", "fold", "bie_sha256", "cohort_membership_sha256", "contrast_membership_sha256")},
                      decisions=decisions, summary=summary, metrics=metrics,
                      affected=scorer.affected_rows(bags, contrasts, policy["maximums"]),
                      eligible_for_nef_compilation=passed, all_declared_limits_pass=passed,
                      bie_receipt_sha256=sha256(run / "vendor/bie_receipt.json"),
                      bie_predictions_sha256=receipt["predictions_sha256"],
                      preparation_sha256=receipt["preparation_sha256"], input_manifest_sha256=receipt["input_manifest_sha256"],
                      output_sha256={name: sha256(out / name) for name in TABLES})
    except Exception as exc:
        report["error"] = repr(exc)
        if "decisions" in report:
            report["decisions"]["incomplete_or_nonfinite_panels"].update(actual=1, status="FAIL", **{"pass": False})
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        save_json(out / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "prepared", "data-root", "policy"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    return 0 if score(args.run, args.prepared, args.data_root, args.policy)["all_declared_limits_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

