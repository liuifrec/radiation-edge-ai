"""Score a completed source-held-out development fold under a prospective policy.

The caller supplies the policy SHA256 recorded before simulator readout. Its JSON
object identifies architecture, fold, training_run_sha256, source_onnx_sha256,
bie_sha256, development_manifest_sha256, fp32_nuclei_sha256 and fp32_bags_sha256.
Its maximums object maps decision metrics in METRICS to inclusive upper limits.
The BIE producer's JSON receipt repeats architecture, fold and the four model /
cohort identities. A completed simulator receipt has status BIE_SIMULATION_COMPLETE,
evaluated_nuclei 2400, and binds policy_sha256 and predictions_sha256. Its CSV
contains fold, sample_id, nucleus_key, sample_name, source_name and bie_burden.
The two FP32 tables are the frozen training outputs. No individual NASA label
is used. This scorer does not execute a model or authorize compilation.

Acceptance v1 uses the hash-bound acceptance document, host preparation receipt
and its prepared-input CSV. In v1 output, fp32 is the original ONNX prediction
on the same NCHW tensor as optimized_fp32 and bie. frozen_training_fp32 is the
previously recorded training-framework prediction.
"""
import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .multimodel_ptq import V1_POLICY_SHA, checked_rows, load_json, require, save_json, sha256
from .multimodel_ptq import FIXED_SIM_API, FIXED_SIM_SHA, IMAGE_ID


IDENTITIES = ("training_run_sha256", "source_onnx_sha256", "bie_sha256",
              "development_manifest_sha256")
METRICS = {"bag_paired_mae", "contrast_paired_mae", "bag_abs_bias",
           "contrast_abs_bias", "bag_mae_increase", "contrast_mae_increase",
           "lost_correct_contrast_directions", "changed_contrast_directions"}
V1_LIMITS = {"bag_paired_mae": 0.05, "bag_paired_max_abs": 0.15,
             "contrast_paired_mae": 0.10, "contrast_paired_max_abs": 0.20,
             "bag_mae_increase": 0.05, "strong_effect_sign_reversals": 0,
             "four_hour_nonpositive": 0, "incomplete_or_nonfinite_panels": 0}
METADATA = ("source_name", "strain", "sex", "particle_type", "dose_Gy", "hr_post_exposure")
EXPOSED_DOSES = {"Fe": 0.82, "X-ray": 1.0}
FOLDS = {"fold_female_role", "fold_male_1_role", "fold_male_2_role"}


def read_rows(path):
    return checked_rows(path)


def index_rows(rows, key):
    indexed = {row[key]: row for row in rows}
    require(len(indexed) == len(rows) and all(indexed), "Duplicate or empty " + key)
    return indexed


def number(value):
    result = float(value)
    require(math.isfinite(result), "Nonfinite prediction, target or limit")
    return result


def sign(value):
    return (value > 0) - (value < 0)


def error_metrics(rows, left, right):
    errors = [number(r[right] - r[left]) for r in rows]
    return {"n": len(errors), "bias": mean(errors), "mae": mean(map(abs, errors)),
            "rmse": number(math.sqrt(mean(x*x for x in errors))), "max_abs": max(map(abs, errors))}


def score_rows(fp32, bie, reference_bags, manifest, fold, prepared=None, strong_effect_min=0.20):
    refs = index_rows(fp32, "sample_id")
    fixed = index_rows(bie, "sample_id")
    reference = index_rows(reference_bags, "sample_name")
    require(refs.keys() == fixed.keys() and len(refs) == 2400, "Incomplete paired nucleus IDs")
    inputs = index_rows(prepared, "sample_id") if prepared is not None else None
    if inputs is not None:
        require(inputs.keys() == refs.keys(), "Prepared tensor membership differs from FP32")
    selected = index_rows([r for r in manifest if r[fold] == "test"], "sample_id")
    require(selected.keys() == refs.keys(), "Pairs differ from frozen development test membership")
    require(len(index_rows(fp32, "nucleus_key")) == 2400, "Incomplete nucleus keys")
    require(len(reference) == 24, "Expected 24 reference bags")
    nuclei, groups = [], defaultdict(list)
    for key, original in refs.items():
        candidate = fixed[key]
        source = selected[key]
        bag = reference[original["sample_name"]]
        require(original["fold"] == candidate["fold"] == bag["fold"] == fold, "Fold mismatch")
        require(all(original[k] == candidate[k] for k in ("nucleus_key", "sample_name", "source_name")),
                "Nucleus identity mismatch")
        require(all(original[k] == bag[k] for k in ("source_name", "strain", "sex", "particle_type")) and
                all(number(original[k]) == number(bag[k]) for k in ("dose_Gy", "hr_post_exposure")),
                "Bag metadata differs from its member nucleus")
        require(source["cohort"] == "development" and
                all(original[k] == source[k] for k in
                    ("nucleus_key", "sample_name", "source_name", "strain", "sex", "particle_type")) and
                all(number(original[k]) == number(source[k]) for k in ("dose_Gy", "hr_post_exposure")),
                "Nucleus differs from frozen development manifest")
        frozen = number(original["predicted_continuous_burden"])
        a = frozen
        input_fields = {}
        if inputs is not None:
            tensor = inputs[key]
            require(all(tensor[k] == original[k] for k in
                        ("fold", "sample_name", "source_name", "nucleus_key")),
                    "Prepared tensor identity mismatch: " + key)
            require(re.fullmatch(r"[0-9a-f]{64}", tensor["array_sha256"]) and
                    tensor["array_sha256"] == candidate["array_sha256"],
                    "The FP32/BIE input tensor hashes differ: " + key)
            a = number(tensor["original_fp32"])
            input_fields = dict(array_sha256=tensor["array_sha256"], frozen_training_fp32=frozen,
                                optimized_fp32=number(tensor["optimized_fp32"]))
        b = number(candidate["bie_burden"])
        row = dict(sample_id=key, nucleus_key=original["nucleus_key"], sample_name=bag["sample_name"],
                   source_name=bag["source_name"], fp32=a, bie=b, bie_minus_fp32=b-a, **input_fields)
        groups[bag["sample_name"]].append(row)
        nuclei.append(row)
    bags, pairs = [], defaultdict(dict)
    for name, original in reference.items():
        members = groups[name]
        require(len(members) == int(original["n_development_nuclei"]) == 100, "Incomplete bag")
        fp = mean(r["fp32"] for r in members)
        frozen = mean(r.get("frozen_training_fp32", r["fp32"]) for r in members)
        require(math.isclose(frozen, number(original["predicted_bag_mean"]), rel_tol=1e-12, abs_tol=1e-12),
                "Frozen FP32 bag mean disagrees with membership")
        b = mean(r["bie"] for r in members)
        bag = dict({k: original[k] for k in METADATA}, sample_name=name, n_nuclei=100,
                   target=number(original["nasa_avg_nfoci"]), fp32=fp, bie=b, bie_minus_fp32=b-fp)
        if inputs is not None:
            bag.update(frozen_training_fp32=frozen, optimized_fp32=mean(r["optimized_fp32"] for r in members))
        bags.append(bag)
        branch = bag["particle_type"]
        dose = number(bag["dose_Gy"])
        hour = number(bag["hr_post_exposure"])
        require(branch in EXPOSED_DOSES and dose in (0, EXPOSED_DOSES[branch]) and hour in (4, 24, 48),
                "Unexpected matched contrast condition")
        pair = pairs[(bag["source_name"], branch, hour)]
        require(dose not in pair, "Duplicate source/branch/time/dose")
        pair[dose] = bag
    require(set(Counter(b["source_name"] for b in bags).values()) == {12} and len(pairs) == 12,
            "Expected two source-held-out panels of six matched contrasts each")
    contrasts = []
    for (source, branch, hour), pair in sorted(pairs.items()):
        require(set(pair) == {0, EXPOSED_DOSES[branch]}, "Missing matched sham/exposed bag")
        sham, exposed = pair[0], pair[EXPOSED_DOSES[branch]]
        backends = ["target", "fp32", "bie"]
        if inputs is not None:
            backends.extend(["frozen_training_fp32", "optimized_fp32"])
        values = {k: exposed[k] - sham[k] for k in backends}
        contrasts.append(dict(values, source_name=source, particle_type=branch, hr_post_exposure=hour,
                              sham_sample_name=sham["sample_name"], exposed_sample_name=exposed["sample_name"],
                              bie_minus_fp32=values["bie"] - values["fp32"],
                              strong_effect_sign_reversal=(abs(values["fp32"]) >= strong_effect_min and
                                                           sign(values["fp32"]) * sign(values["bie"]) == -1),
                              four_hour_nonpositive=(hour == 4 and min(values["fp32"], values["bie"]) <= 0),
                              fp32_direction_correct=sign(values["target"]) == sign(values["fp32"]),
                              bie_direction_correct=sign(values["target"]) == sign(values["bie"])))
    summary = {"paired_nuclei": error_metrics(nuclei, "fp32", "bie")}
    metrics = {}
    for label, rows in (("bag", bags), ("contrast", contrasts)):
        paired = error_metrics(rows, "fp32", "bie")
        baseline = error_metrics(rows, "target", "fp32")
        candidate = error_metrics(rows, "target", "bie")
        summary[label] = {"paired_bie_minus_fp32": paired, "fp32_vs_nasa": baseline,
                          "bie_vs_nasa": candidate}
        metrics[label + "_paired_mae"] = paired["mae"]
        metrics[label + "_paired_max_abs"] = paired["max_abs"]
        metrics[label + "_abs_bias"] = abs(paired["bias"])
        metrics[label + "_mae_increase"] = candidate["mae"] - baseline["mae"]
    metrics["lost_correct_contrast_directions"] = sum(
        r["fp32_direction_correct"] and not r["bie_direction_correct"] for r in contrasts)
    metrics["changed_contrast_directions"] = sum(sign(r["fp32"]) != sign(r["bie"]) for r in contrasts)
    if inputs is not None:
        require(sum(r["hr_post_exposure"] == 4 for r in contrasts) == 4, "Incomplete four-hour panel")
        metrics.update(strong_effect_sign_reversals=sum(r["strong_effect_sign_reversal"] for r in contrasts),
                       four_hour_nonpositive=sum(r["four_hour_nonpositive"] for r in contrasts),
                       incomplete_or_nonfinite_panels=0)
        for stage, rows in (("nucleus", nuclei), ("bag", bags), ("contrast", contrasts)):
            summary[stage + "_backends"] = backend_summary(rows)
        summary["subgroups"] = subgroup_summary(bags, contrasts)
    return nuclei, bags, contrasts, summary, metrics


def backend_summary(rows):
    result = {
        "bie_minus_original_fp32": error_metrics(rows, "fp32", "bie"),
        "optimized_minus_original_fp32": error_metrics(rows, "fp32", "optimized_fp32"),
        "bie_minus_optimized_fp32": error_metrics(rows, "optimized_fp32", "bie"),
        "original_minus_frozen_training_fp32": error_metrics(rows, "frozen_training_fp32", "fp32"),
    }
    if "target" in rows[0]:
        result["against_nasa"] = {k: error_metrics(rows, "target", k) for k in ("fp32", "optimized_fp32", "bie")}
    if "bie_direction_correct" in rows[0]:
        result["directions"] = {
            "total": len(rows),
            "agree_with_nasa": {k: sum(sign(r["target"]) == sign(r[k]) for r in rows)
                                for k in ("fp32", "optimized_fp32", "bie")},
            "bie_agrees_with_fp32": sum(sign(r["fp32"]) == sign(r["bie"]) for r in rows),
            "bie_zero": sum(r["bie"] == 0 for r in rows),
            "strong_effect_sign_reversals": sum(r["strong_effect_sign_reversal"] for r in rows),
        }
    return result


def subgroup_summary(bags, contrasts):
    result = []
    for fields in (("source_name",), ("particle_type",), ("hr_post_exposure",),
                   ("particle_type", "hr_post_exposure")):
        groups = defaultdict(lambda: {"bag": [], "contrast": []})
        for stage, rows in (("bag", bags), ("contrast", contrasts)):
            for row in rows:
                key = tuple(number(row[f]) if f == "hr_post_exposure" else row[f] for f in fields)
                groups[key][stage].append(row)
        for key, stages in sorted(groups.items()):
            result.append({"group": dict(zip(fields, key)), **{
                stage: backend_summary(rows) for stage, rows in stages.items()}})
    return result


def verify_targets(reference_path, bags, fold):
    members = index_rows(bags, "sample_name")
    references = index_rows([r for r in read_rows(reference_path) if r["sample_name"] in members],
                            "sample_name")
    require(members.keys() == references.keys(), "Missing frozen NASA sample reference")
    for name, bag in members.items():
        ref = references[name]
        require(ref[fold] == "test" and ref["source_name"] == bag["source_name"] and
                ref["particle_type"] == bag["particle_type"] and
                all(number(ref[k]) == number(bag[k]) for k in ("dose_Gy", "hr_post_exposure", "nasa_avg_nfoci")),
                "Frozen NASA sample reference mismatch: " + name)


def affected_rows(bags, contrasts, limits):
    keys = ("source_name", "particle_type", "hr_post_exposure")
    return {
        "bags_above_max_abs": [
            {k: r[k] for k in (*keys, "sample_name", "dose_Gy", "bie_minus_fp32")}
            for r in bags if abs(r["bie_minus_fp32"]) > limits["bag_paired_max_abs"]],
        "contrasts_above_max_abs_or_violating_direction": [
            {k: r[k] for k in (*keys, "sham_sample_name", "exposed_sample_name", "fp32", "bie",
                               "bie_minus_fp32", "strong_effect_sign_reversal", "four_hour_nonpositive")}
            for r in contrasts if abs(r["bie_minus_fp32"]) > limits["contrast_paired_max_abs"]
            or r["strong_effect_sign_reversal"] or r["four_hour_nonpositive"]],
    }


def compare(args):
    require(sha256(args.policy) == args.policy_sha256, "Prospective policy SHA256 mismatch")
    policy = load_json(args.policy)
    v1 = policy.get("acceptance_version") == 1
    require("acceptance_version" not in policy or v1, "Unknown acceptance version")
    require(policy.get("fold") in FOLDS and policy.get("architecture") in {"resnet18", "mobilenet_v2"}
            and (policy["architecture"], policy["fold"]) != ("mobilenet_v2", "fold_male_1_role"),
            "Architecture/fold lacks a passing trained export and screen")
    limits = policy["maximums"]
    if v1:
        require(args.policy_sha256 == V1_POLICY_SHA, "Unrecognized prospective V1 policy identity")
        require(limits == V1_LIMITS and policy.get("strong_effect_min_abs_fp32_delta") == 0.20,
                "Incomplete or changed V1 acceptance declaration")
        require(all(getattr(args, k, None) for k in
                    ("acceptance_document", "prepared_inputs", "preparation_receipt", "nasa_reference")),
                "V1 requires the declared document, prepared tensors manifest and preparation receipt")
        require(sha256(args.acceptance_document) == policy["acceptance_document_sha256"],
                "Acceptance document SHA256 mismatch")
        require(sha256(args.nasa_reference) == policy["nasa_reference_sha256"],
                "Frozen NASA reference SHA256 mismatch")
    else:
        require(isinstance(limits, dict) and limits and set(limits) <= METRICS, "Invalid policy maximums")
    for name, value in limits.items():
        require(type(value) in (int, float), "Non-numeric policy maximum: " + name)
        number(value)
    receipt = load_json(args.bie_receipt)
    require(receipt.get("policy_sha256") == args.policy_sha256, "Simulator did not record this policy")
    identities = IDENTITIES + (("optimized_onnx_sha256",) if v1 else ())
    for key in identities + ("fp32_nuclei_sha256", "fp32_bags_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", str(policy.get(key, ""))), "Invalid identity: " + key)
    for key in identities + ("fold", "architecture"):
        require(receipt[key] == policy[key], "BIE receipt mismatch: " + key)
    require(sha256(args.development_manifest) == policy["development_manifest_sha256"] and
            sha256(args.fp32_nuclei) == policy["fp32_nuclei_sha256"] and
            sha256(args.fp32_bags) == policy["fp32_bags_sha256"] and
            sha256(args.bie_nuclei) == receipt["predictions_sha256"], "Prediction table SHA256 mismatch")
    provenance = {}
    if v1:
        require(receipt.get("image_id") == IMAGE_ID and receipt.get("platform") == 720
                and receipt.get("sdk_api") == FIXED_SIM_API and receipt.get("sdk_module_sha256") == FIXED_SIM_SHA
                and all(receipt.get(key) is False for key in ("calibrated", "physical_execution", "nef_compiled")),
                "Expected the pinned KL720 fixed simulator receipt")
        preparation = load_json(args.preparation_receipt)
        require(preparation.get("status") == "PREPARATION_COMPLETE" and
                preparation.get("evaluated_nuclei") == 2400 and preparation.get("error") is None,
                "Incomplete preparation receipt")
        for key in identities + ("fold", "architecture", "policy_sha256"):
            require(preparation[key] == receipt[key], "Preparation identity mismatch: " + key)
        provenance = {"preparation_sha256": sha256(args.preparation_receipt),
                      "input_manifest_sha256": sha256(args.prepared_inputs)}
        require(all(receipt.get(k) == value for k, value in provenance.items()) and
                preparation["input_manifest_sha256"] == provenance["input_manifest_sha256"],
                "Simulator and FP32 preparation identities differ")
    record = {
        "scope": "source_held_out_development_fold", "fold": policy["fold"],
        "architecture": policy["architecture"], "policy": policy, "policy_sha256": args.policy_sha256,
        "bie_receipt_sha256": sha256(args.bie_receipt), "bie_predictions_sha256": receipt["predictions_sha256"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "evaluator_sha256": sha256(__file__),
        "nef_authorized": False, **provenance,
    }
    try:
        require(receipt.get("status") == "BIE_SIMULATION_COMPLETE" and
                receipt.get("evaluated_nuclei") == 2400 and receipt.get("error") is None,
                "A completed 2400-nucleus BIE simulation receipt is required")
        reference_bags = read_rows(args.fp32_bags)
        if v1:
            verify_targets(args.nasa_reference, reference_bags, policy["fold"])
        nuclei, bags, contrasts, summary, metrics = score_rows(
            read_rows(args.fp32_nuclei), read_rows(args.bie_nuclei), reference_bags,
            read_rows(args.development_manifest), policy["fold"],
            read_rows(args.prepared_inputs) if v1 else None)
    except (ValueError, KeyError, OverflowError) as exc:
        if not v1:
            raise
        decisions = {key: {"actual": None, "maximum": limit, "pass": None, "status": "UNEVALUABLE"}
                     for key, limit in limits.items()}
        decisions["incomplete_or_nonfinite_panels"].update(actual=1, status="FAIL", **{"pass": False})
        args.out.mkdir(parents=True, exist_ok=False)
        save_json(args.out / "comparison.json", dict(record, status="INVALID_PANEL", error=str(exc),
                  decisions=decisions, all_declared_limits_pass=False, eligible_for_nef_compilation=False))
        return False
    decisions = {k: {"actual": metrics[k], "maximum": limit, "pass": metrics[k] <= limit}
                 for k, limit in limits.items()}
    args.out.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for name, rows in (("nucleus_pairs.csv", nuclei), ("bag_pairs.csv", bags),
                       ("matched_contrasts.csv", contrasts)):
        path = args.out / name
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        outputs[name] = sha256(path)
    passed = all(result["pass"] for result in decisions.values())
    if v1:
        record["affected"] = affected_rows(bags, contrasts, limits)
    save_json(args.out / "comparison.json", dict(record, output_sha256=outputs, summary=summary,
              metrics=metrics, decisions=decisions, all_declared_limits_pass=passed,
              status="COMPARISON_COMPLETE", eligible_for_nef_compilation=bool(v1 and passed)))
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("policy", "development-manifest", "fp32-nuclei", "fp32-bags", "bie-receipt",
                "bie-nuclei", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    for arg in ("acceptance-document", "prepared-inputs", "preparation-receipt", "nasa-reference"):
        parser.add_argument("--" + arg, type=Path)
    return 0 if compare(parser.parse_args()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
