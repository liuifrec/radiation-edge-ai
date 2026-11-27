"""Check a frozen MobileNetV2 readout and optionally compile its exact BIE under Kneron 0.33.1."""

import argparse
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from . import multimodel_qualification as mobile
    from .multimodel_ptq import IMAGE_ID, load_json, require, sha256
else:
    import multimodel_qualification as mobile
    from multimodel_ptq import IMAGE_ID, load_json, require, sha256


def passing_metrics(comparison, limits):
    decisions = comparison.get("decisions", {})
    if set(decisions) != set(limits) or len(limits) != 8:
        raise ValueError("Expected eight independent engineering decisions")
    failures = []
    for key, limit in limits.items():
        row = decisions[key]
        actual = row.get("actual")
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or row.get("maximum") != limit or comparison["metrics"].get(key) != actual):
            raise ValueError("Invalid engineering measurement: " + key)
        passed = actual <= limit
        if row.get("pass") is not passed:
            raise ValueError("Inconsistent engineering decision: " + key)
        if not passed:
            failures.append(key)
    eligible = not failures
    if (comparison.get("all_declared_limits_pass") is not eligible
            or comparison.get("eligible_for_nef_compilation") is not eligible):
        raise ValueError("Inconsistent NEF eligibility")
    return failures


def check(args):
    policy, result, simulation = (load_json(p) for p in (args.policy, args.comparison, args.simulation))
    digest = sha256(args.policy)
    mobile.verify_policy(policy, digest)
    require(sha256(args.comparison) == args.comparison_sha256, "Unexpected comparison receipt")
    require(result.get("status") == "COMPARISON_COMPLETE" and result.get("policy") == policy
            and result.get("scope") == "source_held_out_development_fold", "Incomplete or unrelated decision")
    require(result["policy_sha256"] == simulation["policy_sha256"] == digest
            and result["bie_receipt_sha256"] == sha256(args.simulation)
            and result["bie_predictions_sha256"] == simulation["predictions_sha256"] == sha256(args.predictions)
            and sha256(args.bie) == simulation["bie_sha256"] == policy["bie_sha256"], "Changed BIE evidence")
    require(simulation["status"] == "BIE_SIMULATION_COMPLETE" and simulation["evaluated_nuclei"] == 2400
            and simulation["image_id"] == IMAGE_ID and simulation["physical_execution"] is False,
            "Unexpected simulator backend or panel")
    for field in ("fold", "architecture", "cohort_membership_sha256", "contrast_membership_sha256"):
        require(result[field] == simulation[field] == policy[field], "Cohort or model changed: " + field)
    outputs = result["output_sha256"]
    require(set(outputs) == {"nucleus_pairs.csv", "bag_pairs.csv", "matched_contrasts.csv"}, "Missing tables")
    for name, expected in outputs.items():
        require(sha256(args.comparison.parent / name) == expected, "Changed comparison table: " + name)
    return passing_metrics(result, policy["maximums"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("policy", "comparison", "simulation", "predictions", "bie"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--comparison-sha256", required=True)
    parser.add_argument("--compile-out", type=Path, help="Compile in the pinned SDK into a new directory")
    args = parser.parse_args()
    failures = check(args)
    print(json.dumps({"eligible_for_nef_compilation": not failures, "failed_criteria": failures}))
    if failures or args.compile_out is None:
        return 2 if failures else 0

    require(Path("/workspace/version.txt").read_text().strip() == "kneron/toolchain:v0.33.1",
            "Expected the pinned compiler image")
    import ktc

    out = args.compile_out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    model = ktc.ModelConfig(32781, "0001", "720", bie_path=str(args.bie.resolve()))
    vendor_output = Path(ktc.compile([model])).resolve()
    nef = out / "model.nef"
    require(vendor_output.is_file() and vendor_output.stat().st_size > 0, "Compiler produced no NEF")
    shutil.copyfile(str(vendor_output), str(nef))
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), architecture=mobile.ARCHITECTURE,
                   fold=mobile.FOLD, model_id=32781, model_version="0001", image_id=IMAGE_ID,
                   policy_sha256=sha256(args.policy), comparison_sha256=args.comparison_sha256,
                   bie_sha256=sha256(args.bie), nef_sha256=sha256(nef), bytes=nef.stat().st_size,
                   compile_seconds=time.perf_counter() - start, physical_execution=False)
    (out / "compile.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
