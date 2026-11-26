import csv
import json
from argparse import Namespace

import pytest

from radiation_edge_ai.nasa_bps import multimodel_bie_comparison as compare


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def case(tmp_path):
    args = Namespace(**{k: tmp_path / k for k in
                        ("policy", "development_manifest", "fp32_nuclei", "fp32_bags",
                         "bie_nuclei", "bie_receipt", "out")})
    fp, fixed, bags, manifest = [], [], [], []
    fold = "fold_female_role"
    for source in ("A", "B"):
        for branch, exposed in (("Fe", 0.82), ("X-ray", 1)):
            for hour in (4, 24, 48):
                for dose in (0, exposed):
                    name = f"{source}/{branch}/{hour}/{dose}"
                    bag = dict(fold=fold, source_name=source, sample_name=name, strain=source, sex="Female",
                               particle_type=branch, dose_Gy=dose, hr_post_exposure=hour)
                    target = 3 if dose else 1
                    bags.append(dict(bag, n_development_nuclei=100, nasa_avg_nfoci=target,
                                     predicted_bag_mean=target))
                    for i in range(100):
                        key = f"{name}/{i}"
                        ids = dict(bag, sample_id=key, nucleus_key=key)
                        manifest.append(dict(ids, cohort="development", **{fold: "test"}))
                        fp.append(dict(ids, predicted_continuous_burden=target))
                        fixed.append(dict(ids, bie_burden=target + (0.25 if dose else 0)))
    for attr, rows in (("fp32_nuclei", fp), ("bie_nuclei", fixed[::-1]), ("fp32_bags", bags),
                       ("development_manifest", manifest)):
        write_csv(getattr(args, attr), rows)
    policy = dict(architecture="resnet18", fold=fold, source_onnx_sha256="1"*64,
                  training_run_sha256="2"*64, bie_sha256="3"*64,
                  maximums={"bag_paired_mae": 0.125, "contrast_mae_increase": 0.24,
                            "lost_correct_contrast_directions": 0})
    for key in ("development_manifest", "fp32_nuclei", "fp32_bags"):
        policy[key + "_sha256"] = compare.sha256(getattr(args, key))
    args.policy.write_text(json.dumps(policy), encoding="utf-8")
    args.policy_sha256 = compare.sha256(args.policy)
    receipt = {k: policy[k] for k in compare.IDENTITIES + ("architecture", "fold")}
    receipt.update(status="BIE_SIMULATION_COMPLETE", evaluated_nuclei=2400,
                   policy_sha256=args.policy_sha256, predictions_sha256=compare.sha256(args.bie_nuclei))
    args.bie_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    return args


def test_source_held_out_pairs_and_inclusive_limits(case):
    assert not compare.compare(case)
    report = compare.load_json(case.out / "comparison.json")
    assert not report["nef_authorized"]
    assert report["decisions"]["bag_paired_mae"] == {"actual": 0.125, "maximum": 0.125, "pass": True}
    assert report["decisions"]["contrast_mae_increase"]["pass"] is False
    assert report["summary"]["contrast"]["paired_bie_minus_fp32"]["bias"] == 0.25
    assert report["metrics"]["changed_contrast_directions"] == 0
    for name, count in (("nucleus_pairs.csv", 2400), ("bag_pairs.csv", 24), ("matched_contrasts.csv", 12)):
        assert len(compare.read_rows(case.out / name)) == count
        assert compare.sha256(case.out / name) == report["output_sha256"][name]
    with pytest.raises(FileExistsError):
        compare.compare(case)


@pytest.mark.parametrize("change,error", [("policy", "policy SHA256"), ("table", "table SHA256"),
                                          ("receipt", "record this policy")])
def test_identity_gate_precedes_readout(case, change, error):
    if change == "policy":
        case.policy_sha256 = "a"*64
    elif change == "receipt":
        receipt = compare.load_json(case.bie_receipt)
        receipt["policy_sha256"] = "a"*64
        case.bie_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    else:
        case.bie_nuclei.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        compare.compare(case)
    assert not case.out.exists()


def test_missing_nucleus_and_changed_fold_role(case):
    a, b, bags, manifest = (compare.read_rows(p) for p in
                            (case.fp32_nuclei, case.bie_nuclei, case.fp32_bags, case.development_manifest))
    with pytest.raises(ValueError, match="Incomplete paired"):
        compare.score_rows(a, b[:-1], bags, manifest, "fold_female_role")
    manifest[0]["fold_female_role"] = "train"
    with pytest.raises(ValueError, match="test membership"):
        compare.score_rows(a, b, bags, manifest, "fold_female_role")


@pytest.mark.parametrize("change", [{"status": "FAILED"}, {"evaluated_nuclei": 2399},
                                     {"error": "simulator exited with an error"}])
def test_failed_or_partial_simulator_receipt(case, change):
    receipt = compare.load_json(case.bie_receipt)
    receipt.update(change)
    case.bie_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="completed 2400-nucleus BIE simulation"):
        compare.compare(case)
    assert not case.out.exists()


def test_manifest_validates_experimental_condition_even_if_predictions_agree(case):
    fp, bie, bags, manifest = (compare.read_rows(p) for p in
                               (case.fp32_nuclei, case.bie_nuclei, case.fp32_bags, case.development_manifest))
    manifest[0]["dose_Gy"] = "0.00"
    bags[0]["hr_post_exposure"] = "4.0"
    compare.score_rows(fp, bie, bags, manifest, "fold_female_role")
    manifest[0]["particle_type"] = "X-ray"
    with pytest.raises(ValueError, match="differs from frozen development manifest"):
        compare.score_rows(fp, bie, bags, manifest, "fold_female_role")


@pytest.fixture
def v1(case, monkeypatch):
    case.acceptance_document = case.out.parent / "policy.md"
    case.prepared_inputs = case.out.parent / "inputs.csv"
    case.preparation_receipt = case.out.parent / "preparation.json"
    case.nasa_reference = case.out.parent / "reference.csv"
    write_csv(case.nasa_reference, [dict(r, fold_female_role="test") for r in compare.read_rows(case.fp32_bags)])
    case.acceptance_document.write_text("Synthetic prospective declaration", encoding="utf-8")
    policy = compare.load_json(case.policy)
    policy.update(acceptance_version=1, acceptance_document_sha256=compare.sha256(case.acceptance_document),
                  nasa_reference_sha256=compare.sha256(case.nasa_reference),
                  maximums=compare.V1_LIMITS, strong_effect_min_abs_fp32_delta=0.20,
                  optimized_onnx_sha256="4"*64)
    case.policy.write_text(json.dumps(policy), encoding="utf-8")
    case.policy_sha256 = compare.sha256(case.policy)
    monkeypatch.setattr(compare, "V1_POLICY_SHA", case.policy_sha256)
    original = compare.read_rows(case.fp32_nuclei)
    fixed = compare.index_rows(compare.read_rows(case.bie_nuclei), "sample_id")
    inputs = []
    for i, row in enumerate(original):
        fp = float(row["predicted_continuous_burden"]) + 0.125
        tensor_sha = f"{i:064x}"
        inputs.append(dict(row, array_sha256=tensor_sha, original_fp32=fp, optimized_fp32=fp - 0.001))
        fixed[row["sample_id"]].update(array_sha256=tensor_sha, bie_burden=fp)
    write_csv(case.prepared_inputs, inputs)
    write_csv(case.bie_nuclei, list(fixed.values()))
    receipt = compare.load_json(case.bie_receipt)
    receipt.update(optimized_onnx_sha256=policy["optimized_onnx_sha256"], policy_sha256=case.policy_sha256,
                   image_id=compare.IMAGE_ID, sdk_api=compare.FIXED_SIM_API, sdk_module_sha256=compare.FIXED_SIM_SHA,
                   platform=720, calibrated=False, physical_execution=False, nef_compiled=False,
                   predictions_sha256=compare.sha256(case.bie_nuclei),
                   input_manifest_sha256=compare.sha256(case.prepared_inputs))
    case.preparation_receipt.write_text(json.dumps(dict(receipt, status="PREPARATION_COMPLETE")), encoding="utf-8")
    receipt["preparation_sha256"] = compare.sha256(case.preparation_receipt)
    case.bie_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    return case


def test_v1_fresh_original_onnx_is_primary_and_reports_backend_subgroups(v1):
    assert compare.compare(v1)
    report = compare.load_json(v1.out / "comparison.json")
    assert set(report["decisions"]) == set(compare.V1_LIMITS)
    assert report["eligible_for_nef_compilation"] and not report["nef_authorized"]
    assert all(decision["actual"] == 0 and decision["pass"] for decision in report["decisions"].values())
    assert report["summary"]["bag_backends"]["original_minus_frozen_training_fp32"]["bias"] == 0.125
    assert report["summary"]["contrast"]["fp32_vs_nasa"]["mae"] == 0
    assert report["summary"]["nucleus_backends"]["optimized_minus_original_fp32"]["bias"] == pytest.approx(-0.001)
    four_hour = next(row for row in report["summary"]["subgroups"] if row["group"] == {"hr_post_exposure": 4})
    assert four_hour["contrast"]["directions"] == {
        "total": 4, "agree_with_nasa": {"fp32": 4, "optimized_fp32": 4, "bie": 4},
        "bie_agrees_with_fp32": 4, "bie_zero": 0, "strong_effect_sign_reversals": 0}
    assert len(report["summary"]["subgroups"]) == 2 + 2 + 3 + 6


def test_all_v1_error_margins_and_sign_rules(v1):
    prepared = compare.read_rows(v1.prepared_inputs)
    fixed = compare.index_rows(compare.read_rows(v1.bie_nuclei), "sample_id")
    for row in prepared:
        exposed = float(row["dose_Gy"]) > 0
        early = row["hr_post_exposure"] == "4"
        branch = row["particle_type"]
        # A strong reversed effect at exactly the cutoff, a strong zero, and weak reversals.
        delta = (0.20 if early else 0.199)
        row["original_fp32"] = delta if exposed else 0
        fixed[row["sample_id"]]["bie_burden"] = -1 if exposed and (branch == "Fe" or not early) else 0
    _, _, contrasts, summary, metrics = compare.score_rows(
        compare.read_rows(v1.fp32_nuclei), list(fixed.values()), compare.read_rows(v1.fp32_bags),
        compare.read_rows(v1.development_manifest), "fold_female_role", prepared)
    assert metrics["strong_effect_sign_reversals"] == 2
    assert metrics["four_hour_nonpositive"] == 4
    assert metrics["incomplete_or_nonfinite_panels"] == 0
    for metric in ("bag_paired_mae", "bag_paired_max_abs", "contrast_paired_mae",
                   "contrast_paired_max_abs", "bag_mae_increase"):
        assert metrics[metric] > compare.V1_LIMITS[metric]
    assert sum(r["bie"] == 0 for r in contrasts) == 2
    assert summary["contrast_backends"]["directions"]["agree_with_nasa"]["bie"] == 0


@pytest.mark.parametrize("problem", ["missing", "NaN", "tensor"])
def test_v1_incomplete_or_nonfinite_panel_returns_failure_receipt(v1, problem):
    fixed = compare.read_rows(v1.bie_nuclei)
    if problem == "missing":
        fixed.pop()
    elif problem == "NaN":
        fixed[0]["bie_burden"] = "nan"
    else:
        fixed[0]["array_sha256"] = "5" * 64
    write_csv(v1.bie_nuclei, fixed)
    receipt = compare.load_json(v1.bie_receipt)
    receipt.update(predictions_sha256=compare.sha256(v1.bie_nuclei))
    v1.bie_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    assert not compare.compare(v1)
    report = compare.load_json(v1.out / "comparison.json")
    assert report["status"] == "INVALID_PANEL"
    assert report["decisions"]["incomplete_or_nonfinite_panels"]["pass"] is False
    assert report["decisions"]["bag_paired_mae"]["pass"] is None
    assert not report["eligible_for_nef_compilation"]


def test_v1_requires_frozen_document(v1):
    v1.acceptance_document.write_text("Changed", encoding="utf-8")
    with pytest.raises(ValueError, match="document SHA256 mismatch"):
        compare.compare(v1)


def test_rehashed_policy_cannot_authorize_nef(v1):
    policy = compare.load_json(v1.policy)
    policy["declared_utc"] = "2099-01-01T00:00:00Z"
    v1.policy.write_text(json.dumps(policy))
    v1.policy_sha256 = compare.sha256(v1.policy)
    with pytest.raises(ValueError, match="prospective V1 policy identity"):
        compare.compare(v1)


@pytest.mark.parametrize("key,value", [("physical_execution", True), ("sdk_api", "CPU"),
                                        ("sdk_module_sha256", "a"*64), ("platform", 730)])
def test_v1_is_specific_to_actual_fixed_simulator_receipts(v1, key, value):
    receipt = compare.load_json(v1.bie_receipt)
    receipt[key] = value
    v1.bie_receipt.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="pinned KL720 fixed simulator"):
        compare.compare(v1)
