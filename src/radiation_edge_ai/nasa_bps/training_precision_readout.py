"""Measure the frozen female-fold training panels with the KL720 BIE simulator."""

import argparse
import csv
import inspect
import shutil
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

if __package__:
    from . import multimodel_bie_inference as sim
    from . import multimodel_ptq as ptq
    from . import training_precision as precision
else:
    import multimodel_bie_inference as sim
    import multimodel_ptq as ptq
    import training_precision as precision


SOURCES = ("BALBCM1", "BALBCM2", "C57BLM1", "C57BLM3")


def panel_inputs(panel, manifest, pairs, policy=None):
    rows = panel["rows"]
    references = {r["sample_id"]: r for r in manifest}
    fp32 = {r["sample_id"]: r for r in pairs}
    if policy is not None and policy["architecture"] == "resnet18":
        ptq.require(
            Counter(r["source_name"] for r in rows) == {s: 12 for s in SOURCES}
            and len({r["sample_name"] for r in rows}) == 48
            and sum(r["sample_id"] == policy["pilot"] for r in rows) == 1,
            "Unexpected ResNet training bags or pilot",
        )
        rows = [
            dict(
                r,
                panel="pilot" if r["sample_id"] == policy["pilot"] else "confirmation",
                stratum="minimum_selection_rank_per_training_bag",
                source_fp32=float(fp32[r["sample_id"]]["source_fp32"]),
            )
            for r in rows
        ]
    else:
        ptq.require(
            Counter((r["source_name"], r["panel"]) for r in rows)
            == {(s, k): n for s in SOURCES for k, n in (("pilot", 4), ("confirmation", 6))},
            "Unexpected training or pilot identities",
        )
    ptq.require(len({r["sample_id"] for r in rows}) == len(rows), "Duplicate training nucleus")
    for r in rows:
        identity = references[r["sample_id"]]
        ptq.require(
            all(
                identity[k] == r[k]
                for k in ("source_name", "array_file", "array_sha256", "sample_name", "nucleus_key")
            )
            and identity[ptq.FOLD] == "train"
            and identity["calibration_fold"] == ptq.FOLD
            and identity["dataset_role"] == "development_train_ptq_calibration"
            and float(fp32[r["sample_id"]]["source_fp32"]) == r["source_fp32"],
            "Unverified training tensor or FP32 prediction",
        )
    return rows


def summarize(rows):
    def metrics(subset):
        errors = [float(r["error"]) for r in subset]
        if not errors:
            return {"n": 0}
        return {
            "n": len(errors),
            "bias": mean(errors),
            "mae": mean(map(abs, errors)),
            "max_abs": max(map(abs, errors)),
        }

    return {
        p: dict(
            metrics([r for r in rows if r["panel"] == p]),
            by_source={
                s: metrics([r for r in rows if r["panel"] == p and r["source_name"] == s])
                for s in SOURCES
            },
        )
        for p in ("pilot", "confirmation")
    }


def training_decision(standard, candidates):
    baseline = standard["summary"]["confirmation"]
    accepted, outcomes = [], {}
    ptq.require(set(candidates) == set(precision.RECIPES), "Missing bounded recipe")
    for key, candidate in candidates.items():
        if candidate["status"] == "BLOCKED":
            outcomes[key] = {"status": "VENDOR_BLOCKED", "error": candidate["error"]}
            continue
        ptq.require(
            candidate["status"] == "TRAINING_READOUT_COMPLETE",
            "Incomplete supported candidate readout",
        )
        score = candidate["summary"]["confirmation"]
        checks = {
            "lower_mae": score["mae"] < baseline["mae"],
            "lower_absolute_bias": abs(score["bias"]) < abs(baseline["bias"]),
            "source_mae_nonincrease": all(
                score["by_source"][s]["mae"] <= baseline["by_source"][s]["mae"] for s in SOURCES
            ),
        }
        if all(checks.values()):
            accepted.append((score["mae"], candidate["bie"]["bytes"], key))
        outcomes[key] = {
            "status": "SCREEN_PASS" if all(checks.values()) else "SCREEN_FAIL",
            "checks": checks,
            "summary": candidate["summary"],
            "bie": candidate["bie"],
        }
    selected = min(accepted)[2] if accepted else None
    return {
        "status": "TRAINING_SELECTION_COMPLETE",
        "selected_configuration": selected,
        "selected_bie_sha256": candidates[selected]["bie"]["sha256"] if selected else None,
        "baseline": standard["summary"],
        "candidates": outcomes,
        "nef_eligible": False,
        "physical_execution": False,
    }


def check_float_word(word, actual, descriptor):
    width = descriptor["width"]
    ptq.require(
        -(1 << (width - 1)) <= word < 1 << (width - 1), "Word exceeds effective signed width"
    )
    value = (
        np.float32(word) / np.float32(2 ** descriptor["radix"]) / np.float32(descriptor["scale"])
    )
    ptq.require(
        np.float32(format(float(value), ".8f")) == np.float32(actual),
        "Decoded word and vendor float disagree",
    )


def run(args):
    import ktc
    from sys_flow.inference import get_model_io, inference_dynasty_fx

    out, policy = args.out, ptq.load_json(args.policy)
    policy_sha = ptq.sha256(args.policy)
    allowed = {
        "mobilenet_v2": precision.POLICY_SHA256,
        "resnet18": "9cecad4e39979589ef211382810a0515ef81cc8ab30c079eb50e88e5782916aa",
    }
    ptq.require(
        policy_sha == allowed.get(policy["architecture"])
        and ptq.sha256(args.panel) == policy["panel_sha256"]
        and ptq.sha256(args.inputs / "calibration_manifest.csv")
        == policy["calibration_manifest_sha256"]
        and ptq.sha256(args.parity / "fp32_optimized_parity.json") == policy["parity_sha256"]
        and ptq.sha256(args.parity / "fp32_optimized_parity.csv")
        == ptq.load_json(args.parity / "fp32_optimized_parity.json")["predictions_sha256"],
        "Frozen policy, panel, calibration or training predictions changed",
    )
    if args.recipe == "baseline":
        expected = policy["standard_bie_sha256"]
    else:
        q = ptq.load_json(args.quantization)
        ptq.require(
            q["status"] == "TRAINING_BIE_GENERATED"
            and q["policy_sha256"] == policy_sha
            and q["overrides"] == policy["recipes"][args.recipe]
            and q["recipe"] == args.recipe
            and q["input"]["optimized_onnx"]["sha256"] == policy["optimized_sha256"],
            "PTQ binding mismatch",
        )
        expected = q["bie"]["sha256"]
    ptq.require(
        ptq.sha256(args.bie) == expected
        and ptq.sha256(inspect.getfile(inference_dynasty_fx)) == ptq.FIXED_SIM_SHA,
        "BIE or SDK changed",
    )
    out.mkdir(exist_ok=False)
    receipt = out / "readout.json"
    record = {
        "status": "STARTED",
        "recipe": args.recipe,
        "panel_sha256": policy["panel_sha256"],
        "policy_sha256": policy_sha,
        "architecture": policy["architecture"],
        "bie": ptq.identity(args.bie),
        "quantization_sha256": ptq.sha256(args.quantization) if args.quantization else None,
        "sdk_sha256": ptq.FIXED_SIM_SHA,
        "worker_sha256": ptq.sha256(__file__),
        "selection_role": "training_only",
    }
    started = time.monotonic()
    try:
        samples = panel_inputs(
            ptq.load_json(args.panel),
            ptq.checked_rows(args.inputs / "calibration_manifest.csv"),
            ptq.checked_rows(args.parity / "fp32_optimized_parity.csv"),
            policy,
        )
        descriptor = precision.scalar_descriptor(args.bie, out / "metadata")
        record["output_descriptor"] = descriptor
        ptq.save_json(receipt, record)
        all_rows = []
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model.bie"
            shutil.copyfile(str(args.bie), str(model))
            ptq.require(ptq.sha256(model) == expected, "Changed simulator model")
            schema = get_model_io(model, 720)
            ptq.require(
                schema[0] == ["input"]
                and schema[1] == ["burden"]
                and tuple(schema[2]["burden"]) == (1,),
                "Unexpected BIE tensor contract",
            )
            record["chunks"] = []
            for start in range(0, len(samples), 8):
                subset = samples[start : start + 8]
                arrays = [
                    ptq.read_calibration_array(
                        ptq.calibration_path(args.inputs, r["array_file"]), r["array_sha256"]
                    )
                    for r in subset
                ]
                run_dir = Path(temp) / str(start)
                run_dir.mkdir()
                log = out / (f"native_{start:03}.zip")
                began = time.monotonic()
                try:
                    values, words = sim.native_batch(
                        inference_dynasty_fx, model, schema, arrays, run_dir, 2
                    )
                finally:
                    with zipfile.ZipFile(str(log), "x", zipfile.ZIP_DEFLATED) as archive:
                        sim.capture_chunk(run_dir, archive)
                elapsed = time.monotonic() - began
                # Cross-check an ordered row from each batch using the public single-image API.
                public = ktc.kneron_inference(
                    [arrays[-1]],
                    input_names=["input"],
                    bie_file=str(model),
                    platform=720,
                    data_type="float",
                )
                ptq.require(
                    len(public) == 1
                    and public[0].shape == (1,)
                    and float(public[0][0]) == values[-1],
                    "Public/native SDK mismatch",
                )
                for row, value, word in zip(subset, values, words):
                    check_float_word(word, value, descriptor)
                    sample = {
                        k: row[k]
                        for k in (
                            "sample_id",
                            "source_name",
                            "sample_name",
                            "panel",
                            "stratum",
                            "array_sha256",
                            "source_fp32",
                        )
                    }
                    all_rows.append(
                        dict(
                            sample,
                            bie_burden=value,
                            fixed_word=word,
                            error=value - row["source_fp32"],
                        )
                    )
                record["chunks"].append(
                    {
                        "offset": start,
                        "count": len(subset),
                        "archive": log.name,
                        "archive_sha256": ptq.sha256(log),
                        "seconds": elapsed,
                    }
                )
                ptq.save_json(receipt, record)
        table = out / "paired.csv"
        with table.open("x", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        record.update(
            status="TRAINING_READOUT_COMPLETE",
            summary=summarize(all_rows),
            paired_sha256=ptq.sha256(table),
        )
    except (Exception, SystemExit) as exc:
        record.update(status="FAILED", error=repr(exc))
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        ptq.save_json(receipt, record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "parity", "panel", "policy", "bie", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--quantization", type=Path)
    parser.add_argument("--recipe", choices=["baseline"] + sorted(precision.RECIPES), required=True)
    run(parser.parse_args())
