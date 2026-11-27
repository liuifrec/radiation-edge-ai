"""Independently reconstruct the neural16 development regression directly from signed native jobs."""
import argparse
import csv
import io
import math
import re
import statistics
import struct
import time
import zipfile
from collections import defaultdict
from pathlib import Path

from .deployment_cohort import require_cohort
from .multimodel_ptq import checked_rows, load_json, require, save_json, sha256
from .neural16_comparison import POLICY_SHA, TABLES
from .prepare_multimodel_bie_evaluation import recorded_data_path


SCALAR = re.compile(r"res_kdp720_[^/]+/in_(\d{6})\.txt/mode_720_piano/layer_output_burden_(fx\.bin|fx\.txt|fl\.txt)")
SCALE = 1.3206359148025513


def check_descriptors(vendor, report, training):
    meta = vendor / "metadata"
    descriptor = report["output_descriptor"]
    require(descriptor == training["output_descriptor"] and report["public_transport_words_observed"] is False,
            "Readout uses a different scalar representation")
    for filename, expected in descriptor["metadata"].items():
        require(sha256(meta / filename) == expected, "Changed simulator descriptor " + filename)
    shape = load_json(meta / "SnrShapeInfo.json")["dp_info"]["burden"]
    require(shape["node_name"] == ["/Gather"] and shape["onnx_shape"] == [1] and shape["hw_shape"] == [1, 1, 1, 1],
            "Ambiguous scalar producer or geometry")
    native = load_json(meta / "radix.json")["/Gather"]
    require(native["output_datapath_bitwidth"] == [15] and native["output_datapath_radix"] == [[12]]
            and native["output_scale"] == [[SCALE]], "Unexpected internal scalar grid")
    public, = load_json(meta / "ioinfo.json")["output"]
    require(public == descriptor["transport"] and public["name"] == "burden" and public["bitw"] == 16
            and public["radix"] == [13] and public["scale"] == [SCALE] and public["data_format"] == "8W1C16B",
            "Unexpected transport descriptor")
    return descriptor


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def native_value(binary, code_text, decoded_text):
    require(len(binary) == 2, "Expected one little-endian signed 16-bit storage word")
    word, = struct.unpack("<h", binary)
    require(-(1 << 14) <= word < (1 << 14) and word == int(code_text), "Invalid signed 15-bit native code")
    value = fp32(fp32(word / 4096.0) / SCALE)
    # Dynasty emits eight fractional digits before its public reader parses binary32.
    text = decoded_text.decode("ascii").strip()
    require(re.fullmatch(r"-?\d+\.\d{8}", text) and fp32(float(text)) == fp32(float(format(value, ".8f"))),
            "Native float disagrees with signed radix-12 decode")
    return word, fp32(float(text))


def check_witnesses(vendor, report, training_rows):
    evidence = report["training_witnesses"]
    path = vendor / "training_order.zip"
    require(evidence["file"] == path.name and sha256(path) == evidence["sha256"], "Changed training order archive")
    controls = [training_rows[key] for key in evidence["sample_ids"]]
    singletons = [float(r["bie_burden"]) for r in controls]
    require(len(set(singletons)) == 2 and singletons == evidence["public_singletons"]
            and evidence["ordered_indices"] == [1, 0, 1], "Public singleton identity or order differs")
    with zipfile.ZipFile(path) as archive:
        for idx, which in enumerate(evidence["ordered_indices"]):
            base, = [n[:-6] for n in archive.namelist() if n.endswith(
                f"in_{idx:06d}.txt/mode_720_piano/layer_output_burden_fl.txt")]
            word, value = native_value(*(archive.read(base + key) for key in ("fx.bin", "fx.txt", "fl.txt")))
            require(value == singletons[which] == evidence["native_burdens"][idx]
                    and word == int(controls[which]["fixed_word"]) == evidence["native_codes"][idx],
                    "Independent training witness replay disagrees")
    return evidence["sha256"]


def read_archive(path, fixed, inputs):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        require(len(names) == len(archive.infolist()) and archive.testzip() is None, "Corrupted archive")
        extracted = list(csv.DictReader(io.StringIO(archive.read("predictions.csv").decode("utf-8"))))
        require(extracted == fixed and len(fixed) == len(inputs), "Archived predictions differ from joined CSV")
        jobs = list(csv.DictReader(io.StringIO(archive.read("jobs.tsv").decode("utf-8")), delimiter="\t"))
        require(sorted(int(r["Seq"]) for r in jobs) == list(range(1, len(inputs) + 1))
                and all(int(r["Exitval"]) == int(r["Signal"]) == 0 for r in jobs), "Incomplete native jobs")
        durations = [float(r["JobRuntime"]) for r in jobs]
        require(all(math.isfinite(x) and x >= 0 for x in durations), "Invalid native job timings")
        scalars = {}
        for name in names:
            match = SCALAR.fullmatch(name)
            if match:
                key = int(match[1]), match[2]
                require(key not in scalars, "Ambiguous scalar output")
                scalars[key] = name
        require(len(scalars) == 3 * len(inputs), "Incomplete binary and text scalar set")
        records = []
        for idx, (row, inp) in enumerate(zip(fixed, inputs)):
            require(all(row[k] == inp[k] for k in ("sample_id", "sample_name", "source_name", "nucleus_key",
                                                    "fold", "array_file", "array_sha256"))
                    and row["chunk"] == path.stem and int(row["row_in_chunk"]) == idx,
                    "Native row disagrees with input order or identity")
            require(all(f"console/{idx+1}/{name}" in names for name in ("stdout", "stderr")), "Missing native console")
            code, burden = native_value(*(archive.read(scalars[idx, key]) for key in ("fx.bin", "fx.txt", "fl.txt")))
            dims = archive.read(scalars[idx, "fx.bin"][:-6] + "output_size.csv").split()
            require(dims == [b"1"] * 4, "Unexpected native scalar dimensions")
            require(code == int(row["bie_fixed_code"]) and burden == float(row["bie_burden"]), "Wrong recorded prediction")
            records.append((code, burden))
    return records, durations


def average(values):
    xs = list(values)
    return math.fsum(xs) / len(xs)


def close(actual, expected, field):
    require(math.isfinite(expected) and math.isclose(float(actual), expected, abs_tol=1e-13, rel_tol=1e-13),
            "Independent reconstruction differs for " + field)


def unique(rows, key):
    grouped = {row[key]: row for row in rows}
    require(len(grouped) == len(rows) and all(grouped), "Duplicate or blank " + key)
    return grouped


def build_means(inputs, observations, target_rows):
    require(len(inputs) == len(observations) == 2400, "Incomplete input observations")
    require(len(unique(inputs, "sample_id")) == len(unique(inputs, "nucleus_key")), "Ambiguous input identities")
    bags = defaultdict(list)
    for inp, (_, value) in zip(inputs, observations):
        bags[inp["sample_name"]].append((float(inp["original_fp32"]), value, float(inp["optimized_fp32"])))
    targets = unique([r for r in target_rows if r["sample_name"] in bags], "sample_name")
    require(len(targets) == len(bags) == 24, "Invalid reference set")
    means = {}
    sides = defaultdict(dict)
    for name, values in bags.items():
        require(len(values) == 100, "Incomplete source bag")
        fp, fixed, optimized = map(average, zip(*values))
        meta = targets[name]
        means[name] = dict(fp32=fp, bie=fixed, optimized_fp32=optimized, target=float(meta["nasa_avg_nfoci"]),
                           bie_minus_fp32=fixed - fp)
        key = (meta["source_name"], meta["particle_type"], float(meta["hr_post_exposure"]))
        dose = float(meta["dose_Gy"])
        require(dose not in sides[key], "Duplicate matched side")
        sides[key][dose] = name
    contrasts = {}
    for (source, particle, timepoint), pair in sides.items():
        dose = {"X-ray": 1., "Fe": .82}[particle]
        require(set(pair) == {0, dose}, "Missing sham or irradiated side")
        sham, exposed = pair[0], pair[dose]
        delta = {backend: means[exposed][backend] - means[sham][backend]
                 for backend in ("target", "fp32", "bie", "optimized_fp32")}
        delta.update(source_name=source, particle_type=particle, hr_post_exposure=timepoint, sham_sample_name=sham,
                     bie_minus_fp32=delta["bie"] - delta["fp32"])
        contrasts[exposed] = delta
    require(len(contrasts) == 12 and sum(r["hr_post_exposure"] == 4 for r in contrasts.values()) == 4,
            "Incomplete paired radiation conditions")
    return means, contrasts


def metrics(bags, contrasts, strong_effect_min):
    errors = [r["bie_minus_fp32"] for r in bags.values()]
    effects = [r["bie_minus_fp32"] for r in contrasts.values()]
    return {
        "bag_paired_mae": average(map(abs, errors)), "bag_paired_max_abs": max(map(abs, errors)),
        "contrast_paired_mae": average(map(abs, effects)), "contrast_paired_max_abs": max(map(abs, effects)),
        "bag_mae_increase": average(abs(r["bie"] - r["target"]) for r in bags.values())
                            - average(abs(r["fp32"] - r["target"]) for r in bags.values()),
        "strong_effect_sign_reversals": sum(abs(r["fp32"]) >= strong_effect_min and r["fp32"] * r["bie"] < 0
                                            for r in contrasts.values()),
        "four_hour_nonpositive": sum(r["hr_post_exposure"] == 4 and min(r["fp32"], r["bie"]) <= 0
                                    for r in contrasts.values()),
        "incomplete_or_nonfinite_panels": 0,
    }


def check_tables(out, official, inputs, observations, bags, contrasts):
    for name in TABLES:
        require(sha256(out / name) == official["output_sha256"][name], "Changed official table " + name)
    nuclei = unique(checked_rows(out / TABLES[0]), "sample_id")
    require(len(nuclei) == len(inputs), "Incomplete paired nucleus table")
    for inp, (_, val) in zip(inputs, observations):
        pair = nuclei[inp["sample_id"]]
        require(all(pair[k] == inp[k] for k in ("source_name", "sample_name", "nucleus_key", "array_sha256"))
                and val == float(pair["bie"]) and float(pair["fp32"]) == float(inp["original_fp32"])
                and float(pair["optimized_fp32"]) == float(inp["optimized_fp32"])
                and val - float(inp["original_fp32"]) == float(pair["bie_minus_fp32"]), "Wrong paired nucleus")
    for name, key, expected in ((TABLES[1], "sample_name", bags), (TABLES[2], "exposed_sample_name", contrasts)):
        rows = unique(checked_rows(out / name), key)
        require(rows.keys() == expected.keys(), "Membership disagreement in " + name)
        for identity, values in expected.items():
            for field, value in values.items():
                if isinstance(value, str):
                    require(rows[identity][field] == value, "Wrong radiation pair identity")
                else:
                    close(rows[identity][field], value, field)


def audit(run, prepared, data_root, policy_path, out=None):
    out = run / "independent_audit" if out is None else out
    out.mkdir(exist_ok=False)
    result = dict(status="AUDIT_FAILED", eligible_for_nef_compilation=False, program_sha256=sha256(__file__))
    started = time.monotonic()
    try:
        require(sha256(policy_path) == POLICY_SHA, "Unexpected policy")
        policy = load_json(policy_path)
        official_dir, vendor = run / "comparison", run / "vendor"
        official = load_json(official_dir / "comparison.json")
        sim = load_json(vendor / "bie_receipt.json")
        require(official["policy"] == policy and official["policy_sha256"] == sim["policy_sha256"] == POLICY_SHA
                and sim["bie_sha256"] == policy["bie_sha256"] and official["status"] == "COMPARISON_COMPLETE"
                and sim["status"] == "BIE_SIMULATION_COMPLETE" and sim["evaluated_nuclei"] == 2400,
                "Incomplete candidate comparison")
        require(sha256(vendor / "bie_receipt.json") == official["bie_receipt_sha256"]
                and sha256(vendor / "bie_nuclei.csv") == sim["predictions_sha256"] == official["bie_predictions_sha256"]
                and sha256(run / "preparation.json") == official["preparation_sha256"] == sim["preparation_sha256"]
                and sha256(prepared / "preparation.json") == policy["historical_fp32_preparation_sha256"],
                "Changed preparation, simulator or prediction identity")
        historical = load_json(prepared / "preparation.json")
        binding = load_json(run / "preparation.json")["artifacts"]
        for key in ("training_readout_sha256", "training_predictions_sha256", "bie_sha256"):
            require(sha256(Path(binding[key]["path"])) == binding[key]["sha256"] == policy[key],
                    "Changed BIE or original training witness evidence")
        descriptor = check_descriptors(vendor, sim, load_json(Path(binding["training_readout_sha256"]["path"])))
        training_rows = unique(checked_rows(Path(binding["training_predictions_sha256"]["path"])), "sample_id")
        witness_sha = check_witnesses(vendor, sim, training_rows)
        require(sha256(prepared / "inputs.csv") == historical["input_manifest_sha256"] == sim["input_manifest_sha256"],
                "Changed input table")
        inputs = checked_rows(prepared / "inputs.csv")
        cohort = require_cohort(inputs, policy)
        combined = checked_rows(vendor / "bie_nuclei.csv")
        require(len(inputs) == len(combined) == 2400, "Incomplete combined predictions")
        observations, durations, recorded = [], [], set()
        for chunk in sim["chunks"]:
            start, count = len(observations), chunk["count"]
            name = "chunks/{:04d}_{:04d}.zip".format(start, start + count - 1)
            path = vendor / name
            require(count > 0 and chunk["file"] == name and sha256(path) == chunk["sha256"], "Changed native archive")
            rows, timings = read_archive(path, combined[start:start+count], inputs[start:start+count])
            observations.extend(rows)
            durations.extend(timings)
            recorded.add(path.name)
        require(len(observations) == len(durations) == 2400 and recorded == {p.name for p in (vendor / "chunks").glob("*.zip")},
                "Missing or unrecorded native observations")
        for row in inputs:
            require(sha256(prepared / row["array_file"]) == row["array_sha256"], "Changed tensor " + row["sample_id"])
        reference = recorded_data_path(data_root, historical["artifacts"]["reference"]["path"])
        require(sha256(reference) == policy["nasa_reference_sha256"], "Changed scientific reference")
        bags, contrasts = build_means(inputs, observations, checked_rows(reference))
        check_tables(official_dir, official, inputs, observations, bags, contrasts)
        measured = metrics(bags, contrasts, policy["strong_effect_min_abs_fp32_delta"])
        require(set(measured) == set(official["decisions"]) == set(policy["maximums"]), "Missing or extra decision")
        decisions = {}
        for key, value in measured.items():
            limit = policy["maximums"][key]
            decisions[key] = value <= limit
            row = official["decisions"][key]
            for measurement in (row["actual"], official["metrics"][key]):
                close(measurement, value, key)
            require(row["maximum"] == limit and row["pass"] is decisions[key], "Gate does not match: " + key)
        passed = all(decisions.values())
        require(official["all_declared_limits_pass"] is passed and official["eligible_for_nef_compilation"] is passed,
                "Overall decision differs")
        words = [item[0] for item in observations]
        result.update(status="AUDIT_COMPLETE", policy_sha256=POLICY_SHA, bie_sha256=policy["bie_sha256"],
                      comparison_sha256=sha256(official_dir / "comparison.json"),
                      bie_receipt_sha256=sha256(vendor / "bie_receipt.json"), predictions_sha256=sim["predictions_sha256"],
                      input_manifest_sha256=sim["input_manifest_sha256"], gate_decisions=decisions, metrics=measured,
                      eligible_for_nef_compilation=passed, native_jobs=len(durations), verified_native_words=len(words),
                      signed_word_range=[min(words), max(words)], nonzero_words=sum(w != 0 for w in words),
                      observed_transport_words=0, archives=len(recorded), output_sha256=official["output_sha256"],
                      output_descriptor=descriptor, training_witness_sha256=witness_sha,
                      native_job_seconds={"mean": average(durations), "median": statistics.median(durations),
                                          "sum": math.fsum(durations), "min": min(durations), "max": max(durations)}, **cohort)
    except Exception as exc:
        result.update(error=repr(exc))
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        save_json(out / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "prepared", "data-root", "policy"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Fresh directory for an offline replay of the independent audit")
    args = parser.parse_args()
    audit(args.run, args.prepared, args.data_root, args.policy, args.out)


if __name__ == "__main__":
    main()

