"""Recompute paired first-block effects from frozen, training-only ResNet18 suffix traces."""

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


SELECTION = "fecec721ddf8ad424ed9f5ac13594a0488d2fe6805aa0031aff11c18a602113a"
TRACE = "9ca9b42c99a52d801b8aaccc9942bfe4ee10c0bf061e17a224983b956b920e06"
CALIBRATION = "79aef81e921a723187d73cef5151ff9b943ad486d7e83af19daf063a5b5a1489"
SOURCES = {"BALBCM1", "BALBCM2", "C57BLM1", "C57BLM3"}


def checked_bytes(path, digest):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"Changed training evidence: {path.name}")
    return content


def source_summary(selection, trace, calibration):
    selected = {r["sample_id"]: r for r in selection["rows"]}
    observed = {r["sample_id"]: r for r in trace["rows"]}
    calibrated = {r["sample_id"]: r for r in calibration}
    if (selection["manifest_sha256"] != CALIBRATION or trace["panel_sha256"] != SELECTION
            or trace["calibration_manifest_sha256"] != CALIBRATION or len(calibrated) != 192
            or len(selected) != 48 or len(observed) != 48 or selected.keys() != observed.keys()
            or len(selected) != len(selection["rows"]) or len(observed) != len(trace["rows"])):
        raise ValueError("Unexpected calibration or training panel")
    samples = defaultdict(list)
    for key, sel in selected.items():
        ref, row = calibrated[key], observed[key]
        if (any(sel[k] != ref[k] or sel[k] != row[k] for k in
                ("source_name", "sample_name", "array_sha256", "particle_type", "dose_Gy", "hr_post_exposure"))
                or ref["fold_female_role"] != "train" or ref["cohort"] != "development"
                or ref["source_name"] not in SOURCES):
            raise ValueError("Training membership mismatch")
        if key == trace["historical_pilot"]:
            continue
        baseline, adjusted = row["opt0_suffix_head"], row["opt1_suffix_head"]
        stem = baseline["stem_pool"] - row["fp32_head"]
        first = baseline["stage1_block0"] - baseline["stem_pool"]
        difference = adjusted["stage1_block0"] - adjusted["stem_pool"] - first
        if not all(map(math.isfinite, (stem, first, difference))):
            raise ValueError("Nonfinite head-equivalent effect")
        samples[ref["source_name"]].append((stem, first, difference))
    if {s for s in samples} != SOURCES or sum(map(len, samples.values())) != 47:
        raise ValueError("Expected all four training sources after excluding the historical pilot")
    result = []
    for source, values in sorted(samples.items()):
        stem, first, difference = zip(*values)
        result.append(dict(source=source, nuclei=len(values), mean_stem_shift=mean(stem),
                           mean_baseline_block0_increment=mean(first),
                           baseline_block0_negative=sum(v < 0 for v in first),
                           mean_opt1_minus_opt0_block0_increment=mean(difference),
                           opt1_minus_opt0_block0_negative=sum(v < 0 for v in difference)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "trace", "calibration", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    selected = json.loads(checked_bytes(args.selection, SELECTION))
    trace = json.loads(checked_bytes(args.trace, TRACE))
    rows = csv.DictReader(io.StringIO(checked_bytes(args.calibration, CALIBRATION).decode("utf-8-sig")))
    result = dict(selection_sha256=SELECTION, stage_trace_sha256=TRACE, calibration_sha256=CALIBRATION,
                  scope="training_only_archived_fp32_suffix_replay", excluded_pilot=trace["historical_pilot"],
                  units="change in pre-ReLU scalar head output, foci/nucleus", by_source=source_summary(selected, trace, rows))
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
