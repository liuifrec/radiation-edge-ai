"""Compare rectified head substitutions on authenticated female-fold training panels."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .multimodel_ptq import checked_rows, load_json, require, save_json, sha256


MANIFEST = "79aef81e921a723187d73cef5151ff9b943ad486d7e83af19daf063a5b5a1489"
PANELS = {
    "mobilenet_v2": "534fffa2f104bf395952fe25ecc8708ca4b91bf032dc521b19298cf5fe35ebfe",
    "resnet18": "fecec721ddf8ad424ed9f5ac13594a0488d2fe6805aa0031aff11c18a602113a",
}
SOURCES = {"BALBCM1", "BALBCM2", "C57BLM1", "C57BLM3"}
RESNET_PILOT = "BPSR1V2D_01501"
COMPONENTS = ("upstream", "head_and_output", "total")


def paired_records(architecture, csv_rows, selected, members, shared_panel=False):
    """Require the separately recorded membership, panel labels and input bytes."""
    require(architecture in PANELS, "Unrecognized architecture")
    selected_by_id = {r["sample_id"]: r for r in selected}
    require(len(selected_by_id) == len(selected), "Duplicate selected nucleus")
    results = {}
    for row in csv_rows:
        sid = row["sample_id"]
        require(sid not in results and sid in selected_by_id and sid in members,
                "Repeated, missing or unexpected training nucleus: " + sid)
        ref, chosen = members[sid], selected_by_id[sid]
        panel = chosen.get("panel") if architecture == "mobilenet_v2" or shared_panel else (
            "pilot" if sid == RESNET_PILOT else "confirmation")
        require(row["panel"] == panel and panel in {"pilot", "confirmation"}, "Panel identity mismatch")
        require(ref["source_name"] in SOURCES and ref["fold_female_role"] == "train"
                and ref["dataset_role"] == "development_train_ptq_calibration"
                and ref["array_sha256"] == chosen["array_sha256"] == row["input_sha256"]
                and ref["source_name"] == chosen["source_name"] == row["source_name"],
                "Input identity or training-source mismatch")
        fp, pooled, bie = (float(row[key]) for key in ("original_fp32", "pooled_head_fp32", "bie"))
        require(all(math.isfinite(x) and x >= 0 for x in (fp, pooled, bie)),
                "Expected three finite rectified burdens")
        results[sid] = dict(architecture=architecture, sample_id=sid, source_name=row["source_name"],
                            input_sha256=row["input_sha256"], panel=panel, original_fp32=fp,
                            pooled_head_fp32=pooled, bie=bie, upstream=pooled-fp,
                            head_and_output=bie-pooled, total=bie-fp)
    require(results.keys() == selected_by_id.keys(), "Incomplete training panel")
    return results


def describe(rows):
    if not rows:
        return {"n": 0}
    return {"n": len(rows), "original_fp32_zero": sum(r["original_fp32"] == 0 for r in rows),
            "bie_zero": sum(r["bie"] == 0 for r in rows),
            **{key: {"bias": mean(r[key] for r in rows), "mae": mean(abs(r[key]) for r in rows),
                     "max_abs": max(abs(r[key]) for r in rows)} for key in COMPONENTS}}


def summarize(records):
    # Only rows that are nonpilot in *both* experiments enter the matched comparison.
    common = sorted(set.intersection(*[set(v) for v in records.values()]))
    for sid in common:
        a, b = (model[sid] for model in records.values())
        require((a["source_name"], a["input_sha256"]) == (b["source_name"], b["input_sha256"]),
                "Common sample ID refers to different training inputs")
    nonpilot = [s for s in common if all(v[s]["panel"] == "confirmation" for v in records.values())]
    comparisons = {}
    for arch, indexed in records.items():
        rows = list(indexed.values())
        confirm = [r for r in rows if r["panel"] == "confirmation"]
        by_source = defaultdict(list)
        for row in confirm:
            by_source[row["source_name"]].append(row)
        comparisons[arch] = {
            "pilot": describe([r for r in rows if r["panel"] == "pilot"]),
            "confirmation": describe(confirm),
            "confirmation_by_source": {s: describe(v) for s, v in sorted(by_source.items())},
            "confirmation_fp32_zero": describe([r for r in confirm if r["original_fp32"] == 0]),
            "confirmation_fp32_positive": describe([r for r in confirm if r["original_fp32"] > 0]),
            "common_nonpilot": describe([indexed[s] for s in nonpilot]),
        }
    return dict(common_all_n=len(common), common_nonpilot_ids=nonpilot, summaries=comparisons)


def matched_panel(records, resnet_matched):
    """Check the four-source replication against each architecture's original panel."""
    mobile = {sid: r for sid, r in records["mobilenet_v2"].items() if r["panel"] == "confirmation"}
    require(mobile.keys() == resnet_matched.keys(), "Expected every sealed MobileNet confirmation row")
    shared = sorted(set(resnet_matched) & set(records["resnet18"]))
    for sid in shared:
        original, new = records["resnet18"][sid], resnet_matched[sid]
        require(all(original[k] == new[k] for k in ("input_sha256", "source_name", "original_fp32", "bie"))
                and math.isclose(original["pooled_head_fp32"], new["pooled_head_fp32"],
                                 abs_tol=1e-5, rel_tol=1e-4),
                "Matched ResNet trace does not reproduce archived endpoint or FP32 head replay")
    paired = summarize({"mobilenet_v2": mobile, "resnet18": resnet_matched})
    return dict(archival_crosscheck_ids=shared, results=paired["summaries"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--matched-resnet", type=Path, help="ResNet standard BIE on the MobileNet confirmation panel")
    parser.add_argument("--matched-resnet-sha256")
    for name in PANELS:
        parser.add_argument("--" + name + "-selection", type=Path, required=True)
        parser.add_argument("--" + name + "-pairs", type=Path, required=True)
        parser.add_argument("--" + name + "-pairs-sha256", required=True)
    args = parser.parse_args()
    require(sha256(args.manifest) == MANIFEST, "Training manifest SHA256 mismatch")
    training = {r["sample_id"]: r for r in checked_rows(args.manifest)}
    records, identities = {}, {}
    for arch, expected in PANELS.items():
        selection = getattr(args, arch + "_selection")
        pairs = getattr(args, arch + "_pairs")
        digest = getattr(args, arch + "_pairs_sha256")
        require(sha256(selection) == expected and sha256(pairs) == digest, "Changed training evidence")
        records[arch] = paired_records(arch, checked_rows(pairs), load_json(selection)["rows"], training)
        identities[arch] = dict(selection_sha256=expected, pairs_sha256=digest)
    summary = summarize(records)
    require(bool(args.matched_resnet) == bool(args.matched_resnet_sha256), "Expected a paired file and digest")
    matched = {}
    if args.matched_resnet:
        require(sha256(args.matched_resnet) == args.matched_resnet_sha256, "Changed matched training evidence")
        selection = load_json(args.mobilenet_v2_selection)["rows"]
        matched = paired_records("resnet18", checked_rows(args.matched_resnet),
                                 [r for r in selection if r["panel"] == "confirmation"],
                                 training, shared_panel=True)
        summary["matched_training_panel"] = matched_panel(records, matched)
        summary["matched_resnet_sha256"] = args.matched_resnet_sha256
    args.out.mkdir(parents=True, exist_ok=False)
    csv_path = args.out / "paired_attribution.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        rows = [r for model in records.values() for r in model.values()]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if matched:
        matched_path = args.out / "matched_attribution.csv"
        with matched_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            for sid in sorted(matched):
                writer.writerows((records["mobilenet_v2"][sid], matched[sid]))
        summary["matched_pairs_sha256"] = sha256(matched_path)
    save_json(args.out / "receipt.json", dict(scope="training_only", units="foci/nucleus",
              manifest_sha256=MANIFEST, inputs=identities, program_sha256=sha256(__file__),
              pairs_sha256=sha256(csv_path), **summary))


if __name__ == "__main__":
    main()
