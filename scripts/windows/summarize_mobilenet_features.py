"""Summarize authenticated MobileNet attribution rows and native-word range occupancy by panel."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from analyze_mobilenet_features import BOUNDARIES, write_table
from mobilenet_feature_capture import FILES, SOURCES, TAPS, checked, digest, read_json, save_json
from radiation_edge_ai.nasa_bps.feature_attribution import FeatureGraph, decode_layer


def replay_original_head(root, attribution, record, rows):
    """Execute the actual released FP32 ONNX suffix on each decoded native pool vector."""
    graph, shapes, radix = (
        checked(root / FILES[key][0], FILES[key][1]) for key in ("original", "shapes", "radix")
    )
    source = FeatureGraph(graph, TAPS)
    session = source.suffix("pool")
    shape = read_json(shapes)["dp_info"][TAPS["pool"]]
    desc = read_json(radix)[shape["node_name"][0]]
    results = []
    for row in rows:
        ident = row["sample_id"]
        capture = Path(record["capture_dir"]) / ident
        rec = read_json(checked(capture / "receipt.json", row["capture_receipt_sha256"]))
        with ZipFile(checked(capture / "native.zip", rec["native_sha256"])) as archive:
            pool = decode_layer(
                archive.read("pool_fx.bin"),
                archive.read("pool_fl.txt"),
                archive.read("pool_output_size.csv"),
                shape,
                desc,
            )["decoded"]
        with np.load(str(attribution / (ident + ".npz")), allow_pickle=False) as data:
            control = data["original_pool"]
        first, last = TAPS["pool"], TAPS["head"]
        replay = float(session.run([last], {first: control})[0].item())
        pre = float(session.run([last], {first: pool})[0].item())
        if (
            abs(replay - row["original_head"]) > 1e-5
            or abs(pre - row["floating_head_on_native_pool"]) > 1e-5
        ):
            raise ValueError(
                "Original FP32 suffix disagrees with its frozen control or analytic replay"
            )
        result = {
            k: row[k]
            for k in ("sample_id", "source_name", "input_sha256", "panel", "original_fp32", "bie")
        }
        results.append(
            dict(
                result,
                pooled_head_fp32=max(0.0, pre),
                pooled_preactivation_fp32=pre,
                pooled_preactivation_analytic=row["floating_head_on_native_pool"],
                fp32_control_abs_error=abs(replay - row["original_head"]),
            )
        )
    return results


def summarize(root, attribution, receipt_sha, out):
    r = read_json(checked(attribution / "receipt.json", receipt_sha))
    rows = read_json(checked(attribution / "nuclear_results.json", r["nuclear_results_sha256"]))
    checked(attribution / "cross_arch_pairs.csv", r["pairs_sha256"])
    if r["scope"] != "prespecified training-only attribution" or len(rows) != 40:
        raise ValueError("A complete, paired training readout is required")
    for row in rows:
        checked(attribution / (row["sample_id"] + ".npz"), row["feature_file_sha256"])
    out.mkdir()
    head_pairs = replay_original_head(root, attribution, r, rows)
    layer_rows, channel_rows, weighted_rows, increments = [], [], [], []
    for panel, expected in (("pilot", 16), ("confirmation", 24)):
        cohort = [row for row in rows if row["panel"] == panel]
        if len(cohort) != expected or len({r["sample_id"] for r in cohort}) != expected:
            raise ValueError("Wrong training panel")
        for source in ("all", *SOURCES):
            group = (
                cohort
                if source == "all"
                else [row for row in cohort if row["source_name"] == source]
            )
            endpoints = np.array([row["bie"] - row["original_fp32"] for row in group])
            previous = np.zeros(len(group))
            for k in BOUNDARIES:
                effect = np.array([row["suffix"][k]["effect"] for row in group])
                step = effect - previous
                increments.append(
                    dict(
                        panel=panel,
                        source=source,
                        boundary=k,
                        n=len(group),
                        mean_cumulative=effect.mean(),
                        mean_increment=step.mean(),
                        mean_abs_increment=abs(step).mean(),
                        negative_increments=int((step < 0).sum()),
                        terminal_signed_bias=endpoints.mean(),
                        terminal_mae=abs(endpoints).mean(),
                    )
                )
                previous = effect
        weighted = []
        channels = {k: [] for k in TAPS}
        for row in cohort:
            with np.load(
                str(attribution / (row["sample_id"] + ".npz")), allow_pickle=False
            ) as data:
                weighted.append(data["pooled_weighted_head_change"])
                for k in TAPS:
                    channels[k].append(data["channels_" + k])
        for k in TAPS:
            data = np.stack(channels[k])
            records = [row["layers"][k] for row in cohort]
            elements = sum(row["elements"] for row in records)
            counts = dict(
                (col, sum(row[col] for row in records))
                for col in ("at_low", "at_high", "fp_below", "fp_above")
            )
            means = data.mean(axis=0)
            observed, ref_above = (
                sum(1 for v in records if v["at_high"]),
                sum(1 for v in records if v["fp_above"]),
            )
            layer_rows.append(
                dict(
                    panel=panel,
                    boundary=k,
                    n=len(cohort),
                    channels=means.shape[1],
                    elements=elements,
                    width=records[0]["bits"],
                    mean_error=means[2].mean(),
                    mae=means[3].mean(),
                    fraction_at_high=counts["at_high"] / elements,
                    fraction_fp_above=counts["fp_above"] / elements,
                    nuclei_any_high=observed,
                    nuclei_any_fp_above=ref_above,
                    **counts,
                )
            )
            for c in range(means.shape[1]):
                channel_rows.append(
                    dict(
                        panel=panel,
                        boundary=k,
                        channel=c,
                        fp32_mean=means[0, c],
                        bie_mean=means[1, c],
                        bias=means[2, c],
                        mae=means[3, c],
                        at_low=int(data[:, 4, c].sum()),
                        at_high=int(data[:, 5, c].sum()),
                        fp_below=int(data[:, 6, c].sum()),
                        fp_above=int(data[:, 7, c].sum()),
                    )
                )
        w = np.stack(weighted)
        for c in range(w.shape[1]):
            weighted_rows.append(
                dict(
                    panel=panel,
                    channel=c,
                    mean_signed=float(w[:, c].mean()),
                    mae=float(np.abs(w[:, c]).mean()),
                )
            )
    outputs = {}
    for name, records in (
        ("layer_summary.csv", layer_rows),
        ("channel_summary.csv", channel_rows),
        ("pooled_weighted_errors.csv", weighted_rows),
        ("source_boundary_increments.csv", increments),
        ("cross_arch_pairs.csv", head_pairs),
    ):
        write_table(out / name, records)
        outputs[name] = digest(out / name)
    save_json(
        out / "receipt.json",
        dict(
            attribution_receipt_sha256=receipt_sha,
            source_sha256=digest(__file__),
            files_sha256=outputs,
            training_only=True,
        ),
    )
    print(out, digest(out / "receipt.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attribution", type=Path, required=True)
    p.add_argument("--receipt-sha", required=True)
    p.add_argument("--root", type=Path, default=Path("D:/radiation-edge-ai-data"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    summarize(args.root, args.attribution, args.receipt_sha, args.out)
