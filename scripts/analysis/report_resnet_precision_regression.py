"""Compare the original INT8 decision with the audited neural16 regression on identical development inputs."""
import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from radiation_edge_ai.audit_utils import write_csv_rows
from radiation_edge_ai.nasa_bps.multimodel_ptq import checked_rows, load_json, require, save_json, sha256
from radiation_edge_ai.nasa_bps.neural16_comparison import POLICY_SHA, TABLES


STANDARD_RECEIPT = "4f8efacd1f5bd3223deb08319f37d72ea2604029f0ddc29880d52483cef75146"


def compare_tables(standard, candidate, stage):
    keys = {"nucleus": "sample_id", "bag": "sample_name", "contrast": "exposed_sample_name"}
    shared = {
        "nucleus": ("sample_id", "nucleus_key", "array_sha256", "sample_name", "source_name"),
        "bag": ("sample_name", "source_name", "strain", "sex", "dose_Gy", "particle_type", "hr_post_exposure"),
        "contrast": ("sham_sample_name", "exposed_sample_name", "source_name", "particle_type", "hr_post_exposure"),
    }
    std = {r[keys[stage]]: r for r in standard}
    ids = {r[keys[stage]] for r in candidate}
    require(len(standard) == len(std) == len(ids) == len(candidate) and std.keys() == ids, "Mismatched " + stage)
    pairs = []
    for new in candidate:
        old = std[new[keys[stage]]]
        require(all(old[k] == new[k] for k in shared[stage]) and float(old["fp32"]) == float(new["fp32"])
                and float(old["optimized_fp32"]) == float(new["optimized_fp32"]), "Mismatched FP32 or " + stage)
        if stage != "nucleus":
            require(float(old["target"]) == float(new["target"]), "Changed NASA reference")
        row = {k: new[k] for k in shared[stage]}
        row.update(fp32=float(new["fp32"]), optimized_fp32=float(new["optimized_fp32"]))
        if stage != "nucleus":
            row["nasa_target"] = float(new["target"])
        for label, data in (("standard_int8", old), ("full_neural_int16", new)):
            row[label] = float(data["bie"])
            row[label + "_minus_fp32"] = row[label] - row["fp32"]
        pairs.append(row)
    return pairs


def source_stats(source, stage, pairs, backend):
    errors = [r[backend + "_minus_fp32"] for r in pairs]
    stats = dict(source=source, stage=stage, model=backend, n=len(pairs), mean_signed_shift=mean(errors),
                 mae=mean(map(abs, errors)), max_abs=max(map(abs, errors)), rmse=math.sqrt(mean(e*e for e in errors)))
    if stage != "nucleus":
        stats["mae_vs_nasa"] = mean(abs(r[backend] - r["nasa_target"]) for r in pairs)
    if stage == "contrast":
        stats["fp32_direction_agreement"] = sum(
            ((r[backend] > 0) - (r[backend] < 0)) == ((r["fp32"] > 0) - (r["fp32"] < 0)) for r in pairs)
    return stats


def plot(tables, out, sources, limits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    old_colour, new_colour = "#ac563b", "#087e83"
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 7.0), sharey=True)
    extent = max(.21, 1.08 * max(abs(r[k]) for stage in ("bag", "contrast") for r in tables[stage]
                                  for k in ("standard_int8_minus_fp32", "full_neural_int16_minus_fp32")))
    for col, source in enumerate(sources):
        ax = axes[col]
        by_y = {}
        for n, (particle, hour) in enumerate((p, h) for p in ("Fe", "X-ray") for h in (4, 24, 48)):
            match = [r for r in tables["bag"] if r["source_name"] == source and r["particle_type"] == particle
                     and float(r["hr_post_exposure"]) == hour]
            match.sort(key=lambda r: float(r["dose_Gy"]))
            contrast, = [r for r in tables["contrast"] if r["source_name"] == source and r["particle_type"] == particle
                         and float(r["hr_post_exposure"]) == hour]
            for j, r in enumerate([*match, contrast]):
                y = 4*n + j
                by_y[y] = f"{particle} {hour} h   " + ("sham", "exposed", "delta")[j]
                left, right = r["standard_int8_minus_fp32"], r["full_neural_int16_minus_fp32"]
                ax.plot([left, right], [y, y], color="#a8b2b6", lw=1.5, zorder=1)
                marker = "D" if j == 2 else "o"
                ax.scatter(left, y, c=old_colour, marker=marker, s=29, zorder=2)
                ax.scatter(right, y, c=new_colour, marker=marker, s=29, zorder=3)
        ax.axvline(0, color="#333333", lw=1, zorder=0)
        for x in (-limits["bag_paired_max_abs"], limits["bag_paired_max_abs"]):
            ax.axvline(x, color="#c9c9c9", lw=.8, ls=":", zorder=0)
        ax.set_title(source, loc="left", weight="bold")
        ax.set_xlabel("BIE minus FP32 (foci/nucleus)")
        ax.set_yticks(list(by_y))
        ax.set_yticklabels(list(by_y.values()), fontsize=8)
        ax.set_xlim(-extent, extent)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("ResNet18 precision on the known female development panel", x=.155, ha="left", weight="bold")
    legend = [Line2D([], [], ls="", marker="o", color=old_colour, label="Standard INT8"),
              Line2D([], [], ls="", marker="o", color=new_colour, label="Full neural INT16"),
              Line2D([], [], ls="", marker="D", color="#657176", label="Matched exposed minus sham"),
              Line2D([], [], ls=":", color="#888888", label="Bag maximum absolute shift limit")]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.17, right=.98, top=.88, bottom=.16, wspace=.10)
    fig.savefig(out / "paired_development_regression.pdf", bbox_inches="tight")
    fig.savefig(out / "paired_development_regression.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def report(run, historical, out):
    current = run / "comparison"
    verified = load_json(run / "independent_audit/verification.json")
    require(verified["status"] == "AUDIT_COMPLETE" and verified["policy_sha256"] == POLICY_SHA
            and verified["comparison_sha256"] == sha256(current / "comparison.json"), "Unverified neural16 comparison")
    require(sha256(historical / "comparison.json") == STANDARD_RECEIPT, "Historical INT8 comparison changed")
    previous = load_json(historical / "comparison.json")
    selected = load_json(current / "comparison.json")
    require(previous["policy"]["bie_sha256"] == selected["policy"]["standard_bie_sha256"]
            and previous["input_manifest_sha256"] == selected["input_manifest_sha256"]
            and previous["policy"]["maximums"] == selected["policy"]["maximums"], "Incompatible regression inputs")
    out.mkdir(exist_ok=False)
    tables, by_source = {}, []
    sources = selected["policy"]["sources"]
    for stage, name in zip(("nucleus", "bag", "contrast"), TABLES):
        for folder, sha in ((historical, previous["output_sha256"]), (current, verified["output_sha256"])):
            require(sha256(folder / name) == sha[name], "Changed paired table " + name)
        rows = compare_tables(checked_rows(historical / name), checked_rows(current / name), stage)
        tables[stage] = rows
        write_csv_rows(out / name, rows)
        for source in [*sources, "all"]:
            members = [r for r in rows if source == "all" or r["source_name"] == source]
            for backend in ("standard_int8", "full_neural_int16"):
                by_source.append(source_stats(source, stage, members, backend))
    write_csv_rows(out / "source_metrics.csv", by_source)
    plot(tables, out, sources, selected["policy"]["maximums"])
    outputs = {p.name: sha256(p) for p in sorted(out.iterdir())}
    save_json(out / "report.json", dict(scope=selected["scope"], units="foci/nucleus",
              created_utc=datetime.now(timezone.utc).isoformat(), program_sha256=sha256(__file__),
              policy_sha256=POLICY_SHA, audit_sha256=sha256(run / "independent_audit/verification.json"),
              int8_comparison_sha256=STANDARD_RECEIPT, neural16_comparison_sha256=verified["comparison_sha256"],
              source_metrics=by_source, outputs=outputs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("run", "historical", "out"):
        parser.add_argument("--" + option, required=True, type=Path)
    args = parser.parse_args()
    report(args.run, args.historical, args.out)

