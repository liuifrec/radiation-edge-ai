"""Plot rectified-head error attribution for the two frozen training panels."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from radiation_edge_ai.nasa_bps.multimodel_ptq import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True,
                        help="Directory written by cross_arch_attribution")
    parser.add_argument("--out", type=Path, required=True, help="New directory")
    parser.add_argument("--matched", action="store_true", help="Draw only identical nuclei across architectures")
    args = parser.parse_args()
    receipt = json.loads((args.comparison / "receipt.json").read_text(encoding="utf-8"))
    pairs = args.comparison / ("matched_attribution.csv" if args.matched else "paired_attribution.csv")
    expected = receipt["matched_pairs_sha256" if args.matched else "pairs_sha256"]
    if sha256(pairs) != expected or receipt["scope"] != "training_only":
        raise ValueError("Training comparison receipt and predictions disagree")
    with pairs.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["panel"] == "confirmation"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), sharex=True, sharey=True)
    colors = ["#226e8b", "#b65b25"]
    for ax, (arch, name) in zip(axes, (("mobilenet_v2", "MobileNetV2"), ("resnet18", "ResNet18"))):
        selected = [row for row in rows if row["architecture"] == arch]
        for key, color, marker in zip(("upstream", "head_and_output"), colors, ("o", "^")):
            ax.scatter([float(r["original_fp32"]) for r in selected],
                       [float(r[key]) for r in selected], s=30, color=color,
                       alpha=0.68, marker=marker, edgecolors="none")
        ax.axhline(0, color="#4b545d", linewidth=0.8, zorder=0)
        ax.set_title(f"{name}   ·   {len(selected)} training nuclei", loc="left", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Original FP32 burden (foci/nucleus)")
        ax.grid(axis="y", color="#646f78", alpha=.16, linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Contribution to BIE shift (foci/nucleus)")
    title = ("Regression error on identical training nuclei" if args.matched
             else "Where prediction shifts enter the regression output")
    fig.suptitle(title, ha="left",
                 x=.09, fontsize=15, fontweight="bold")
    handles = [Line2D([], [], marker=m, color=c, ls="", markersize=7)
               for m, c in zip(("o", "^"), colors)]
    fig.legend(handles, ["Feature extractor and pooling", "Regression head and output"],
               loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(top=.81, bottom=.25, wspace=.09, left=.09, right=.98)
    args.out.mkdir(parents=True, exist_ok=False)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / ("training_error_attribution." + ext), dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    main()
