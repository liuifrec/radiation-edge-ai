"""Replay the frozen MobileNetV2 training diagnostics using original BIE words and FP32 boundaries."""

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from mobilenet_feature_capture import (
    EXPORT,
    FILES,
    IMAGE,
    SOURCES,
    TAPS,
    check,
    checked,
    digest,
    read_json,
    save_json,
    select_panel,
    table,
)
from radiation_edge_ai.nasa_bps.feature_attribution import (
    FeatureGraph,
    decode_layer,
    head_decomposition,
    layout_scores,
)


BOUNDARIES = (
    "input",
    "stem_conv",
    "stem_clip",
    "block1",
    "block2",
    "block13",
    "block17",
    "spatial",
)


def write_table(path, rows):
    with path.open("x", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def difference(a, b):
    delta = b.astype(np.float64) - a
    return dict(
        mean=float(np.mean(delta)),
        mae=float(np.mean(np.abs(delta))),
        max_abs=float(np.max(np.abs(delta))),
    )


def moments(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(
        n=values.size,
        mean=float(values.mean()),
        mae=float(np.abs(values).mean()),
        min=float(values.min()),
        max=float(values.max()),
    )


class Analyzer:
    def __init__(self, root, capture, sha):
        import onnxruntime as ort

        self.capture = capture
        self.root = root
        self.sha = sha
        paths = {k: checked(root / p, h) for k, (p, h) in FILES.items()}
        export = read_json(paths["export"])
        checked(Path(export["checkpoint"]["path"]), export["checkpoint"]["sha256"])
        selection = read_json(checked(capture / "selection.json", sha))
        checked(capture / "capture_source.py", selection["capture_source_sha256"])
        self.rows = select_panel(
            table(paths["manifest"]),
            table(paths["parity"]),
            read_json(paths["pilot"])["sample_ids"],
        )
        check(
            selection["rows"] == self.rows
            and selection["image"] == IMAGE
            and selection["inputs"] == {k: h for k, (_, h) in FILES.items()}
            and selection["checkpoint"] == export["checkpoint"]["sha256"],
            "Invalid frozen selection",
        )
        self.pilot = {r["sample_id"]: r for r in table(paths["pilot_predictions"])}
        self.shapes = read_json(paths["shapes"])["dp_info"]
        self.radix = read_json(paths["radix"])
        self.original = FeatureGraph(paths["original"], TAPS)
        self.optimized = FeatureGraph(paths["optimized"], TAPS)
        check(
            np.array_equal(self.original.w, self.optimized.w)
            and self.original.b == self.optimized.b,
            "Optimized parameters differ from the original frozen head",
        )
        self.suffixes = {k: self.optimized.suffix(k) for k in BOUNDARIES}
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.release = ort.InferenceSession(
            str(paths["original"]), options, providers=["CPUExecutionProvider"]
        )

    def one(self, row, out):
        name = row["sample_id"]
        case = self.capture / name
        rec = read_json(case / "receipt.json")
        check(
            rec["selection_sha256"] == self.sha
            and rec["input_sha256"] == row["array_sha256"]
            and rec["sample_id"] == name
            and rec["bie_sha256"] == FILES["bie"][1],
            "Native receipt identity mismatch",
        )
        archive = checked(case / "native.zip", rec["native_sha256"])
        x = np.load(
            str(
                checked(
                    self.root / EXPORT / row["array_file"].replace("\\", "/"), row["array_sha256"]
                )
            ),
            allow_pickle=False,
        )
        original = self.original.run(x)
        opt = self.optimized.run(x)
        release = float(self.release.run(["burden"], {"input": x})[0].item())
        check(
            abs(release - row["source_fp32"]) <= 1e-6
            and abs(original["burden"].item() - release) <= 1e-5,
            "Original FP32 reference changed",
        )
        fixed, layers, channels, arrays, scalar_words = {}, {}, {}, {}, {}
        with ZipFile(archive) as raw:
            for k, tensor in TAPS.items():
                shape = self.shapes[tensor]
                (node,) = shape["node_name"]
                decoded = decode_layer(
                    raw.read(k + "_fx.bin"),
                    raw.read(k + "_fl.txt"),
                    raw.read(k + "_output_size.csv"),
                    shape,
                    self.radix[node],
                )
                f, codes, step = decoded["decoded"], decoded["codes"], decoded["step"]
                a, b = original[k], opt[k]
                check(
                    f.shape == a.shape == b.shape and np.isfinite(a).all(),
                    "Tensor geometry mismatch",
                )
                fixed[k] = f
                if codes.size == 1:
                    scalar_words[k] = int(codes.item())
                s = layout_scores(f, b)
                c = len(step)
                low, high = -(2 ** (decoded["bits"] - 1)), 2 ** (decoded["bits"] - 1) - 1
                ref, actual = b.reshape(c, -1).astype(np.float64), f.reshape(c, -1)
                d = actual - ref
                ref_below = ref < low * step[:, None]
                ref_above = ref > high * step[:, None]
                layers[k] = dict(
                    tensor=tensor,
                    node=node,
                    bits=decoded["bits"],
                    shape=list(f.shape),
                    decode_max_abs=decoded["max_decode_abs"],
                    layout_mae=s,
                    original_to_optimized=difference(a, b),
                    optimized_to_bie=difference(b, f),
                    original_range=[float(a.min()), float(a.max())],
                    fixed_range=[float(f.min()), float(f.max())],
                    original_zero=float((a == 0).mean()),
                    fixed_zero=float((codes == 0).mean()),
                    at_low=int((codes == low).sum()),
                    at_high=int((codes == high).sum()),
                    fp_below=int(ref_below.sum()),
                    fp_above=int(ref_above.sum()),
                    elements=f.size,
                )
                channels[k] = np.stack(
                    [
                        ref.mean(axis=1),
                        actual.mean(axis=1),
                        d.mean(axis=1),
                        np.abs(d).mean(axis=1),
                        (codes.reshape(c, -1) == low).sum(axis=1),
                        (codes.reshape(c, -1) == high).sum(axis=1),
                        ref_below.sum(axis=1),
                        ref_above.sum(axis=1),
                    ],
                    axis=0,
                )
                arrays["original_" + k] = a
                arrays["optimized_" + k] = b
        check(
            np.array_equal(fixed["pool"].ravel(), fixed["flatten"].ravel()),
            "Fixed flatten changed pooled values",
        )
        check(
            np.array_equal(np.clip(np.rint(x * 128), -128, 127) / 128, fixed["input"]),
            "Input Q0.7 mismatch",
        )
        check(
            np.allclose(
                opt["spatial"].mean(axis=(2, 3), dtype=np.float64).ravel(),
                opt["pool"].ravel(),
                atol=2e-6,
                rtol=2e-6,
            ),
            "FP32 global pooling mismatch",
        )
        check(
            abs(fixed["burden"].item() - rec["burden"]) <= 1e-8,
            "API return differs from the native output",
        )
        term = scalar_words["burden"]
        if row["panel"] == "pilot":
            pilot = self.pilot[name]
            check(
                rec["burden"] == float(pilot["bie_burden"])
                and term == int(pilot["bie_fixed_code"]),
                "Original pilot failed to reproduce",
            )
        suffix = {}
        head = float(opt["head"].item())
        for k, session in self.suffixes.items():
            start, end = TAPS[k], TAPS["head"]
            baseline = float(session.run([end], {start: opt[k]})[0].item())
            check(abs(baseline - head) <= 1e-5, "FP32 suffix failed to reproduce head at " + k)
            substituted = float(session.run([end], {start: fixed[k]})[0].item())
            suffix[k] = dict(
                fp32_replay=baseline,
                fixed_boundary_fp32_suffix=substituted,
                effect=substituted - head,
            )
        spatial = self.original.affine(fixed["spatial"].mean(axis=(2, 3), dtype=np.float64))
        pooled = self.original.affine(fixed["pool"])
        check(
            abs(suffix["spatial"]["fixed_boundary_fp32_suffix"] - spatial) <= 1e-5,
            "Spatial suffix disagreement",
        )
        o = float(original["head"].item())
        fixed_head, fixed_relu = fixed["head"].item(), fixed["relu"].item()
        signed, endpoint = head_decomposition(
            o,
            head,
            suffix["input"]["fixed_boundary_fp32_suffix"],
            spatial,
            pooled,
            fixed_head,
            fixed_relu,
            rec["burden"],
        )
        arrays["pooled_weighted_head_change"] = (
            fixed["pool"].ravel().astype(np.float64) - opt["pool"].ravel()
        ) * self.original.w
        check(
            abs(arrays["pooled_weighted_head_change"].sum() - (pooled - head)) <= 1e-5,
            "Channel-weighted head mismatch",
        )
        data = out / (name + ".npz")
        with data.open("xb") as stream:
            np.savez_compressed(
                stream, **arrays, **{"channels_" + k: v for k, v in channels.items()}
            )
        common = dict(
            sample_id=name,
            source_name=row["source_name"],
            input_sha256=row["array_sha256"],
            panel=row["panel"],
            original_fp32=row["source_fp32"],
            pooled_head_fp32=max(0.0, pooled),
            bie=rec["burden"],
        )
        info = dict(
            **common,
            stratum=row["stratum"],
            sample_name=row["sample_name"],
            dose_Gy=row["dose_Gy"],
            particle_type=row["particle_type"],
            hr_post_exposure=row["hr_post_exposure"],
            input_rms=float(np.sqrt(np.square(x[:, :2], dtype=np.float64).mean())),
            original_default_ort=release,
            original_tapped=original["burden"].item(),
            optimized_tapped=opt["burden"].item(),
            original_head=o,
            optimized_head=head,
            floating_pool_on_native_spatial_head=spatial,
            floating_head_on_native_pool=pooled,
            fixed_head=fixed_head,
            fixed_relu=fixed_relu,
            terminal_word=term,
            scalar_words=scalar_words,
            native_seconds=rec["job_seconds"],
            capture_receipt_sha256=digest(case / "receipt.json"),
            feature_file_sha256=digest(data),
            layers=layers,
            suffix=suffix,
            signed=signed,
            endpoint=endpoint,
        )
        return common, info


def summary(part):
    return dict(
        n=len(part),
        signed_bias=moments([r["bie"] - r["original_fp32"] for r in part]),
        pooled_upstream=moments([r["pooled_head_fp32"] - r["original_fp32"] for r in part]),
        downstream=moments([r["bie"] - r["pooled_head_fp32"] for r in part]),
        endpoint={k: moments([r["endpoint"][k] for r in part]) for k in part[0]["endpoint"]},
        signed={k: moments([r["signed"][k] for r in part]) for k in part[0]["signed"]},
        suffix_preactivation={
            k: moments([r["suffix"][k]["effect"] for r in part]) for k in BOUNDARIES
        },
    )


def analyze(root, capture, sha, out, sample=None, wait_seconds=0):
    import onnx
    import onnxruntime as ort
    from radiation_edge_ai.nasa_bps import feature_attribution

    t = time.monotonic()
    a = Analyzer(root, capture, sha)
    cases = a.rows if sample is None else [r for r in a.rows if r["sample_id"] == sample]
    check(bool(cases), "Sample absent from selected training panel")
    out.mkdir()
    pairs, full = [], []
    for row in cases:
        deadline = time.monotonic() + wait_seconds
        receipt_file = capture / row["sample_id"] / "receipt.json"
        while not receipt_file.is_file() and time.monotonic() < deadline:
            time.sleep(2)
        pair, data = a.one(row, out)
        save_json(out / (row["sample_id"] + ".json"), data)
        pairs.append(pair)
        full.append(data)
        print(row["sample_id"], "verified", flush=True)
    write_table(out / "cross_arch_pairs.csv", pairs)
    save_json(out / "nuclear_results.json", full)
    panels = {}
    for p in sorted({row["panel"] for row in cases}):
        cohort = [r for r in full if r["panel"] == p]
        conditions = sorted(
            {(r["particle_type"], r["dose_Gy"], r["hr_post_exposure"]) for r in cohort}
        )
        levels = sorted({r["stratum"] for r in cohort})
        median = np.median([r["input_rms"] for r in cohort])
        panels[p] = dict(
            all=summary(cohort),
            sources={
                s: summary([r for r in cohort if r["source_name"] == s])
                for s in SOURCES
                if any(r["source_name"] == s for r in cohort)
            },
            prediction_stratum={
                s: summary([r for r in cohort if r["stratum"] == s]) for s in levels
            },
            condition={
                "|".join(c): summary(
                    [
                        r
                        for r in cohort
                        if (r["particle_type"], r["dose_Gy"], r["hr_post_exposure"]) == c
                    ]
                )
                for c in conditions
            },
            median_input_rms=float(median),
            input_rms={
                name: summary(group)
                for name, group in (
                    ("at_or_below_median", [r for r in cohort if r["input_rms"] <= median]),
                    ("above_median", [r for r in cohort if r["input_rms"] > median]),
                )
                if group
            },
        )
    receipt = dict(
        utc=datetime.now(timezone.utc).isoformat(),
        study="mobilenet_v2_training_only",
        scope="partial diagnostic" if sample else "prespecified training-only attribution",
        selection_sha256=sha,
        capture_dir=str(capture),
        policy_inputs={k: h for k, (_, h) in FILES.items()},
        source_sha256=digest(__file__),
        decoder_sha256=digest(feature_attribution.__file__),
        runtime=dict(numpy=np.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__),
        channel_rows=[
            "optimized_mean",
            "native_mean",
            "native_minus_optimized_mean",
            "mean_absolute_error",
            "at_low",
            "at_high",
            "fp_below_range",
            "fp_above_range",
        ],
        native_diagnostic="original binary, native dump_level 2",
        held_out=False,
        compile=False,
        usb=False,
        elapsed_seconds=time.monotonic() - t,
        panels=panels,
        pairs_sha256=digest(out / "cross_arch_pairs.csv"),
        nuclear_results_sha256=digest(out / "nuclear_results.json"),
    )
    save_json(out / "receipt.json", receipt)
    print(out / "receipt.json", digest(out / "receipt.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("D:/radiation-edge-ai-data"))
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--selection-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--sample", help="Optional single prespecified training case to verify the implementation"
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=0,
        help="Maximum wait per sample for the concurrently running native capture",
    )
    args = parser.parse_args()
    analyze(args.root, args.capture, args.selection_sha, args.out, args.sample, args.wait_seconds)
