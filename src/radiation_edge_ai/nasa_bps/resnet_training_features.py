"""Replay the frozen ResNet18 training panel from authenticated native activations."""

import argparse
import csv
import hashlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .resnet_stage1_source_audit import CALIBRATION, SELECTION, SOURCES, checked_bytes

CAPTURE = Path("scratch/nasa_multisource_bie_20261118T232421Z_3a8998bc")
ANALYSIS = "d1112aebf7a4ff7f25e85c9c613145913c4883ce509cd71d8de26f91a712684f"
PILOT = "BPSR1V2D_01501"
STEM = ("stem_conv", "stem_relu", "stem_pool")
MAP = "stage4_block1"
TAPS = {
    "stem_conv": "/features/features.0/Conv_output_0",
    "stem_relu": "/features/features.2/Relu_output_0",
    "stem_pool": "/features/features.3/MaxPool_output_0",
    MAP: "/features/features.7/features.7.1/relu_1/Relu_output_0",
    "global_pool": "/pool/GlobalAveragePool_output_0",
    "flatten": "/Flatten_output_0",
    "head": "/head/Gemm_output_0",
    "nonnegative": "/nonnegative/Relu_output_0",
    "burden": "burden",
}


def exact(a, b, name):
    if not np.array_equal(a, b):
        raise ValueError(f"{name} differs")


def close(a, b, name, atol=1e-5):
    if not np.allclose(a, b, atol=atol, rtol=0):
        raise ValueError(f"{name} differs")


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def session(model):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


class FrozenGraph:
    def __init__(self, binary, stem_suffixes=False):
        import onnx

        model = onnx.load_from_string(binary)
        self.plain = session(model)
        self.suffix = {}
        extractor = onnx.utils.Extractor(model)
        if stem_suffixes:
            for key in STEM:
                self.suffix[key] = session(extractor.extract_model([TAPS[key]], [TAPS["head"]]))
            pool = extractor.extract_model([TAPS["stem_relu"]], [TAPS["stem_pool"]])
            if [n.op_type for n in pool.graph.node] != ["MaxPool"]:
                raise ValueError("Unexpected ResNet stem pool subgraph")
            self.stem_pool = session(pool)
        (head,) = [n for n in model.graph.node if n.op_type == "Gemm"]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in head.attribute}
        if (
            list(head.input) != [TAPS["flatten"], "head.weight", "head.bias"]
            or attrs.get("transA", 0)
            or attrs.get("transB", 0) not in (0, 1)
            or attrs.get("alpha", 1) != 1
            or attrs.get("beta", 1) != 1
        ):
            raise ValueError("Unexpected frozen affine head")
        params = {
            t.name: onnx.numpy_helper.to_array(t)
            for t in model.graph.initializer
            if t.name in head.input[1:]
        }
        for node in model.graph.node:
            if node.op_type == "Constant" and node.output[0] in head.input[1:]:
                params[node.output[0]] = onnx.numpy_helper.to_array(
                    onnx.helper.get_attribute_value(node.attribute[0])
                )
        weights = params["head.weight"]
        if attrs.get("transB", 0):
            weights = weights.T
        if weights.shape != (512, 1) or params["head.bias"].size != 1:
            raise ValueError("Unexpected head parameter shapes")
        self.weights = weights[:, 0].astype(np.float64)
        self.bias = params["head.bias"].item()
        for key, name in TAPS.items():
            if key != "burden":
                model.graph.output.extend(
                    [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)]
                )
        self.tapped = session(model)

    def run(self, x):
        y = dict(zip(TAPS, self.tapped.run(list(TAPS.values()), {"input": x})))
        exact(y["burden"], self.plain.run(["burden"], {"input": x})[0], "tapped FP32 terminal")
        return y

    def affine(self, pooled):
        return float(self.weights @ pooled.reshape(512).astype(np.float64) + self.bias)


def signed_word(raw, descriptor, decoded):
    if (
        descriptor["name"] != "burden"
        or descriptor["onnx_shape"] != [1]
        or descriptor["bitw"] != 8
        or len(raw) != 1
    ):
        raise ValueError("Unexpected native terminal descriptor")
    (radix,) = descriptor["radix"]
    (scale,) = descriptor["scale"]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid terminal scale")
    code = int.from_bytes(raw, "little", signed=True)
    close(code / (2.0**radix * scale), decoded, "terminal word", atol=1e-6)
    return code


def scalar_effects(fp, fixed, rounded, suffix, affine):
    """Signed preactivation increments and observable endpoint-scale error."""
    ref = fp["head"].item()
    native_head = fixed["head"].item()
    upstream = affine(fixed["flatten"])
    native_map = affine(fixed[MAP].mean(axis=(2, 3), dtype=np.float64))
    pred = fixed["burden"].item()
    original = fp["burden"].item()
    effects = {"input_grid": rounded - ref}
    previous = rounded
    for key in STEM:
        effects[key] = suffix[key] - previous
        previous = suffix[key]
    effects.update(
        remaining_extractor=native_map - previous,
        pool=upstream - native_map,
        affine_head=native_head - upstream,
        floor=(max(0, native_head) - native_head) - (max(0, ref) - ref),
        relu_boundary=fixed["nonnegative"].item() - max(0, native_head),
        terminal=pred - fixed["nonnegative"].item(),
    )
    close(sum(effects.values()), pred - original, "signed decomposition", atol=1e-6)
    return effects, max(0, upstream), native_map, upstream


def observed_steps(codes, native):
    """Use one nonzero anchor and independently check every decoded value in its channel."""
    if codes.dtype not in (np.dtype("i1"), np.dtype("<i2")) or codes.shape != native.shape:
        raise ValueError("Stem signed-code geometry differs")
    c = codes.reshape(codes.shape[1], -1).astype(np.float64)
    f = native.reshape(c.shape).astype(np.float64)
    high = np.argmax(np.abs(c), axis=1)
    anchor = c[np.arange(len(c)), high]
    steps = np.divide(f[np.arange(len(c)), high], anchor, out=np.zeros(len(c)), where=anchor != 0)
    close(c * steps[:, None], f, "native stem code layout", atol=1e-5)
    if np.any(steps[anchor != 0] <= 0):
        raise ValueError("Invalid native stem code scale")
    return steps, anchor != 0


def code_range(codes, native, fp, relu):
    """Count range excursions using the verified channel spacing."""
    if not (codes.shape == fp.shape == relu.shape):
        raise ValueError("Stem reference geometry differs")
    steps, active = observed_steps(codes, native)
    bounds = np.iinfo(codes.dtype)
    below, above = codes == bounds.min, codes == bounds.max
    ref = fp.reshape(len(steps), -1)
    return {
        "elements": int(codes.size),
        "lower": int(below.sum()),
        "upper": int(above.sum()),
        "lower_relu_positive": int(np.count_nonzero(relu[below] > 0)),
        "reference_below": int((ref[active] < bounds.min * steps[active, None] - 1e-5).sum()),
        "reference_above": int((ref[active] > bounds.max * steps[active, None] + 1e-5).sum()),
        "range_checked_elements": int(active.sum() * ref.shape[1]),
    }


def pool_codes(codes, native, pooled_relu):
    if codes.dtype != np.int8 or codes.shape != pooled_relu.shape:
        raise ValueError("Unexpected stem pool storage width or shape")
    steps, active = observed_steps(codes, native)
    target = pooled_relu.reshape(len(steps), -1)[active].astype(np.float64) / steps[active, None]
    actual = codes.reshape(len(steps), -1)[active]
    bounds = np.iinfo(codes.dtype)
    floor = np.clip(np.floor(target), bounds.min, bounds.max)
    nearest = np.clip(np.rint(target), bounds.min, bounds.max)
    inside = (target >= bounds.min) & (target <= bounds.max)
    away = np.abs(2 * target - np.rint(2 * target)) > 2e-4
    distinguish = inside & (floor != nearest) & away
    return {
        "checked": int(actual.size),
        "in_range": int(inside.sum()),
        "floor_match": int((actual == floor).sum()),
        "nearest_match": int((actual == nearest).sum()),
        "distinguish": int(distinguish.sum()),
        "distinguish_floor": int((distinguish & (actual == floor)).sum()),
        "distinguish_nearest": int((distinguish & (actual == nearest)).sum()),
        "distinguish_neither": int((distinguish & (actual != nearest) & (actual != floor)).sum()),
    }


def feature_distributions(tensors, rows, weights):
    """Equal-nucleus pooled/map distributions and per-channel errors for one group."""
    metrics = []
    distribution = {}
    for name in (MAP, "global_pool"):
        fp = np.stack([tensors[i][name][0] for i in rows]).astype(np.float64)
        fx = np.stack([tensors[i][name][1] for i in rows]).astype(np.float64)
        d = fx - fp
        distribution[name] = {
            k: {
                "mean": float(v.mean()),
                "zero_fraction": float((v == 0).mean()),
                "p01_p50_p99": np.quantile(v, [0.01, 0.5, 0.99]).tolist(),
            }
            for k, v in (("original", fp), ("bie", fx), ("error", d))
        }
        for channel in range(fp.shape[2]):
            err = d[:, 0, channel]
            metrics.append(
                {
                    "stage": name,
                    "channel": channel,
                    "mean_original": float(fp[:, 0, channel].mean()),
                    "mean_bie": float(fx[:, 0, channel].mean()),
                    "bias": float(err.mean()),
                    "mae": float(np.abs(err).mean()),
                    "head_weight": float(weights[channel]),
                    "mean_weighted_error": float(err.mean() * weights[channel]),
                }
            )
    return distribution, metrics


def replay(root, out):
    import onnxruntime as ort

    started = time.perf_counter()
    capture = root / CAPTURE
    analysis = json.loads(checked_bytes(capture / "analysis.json", ANALYSIS))
    selected = json.loads(checked_bytes(capture / "selection.json", SELECTION))
    graphs = {k: root / p for k, (p, _) in analysis["graphs"].items()}
    payloads = {k: checked_bytes(p, analysis["graphs"][k][1]) for k, p in graphs.items()}
    with ZipFile(io.BytesIO(payloads["bie"])) as vendor:
        metadata = json.loads(vendor.read("ioinfo.json"))
    (output,) = metadata["output"]
    (inp,) = metadata["input"]
    if (
        inp["name"] != "input"
        or inp["onnx_shape"] != [1, 3, 256, 256]
        or inp["bitw"] != 8
        or inp["radix"] != [7, 7, 7]
        or inp["scale"] != [1, 1, 1]
        or inp["ch_dim"] != 1
    ):
        raise ValueError("Unexpected input quantization descriptor")
    exact(output == analysis["output_descriptor"], True, "archived IO descriptor")
    manifest = csv.DictReader(
        io.StringIO(
            checked_bytes(
                graphs["original"].with_name("calibration_manifest.csv"), CALIBRATION
            ).decode("utf-8-sig")
        )
    )
    calibration = {r["sample_id"]: r for r in manifest}
    original = FrozenGraph(payloads["original"])
    optimized = FrozenGraph(payloads["optimized"], stem_suffixes=True)
    exact(original.weights, optimized.weights, "head weights")
    exact(original.bias, optimized.bias, "head bias")
    archived = {r["sample_id"]: r for r in analysis["rows"]}
    selection = selected["rows"]
    if (
        len(calibration) != 192
        or len(archived) != 48
        or len(selection) != 48
        or len({r["sample_name"] for r in selection}) != 48
        or set(archived) != {r["sample_id"] for r in selection}
        or {r["source_name"] for r in selection} != SOURCES
        or selected["manifest_sha256"] != CALIBRATION
    ):
        raise ValueError("Unexpected training selection")
    rows, tensors = [], []
    observed_widths = {}
    for sel in selection:
        sid = sel["sample_id"]
        prev = archived[sid]
        if (
            sel != calibration[sid]
            or sel["fold_female_role"] != "train"
            or sel["dataset_role"] != "development_train_ptq_calibration"
            or sel["array_sha256"] != prev["array_sha256"]
        ):
            raise ValueError("Incorrect training membership")
        folder = capture / sid
        receipt = json.loads(checked_bytes(folder / "receipt.json", prev["receipt_sha256"]))
        if (
            receipt["bie_sha256"] != analysis["graphs"]["bie"][1]
            or receipt["tensor_sha256"] != sel["array_sha256"]
            or receipt["sample_id"] != sid
            or receipt["source"] != sel["source_name"]
            or receipt["selection_sha256"] != SELECTION
        ):
            raise ValueError("Native lineage mismatch")
        for file, sha in receipt["files"].items():
            checked_bytes(folder / file, sha)
        tensor_path = graphs["original"].parent / sel["array_file"].replace("\\", "/")
        x = np.load(io.BytesIO(checked_bytes(tensor_path, sel["array_sha256"])), allow_pickle=False)
        fp, opt = original.run(x), optimized.run(x)
        for key in TAPS:
            exact(fp[key], opt[key], "original/optimized " + key)
        close(original.affine(fp["flatten"]), fp["head"], "original affine")
        fixed = {}
        with (
            np.load(folder / "activations.npz", allow_pickle=False) as floats,
            ZipFile(folder / "native.zip") as raw,
        ):
            for key, name in TAPS.items():
                tap = receipt["taps"][key]
                prefix = tap["text"][: -len("_fl.txt")]
                expected = (
                    "layer_output_burden" if key == "burden" else "layer_" + name.replace("/", "_")
                )
                exact(prefix, expected, "tap name")
                dims = [
                    int(i)
                    for i in raw.read(prefix + "_output_size.csv").replace(b",", b" ").split()
                ]
                c, h, w = (fp[key].shape[1:] + (1, 1, 1))[:3]
                if (
                    dims != [c, h, w, 1]
                    or tap["elements"] != fp[key].size
                    or not prev["layers"][key]["native_layout"]["accepted"]
                ):
                    raise ValueError(f"Invalid native geometry for {key}")
                bit_count = raw.getinfo(prefix + "_fx.bin").file_size * 8
                width, extra = divmod(bit_count, fp[key].size)
                if extra or width not in (8, 16) or observed_widths.setdefault(key, width) != width:
                    raise ValueError(f"Inconsistent native storage width for {key}")
                fixed[key] = floats[key].reshape(fp[key].shape)
                if not np.isfinite(fixed[key]).all():
                    raise ValueError("Non-finite native activation")
            code = signed_word(
                raw.read("layer_output_burden_fx.bin"), output, fixed["burden"].item()
            )
            stem_raw = raw.read(receipt["taps"]["stem_conv"]["text"][: -len("_fl.txt")] + "_fx.bin")
            bits = prev["layers"]["stem_conv"]["verified_code_endpoints"]
            if (
                not bits["verified_flat_signed_codes"]
                or bits["bits"] not in (8, 16)
                or len(stem_raw) * 8 != fp["stem_conv"].size * bits["bits"]
            ):
                raise ValueError("Unresolved native stem signed width")
            limits = code_range(
                np.frombuffer(stem_raw, dtype="i1" if bits["bits"] == 8 else "<i2").reshape(
                    fp["stem_conv"].shape
                ),
                fixed["stem_conv"],
                fp["stem_conv"],
                fixed["stem_relu"],
            )
            rounded = floats["input"].reshape(x.shape)
            exact(rounded, np.clip(np.rint(x * 128), -128, 127) / 128, "Q0.7 input")
            pool_fp = optimized.stem_pool.run(None, {TAPS["stem_relu"]: fp["stem_relu"]})[0]
            exact(pool_fp, fp["stem_pool"], "FP32 stem pool operation")
            pool_fx_relu = optimized.stem_pool.run(None, {TAPS["stem_relu"]: fixed["stem_relu"]})[0]
            words = raw.read(receipt["taps"]["stem_pool"]["text"][: -len("_fl.txt")] + "_fx.bin")
            pool_diagnostic = pool_codes(
                np.frombuffer(words, np.int8).reshape(pool_fp.shape),
                fixed["stem_pool"],
                pool_fx_relu,
            )
            prefix = receipt["taps"]["global_pool"]["text"][: -len("_fl.txt")]
            pooled_words = np.frombuffer(raw.read(prefix + "_fx.bin"), np.int8)
            channels = analysis["cross_nucleus_pooled_code_ranges"]["global_pool"]["per_channel"]
            if (
                len(channels) != 512
                or pooled_words.shape != (512,)
                or any(
                    c["channel"] != j or c["status"] != "verified" for j, c in enumerate(channels)
                )
            ):
                raise ValueError("Unexpected archived channel scales")
            pool_steps = np.asarray([c["scale"] for c in channels])
            pool_ref = fp["global_pool"].reshape(-1)
            close(
                pooled_words * pool_steps,
                fixed["global_pool"].reshape(-1),
                "archived native pooled spacing",
            )
            pool_diagnostic.update(
                global_pool_at_limit=int(
                    np.count_nonzero((pooled_words == -128) | (pooled_words == 127))
                ),
                global_pool_fp_outside=int(
                    np.count_nonzero(
                        (pool_ref < -128 * pool_steps - 1e-5) | (pool_ref > 127 * pool_steps + 1e-5)
                    )
                ),
            )
        exact(fixed["global_pool"].reshape(-1), fixed["flatten"].reshape(-1), "native flatten")
        rounded_head = optimized.tapped.run([TAPS["head"]], {"input": rounded})[0].item()
        suffix = {}
        replay_error = []
        for key, sess in optimized.suffix.items():
            replayed = sess.run(None, {TAPS[key]: fp[key]})[0].item()
            replay_error.append(abs(replayed - fp["head"].item()))
            close(replayed, fp["head"].item(), "FP32 suffix " + key)
            suffix[key] = sess.run(None, {TAPS[key]: fixed[key]})[0].item()
        close(
            suffix["stem_pool"],
            prev["native_boundary_fp32_suffix_head"]["stem_pool"],
            "historical suffix",
        )
        ideal_pool_head = (
            optimized.suffix["stem_pool"].run(None, {TAPS["stem_pool"]: pool_fx_relu})[0].item()
        )
        close(ideal_pool_head, suffix["stem_relu"], "ideal pool on BIE ReLU vs ReLU suffix")
        effects, pooled, spatial, pooled_pre = scalar_effects(
            fp, fixed, rounded_head, suffix, original.affine
        )
        close(pooled_pre, prev["heads"]["float_head_on_native_pool"], "archived pooled replay")
        close(spatial, prev["heads"]["float_pool_on_native_extractor"], "archived spatial replay")
        close(fp["burden"].item(), prev["predictions"]["original"], "archived original")
        close(fixed["burden"].item(), prev["predictions"]["bie"], "archived BIE")
        tensors.append({k: (fp[k], fixed[k]) for k in (MAP, "global_pool")})
        reference, bie = fp["burden"].item(), fixed["burden"].item()
        rows.append(
            dict(
                sample_id=sid,
                source_name=sel["source_name"],
                input_sha256=sel["array_sha256"],
                panel="pilot" if sid == PILOT else "confirmation",
                original_fp32=reference,
                pooled_head_fp32=pooled,
                bie=bie,
                input_rms_group=prev["quantile_strata"]["combined_rms"],
                prediction_group=prev["quantile_strata"]["positive_original_burden"],
                condition="/".join(
                    sel[k] for k in ("particle_type", "hr_post_exposure", "dose_Gy")
                ),
                terminal_code=code,
                original_pre=fp["head"].item(),
                pooled_pre=pooled_pre,
                bie_pre=fixed["head"].item(),
                total=bie - reference,
                upstream=pooled - reference,
                downstream=bie - pooled,
                **effects,
                **{"stem_" + k: v for k, v in limits.items()},
                **{"codes_" + k: v for k, v in pool_diagnostic.items()},
                ideal_pool_suffix=ideal_pool_head,
                native_pool_suffix=suffix["stem_pool"],
                fp_suffix_max_replay_error=max(replay_error),
                archived_simulator_seconds=receipt["native_runtime_seconds"],
            )
        )
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        panel = row["panel"]
        groups[(panel, "all", "all")].append(i)
        if panel == "confirmation":
            for by in ("source_name", "prediction_group", "condition", "input_rms_group"):
                groups[(panel, by, row[by])].append(i)
    strata = []
    for (panel, by, value), idx in groups.items():
        item = {"panel": panel, "factor": by, "group": value, "n": len(idx)}
        for name in ("total", "upstream", "downstream", *effects):
            a = np.array([rows[i][name] for i in idx])
            item.update({name + "_bias": a.mean(), name + "_mae": abs(a).mean()})
        strata.append(item)
    out.mkdir(exist_ok=False)
    write_csv(out / "paired.csv", rows)
    write_csv(out / "groups.csv", strata)
    distributions = {}
    channels = []
    for panel in ("pilot", "confirmation"):
        d, ch = feature_distributions(tensors, groups[(panel, "all", "all")], original.weights)
        distributions[panel] = d
        channels.extend(dict(panel=panel, **v) for v in ch)
    write_csv(out / "channels.csv", channels)
    with (out / "feature_pairs.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            sample_id=np.asarray([r["sample_id"] for r in rows]),
            head_weights=original.weights,
            head_bias=original.bias,
            **{
                k + "_" + label: np.concatenate([sample[k][j] for sample in tensors])
                for k in (MAP, "global_pool")
                for j, label in enumerate(("fp32", "bie"))
            },
        )
    result = {
        "schema": "resnet_training_feature_replay_v1",
        "captured_bie": "unchanged_archived_native_dump_level_2",
        "selection_sha256": SELECTION,
        "calibration_sha256": CALIBRATION,
        "archival_analysis_sha256": ANALYSIS,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "graphs": analysis["graphs"],
        "sdk_image": analysis["image"],
        "io": metadata,
        "onnxruntime": ort.__version__,
        "archive_capture_count": len(rows),
        "stem_limits": {
            p: {k: sum(r["stem_" + k] for r in rows if r["panel"] == p) for k in limits}
            for p in distributions
        },
        "pool_codes": {
            p: {k: sum(r["codes_" + k] for r in rows if r["panel"] == p) for k in pool_diagnostic}
            for p in distributions
        },
        "observed_word_bits": observed_widths,
        "original_suffix_max_replay_error": max(r["fp_suffix_max_replay_error"] for r in rows),
        "distributions": distributions,
        "units": "burden in foci/nucleus; activation statistics in feature units",
        "historical_native_seconds": sum(r["archived_simulator_seconds"] for r in rows),
        "host_replay_seconds": time.perf_counter() - started,
        "files": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir())
        },
    }
    with (out / "receipt.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replay(args.root, args.out)
