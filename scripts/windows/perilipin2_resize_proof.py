"""Experimental, immutable Resize-compatibility proof for Perilipin2 M3C.

READ-ONLY source: a frozen, already imported Windows M3C work directory.
WRITE-ONLY destination: a new sibling D: scratch directory.

This tool does not run Kneron, PTQ, BIE, NEF, or an attached KL720.
It tests whether the four shape-derived ONNX Resize operations can be
rewritten as constant sizes or integer scales without changing FP32 outputs.
The validated source ONNX and previous compiler artifacts are never modified.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from radiation_edge_ai.perilipin2 import windows_cli as w  # noqa: E402


def resize_nodes(model):
    nodes = [(index, node) for index, node in enumerate(model.graph.node)
             if node.op_type == "Resize"]
    if len(nodes) != 4:
        raise ValueError("Expected exactly four Resize nodes; found {}".format(len(nodes)))
    for index, node in nodes:
        attrs = {}
        for attr in node.attribute:
            attrs[attr.name] = attr.s.decode("utf-8") if attr.type == 3 else None
        actual = (attrs.get("mode"), attrs.get("coordinate_transformation_mode"),
                  attrs.get("nearest_mode"))
        if actual != ("nearest", "asymmetric", "floor"):
            raise ValueError("Resize {} has unexpected interpolation contract: {}".format(
                index, actual))
        if len(node.input) != 4 or not node.input[3]:
            raise ValueError("Resize {} does not have a sizes input".format(index))
    return nodes


def two_inputs(np):
    rng = np.random.default_rng(20261110)
    x1 = rng.uniform(-0.5, 0.5, size=w.INPUT_SHAPE).astype(np.float32)
    # A deterministic, spatially structured second probe is independent of x1.
    yy = np.linspace(-0.5, 0.5, 512, dtype=np.float32)
    xx = np.linspace(0.5, -0.5, 512, dtype=np.float32)
    x2 = np.empty(w.INPUT_SHAPE, dtype=np.float32)
    for c in range(3):
        x2[0, c] = np.clip(yy[:, None] + xx[None, :] + c * 0.07, -0.5, 0.5)
    return (x1, x2)


def session(ort, model):
    # Do not let the test runtime fold away named intermediate tensors.
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), sess_options=options,
                                providers=["CPUExecutionProvider"])


def delta_stats(np, expected, observed):
    if expected.shape != observed.shape or not np.isfinite(observed).all():
        raise ValueError("FP32 model output shape or finiteness mismatch")
    delta = np.abs(expected.astype(np.float64) - observed.astype(np.float64))
    bd_cutoff = math.log(.45 / .55)
    return {
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "foreground_decision_mismatches": int(np.count_nonzero(
            (expected[:, 0] >= 0) != (observed[:, 0] >= 0))),
        "boundary_decision_mismatches": int(np.count_nonzero(
            (expected[:, 1] >= bd_cutoff) != (observed[:, 1] >= bd_cutoff))),
    }


def strict_parity_pass(rows, atol, mean_limit):
    return all(
        row["max_abs_error"] <= atol
        and row["mean_abs_error"] <= mean_limit
        and row["foreground_decision_mismatches"] == 0
        and row["boundary_decision_mismatches"] == 0
        for row in rows
    )


def inspect_resize_inputs(onnx, np, ort, model, inputs):
    """Add in-memory intermediate outputs; no modified graph is written."""
    observed = []
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    nodes = resize_nodes(probe)
    existing = {value.name for value in probe.graph.output}
    output_pairs = []
    for index, node in nodes:
        size_name = node.input[3]
        feature_name = node.input[0]
        output_pairs.append((index, node.name, size_name, feature_name))
        for name, tensor_type, shape in (
            (size_name, onnx.TensorProto.INT64, [4]),
            (feature_name, onnx.TensorProto.FLOAT, [None, None, None, None]),
        ):
            if name not in existing:
                probe.graph.output.append(
                    onnx.helper.make_tensor_value_info(name, tensor_type, shape)
                )
                existing.add(name)
    onnx.checker.check_model(probe)
    runtime = session(ort, probe)
    probes = []
    for x in inputs:
        fetched = runtime.run(None, {w.INPUT_NAME: x})
        outputs = {info.name: val for info, val in zip(runtime.get_outputs(), fetched)}
        probes.append(outputs)
    for index, node_name, size_name, feature_name in output_pairs:
        sizes = []
        source_dims = []
        for fetched in probes:
            size = np.asarray(fetched[size_name])
            if size.shape != (4,) or size.dtype.kind not in ("u", "i"):
                raise ValueError("Resize {} sizes are not a fixed int64[4]".format(index))
            sizes.append([int(v) for v in size])
            source_dims.append(list(fetched[feature_name].shape))
        if sizes[0] != sizes[1] or source_dims[0] != source_dims[1]:
            raise ValueError("Resize {} sizes changed between deterministic inputs".format(index))
        input_dims = source_dims[0]
        target_dims = sizes[0]
        if any(x <= 0 for x in input_dims + target_dims):
            raise ValueError("Non-positive Resize dimension at {}".format(index))
        if target_dims[:2] != input_dims[:2]:
            raise ValueError("Resize {} unexpectedly changes batch/channels".format(index))
        integer_scales = [
            target_dims[k] // input_dims[k] if target_dims[k] % input_dims[k] == 0 else None
            for k in range(4)
        ]
        observed.append({
            "node_index": index,
            "node_name": node_name,
            "sizes_input": size_name,
            "size_producer": next(
                (n.op_type for n in model.graph.node if size_name in n.output), None),
            "input_shape": input_dims,
            "target_shape": target_dims,
            "integer_scales": integer_scales,
        })
    references = [fetched[w.OUTPUT_NAME] for fetched in probes]
    return observed, references


def make_variant(onnx, np, model, measurements, mode):
    updated = onnx.ModelProto()
    updated.CopyFrom(model)
    by_index = {entry["node_index"]: entry for entry in measurements}
    for index, node in resize_nodes(updated):
        entry = by_index[index]
        name = "m3c_resize_{}_{}".format(mode, index)
        if any(initializer.name == name for initializer in updated.graph.initializer):
            raise ValueError("Unexpected generated initializer collision: " + name)
        if mode == "static-sizes":
            values = np.array(entry["target_shape"], dtype=np.int64)
            node.input[3] = name
        elif mode == "integer-scales":
            if any(factor is None or factor <= 0 for factor in entry["integer_scales"]):
                raise ValueError("Noninteger Resize ratio at node {}".format(index))
            values = np.array(entry["integer_scales"], dtype=np.float32)
            node.input[2] = name
            del node.input[3:]
        else:
            raise ValueError("Unknown variant " + mode)
        updated.graph.initializer.append(onnx.numpy_helper.from_array(values, name))
    onnx.checker.check_model(updated)
    return updated


def run(args):
    import numpy as np
    import onnx
    import onnxruntime as ort

    work, receipt = w.check_bundle(args.work)
    optimized_path = work / w.WORK_OPT
    if not optimized_path.is_file():
        raise ValueError("Missing already optimized ONNX; use an evaluated M3C work folder")
    output = Path(args.out).expanduser().resolve()
    if output.exists() or output.is_symlink() or output == work or work in output.parents:
        raise ValueError("Output must be a NEW directory outside the frozen work folder")
    if output in work.parents:
        raise ValueError("Output may not be an ancestor of the frozen work folder")
    staging = output.with_name(output.name + ".BUILDING")
    if staging.exists() or staging.is_symlink():
        raise ValueError("Existing staging output: " + str(staging))

    original = onnx.load(str(work / w.WORK_MODEL))
    optimized = onnx.load(str(optimized_path))
    onnx.checker.check_model(optimized)
    w.validate_contract(w.graph_contract(optimized))
    inputs = two_inputs(np)
    measurements, optimized_values = inspect_resize_inputs(
        onnx, np, ort, optimized, inputs)
    source_sess = session(ort, original)
    original_values = [
        source_sess.run([w.OUTPUT_NAME], {w.INPUT_NAME: x})[0] for x in inputs
    ]
    baseline = [delta_stats(np, source, optimized_output)
                for source, optimized_output in zip(original_values, optimized_values)]
    baseline_pass = strict_parity_pass(baseline, 1e-3, 1e-5)

    report = {
        "schema": "perilipin2-m3c-resize-proof-v1",
        "status": "BLOCKED",
        "original_sha256": receipt["model_sha256"],
        "optimized_sha256": w.sha256(optimized_path),
        "optimized_vs_frozen": baseline,
        "optimized_vs_frozen_parity_pass": baseline_pass,
        "resize_nodes": measurements,
        "test_inputs": "two deterministic synthetic normalized tensors, no microscopy",
        "compiled_for_kl720": False,
        "quantized": False,
        "bie_generated": False,
        "nef_compiled": False,
        "on_device": False,
        "biological_validation": False,
        "variants": {},
    }
    variants = {}
    for mode in ("static-sizes", "integer-scales"):
        try:
            model = make_variant(onnx, np, optimized, measurements, mode)
            candidate_sess = session(ort, model)
            candidate_rows = []
            for x, ref in zip(inputs, optimized_values):
                predicted = candidate_sess.run([w.OUTPUT_NAME], {w.INPUT_NAME: x})[0]
                candidate_rows.append(delta_stats(np, ref, predicted))
            accepted = strict_parity_pass(candidate_rows, 1e-6, 1e-7) and baseline_pass
            report["variants"][mode] = {
                "parity_vs_optimized": candidate_rows,
                "parity_pass": accepted,
                "reason": "candidate numerically checked against optimized and frozen ONNX",
            }
            if accepted:
                variants[mode] = model
        except (ValueError, RuntimeError, ort.OnnxRuntimeException) as exc:
            report["variants"][mode] = {
                "parity_pass": False,
                "error": "{}: {}".format(type(exc).__name__, exc),
            }

    if len(variants) == 0:
        print("M3C_RESIZE_PROOF_BLOCKED; no variant passed numeric gates")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    for mode, model in variants.items():
        filename = "nuclear_candidate.kneron.opt.{}.onnx".format(mode)
        path = staging / filename
        onnx.save(model, str(path))
        report["variants"][mode]["filename"] = filename
        report["variants"][mode]["sha256"] = w.sha256(path)
    report["status"] = "FP32_COMPATIBILITY_CANDIDATES" if variants else "BLOCKED"
    w.dump_json_new(staging / "M3C_RESIZE_PROOF.json", report)
    staging.rename(output)
    print("M3C_RESIZE_PROOF:", report["status"])
    print("Report:", output / "M3C_RESIZE_PROOF.json")
    for mode in ("static-sizes", "integer-scales"):
        print(mode, "parity:", report["variants"][mode]["parity_pass"])
    print("KL720 compiler tested: NO. Models are experimental and must not be deployed.")
    return 0 if variants else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, help="frozen evaluated M3C work folder")
    parser.add_argument("--out", required=True, help="new directory outside the M3C work folder")
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError, ImportError) as exc:
        print("M3C_RESIZE_PROOF_STOP: {}: {}".format(type(exc).__name__, exc),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
