"""Prove an exact nearest-2x replacement that avoids ONNX Resize entirely.

Convert the four already-pruned Perilipin2 decoder Resize ops into grouped
depthwise ConvTranspose with stride (2,2), 2x2 all-one, fixed kernels.
For each spatial element only one source element contributes to each output
location, so this implements nearest/asymmetric/floor 2x tiling.

This is an EXPERIMENTAL graph-representation test, NOT hardware support.
All learned weights and live tensor names are preserved; only four Resize
operators and their now-unused size initializers are changed. Work folders,
frozen ONNX and earlier reports remain immutable. No Docker, BIE, NEF, PTQ.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from radiation_edge_ai.perilipin2 import windows_cli as w  # noqa: E402

import perilipin2_resize_proof as prior  # noqa: E402
import perilipin2_resize_vendor_gate as vendor  # noqa: E402

MODEL_NAME = "nuclear_candidate.kneron.opt.depthwise-deconv.onnx"
PROOF_NAME = "M3C_RESIZE_PROOF.json"


def validated_geometry(measurements):
    """Check the four previously measured fixed-size decoder stages."""
    if len(measurements) != 4:
        raise ValueError("Exactly four decoder stages required")
    results = []
    names = set()
    for entry in measurements:
        name = entry.get("node_name")
        before, after = entry.get("input_shape"), entry.get("target_shape")
        scales = entry.get("integer_scales")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Nonunique or missing original Resize node name")
        names.add(name)
        if (not isinstance(before, list) or not isinstance(after, list)
                or len(before) != 4 or len(after) != 4
                or any(type(x) is not int or x <= 0 for x in before + after)):
            raise ValueError("Invalid decoder tensor dimensions")
        n, channels, height, width = before
        if (after != [n, channels, 2 * height, 2 * width]
                or scales != [1, 1, 2, 2]):
            raise ValueError("Not an exact 2x nearest spatial upsampling: " + name)
        if n != 1:
            raise ValueError("Deployment batch dimension must equal one")
        results.append({"node_name": name, "channels": channels,
                        "input_shape": before, "target_shape": after})
    return results


def convert(onnx, np, model, measurements):
    """Replace only the four 2x nearest Resize ops; no learned parameters edit."""
    geometries = validated_geometry(measurements)
    replacement = {item["node_name"]: item for item in geometries}
    candidate = onnx.ModelProto()
    candidate.CopyFrom(model)
    graph = candidate.graph
    existing_initializers = {entry.name for entry in graph.initializer}
    old_ops = dict(sorted(Counter(node.op_type for node in graph.node).items()))
    removed_size_names = []
    created = []
    for index, node in enumerate(graph.node):
        if node.op_type != "Resize":
            continue
        if node.name not in replacement:
            raise ValueError("Unexpected Resize node: " + node.name)
        data = replacement.pop(node.name)
        if len(node.input) != 4 or len(node.output) != 1:
            raise ValueError("Unexpected Resize IO contract at " + node.name)
        size_name = node.input[3]
        if size_name not in existing_initializers:
            raise ValueError("Resize sizes are not static proof constants: " + node.name)
        weight_name = "m3c_deconv_kernel_{}".format(index)
        if weight_name in existing_initializers:
            raise ValueError("Generated ConvTranspose weight collision")
        weight = np.ones((data["channels"], 1, 2, 2), dtype=np.float32)
        new_node = onnx.helper.make_node(
            "ConvTranspose", inputs=[node.input[0], weight_name],
            outputs=list(node.output), name=node.name,
            group=data["channels"], kernel_shape=[2, 2],
            strides=[2, 2], pads=[0, 0, 0, 0],
        )
        graph.node[index].CopyFrom(new_node)
        graph.initializer.append(onnx.numpy_helper.from_array(weight, weight_name))
        removed_size_names.append(size_name)
        created.append({"node_name": node.name, "input_name": node.input[0],
                        "output_name": node.output[0], "weight_name": weight_name,
                        "weight_shape": list(weight.shape),
                        "input_shape": data["input_shape"],
                        "target_shape": data["target_shape"]})
        existing_initializers.add(weight_name)
    if replacement or len(created) != 4:
        raise ValueError("Not all original Resize nodes were transformed")
    if len(set(removed_size_names)) != 4:
        raise ValueError("Expected four distinct size initializers")
    for node in graph.node:
        if any(name in node.input for name in removed_size_names):
            raise ValueError("Former Resize size tensor remains active")
    old_initializers = list(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(
        item for item in old_initializers if item.name not in removed_size_names
    )
    # A legacy ONNX exporter may also list initializers as graph inputs.
    # Remove only the now-dead size parameters, never an actual feature input.
    old_inputs = list(graph.input)
    del graph.input[:]
    graph.input.extend(
        item for item in old_inputs if item.name not in removed_size_names
    )
    old_info = list(graph.value_info)
    del graph.value_info[:]
    graph.value_info.extend(
        item for item in old_info if item.name not in removed_size_names
    )
    if any(node.op_type == "Resize" for node in graph.node):
        raise ValueError("Resize remains after rewrite")
    onnx.checker.check_model(candidate)
    vendor.resize_graph_contract(candidate, "depthwise-deconv", measurements)
    return candidate, {
        "replacements": created,
        "nodes_before": len(model.graph.node),
        "nodes_after": len(candidate.graph.node),
        "operators_before": old_ops,
        "operators_after": dict(sorted(Counter(n.op_type for n in graph.node).items())),
        "former_size_initializers_removed": sorted(removed_size_names),
        "weight_type": "float32 all-ones, grouped [C,1,2,2]",
        "kernel_stride": [2, 2],
        "trained_parameter_values_changed": False,
        "description": "fixed nearest-2x -> nonoverlapping depthwise ConvTranspose",
    }


def run(args):
    import numpy as np
    import onnx
    import onnxruntime as ort

    work, receipt = w.check_bundle(args.work)
    proof_file, source_path, parent_candidate = vendor.checked_proof(
        args.pruned_proof, "static-sizes"
    )
    parent = vendor.read_json(proof_file)
    if parent.get("pruning", {}).get("nodes_removed", 0) < 1:
        raise ValueError("Expected previously pruned, parity-proven Resize graph")
    if receipt["model_sha256"] != parent["original_sha256"]:
        raise ValueError("Frozen original model disagrees with source proof")
    optimized_path = work / w.WORK_OPT
    if w.sha256(optimized_path) != parent["optimized_sha256"]:
        raise ValueError("Kneron optimized ONNX changed after proof")

    output = Path(args.out).expanduser().resolve()
    source_dir = Path(args.pruned_proof).expanduser().resolve(strict=True)
    for previous in (work.resolve(strict=True), source_dir):
        if output == previous or previous in output.parents or output in previous.parents:
            raise ValueError("Output must be a NEW sibling directory")
    if output.exists() or output.is_symlink():
        raise ValueError("Refusing existing output: " + str(output))
    stage = output.with_name(output.name + ".BUILDING")
    if stage.exists() or stage.is_symlink():
        raise ValueError("Existing BUILDING folder: " + str(stage))

    source = onnx.load(str(source_path))
    candidate, audit = convert(onnx, np, source, parent["resize_nodes"])
    w.validate_contract(w.graph_contract(candidate))
    original = onnx.load(str(work / w.WORK_MODEL))
    optimized = onnx.load(str(optimized_path))
    sessions = {
        "frozen": prior.session(ort, original),
        "optimized": prior.session(ort, optimized),
        "source": prior.session(ort, source),
        "candidate": prior.session(ort, candidate),
    }
    comparisons = {"optimized_vs_frozen": [], "source_vs_optimized": [],
                   "candidate_vs_source": [], "candidate_vs_optimized": []}
    for x in prior.two_inputs(np):
        outputs = {
            name: sess.run([w.OUTPUT_NAME], {w.INPUT_NAME: x})[0]
            for name, sess in sessions.items()
        }
        for title, left, right in (
            ("optimized_vs_frozen", "frozen", "optimized"),
            ("source_vs_optimized", "optimized", "source"),
            ("candidate_vs_source", "source", "candidate"),
            ("candidate_vs_optimized", "optimized", "candidate"),
        ):
            comparisons[title].append(
                prior.delta_stats(np, outputs[left], outputs[right])
            )
    baseline_ok = prior.strict_parity_pass(
        comparisons["optimized_vs_frozen"], 1e-3, 1e-5
    )
    source_ok = prior.strict_parity_pass(
        comparisons["source_vs_optimized"], 1e-6, 1e-7
    )
    candidate_ok = prior.strict_parity_pass(
        comparisons["candidate_vs_source"], 1e-6, 1e-7
    ) and prior.strict_parity_pass(
        comparisons["candidate_vs_optimized"], 1e-6, 1e-7
    )
    accepted = bool(baseline_ok and source_ok and candidate_ok)
    report = {
        "schema": vendor.PROOF_SCHEMA,
        "status": "FP32_COMPATIBILITY_CANDIDATES" if accepted else "BLOCKED",
        "original_sha256": receipt["model_sha256"],
        "optimized_sha256": w.sha256(optimized_path),
        "optimized_vs_frozen_parity_pass": baseline_ok,
        "optimized_vs_frozen": comparisons["optimized_vs_frozen"],
        "resize_nodes": parent["resize_nodes"],
        "parent_pruned_proof_sha256": w.sha256(proof_file),
        "parent_pruned_candidate_sha256": parent_candidate["sha256"],
        "deconv": audit,
        "test_inputs": "two deterministic synthetic normalized tensors, no microscopy",
        "compiled_for_kl720": False, "quantized": False, "bie_generated": False,
        "nef_compiled": False, "on_device": False, "biological_validation": False,
        "variants": {
            "depthwise-deconv": {
                "parity_pass": accepted,
                "parity_vs_optimized": comparisons["candidate_vs_optimized"],
                "parity_vs_pruned_static": comparisons["candidate_vs_source"],
                "reason": "Exact 2x nonoverlapping ConvTranspose replacement of nearest Resize",
            }
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=False, exist_ok=False)
    if accepted:
        destination = stage / MODEL_NAME
        onnx.save(candidate, str(destination))
        onnx.checker.check_model(onnx.load(str(destination)))
        report["variants"]["depthwise-deconv"]["filename"] = MODEL_NAME
        report["variants"]["depthwise-deconv"]["sha256"] = w.sha256(destination)
    w.dump_json_new(stage / PROOF_NAME, report)
    stage.rename(output)
    print("M3C_DEPTHWISE_DECONV_PROOF:", report["status"])
    print("Operators:", audit["operators_before"], "->", audit["operators_after"])
    print("Candidate vs pruned static FP32 parity:", candidate_ok)
    print("Output:", output / PROOF_NAME)
    print("KL720 vendor compilation, PTQ and device inference: NOT PERFORMED")
    return 0 if accepted else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--pruned-proof", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print("M3C_DEPTHWISE_DECONV_PROOF_STOP: {}: {}".format(
            type(exc).__name__, exc), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
