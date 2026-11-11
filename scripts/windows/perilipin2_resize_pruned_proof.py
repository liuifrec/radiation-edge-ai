"""Prune dead ONNX shape branches after the already proven static-Resize rewrite.

Windows/Alienware host only. Preserves the source Perilipin2 ONNX, frozen
M3C eval folder, original Resize proof and all failed KL720 vendor artifacts.
Produces a new proof-compatible folder, but NEVER runs Docker or the KL720.

No model weights are altered: live ONNX graph paths are copied exactly.
The numerical gate compares the new model with the original, Kneron-optimized,
and previously parity-verified static-sizes model on two locked test inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from radiation_edge_ai.perilipin2 import windows_cli as w  # noqa: E402

import perilipin2_resize_proof as prior  # noqa: E402
import perilipin2_resize_vendor_gate as vendor  # noqa: E402

MODEL_NAME = "nuclear_candidate.kneron.opt.static-sizes.pruned.onnx"
PROOF_NAME = "M3C_RESIZE_PROOF.json"


def reachable_nodes(graph):
    """Find exact producer dependency closure of graph outputs.

    The input ONNX must be a regular feed-forward graph. Nested control flow
    would require scope-aware dependency analysis and is rejected.
    """
    producers = {}
    for index, node in enumerate(graph.node):
        for attr in node.attribute:
            if attr.type in (5, 10):  # ONNX AttributeProto GRAPH and GRAPHS
                raise ValueError("Control-flow/subgraph attribute not supported by pruner")
        for output in node.output:
            if not output:
                continue
            if output in producers:
                raise ValueError("Duplicate ONNX tensor producer: " + output)
            producers[output] = index
    available_inputs = {x.name for x in graph.input}
    available_initializers = {x.name for x in graph.initializer}
    if getattr(graph, "sparse_initializer", None):
        if len(graph.sparse_initializer) != 0:
            raise ValueError("Sparse initializers not supported by pruner")
    live_names = set()
    reachable = set()
    pending = [x.name for x in graph.output]
    while pending:
        tensor = pending.pop()
        if not tensor or tensor in live_names:
            continue
        live_names.add(tensor)
        if tensor in producers:
            index = producers[tensor]
            if index not in reachable:
                reachable.add(index)
                pending.extend(x for x in graph.node[index].input if x)
        elif tensor not in available_inputs and tensor not in available_initializers:
            raise ValueError("Unresolved live ONNX tensor: " + tensor)
    return reachable, live_names


def prune_dead(model):
    """Graph surgery is deletion-only; keep surviving node order and attributes."""
    import onnx
    cleaned = onnx.ModelProto()
    cleaned.CopyFrom(model)
    graph = cleaned.graph
    keep_indices, live_names = reachable_nodes(graph)
    old_nodes = list(graph.node)
    discarded = [node for i, node in enumerate(old_nodes) if i not in keep_indices]
    if not discarded:
        raise ValueError("No unreachable nodes: dead-graph hypothesis not applicable")
    del graph.node[:]
    graph.node.extend(node for i, node in enumerate(old_nodes) if i in keep_indices)

    old_initializers = list(graph.initializer)
    live_initializers = [item for item in old_initializers if item.name in live_names]
    del graph.initializer[:]
    graph.initializer.extend(live_initializers)

    old_value_info = list(graph.value_info)
    del graph.value_info[:]
    graph.value_info.extend(item for item in old_value_info if item.name in live_names)

    removed_ops = dict(sorted(Counter(node.op_type for node in discarded).items()))
    if not any(name in removed_ops for name in ("Pad", "Shape", "Gather", "Concat")):
        raise ValueError("No shape-branch operators removed; refuse speculative rewrite")

    onnx.checker.check_model(cleaned)
    return cleaned, {
        "nodes_before": len(old_nodes),
        "nodes_after": len(graph.node),
        "nodes_removed": len(discarded),
        "removed_operators": removed_ops,
        "initializers_before": len(old_initializers),
        "initializers_after": len(live_initializers),
        "value_info_before": len(old_value_info),
        "value_info_after": len(graph.value_info),
        "method": "backward graph-output dependency closure; deletion only",
    }


def run(args):
    import numpy as np
    import onnx
    import onnxruntime as ort

    work, host = w.check_bundle(args.work)
    proof_file, source_path, source_entry = vendor.checked_proof(
        args.proof_dir, "static-sizes")
    parent = vendor.read_json(proof_file)
    if host["model_sha256"] != parent["original_sha256"]:
        raise ValueError("Original frozen M3C source disagrees with Resize proof")
    optimized_path = work / w.WORK_OPT
    if w.sha256(optimized_path) != parent["optimized_sha256"]:
        raise ValueError("Original Kneron optimized model changed after Resize proof")
    if not parent["variants"]["static-sizes"]["parity_pass"]:
        raise ValueError("Source static-sizes variant did not pass parity")
    if w.sha256(source_path) != source_entry["sha256"]:
        raise ValueError("Source static-sizes bytes were modified")

    output = Path(args.out).expanduser().resolve()
    work = work.resolve(strict=True)
    source_dir = Path(args.proof_dir).expanduser().resolve(strict=True)
    for previous in (work, source_dir):
        if output == previous or previous in output.parents or output in previous.parents:
            raise ValueError("Output must be a separate new sibling folder")
    if output.exists() or output.is_symlink():
        raise ValueError("Refusing existing output folder: " + str(output))
    staging = output.with_name(output.name + ".BUILDING")
    if staging.exists() or staging.is_symlink():
        raise ValueError("Existing staging folder: " + str(staging))

    source = onnx.load(str(source_path))
    onnx.checker.check_model(source)
    vendor.resize_graph_contract(source, "static-sizes", parent["resize_nodes"])
    pruned, audit = prune_dead(source)
    vendor.resize_graph_contract(pruned, "static-sizes", parent["resize_nodes"])
    w.validate_contract(w.graph_contract(pruned))
    # A deletion-only rewrite must not remove or alter active Resize nodes.
    if len([node for node in pruned.graph.node if node.op_type == "Resize"]) != 4:
        raise ValueError("Required Resize nodes were pruned")

    frozen = onnx.load(str(work / w.WORK_MODEL))
    optimized = onnx.load(str(optimized_path))
    inputs = prior.two_inputs(np)
    models = {
        "frozen": prior.session(ort, frozen),
        "optimized": prior.session(ort, optimized),
        "static": prior.session(ort, source),
        "pruned": prior.session(ort, pruned),
    }
    base_rows, static_rows, pruned_rows, direct_rows = [], [], [], []
    for x in inputs:
        predictions = {
            name: sess.run([w.OUTPUT_NAME], {w.INPUT_NAME: x})[0]
            for name, sess in models.items()
        }
        base_rows.append(prior.delta_stats(
            np, predictions["frozen"], predictions["optimized"]))
        static_rows.append(prior.delta_stats(
            np, predictions["optimized"], predictions["static"]))
        pruned_rows.append(prior.delta_stats(
            np, predictions["optimized"], predictions["pruned"]))
        direct_rows.append(prior.delta_stats(
            np, predictions["static"], predictions["pruned"]))

    baseline_ok = prior.strict_parity_pass(base_rows, 1e-3, 1e-5)
    static_ok = prior.strict_parity_pass(static_rows, 1e-6, 1e-7)
    pruned_ok = prior.strict_parity_pass(pruned_rows, 1e-6, 1e-7)
    direct_ok = prior.strict_parity_pass(direct_rows, 1e-6, 1e-7)
    accepted = baseline_ok and static_ok and pruned_ok and direct_ok
    report = {
        "schema": vendor.PROOF_SCHEMA,
        "status": "FP32_COMPATIBILITY_CANDIDATES" if accepted else "BLOCKED",
        "original_sha256": parent["original_sha256"],
        "optimized_sha256": parent["optimized_sha256"],
        "optimized_vs_frozen_parity_pass": baseline_ok,
        "optimized_vs_frozen": base_rows,
        "resize_nodes": parent["resize_nodes"],
        "parent_proof_sha256": w.sha256(proof_file),
        "parent_candidate_sha256": source_entry["sha256"],
        "pruning": audit,
        "test_inputs": "two deterministic synthetic normalized tensors, no microscopy",
        "compiled_for_kl720": False,
        "quantized": False, "bie_generated": False, "nef_compiled": False,
        "on_device": False, "biological_validation": False,
        "variants": {
            "static-sizes": {
                "parity_pass": accepted,
                "parity_vs_optimized": pruned_rows,
                "parity_vs_unpruned_static": direct_rows,
                "unpruned_static_parity_pass": static_ok,
                "reason": "Deletion-only dead-graph elimination; repeat strict FP32 parity",
            }
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    if accepted:
        candidate_path = staging / MODEL_NAME
        onnx.save(pruned, str(candidate_path))
        # The hash is computed from the serialized artifact, not memory.
        onnx.checker.check_model(onnx.load(str(candidate_path)))
        report["variants"]["static-sizes"]["filename"] = MODEL_NAME
        report["variants"]["static-sizes"]["sha256"] = w.sha256(candidate_path)
    w.dump_json_new(staging / PROOF_NAME, report)
    staging.rename(output)
    print("M3C_PRUNED_RESIZE_PROOF:", report["status"])
    print("Dead graph nodes removed:", audit["nodes_removed"])
    print("Removed operators:", json.dumps(audit["removed_operators"], sort_keys=True))
    print("FP32 parity preserved:", accepted)
    print("Report:", output / PROOF_NAME)
    print("KL720 compiler tested: NO; do not deploy, quantize, or infer.")
    return 0 if accepted else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, help="frozen M3C eval work folder")
    parser.add_argument("--proof-dir", required=True, help="original successful Resize proof")
    parser.add_argument("--out", required=True, help="new output folder, never an existing path")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print("M3C_PRUNED_RESIZE_PROOF_STOP: {}: {}".format(type(exc).__name__, exc),
              file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
