"""Five dependency-closed KL720 compiler probes from a parity-proven Perilipin2 ONNX.

Uses ORIGINAL learned weights and source image input. Does not compile, quantize,
produce NEF, access KL720, or upload private model artifacts to Git.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import perilipin2_resize_vendor_gate as vendor

SCHEMA = "perilipin2-kl720-stagewise-proof-v1"
STAGES = ("encoder_before_first_upsample", "upsample_1", "upsample_2",
          "upsample_3", "upsample_4")
REPORT = "STAGEWISE_PROOF.json"


def dependency_closure(graph, endpoint):
    """Backward dependency closure; preserve original node order and weights."""
    producers = {}
    for index, node in enumerate(graph.node):
        for attribute in node.attribute:
            if attribute.type in (5, 10):
                raise ValueError("Nested graph/control flow not supported")
        for name in node.output:
            if name in producers:
                raise ValueError("Duplicate tensor producer: " + name)
            if name:
                producers[name] = index
    inputs = {x.name for x in graph.input}
    initializers = {x.name for x in graph.initializer}
    if len(inputs) != 1 or graph.sparse_initializer:
        raise ValueError("Expected one model input and no sparse initializers")
    selected, touched = set(), set()
    pending = [endpoint]
    while pending:
        name = pending.pop()
        if name in touched:
            continue
        touched.add(name)
        if name in producers:
            index = producers[name]
            if index not in selected:
                selected.add(index)
                pending.extend(x for x in graph.node[index].input if x)
        elif name not in inputs and name not in initializers:
            raise ValueError("Unresolved dependency: " + name)
    if not selected:
        raise ValueError("No nodes in probe")
    return selected, touched


def isolate(onnx, source, endpoint, shape):
    candidate = onnx.ModelProto()
    candidate.CopyFrom(source)
    graph = candidate.graph
    indices, tensors = dependency_closure(source.graph, endpoint)
    original_nodes = list(source.graph.node)
    del graph.node[:]
    graph.node.extend(n for i, n in enumerate(original_nodes) if i in indices)
    initializers = list(source.graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(n for n in initializers if n.name in tensors)
    info = list(source.graph.value_info)
    del graph.value_info[:]
    graph.value_info.extend(x for x in info if x.name in tensors and x.name != endpoint)
    del graph.output[:]
    graph.output.append(onnx.helper.make_tensor_value_info(
        endpoint, onnx.TensorProto.FLOAT, shape))
    onnx.checker.check_model(candidate)
    ops = dict(sorted(Counter(n.op_type for n in graph.node).items()))
    return candidate, {"node_count": len(graph.node),
                       "initializers": len(graph.initializer),
                       "convtranspose_count": ops.get("ConvTranspose", 0),
                       "operators": ops}


def cutpoints(proof, model):
    data = proof.get("resize_nodes", [])
    if len(data) != 4 or len({x.get("node_name") for x in data}) != 4:
        raise ValueError("Four original decoder nodes required")
    data = sorted(data, key=lambda x: x["input_shape"][2])
    expected = [(96, 32), (72, 64), (48, 128), (24, 256)]
    nodes = {n.name: n for n in model.graph.node}
    result = []
    for i, (entry, (c, h)) in enumerate(zip(data, expected)):
        if entry["input_shape"] != [1, c, h, h] or entry["target_shape"] != [1, c, 2*h, 2*h]:
            raise ValueError("Decoder shape no longer matches frozen proof")
        node = nodes.get(entry["node_name"])
        if node is None or node.op_type != "ConvTranspose" or len(node.input) != 2 or len(node.output) != 1:
            raise ValueError("Expected named frozen ConvTranspose node")
        if i == 0:
            result.append((STAGES[0], node.input[0], entry["input_shape"]))
        result.append((STAGES[i+1], node.output[0], entry["target_shape"]))
    return result


def verify_monotone(rows, original_output):
    if len(rows) != 5 or tuple(row["stage"] for row in rows) != STAGES:
        raise ValueError("Expected ordered encoder control and four upsampling stages")
    previous = 0
    for i, row in enumerate(rows):
        if (row["node_count"] <= previous or row["convtranspose_count"] != i or
                row["tensor"] == original_output):
            raise ValueError("Non-monotone or incompatible staged topology")
        previous = row["node_count"]


def source_inputs(np):
    rng = np.random.default_rng(20261110)
    a = rng.uniform(-0.5, 0.5, (1, 3, 512, 512)).astype(np.float32)
    yy = np.linspace(-0.5, 0.5, 512, dtype=np.float32)
    xx = np.linspace(0.5, -0.5, 512, dtype=np.float32)
    b = np.empty_like(a)
    for c in range(3):
        b[0, c] = np.clip(yy[:, None] + xx[None, :] + c*0.07, -0.5, 0.5)
    return a, b


def exact_fp32_parity(onnx, ort, np, source, probes, records):
    instrumented = onnx.ModelProto()
    instrumented.CopyFrom(source)
    already = {x.name for x in instrumented.graph.output}
    for r in records:
        if r["tensor"] not in already:
            instrumented.graph.output.append(onnx.helper.make_tensor_value_info(
                r["tensor"], onnx.TensorProto.FLOAT, r["shape"]))
    onnx.checker.check_model(instrumented)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    whole = ort.InferenceSession(instrumented.SerializeToString(), options,
                                 providers=["CPUExecutionProvider"])
    pieces = {stage: ort.InferenceSession(probe.SerializeToString(), options,
                                         providers=["CPUExecutionProvider"])
              for stage, probe in probes.items()}
    input_name = source.graph.input[0].name
    for r in records:
        r["parity"] = []
    for array in source_inputs(np):
        feed = {input_name: array}
        tensors = [r["tensor"] for r in records]
        expected = dict(zip(tensors, whole.run(tensors, feed)))
        for r in records:
            actual = pieces[r["stage"]].run([r["tensor"]], feed)[0]
            target = expected[r["tensor"]]
            if (list(actual.shape) != r["shape"] or not np.isfinite(actual).all() or
                    not np.array_equal(actual, target)):
                raise ValueError("Stage FP32 parity differs: " + r["stage"])
            r["parity"].append({"exact_values": True, "max_abs_error": 0.0})


def run(args):
    import numpy as np
    import onnx
    import onnxruntime as ort

    receipt_path, model_path, _ = vendor.checked_proof(args.proof_dir, "depthwise-deconv")
    proof = vendor.read_json(receipt_path)
    original = onnx.load(str(model_path))
    onnx.checker.check_model(original)
    root = Path(args.proof_dir).expanduser().resolve(strict=True)
    output = Path(args.out).expanduser().resolve()
    if (output.exists() or output.is_symlink() or output == root or
            root in output.parents or output in root.parents):
        raise ValueError("Use a NEW folder outside the original proof")
    staging = output.with_name(output.name + ".BUILDING")
    if staging.exists() or staging.is_symlink():
        raise ValueError("Existing BUILDING folder")
    candidates, records = {}, []
    for stage, tensor, shape in cutpoints(proof, original):
        subgraph, audit = isolate(onnx, original, tensor, shape)
        candidates[stage] = subgraph
        records.append({"stage": stage, "tensor": tensor, "shape": shape, **audit})
    verify_monotone(records, original.graph.output[0].name)
    exact_fp32_parity(onnx, ort, np, original, candidates, records)
    if vendor.sha256(model_path) != proof["variants"]["depthwise-deconv"]["sha256"]:
        raise ValueError("Source model changed during extraction")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(exist_ok=False)
    for r in records:
        r["filename"] = r["stage"] + ".onnx"
        destination = staging / r["filename"]
        onnx.save(candidates[r["stage"]], str(destination))
        onnx.checker.check_model(str(destination))
        r["sha256"] = vendor.sha256(destination)
    vendor.write_new_json(staging / REPORT, {
        "schema": SCHEMA, "status": "STAGED_FP32_PARITY_EXACT",
        "source_frozen_sha256": vendor.FROZEN_SOURCE_SHA,
        "source_onnx_sha256": vendor.sha256(model_path),
        "reference_proof_sha256": vendor.sha256(receipt_path),
        "stages": records, "new_decoder_rewrite": False,
        "learned_weights_retrained": False, "nef_compiled": False,
        "bie_generated": False, "physical_kl720_inference": False,
    })
    staging.rename(output)
    print("STAGEWISE_FP32_PROOF_PASS: 5 exact-parity stage probes")
    for row in records:
        print(row["stage"], "nodes:", row["node_count"],
              "ConvTranspose:", row["convtranspose_count"])
    print("Private outputs:", output)
    print("Vendor compiler, NEF, KL720 inference: NOT TESTED")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print("STAGEWISE_PROOF_STOP: {}: {}".format(type(exc).__name__, exc),
              file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
