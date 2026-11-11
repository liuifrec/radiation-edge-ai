"""Generate shareable synthetic ONNX repro candidates for Kneron KL720.

Creates tiny standalone ConvTranspose-only graphs and a strict ONNX Runtime
nearest-2x equivalence receipt. No trained weights, microscopy images, donor
data or proprietary Perilipin2 model bytes are read or committed.

These models are NOT proven to reproduce the vendor failure until separately
evaluated in the pinned KL720 toolchain. Generation is host-only: no Docker,
BIE, NEF, PTQ or physical accelerator inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Deliberate small controls and one geometry-matched first decoder stage.
# Never import or load the user's frozen segmentation model.
CASES = (
    ("control_c1_h8", 1, 8),
    ("group_c24_h8", 24, 8),
    ("group_c96_h8", 96, 8),
    ("stage_c96_h32", 96, 32),
)
SCHEMA = "perilipin2-kneron-synthetic-deconv-repro-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(onnx, np, channels, height):
    import onnx.helper as helper
    from onnx import TensorProto, numpy_helper

    if channels <= 0 or height <= 0:
        raise ValueError("Invalid dimensions")
    shape = [1, channels, height, height]
    target = [1, channels, height * 2, height * 2]
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, target)
    weights = np.ones((channels, 1, 2, 2), dtype=np.float32)
    kernel = numpy_helper.from_array(weights, name="fixed_all_ones_kernel")
    conv = helper.make_node(
        "ConvTranspose",
        inputs=["input", "fixed_all_ones_kernel"],
        outputs=["output"],
        name="synthetic_stride2_grouped_deconvolution",
        strides=[2, 2],
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],
        group=channels,
    )
    graph = helper.make_graph(
        [conv], "kneron_kl720_synthetic_convtranspose", [x], [y],
        initializer=[kernel],
    )
    model = helper.make_model(graph, producer_name="radiation-edge-ai-synthetic-repro",
                              opset_imports=[helper.make_opsetid("", 11)])
    # Explicit IR version in the range historically supported by vendor ONNX 1.7.
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def run(out_dir):
    output = Path(out_dir).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if output == repo_root or repo_root in output.parents:
        raise ValueError("Generated ONNX files must stay outside the Git repository")
    if output.exists() or output.is_symlink():
        raise ValueError("Output already exists; never overwrite evidence: " + str(output))
    stage = output.with_name(output.name + ".BUILDING")
    if stage.exists() or stage.is_symlink():
        raise ValueError("Staging directory already exists: " + str(stage))
    import numpy as np
    import onnx
    import onnxruntime as ort

    output.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=False, exist_ok=False)
    receipt = {
        "schema": SCHEMA,
        "source": "fully synthetic; no microscopy images or learned model weights",
        "target": "KL720 Toolchain v0.33.1 (not evaluated by this generator)",
        "vendor_evaluate_performed": False,
        "quantized": False, "nef_compiled": False,
        "physical_kl720_inference": False,
        "cases": {},
    }
    for name, channels, height in CASES:
        model = generate(onnx, np, channels, height)
        path = stage / (name + ".onnx")
        onnx.save(model, str(path))
        onnx.checker.check_model(onnx.load(str(path)))
        rng = np.random.default_rng(720 + channels + height)
        x = rng.uniform(-0.5, 0.5, size=(1, channels, height, height)).astype(np.float32)
        expected = np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"])
        observed = session.run(["output"], {"input": x})[0]
        if observed.shape != expected.shape:
            raise ValueError("Synthetic ConvTranspose output shape differs")
        maximum_error = float(np.max(np.abs(observed.astype(np.float64)
                                           - expected.astype(np.float64))))
        if maximum_error != 0:
            raise ValueError("Synthetic ConvTranspose not bit-identical: " + name)
        receipt["cases"][name] = {
            "filename": path.name, "sha256": sha256(path),
            "operator": "ConvTranspose", "group": channels,
            "input_shape": list(x.shape), "output_shape": list(observed.shape),
            "kernel_shape": [2, 2], "strides": [2, 2],
            "pads": [0, 0, 0, 0],
            "fixed_weight_shape": [channels, 1, 2, 2],
            "opset": 11, "ir_version": 7,
            "onnxruntime_vs_nearest_2x_max_abs_error": maximum_error,
            "vendor_compilation_tested": False,
        }
    report = stage / "SYNTHETIC_REPRO_MANIFEST.json"
    with report.open("x", encoding="utf-8") as fh:
        json.dump(receipt, fh, sort_keys=True, indent=2, allow_nan=False)
        fh.write("\n")
    stage.rename(output)
    print("M3C_SYNTHETIC_CONVTRANSPOSE_MODELS_GENERATED")
    print("Folder:", output)
    for name in receipt["cases"]:
        print("  {}: ONNX Runtime nearest-2x parity PASS".format(name))
    print("KL720 compilation: NOT TESTED; share only if requested by engineering.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="new scratch folder, not the source tree")
    args = parser.parse_args()
    try:
        return run(args.out)
    except Exception as exc:
        print("M3C_SYNTHETIC_REPRO_STOP: {}: {}".format(type(exc).__name__, exc),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
