"""Verified native feature decoding and regression-head attribution for training diagnostics."""

from itertools import permutations

import numpy as np


def decode_layer(raw, native_float, native_size, shape_info, descriptor):
    """Decode signed words by named NCHW channel and verify against the native text."""
    n, c, h, w = shape_info["hw_shape"]
    shape = tuple(shape_info["onnx_shape"])
    if n != 1 or tuple(int(k) for k in native_size.split()) != (c, w, h, n):
        raise ValueError("Native geometry differs from the bound hardware shape")
    if np.prod(shape) != c * h * w or shape_info["hw_c_in_onnx"] != (
        [1] if len(shape) >= 2 else [0]
    ):
        raise ValueError("Channel axis or logical geometry disagrees with metadata")
    if tuple(shape) != (n, c, h, w)[: len(shape)]:
        raise ValueError("Logical dimensions differ from native dimensions")
    (bits,) = descriptor["output_datapath_bitwidth"]
    dtype = {8: "i1", 15: "<i2", 16: "<i2"}[bits]
    codes = np.frombuffer(raw, dtype=dtype)
    if len(codes) != c * h * w:
        raise ValueError("Signed storage width or element count differs from descriptor")
    codes = codes.reshape(c, -1)
    scales = np.asarray(descriptor["output_scale"], dtype=np.float32)
    radix = np.asarray(descriptor["output_datapath_radix"])
    if (
        scales.shape != (1, c)
        or radix.shape != (1, c)
        or not np.issubdtype(radix.dtype, np.integer)
    ):
        raise ValueError("Expected one scale and one integer radix per channel")
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Invalid fixed-point scale")
    power = np.exp2(radix.T.astype(np.float32))
    decoded = codes.astype(np.float32) / power / scales.T
    floats = np.fromstring(native_float, sep=" ", dtype=np.float32)
    if floats.size != codes.size or not np.allclose(decoded.ravel(), floats, rtol=0, atol=1e-8):
        raise ValueError("Native floats disagree with independently decoded signed words")
    bound = 2 ** (bits - 1)
    if codes.min() < -bound or codes.max() >= bound:
        raise ValueError("Value outside declared signed bit width")
    delta = np.abs(decoded.ravel() - floats)
    return dict(
        codes=codes.reshape(shape),
        decoded=decoded.reshape(shape),
        native=floats.reshape(shape),
        bits=bits,
        step=(1 / power / scales.T).ravel(),
        max_decode_abs=float(delta.max()),
    )


def layout_scores(native, reference):
    """Audit ordering with actual FP32 values for all distinct axis permutations."""
    if native.shape != reference.shape or not np.isfinite(reference).all():
        raise ValueError("Invalid reference geometry or values")
    if native.ndim != 4:
        return {"CHW": 0.0} if native.size == 1 else {"C": float(np.abs(native - reference).mean())}
    lengths = dict(zip("CHW", native.shape[1:]))
    flat = native.reshape(-1)
    scores = {}
    for order in permutations("CHW"):
        view = flat.reshape(tuple(lengths[k] for k in order)).transpose(
            [order.index(k) for k in "CHW"]
        )
        scores["".join(order)] = float(np.abs(view.astype(np.float64) - reference[0]).mean())
    if scores["CHW"] > min(scores.values()) + 1e-14:
        raise ValueError("An alternative spatial ordering better matches the FP32 boundary")
    return scores


def head_decomposition(
    original, optimized, rounded, spatial, pooled, fixed_head, fixed_relu, terminal
):
    """Close signed preactivation and rectified endpoint effects on the same FP32 reference."""
    vals = (original, optimized, rounded, spatial, pooled, fixed_head, fixed_relu, terminal)
    if not np.isfinite(vals).all():
        raise ValueError("Nonfinite scalar")

    def h(x):
        return max(0.0, x)

    effects = dict(
        optimization=h(optimized) - h(original),
        input_rounding=rounded - optimized,
        extractor=spatial - rounded,
        pooling=pooled - spatial,
        head=fixed_head - pooled,
        rectification=(h(fixed_head) - fixed_head) - (h(optimized) - optimized),
        relu_encoding=fixed_relu - h(fixed_head),
        final_transfer=terminal - fixed_relu,
    )
    endpoint = dict(
        optimization=h(optimized) - h(original),
        input_rounding=h(rounded) - h(optimized),
        extractor=h(spatial) - h(rounded),
        pooling=h(pooled) - h(spatial),
        head=h(fixed_head) - h(pooled),
        relu_encoding=fixed_relu - h(fixed_head),
        final_transfer=terminal - fixed_relu,
    )
    for values in (effects, endpoint):
        if abs(sum(values.values()) - (terminal - h(original))) > 1e-8:
            raise ValueError("Attribution does not reconstruct the observed error")
    return effects, endpoint


class FeatureGraph:
    """Add output declarations to the FP32 graph; verify its unchanged endpoint at each use."""

    def __init__(self, path, names):
        import onnx
        import onnxruntime as ort

        model = onnx.load(str(path))
        self.names = names
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.options = options
        self.full = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        # Save the original graph for extracting suffixes. Inference adds type/shape annotations.
        self.source = onnx.shape_inference.infer_shapes(model)
        typed = {
            v.name: v for v in list(self.source.graph.value_info) + list(self.source.graph.output)
        }
        for name in names.values():
            if name not in ("input", "burden"):
                model.graph.output.add().CopyFrom(typed[name])
        self.tapped = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        (node,) = [n for n in model.graph.node if n.op_type == "Gemm"]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get("alpha", 1) != 1 or attrs.get("beta", 1) != 1 or attrs.get("transA", 0) != 0:
            raise ValueError("Unexpected affine head attributes")
        constants = {v.name: onnx.numpy_helper.to_array(v) for v in model.graph.initializer}
        for n in model.graph.node:
            if n.op_type == "Constant" and n.output[0] in node.input[1:]:
                constants[n.output[0]] = onnx.numpy_helper.to_array(
                    onnx.helper.get_attribute_value(n.attribute[0])
                )
        weights = constants[node.input[1]]
        weights = weights.T if attrs.get("transB", 0) else weights
        if (
            weights.shape[1] != 1
            or constants[node.input[2]].size != 1
            or node.input[0] != names["flatten"]
        ):
            raise ValueError("Unexpected input or affine head width")
        self.w = weights[:, 0].astype(np.float64)
        self.b = float(constants[node.input[2]].item())

    def run(self, tensor):
        selected = {k: n for k, n in self.names.items() if k != "input"}
        feed = {"input": tensor}
        results = dict(zip(selected, self.tapped.run(list(selected.values()), feed)), input=tensor)
        if not np.array_equal(results["burden"], self.full.run(["burden"], feed)[0]):
            raise ValueError("FP32 output changed with diagnostic taps")
        if abs(self.affine(results["flatten"]) - results["head"].item()) > 1e-5:
            raise ValueError("Analytic FP32 head replay failed")
        return results

    def affine(self, x):
        return float(np.asarray(x).reshape(-1).astype(np.float64) @ self.w + self.b)

    def suffix(self, label):
        import onnx
        import onnxruntime as ort

        segment = onnx.utils.Extractor(self.source).extract_model(
            [self.names[label]], [self.names["head"]]
        )
        if [v.name for v in segment.graph.input] != [self.names[label]]:
            raise ValueError("Requested boundary does not define a self-contained suffix")
        return ort.InferenceSession(
            segment.SerializeToString(), self.options, providers=["CPUExecutionProvider"]
        )
