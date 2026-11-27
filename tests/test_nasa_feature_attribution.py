"""Training-only quantization diagnostics must reject ambiguous geometry, graphs and cohorts."""

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
attribution = pytest.importorskip("radiation_edge_ai.nasa_bps.feature_attribution")
decode_layer = attribution.decode_layer
head_decomposition = attribution.head_decomposition
layout_scores = attribution.layout_scores


def test_decode_per_channel_signed_binary_and_boundaries():
    codes = np.array([[-128, 127], [1, -2]], dtype="i1")
    desc = dict(
        output_datapath_bitwidth=[8], output_datapath_radix=[[1, 2]], output_scale=[[2.0, 0.5]]
    )
    shape = dict(hw_shape=[1, 2, 1, 2], onnx_shape=[1, 2, 1, 2], hw_c_in_onnx=[1])
    native = b"-32.00000000 31.75000000 0.50000000 -1.00000000"
    result = decode_layer(codes.tobytes(), native, b"2 2 1 1", shape, desc)
    assert result["max_decode_abs"] == 0.0
    np.testing.assert_array_equal(result["codes"].ravel(), codes.ravel())
    wrong = dict(desc, output_scale=[[0.5, 2.0]])
    with pytest.raises(ValueError, match="Native floats"):
        decode_layer(codes.tobytes(), native, b"2 2 1 1", shape, wrong)
    with pytest.raises(ValueError, match="geometry"):
        decode_layer(codes.tobytes(), native, b"1 2 2 1", shape, desc)
    with pytest.raises(ValueError, match="width"):
        decode_layer(
            codes.tobytes(), native, b"2 2 1 1", shape, dict(desc, output_datapath_bitwidth=[16])
        )


def test_effective_15_bit_range_checked_independently_of_storage():
    shape = dict(hw_shape=[1, 1, 1, 1], onnx_shape=[1], hw_c_in_onnx=[0])
    desc = dict(output_datapath_bitwidth=[15], output_datapath_radix=[[0]], output_scale=[[1.0]])
    for code in (-16384, 16383):
        r = decode_layer(
            code.to_bytes(2, "little", signed=True), str(code).encode(), b"1 1 1 1", shape, desc
        )
        assert r["codes"].item() == code
    with pytest.raises(ValueError, match="bit width"):
        decode_layer(b"\x00\x40", b"16384", b"1 1 1 1", shape, desc)


def test_signed_and_endpoint_attribution_cross_zero():
    signed, endpoint = head_decomposition(1.0, 1.0, 0.8, -0.4, -0.3, -0.1, 0.0, 0.0)
    assert signed["pooling"] == pytest.approx(0.1)
    assert signed["head"] == pytest.approx(0.2)
    assert signed["rectification"] == pytest.approx(0.1)
    assert endpoint["pooling"] == endpoint["head"] == 0.0
    assert sum(signed.values()) == pytest.approx(-1.0)
    assert sum(endpoint.values()) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="Nonfinite"):
        head_decomposition(0.0, 0.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0)


def test_geometry_and_axis_order():
    reference = np.arange(12).reshape(1, 2, 2, 3)
    assert layout_scores(reference, reference)["CHW"] == 0.0
    with pytest.raises(ValueError, match="ordering"):
        layout_scores(reference.transpose(0, 1, 3, 2).reshape(reference.shape), reference)


def capture_module():
    file = Path(__file__).resolve().parents[1] / "scripts/windows/mobilenet_feature_capture.py"
    spec = importlib.util.spec_from_file_location("mobile_training_capture", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_uses_only_fp32_order_and_is_train_exclusive():
    capture = capture_module()
    manifest, pred, pilot = [], [], []
    for source in capture.SOURCES:
        for i in range(48):
            ident = f"{source}-{i:02}"
            manifest.append(
                dict(
                    sample_id=ident,
                    source_name=source,
                    fold_female_role="train",
                    calibration_fold="fold_female_role",
                    cohort="development",
                    dataset_role="development_train_ptq_calibration",
                )
            )
            pred.append(dict(sample_id=ident, source_fp32=str(max(0, i - 10))))
            if i < 4:
                pilot.append(ident)
    selected = capture.select_panel(manifest, pred, pilot)
    assert len({r["sample_id"] for r in selected}) == 40
    for s in capture.SOURCES:
        group = [r for r in selected if r["source_name"] == s and r["panel"] == "confirmation"]
        assert len(group) == 6
        assert {r["source_fp32"] for r in group} == {0, 1, 10, 19, 28, 37}
    manifest[0]["fold_female_role"] = "test"
    with pytest.raises(ValueError, match="Nontraining"):
        capture.select_panel(manifest, pred, pilot)


def test_synthetic_frozen_suffices_agree_and_identity_check_fails_closed(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    h = onnx.helper
    dtype = onnx.TensorProto.FLOAT
    inputs = [h.make_tensor_value_info("input", dtype, [1, 2, 2, 2])]
    outputs = [h.make_tensor_value_info("burden", dtype, [1])]
    nodes = [
        h.make_node("Identity", ["input"], ["spatial"]),
        h.make_node("GlobalAveragePool", ["spatial"], ["pool"]),
        h.make_node("Flatten", ["pool"], ["flatten"]),
        h.make_node("Gemm", ["flatten", "weights", "bias"], ["head"], transB=1),
        h.make_node("Relu", ["head"], ["relu"]),
        h.make_node("Gather", ["relu", "index"], ["burden"], axis=1),
    ]
    constants = [
        onnx.numpy_helper.from_array(np.array([[1, -2]], np.float32), "weights"),
        onnx.numpy_helper.from_array(np.array([0.25], np.float32), "bias"),
        onnx.numpy_helper.from_array(np.array(0, np.int64), "index"),
    ]
    model = h.make_model(
        h.make_graph(nodes, "regression", inputs, outputs, constants),
        opset_imports=[h.make_opsetid("", 11)],
    )
    model.ir_version = 7
    path = tmp_path / "model.onnx"
    onnx.save(model, str(path))
    capture = capture_module()
    sha = capture.digest(path)
    assert capture.checked(path, sha) == path
    names = {k: k for k in ("input", "spatial", "pool", "flatten", "head", "relu", "burden")}
    graph = attribution.FeatureGraph(path, names)
    x = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
    values = graph.run(x)
    assert values["burden"].item() == 0
    assert graph.affine(values["pool"]) == values["head"].item() == -9.25
    suffix = graph.suffix("pool")
    assert suffix.run(["head"], {"pool": values["pool"]})[0].item() == -9.25
    modified = values["pool"].copy()
    modified[0, 0] += 1
    assert suffix.run(["head"], {"pool": modified})[0].item() == -8.25
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Changed file"):
        capture.checked(path, sha)
