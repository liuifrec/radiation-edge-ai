"""Physical panel control-path tests. The synthetic transport never opens a device."""

import ctypes
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/windows"))
    import run_spotiflow_physical_panel
    return run_spotiflow_physical_panel


def test_complete_simulator_receipt_required(runner):
    ids = [str(i) for i in range(63)]
    info = {"model": "general", "panel_sha256": "p", "nef_sha256": "n"}
    sim = dict(info, stage="nef_simulator", rows=[{"case_id": k} for k in ids])
    gate = dict(info, stage="nef_simulator_parity", passed=True, output_manifest_sha256="s",
                rows=[{"case_id": k, "passed": True} for k in ids])
    assert set(runner.require_simulator_binding("s", gate, sim, "p", "n", "general", ids)) == set(ids)
    with pytest.raises(ValueError, match="matching"):
        runner.require_simulator_binding("changed", gate, sim, "p", "n", "general", ids)
    with pytest.raises(ValueError, match="63"):
        runner.require_simulator_binding("s", gate, sim, "p", "n", "general", ids[:-1])
    sim["rows"][0]["sdk_packed_input"] = {"available": False}
    with pytest.raises(ValueError, match="SDK input identities"):
        runner.require_simulator_binding("s", gate, sim, "p", "n", "general", ids)
    gate["rows"][0]["sdk_packed_input"] = {"available": False}
    gate["rows"][-1]["passed"] = False
    with pytest.raises(ValueError, match="Each"):
        runner.require_simulator_binding("s", gate, sim, "p", "n", "general", ids)


def test_sdk_dram_input_binding(tmp_path, runner):
    assert runner.load_sdk_input(tmp_path, None, "input", required=False) is None
    with pytest.raises(ValueError, match="Missing"):
        runner.load_sdk_input(tmp_path, {"available": False}, "input", required=True)
    path = tmp_path / "case.bin"
    path.write_bytes(b"real registered input")
    spec = {"available": True, "path": path.name, "input_name": "input", "bytes": path.stat().st_size,
            "sha256": runner.sha256_file(path)}
    assert runner.load_sdk_input(tmp_path, spec, "input", True) == path.read_bytes()
    with pytest.raises(ValueError, match="checksum"):
        runner.load_sdk_input(tmp_path, dict(spec, bytes=0), "input", True)


def sdk_reference(tmp_path, runner, contract, raw_heads):
    records = []
    for index, (spec, raw) in enumerate(zip(contract, raw_heads)):
        path = tmp_path / f"sdk-{index}.npy"
        np.save(path, raw)
        records.append({"returned_index": index, "graph_name": spec["name"], "shape": list(raw.shape),
                        "dtype": str(raw.dtype), "path": path.name, "sha256": runner.sha256_file(path)})
    row = {"sdk_returned_tensors": records, "separate_tensors": [dict(r) for r in records]}
    hosted = np.concatenate(sorted(raw_heads, key=lambda h: h.shape[1]), axis=1).astype(np.float32)
    return row, hosted


def test_physical_parity_compares_to_original_double_before_host_rounding(tmp_path, runner, monkeypatch):
    contract = [{"name": name, "shape": shape} for name, shape in runner.JOINED.items()]
    raw = np.full(contract[0]["shape"], 7.9999998, dtype=np.float64)
    row, hosted = sdk_reference(tmp_path, runner, contract, [raw])
    originals = runner.load_sdk_heads(tmp_path, row, contract, hosted)
    board = np.nextafter(hosted, np.float32(np.inf))
    detection = {"points": np.empty((0, 2))}
    monkeypatch.setattr(runner, "compare_points", lambda a, b: {"equal_count": True, "max_coordinate_abs_px": 0.})
    result = runner.compare_case({"heads": hosted, "originals": originals}, board, detection, detection, {}, {})
    assert result["tensor"]["within_tolerance"]
    assert result["original_return_parity"][0]["max_abs_error"] == float(board.flat[0]) - 7.9999998
    assert result["original_return_parity"][0]["dtype"] == "float64"
    assert not result["original_return_parity"][0]["within_tolerance"] and not result["passed"]


def test_original_head_names_indices_and_cast_match_hosted_heads(tmp_path, runner):
    # Reverse SDK order; each logical head must still match its own channels and dtype.
    contract = [{"name": key, "shape": shape} for key, shape in reversed(list(runner.SEPARATE.items()))]
    arrays = [np.full(contract[0]["shape"], 2., np.float32), np.full(contract[1]["shape"], 7.9999998)]
    row, hosted = sdk_reference(tmp_path, runner, contract, arrays)
    ordered = runner.load_sdk_heads(tmp_path, row, contract, hosted)
    assert [record["returned_index"] for record, _ in ordered] == [1, 0]
    with pytest.raises(ValueError, match="projection"):
        runner.load_sdk_heads(tmp_path, row, contract, np.nextafter(hosted, np.float32(np.inf)))
    with pytest.raises(ValueError, match="identity"):
        runner.load_sdk_heads(tmp_path, {"host_output_dtype": "float32"}, contract, hosted)
    with pytest.raises(ValueError, match="indices"):
        wrong = [dict(r, returned_index=1-r["returned_index"]) for r in row["separate_tensors"]]
        runner.load_sdk_heads(tmp_path, dict(row, separate_tensors=wrong), contract, hosted)
    with pytest.raises(ValueError, match="identity"):
        runner.load_sdk_heads(tmp_path, dict(row, sdk_returned_tensors=[row["sdk_returned_tensors"][0]]*2),
                              contract, hosted)


def test_mock_transport_resident_pair_writes_raw_and_aligns_reversed_descriptors(tmp_path, runner, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    pytest.importorskip("spotiflow")
    def node(c):
        return NS(tensor_shape_info=NS(v1=NS(shape_npu=(1, c, 128, 128))),
                  get_member_variable_dict=lambda: {"channels": c})
    input_node = node(1)
    input_node.data_layout = 3
    input_node.quantization_parameters = NS(v1=NS(quantized_fixed_point_descriptor_list=[NS(radix=0, scale=1.)]))
    loaded = NS(id=32821, input_nodes=[input_node], output_nodes=[node(3), node(1)])
    h = np.zeros((1, 4, 128, 128), np.float32)
    h[:, 0] = -10
    h[:, 0, 24, 35] = 8
    h[:, 1] = 1
    order = []
    def send(device_group, generic_inference_input_descriptor):
        order.append(generic_inference_input_descriptor.inference_number)
    payload = ctypes.create_string_buffer(b"captured bytes")
    def receive(device_group):
        i = order[-1]
        return NS(header=NS(inference_number=i, num_output_node=2, get_member_variable_dict=lambda: {"inference_number": i}),
                  raw_result=NS(buffer_size=len(payload), _get_element_buffer=lambda: NS(
                      _LP_raw_out_buffer=ctypes.addressof(payload))))
    kp = NS(GenericDataInferenceDescriptor=NS, GenericInputNodeData=NS,
            ModelTensorDataLayout=NS(KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B=1, KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B=2,
                                     KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B=3),
            ChannelOrdering=NS(KP_CHANNEL_ORDERING_CHW=1),
            inference=NS(generic_data_inference_send=send, generic_data_inference_receive=receive,
                         generic_inference_retrieve_fixed_node=lambda node_idx, **kwargs: NS(
                             ndarray=(h[:, 1:], h[:, :1])[node_idx].astype(np.int16)),
                         generic_inference_retrieve_float_node=lambda node_idx, **kwargs: (h[:, 1:], h[:, :1])[node_idx]))
    raw = tmp_path / "raw.npy"
    mask = tmp_path / "mask.npy"
    np.save(raw, np.ones((128, 128), dtype=np.uint16))
    np.save(mask, np.ones((128, 128), dtype=np.uint16))
    tensor = tmp_path / "tensor.npy"
    normalized = runner.spotiflow.prepare_input(np.load(raw))
    np.save(tensor, normalized)
    rows = [{"case_id": k, "raw_path": raw.name, "raw_sha256": runner.sha256_file(raw),
             "source_id": "synthetic", "domain": "synthetic",
             "target_mask_path": mask.name, "target_mask_sha256": runner.sha256_file(mask),
             "target_label": 1, "tensor_path": tensor.name, "tensor_sha256": runner.sha256_file(tensor)} for k in ("a", "b")]
    contract = [{"name": "/finest_output/block/block.0/Conv_output_0", "shape": [1, 1, 128, 128]},
                {"name": "/flow/flow.6/Conv_output_0", "shape": [1, 3, 128, 128]}]
    receipt = {"rows": []}
    sdk_input = runner.usb.pack_kl720_input(kp, input_node, normalized)[0]
    sdk_row, _ = sdk_reference(tmp_path, runner, contract, [h[:, :1], h[:, 1:]])
    originals = runner.load_sdk_heads(tmp_path, sdk_row, contract, h)
    fixed = [(dict(record, fixed_sha256="fixture"), raw.astype(np.int16)) for record, raw in originals]
    reference = {key: {"heads": h, "sdk_input": sdk_input, "originals": originals, "fixed": fixed} for key in ("a", "b")}
    wrong = dict(reference, b={"heads": h, "sdk_input": b"wrong case"})
    with pytest.raises(ValueError, match="before inference"):
        runner.resident_loop(kp, object(), loaded, contract, tmp_path, rows, wrong, 0.5, NS(out=tmp_path), receipt)
    assert not order and receipt["packing_preflight"][-1]["equal"] is False
    runner.resident_loop(kp, object(), loaded, contract, tmp_path, rows, reference, 0.5,
                         NS(out=tmp_path), receipt)
    assert order == [0, 1] and receipt["joined_descriptor_indices"] == [1, 0]
    for key, row in zip(("a", "b"), receipt["rows"]):
        assert row["passed"] and row["physical_counts"] == {"crop": 1, "target": 1}
        assert all(r["exact"] and r["dtype"] == "float32" for r in row["original_return_parity"])
        assert row["sdk_packing"]["equal"] and row["sdk_packing"]["sdk_sha256"] == row["packed_input_sha256"]
        assert [f["device_index"] for f in row["signed_transport_outputs"]] == [1, 0]
        assert all(f["equal"] for f in row["signed_transport_outputs"])
        for f, (_, expected_fixed) in zip(row["signed_transport_outputs"], fixed):
            np.testing.assert_array_equal(np.load(tmp_path / f["path"]), expected_fixed)
        np.testing.assert_array_equal(np.load(tmp_path / row["raw_heads_path"]), h)
        assert (tmp_path / (key + ".bin")).read_bytes() == payload.raw
        assert row["timings_ms"]["end_to_end"] >= row["timings_ms"]["send_receive"]
    failed_dir = tmp_path / "invalid_receive"
    failed_dir.mkdir()
    def decode_failure(**kwargs):
        raise ValueError("simulated vendor decode error")
    monkeypatch.setattr(kp.inference, "generic_inference_retrieve_float_node", decode_failure)
    failed = {"rows": []}
    with pytest.raises(ValueError, match="simulated vendor decode"):
        runner.resident_loop(kp, object(), loaded, contract, tmp_path, rows[:1], reference, 0.5,
                             NS(out=failed_dir), failed)
    assert (failed_dir / "a.bin").read_bytes() == payload.raw
    assert not failed["rows"] and failed["raw_receives"][0]["submitted_model_id"] == 32821


def test_panel_cli_requires_explicit_mode(tmp_path, runner):
    base = ["--model", "spotiflow/general", "--variant", "full_int16_transport",
            "--data-root", str(tmp_path), "--out", str(tmp_path / "output")]
    parser = runner.argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--replay-simulator", "--panel"])
    offline = parser.parse_args(base + ["--replay-simulator"])
    assert offline.replay_simulator and not offline.panel and not offline.smoke


def test_transport_file_identity_and_signed_extrema(tmp_path):
    import struct
    from radiation_edge_ai.hardware.spotiflow_plus import transport_words

    head = {"name": "test", "index": 0, "scale": 1.5, "radix": 2, "shape": [1, 1, 1, 4]}
    data = tmp_path / "dma2seq_0.seq.16B"
    data.write_text("-32768 -1 0 32767")
    info = tmp_path / "dma2seq_0.info"
    scale_bits, = struct.unpack("<I", struct.pack("<f", head["scale"]))
    fields = [*head["shape"], head["radix"], scale_bits]
    info.write_text(chr(10).join(map(str, fields)))
    np.testing.assert_array_equal(transport_words(head, data, info).ravel(), [-32768, -1, 0, 32767])
    with pytest.raises(ValueError, match="file and metadata"):
        transport_words(dict(head, index=1), data, info)
    with pytest.raises(ValueError, match="descriptor"):
        transport_words(dict(head, scale=2), data, info)
    data.write_text("32768 0 0 0")
    with pytest.raises(ValueError, match="range"):
        transport_words(head, data, info)
