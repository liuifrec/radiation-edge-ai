"""Exercise the KL720 replay entrypoint without loading the hardware SDK or opening USB."""

import csv
import ctypes
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def panel(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts" / "windows"))
    return importlib.import_module("run_kl720_dnai_panel")


def device(port, pid=0x720, connectable=True):
    return SimpleNamespace(
        usb_port_id=port, product_id=pid, is_connectable=connectable,
        firmware="KDP2 Comp/F",
        get_member_variable_dict=lambda: {"usb_port_id": port, "firmware": "KDP2 Comp/F"},
    )


def test_detection_selects_only_one_connectable_kl720(panel):
    other, first, second = device(10, 0x630), device(11), device(12)
    assert panel.replay_device([other, first], None) is first
    assert panel.replay_device([first, second], 12) is second
    for choices, port in [([], None), ([other], None), ([first, second], None), ([other, first], 10)]:
        with pytest.raises(RuntimeError, match="one matching KL720"):
            panel.replay_device(choices, port)
    with pytest.raises(RuntimeError, match="unconnectable"):
        panel.replay_device([device(11, connectable=False)], 11)


def test_raw_capture_rejects_mismatched_inference_number(panel, tmp_path):
    header = SimpleNamespace(get_member_variable_dict=lambda: {"inference_number": 9})
    destination = tmp_path / "raw.bin"
    with pytest.raises(RuntimeError, match="sent 0"):
        panel.save_receive_buffer(SimpleNamespace(header=header), destination, 0)
    assert not destination.exists()


@pytest.mark.parametrize("devices,code", [
    ([device(10, pid=0x630)], 4),
    ([device(11, connectable=False)], 4),
    ([device(12)], 0),
])
def test_discovery_checks_without_connecting(devices, code, monkeypatch):
    from radiation_edge_ai.hardware import kl720_check

    core = SimpleNamespace(scan_devices=lambda: SimpleNamespace(
        device_descriptor_number=len(devices), device_descriptor_list=devices))
    monkeypatch.setitem(sys.modules, "kp", SimpleNamespace(core=core))
    assert kl720_check.main() == code


def test_replay_captures_fresh_raw_storage_and_hashes(panel, tmp_path, monkeypatch):
    model_path = tmp_path / "model.nef"
    model_path.write_bytes(b"synthetic test only")
    monkeypatch.setattr(panel, "EXPECTED_NEF_SHA256", panel.sha256_file(model_path))
    array = np.ones(panel.EXPECTED_INPUT_SHAPE, dtype=np.float32)
    input_path = tmp_path / "input.npy"
    np.save(input_path, array)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "tile": array.shape, "overlap": 0.5,
        "records": [{"image_index": 0, "window_index": i, "sample_id": "sample",
                     "key": "test", "tensor": input_path.name} for i in range(2)],
    }), encoding="utf-8")
    monkeypatch.setattr(panel, "FROZEN_MANIFEST_SHA256", panel.sha256_file(manifest_path))

    node = SimpleNamespace(
        tensor_shape_info=SimpleNamespace(v1=SimpleNamespace(shape_npu=array.shape)),
        data_layout=1,
        quantization_parameters=SimpleNamespace(v1=SimpleNamespace(
            quantized_fixed_point_descriptor_list=[SimpleNamespace(radix=5, scale=1.0)])),
    )
    model = SimpleNamespace(id=32769, version=0x8B28, input_nodes=[node], output_nodes=[node])
    metadata = SimpleNamespace(get_member_variable_dict=lambda: {"compiler_version": "fixture"})
    core = SimpleNamespace(
        scan_devices=Mock(return_value=[device(42)]),
        connect_devices=Mock(return_value="group"), set_timeout=Mock(),
        get_version=lambda: "3.2.0",
        get_system_info=lambda group, port: SimpleNamespace(
            get_member_variable_dict=lambda: {"firmware_version": "2.2.0-build.615"}),
        load_model_from_file=Mock(return_value=SimpleNamespace(models=[model], metadata=metadata)),
    )
    payload = b"\x00arbitrary\x00output\x00"
    allocation = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
    raw = SimpleNamespace(
        buffer_size=len(payload),
        _get_element_buffer=lambda: SimpleNamespace(_LP_raw_out_buffer=allocation),
    )
    def response(number):
        header = SimpleNamespace(num_output_node=1, get_member_variable_dict=lambda: {"inference_number": number})
        return SimpleNamespace(header=header, raw_result=raw)

    inference = SimpleNamespace(
        generic_data_inference_send=Mock(),
        generic_data_inference_receive=Mock(side_effect=[response(0), response(1)]),
        generic_inference_retrieve_float_node=Mock(return_value=array[0]),
    )
    monkeypatch.setitem(sys.modules, "kp", SimpleNamespace(
        __file__=str(tmp_path / "mock_sdk.py"), core=core, inference=inference,
        GenericDataInferenceDescriptor=SimpleNamespace, GenericInputNodeData=SimpleNamespace,
        ModelTensorDataLayout=SimpleNamespace(KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B=1),
        ChannelOrdering=SimpleNamespace(KP_CHANNEL_ORDERING_DEFAULT=0),
    ))
    run_dir = tmp_path / "fresh"
    args = ["--baseline-replay", "--nef", str(model_path), "--validation-manifest",
            str(manifest_path), "--output-dir", str(run_dir)]
    assert panel.main(args) == 0
    core.connect_devices.assert_called_once_with(usb_port_ids=[42])
    core.load_model_from_file.assert_called_once_with(device_group="group", file_path=str(model_path))
    assert inference.generic_data_inference_receive.call_count == 2
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_ran"] == 2 and summary["n_existing_skipped"] == 0
    assert summary["runtime"]["plus_version"] == "3.2.0"
    assert summary["device"]["usb_port_id"] == 42
    assert summary["system_info"]["firmware_version"] == "2.2.0-build.615"
    assert summary["nef_metadata"]["compiler_version"] == "fixture" and summary["model_version"] == 0x8B28
    csv_path = Path(summary["per_window_csv"])
    assert summary["per_window_sha256"] == panel.sha256_file(csv_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        row, second = csv.DictReader(handle)
    assert second["inference_number_received"] == second["inference_number_sent"] == "1"
    assert second["raw_receive_path"] != row["raw_receive_path"]
    sent = inference.generic_data_inference_send.call_args.kwargs["generic_inference_input_descriptor"]
    assert row["packed_sha256"] == hashlib.sha256(sent.input_node_data_list[0].buffer).hexdigest()
    assert row["inference_number_received"] == row["inference_number_sent"] == "0"
    assert row["input_sha256"] == panel.sha256_file(input_path)
    assert Path(row["raw_receive_path"]).read_bytes() == payload
    assert row["raw_receive_sha256"] == hashlib.sha256(payload).hexdigest()
    out = Path(row["physical_output"])
    assert panel.sha256_file(out) == row["output_sha256"]
    np.testing.assert_array_equal(np.load(out, allow_pickle=False), array)
    with pytest.raises(FileExistsError):
        panel.main(args)
    core.scan_devices.assert_called_once()
