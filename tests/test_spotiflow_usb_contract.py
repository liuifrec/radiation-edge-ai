import hashlib
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from radiation_edge_ai.audit_utils import sha256_file
from radiation_edge_ai.hardware import spotiflow_plus as probe, spotiflow_transport


def encode_packet(heads, word_arrays, inference_number):
    """Pack both CHW word tensors under the pinned KL720 v1 result ABI."""
    if len(word_arrays) != len(heads):
        raise ValueError("Missing output head")
    nbytes = sum(head["bytes"] for head in heads)
    payload = bytearray(probe.DATA_OFFSET + nbytes)
    probe.STAMP.pack_into(payload, 0, 0x11FF22AA, len(payload), 17, 0, 1, 0)
    probe.PRODUCT.pack_into(payload, probe.PRODUCT_OFFSET, 0x720, inference_number, 0, 1)
    struct.pack_into("<II", payload, probe.CNN_OFFSET, nbytes, len(heads))
    for head, words in zip(heads, word_arrays):
        _, channels, height, width = head["shape"]
        count = channels * height * width
        if (head["bytes"] != 2 * count or width % 8 or head["offset"] + 2 * count > nbytes
                or words.dtype != np.dtype("<i2") or list(words.shape) != head["shape"]):
            raise ValueError("Expected contiguous width-aligned 8W1C16B output")
        scale_bits, = struct.unpack("<I", struct.pack("<f", head["scale"]))
        probe.NODE.pack_into(payload, probe.CNN_OFFSET + 8 + head["index"] * probe.NODE.size,
                       head["offset"], head["bytes"], head["index"], 1, 6, 0, 0, 0, height,
                       width, channels, head["index"], head["radix"], scale_bits)
        start = probe.DATA_OFFSET + head["offset"]
        payload[start:start + head["bytes"]] = words.tobytes()
    return bytes(payload)


def test_general_image_worker_rejects_another_device_before_connect(tmp_path, monkeypatch):
    from radiation_edge_ai import _kl720_worker

    nef, image = tmp_path / "model.nef", tmp_path / "input.npy"
    nef.write_bytes(b"synthetic unexecuted NEF")
    np.save(image, np.zeros((1, 1, 4, 4), np.float32))
    device = SimpleNamespace(product_id=0x720, usb_port_id=9, kn_number=1)
    kp = SimpleNamespace(core=SimpleNamespace(scan_devices=lambda: SimpleNamespace(device_descriptor_list=[device])))
    monkeypatch.setitem(sys.modules, "kp", kp)
    monkeypatch.setattr(sys, "argv", ["worker", "--model", str(nef), "--input", str(image),
                                    "--output", str(tmp_path / "unused.npy"), "--manifest", str(tmp_path / "unused.json"),
                                    "--expected-kn", "0x12345678"])
    with pytest.raises(RuntimeError, match="different KN number"):
        _kl720_worker.main()


@pytest.mark.parametrize("sdk", ["fixture", "installed_plus"])
def test_full_panel_reauthenticates_smoke_raw_and_signed_outputs(tmp_path, monkeypatch, sdk):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/windows"))
    from radiation_edge_ai.hardware import spotiflow_plus as probe
    from spotiflow_physical_smoke import require_smoke

    if sdk == "installed_plus":
        kp = pytest.importorskip("kp")
        assert kp.core.get_version() == "3.2.0"
        assert sha256_file(Path(kp.__file__).parent / "lib/libkplus.dll") == probe.DLL_SHA
    else:
        kp = SimpleNamespace(ChannelOrdering=SimpleNamespace(KP_CHANNEL_ORDERING_CHW=1))
        monkeypatch.setattr(probe, "raw_frame", lambda sdk, packet: packet)
        def retrieve(sdk, packet, head, expected, ordering):
            words = np.frombuffer(packet, "<i2", count=expected.size,
                                  offset=probe.DATA_OFFSET + head["offset"]).reshape(expected.shape)
            np.testing.assert_array_equal(words, expected)
            return {}, words, words.astype(np.float32)
        monkeypatch.setattr(probe, "retrieve_node", retrieve)
    root = tmp_path / "smoke"
    root.mkdir()
    received = root / "a.bin"
    first = {"case_id": "a", "tensor_sha256": "tensor", "raw_sha256": "image"}
    fixed_reference = []
    fixed_outputs = []
    descriptors, float_files = [], []
    for i, c in enumerate((1, 3)):
        fixed = np.arange(c * 8, dtype=np.int16).reshape(1, c, 1, 8)
        descriptors.append({"name": str(c), "shape": list(fixed.shape), "index": i, "bytes": fixed.nbytes,
                            "offset": 16 * i, "radix": 0, "scale": 1.0})
        path = root / f"a_{c}.npy"
        np.save(path, fixed)
        floating = root / f"a_{c}_float.npy"
        np.save(floating, fixed.astype(np.float32))
        float_files.append({"graph_name": str(c), "path": floating.name, "sha256": sha256_file(floating)})
        stub = {"graph_name": str(c), "fixed_sha256": sha256_file(path)}
        fixed_reference.append((stub, fixed))
        fixed_outputs.append({"equal": True, "path": path.name, "sha256": sha256_file(path),
                              "reference_sha256": stub["fixed_sha256"], "graph_name": str(c)})
    packet = encode_packet(descriptors, [v for _, v in fixed_reference], 0)
    received.write_bytes(packet)
    heads = np.concatenate([v for _, v in fixed_reference], axis=1).astype(np.float32)
    np.save(root / "a.npy", heads)
    ref = {"sdk_input": b"packed", "heads": heads, "fixed": fixed_reference}
    row = {"case_id": "a", "passed": True, "inference_number": 0, "tensor_sha256": "tensor", "raw_image_sha256": "image",
           "sdk_packing": {"equal": True}, "packed_input_sha256": hashlib.sha256(b"packed").hexdigest(),
           "timings_ms": {"send_receive": 4, "end_to_end": 6},
           "submitted_model_id": 32821, "separate_tensors": float_files,
           "raw_receive": {"header": {"product_id": 0x720, "inference_number": 0, "crop_number": 0, "num_output_node": 2},
                           "allocation": {"path": str(received), "size_bytes": len(packet), "sha256": sha256_file(received)}},
           "raw_heads_path": "a.npy", "raw_heads_sha256": sha256_file(root / "a.npy"), "signed_transport_outputs": fixed_outputs}
    utc = "2026-11-20T00:00:00+00:00"
    receipt = {"stage": "physical_kl720_smoke", "variant": spotiflow_transport.VARIANT, "passed": True, "completed": True,
               "rows": [row], "started_utc": utc, "ended_utc": utc, "device": {"product_id": "0x720", "kn_number": "0x12345678"},
               "system_info": {"kn_number": "0x12345678", "firmware_version": "2.2.0-build.615"},
               "model_id": 32821, "nef_sha256": "nef"}
    file = root / "receipt.json"
    file.write_text(json.dumps(receipt))
    def check(path=file, binding=None, minimum=utc):
        return require_smoke(path, sha256_file(file), tmp_path, binding or {"nef_sha256": "nef", "model_id": 32821},
                             first, ref, minimum, kp, descriptors, expected_kn=0x12345678)
    assert check()["sha256"] == sha256_file(file)
    row["submitted_model_id"] = 111
    file.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="first registered input"):
        check()
    row["submitted_model_id"] = 32821
    file.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Supply"):
        check(path=None)
    with pytest.raises(ValueError, match="qualification"):
        check(binding={"nef_sha256": "other"})
    with pytest.raises(ValueError, match="predates"):
        check(minimum="2026-11-21T00:00:00+00:00")
    received.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact"):
        check()
    for offset, error in ((0, "header"), (probe.CNN_OFFSET + 8, "descriptor"), (probe.DATA_OFFSET, "raw|Raw")):
        changed = bytearray(packet)
        changed[offset] ^= 1
        received.write_bytes(changed)
        row["raw_receive"]["allocation"]["sha256"] = sha256_file(received)
        file.write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match=error):
            check()
