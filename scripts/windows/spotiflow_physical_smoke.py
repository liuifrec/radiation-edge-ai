"""Authenticate the matched physical smoke before any resident KL720 panel submission."""

import hashlib
import json
import struct
from datetime import datetime
from pathlib import Path

import numpy as np
from radiation_edge_ai.hardware import spotiflow_plus as probe

from radiation_edge_ai.audit_utils import sha256_file
from radiation_edge_ai.hardware.spotiflow_transport import VARIANT, check_device


def decode_receive(kp, packet, header, heads, reference, *, expected_sequence=0):
    """Bounds-check the KDP2 allocation before giving its complete bytes to the native DLL."""
    size = sum(h["bytes"] for h in heads)
    if len(packet) < probe.DATA_OFFSET + size:
        raise ValueError("Truncated KL720 receive allocation")
    magic, length, job, status, _, _ = probe.STAMP.unpack_from(packet)
    product, sequence, crop, last = probe.PRODUCT.unpack_from(packet, probe.PRODUCT_OFFSET)
    tensor_size, nodes = struct.unpack_from("<II", packet, probe.CNN_OFFSET)
    if (magic != 0x11FF22AA or length != probe.DATA_OFFSET + size or job != 17 or status != 0
            or (product, sequence, crop, last) != (0x720, expected_sequence, 0, 1) or tensor_size != size or nodes != len(heads)
            or any(int(str(header[k]), 0) != v for k, v in (("product_id", product), ("inference_number", sequence),
                                                             ("crop_number", crop), ("num_output_node", nodes)))):
        raise ValueError("Physical smoke KDP2 receive header does not match the expected result")
    fixed = {p["graph_name"]: a for p, a in reference["fixed"]}
    for h in heads:
        fields = probe.NODE.unpack_from(packet, probe.CNN_OFFSET + 8 + h["index"] * probe.NODE.size)
        scale_bits, = struct.unpack("<I", struct.pack("<f", h["scale"]))
        _, ch, height, width = h["shape"]
        if (fields[:2] != (h["offset"], h["bytes"]) or fields[4] != 6 or fields[8:12] != (height, width, ch, h["index"])
                or fields[12:] != (h["radix"], scale_bits) or h["offset"] + h["bytes"] > size):
            raise ValueError("Unsafe or mismatched KL720 output descriptor in receive allocation")
    frame = probe.raw_frame(kp, packet)
    decoded = {}
    for h in heads:
        try:
            _, words, floats = probe.retrieve_node(kp, frame, h, fixed[h["name"]], kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW)
        except AssertionError as exc:
            raise ValueError("Raw physical packet differs from the signed or floating transport reference") from exc
        decoded[h["name"]] = (words.copy(), floats.copy())
    return decoded


def require_smoke(path, digest, data_root, binding, first_case, reference, gate_utc, kp, descriptors, *, expected_kn):
    if not path or not digest:
        raise ValueError("Supply the passing physical --smoke-receipt and its --smoke-sha256")
    path = path.resolve(strict=True)
    if not path.is_relative_to(data_root.resolve()) or path.name != "receipt.json" or sha256_file(path) != digest:
        raise ValueError("Unexpected smoke receipt directory or digest")
    receipt = json.loads(path.read_bytes())
    if (receipt.get("stage") != "physical_kl720_smoke" or receipt.get("passed") is not True
            or receipt.get("completed") is not True or receipt.get("variant") != VARIANT or len(receipt["rows"]) != 1
            or any(receipt.get(k) != v for k, v in binding.items())
            or datetime.fromisoformat(receipt["started_utc"]) < datetime.fromisoformat(gate_utc)
            or datetime.fromisoformat(receipt["ended_utc"]) < datetime.fromisoformat(receipt["started_utc"])):
        raise ValueError("Physical smoke predates, failed, or differs from the current transport qualification")
    check_device(receipt["device"], receipt["system_info"], expected_kn=expected_kn)
    case = receipt["rows"][0]
    name = first_case["case_id"]
    if (case["case_id"] != name or case["passed"] is not True or case["inference_number"] != 0
            or case["submitted_model_id"] != binding["model_id"] or receipt["model_id"] != binding["model_id"]
            or case["tensor_sha256"] != first_case["tensor_sha256"] or case["raw_image_sha256"] != first_case["raw_sha256"]
            or case["sdk_packing"]["equal"] is not True
            or case["packed_input_sha256"] != hashlib.sha256(reference["sdk_input"]).hexdigest()
            or not 0 < case["timings_ms"]["send_receive"] <= case["timings_ms"]["end_to_end"]):
        raise ValueError("Physical smoke did not pass the first registered input")
    proof = authenticate_received_outputs(kp, path.parent, case, descriptors, reference, 0)
    return {"path": str(path), "sha256": digest, "case_id": name, "firmware_version": receipt["system_info"]["firmware_version"],
            **proof}


def authenticate_received_outputs(kp, folder, case, heads, reference, expected_sequence):
    name = case["case_id"]
    def evidence(rel, checksum):
        item = (folder / rel).resolve(strict=True)
        if item.parent != folder.resolve() or sha256_file(item) != checksum:
            raise ValueError("Previously recorded physical smoke artifact mismatch")
        return item
    allocation = case["raw_receive"]["allocation"]
    if Path(allocation["path"]).name != name + ".bin" or allocation["size_bytes"] <= 0:
        raise ValueError("Missing raw smoke receive buffer")
    packet = evidence(name + ".bin", allocation["sha256"]).read_bytes()
    if len(packet) != allocation["size_bytes"]:
        raise ValueError("Physical smoke receive allocation size differs")
    decoded = decode_receive(kp, packet, case["raw_receive"]["header"], heads, reference, expected_sequence=expected_sequence)
    measured = np.load(evidence(case["raw_heads_path"], case["raw_heads_sha256"]), allow_pickle=False)
    decoded_joined = np.concatenate([decoded[h["name"]][1] for h in heads], axis=1)
    if (measured.shape != reference["heads"].shape or measured.dtype != np.float32 or
            not np.isfinite(measured).all() or np.max(np.abs(measured - reference["heads"])) > 1e-6
            or not np.array_equal(measured.view(np.uint32), decoded_joined.view(np.uint32))):
        raise ValueError("Physical smoke floating outputs differ from the qualified transport")
    separate = case["separate_tensors"]
    if len(separate) != len(heads) or {r["graph_name"] for r in separate} != set(decoded):
        raise ValueError("Physical smoke is missing a separate float head")
    for r in separate:
        arr = np.load(evidence(r["path"], r["sha256"]), allow_pickle=False)
        if arr.dtype != np.float32 or not np.array_equal(decoded[r["graph_name"]][1].view(np.uint32), arr.view(np.uint32)):
            raise ValueError("Saved physical float head differs from raw PLUS retrieval")
    expected = {r["graph_name"]: (r["fixed_sha256"], a) for r, a in reference["fixed"]}
    fixed = case["signed_transport_outputs"]
    if len(fixed) != 2 or {r["graph_name"]: r["reference_sha256"] for r in fixed} != {k: v[0] for k, v in expected.items()}:
        raise ValueError("Physical smoke signed output references differ")
    for r in fixed:
        words = np.load(evidence(r["path"], r["sha256"]), allow_pickle=False)
        if r["equal"] is not True or words.dtype != np.int16 or not np.array_equal(words, decoded[r["graph_name"]][0]):
            raise ValueError("Physical smoke did not reproduce signed transport outputs")
    return {"raw_receive_sha256": allocation["sha256"], "raw_plus_retrieval_verified": True,
            "heads": [{"graph_name": h["name"], "words": decoded[h["name"]][0].size,
                       "raw_fixed_sha256": hashlib.sha256(decoded[h["name"]][0].tobytes()).hexdigest(),
                       "raw_float_sha256": hashlib.sha256(decoded[h["name"]][1].tobytes()).hexdigest()} for h in heads]}
