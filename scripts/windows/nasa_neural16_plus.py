"""Offline PLUS 3.2.0 fixed/float retrieval of the KL720 scalar output."""

import struct

import numpy as np

from radiation_edge_ai.hardware import spotiflow_plus as abi
from radiation_edge_ai.nasa_bps.multimodel_ptq import require
from radiation_edge_ai.nasa_bps.neural16_transport import SCALE, descriptor_bits
from spotiflow_physical_smoke import decode_receive


def scalar_head(setup):
    node, = setup["outputs"]
    buf = node["buffer"]
    require(node["name"] == "burden" and node["shape"] == [1, 1, 1, 1] and node["raw_shape"] == [1]
            and node["format"] == 6 and buf["segment"] == "OUTPUT" and buf["entry"] == 0
            and type(buf["len"]) is int and 16 <= buf["len"] <= 65536, "Invalid aligned scalar output descriptor")
    return {"index": 0, "name": "burden", "shape": [1, 1, 1, 1], "radix": 13, "scale": SCALE,
            "offset": 0, "bytes": buf["len"]}


def case_reference(word):
    require(type(word) is int and -32768 <= word <= 32767, "Invalid signed transport code")
    scalar = np.array([[[[word]]]], dtype="<i2")
    return {"fixed": [({"graph_name": "burden"}, scalar)]}


def csim_packet(head, word, sequence):
    """Embed an already observed CSIM word in a synthetic, zero-padded transport frame."""
    size = head["bytes"]
    payload = bytearray(abi.DATA_OFFSET + size)
    abi.STAMP.pack_into(payload, 0, 0x11FF22AA, len(payload), 17, 0, 1, 0)
    abi.PRODUCT.pack_into(payload, abi.PRODUCT_OFFSET, 0x720, sequence, 0, 1)
    struct.pack_into("<II", payload, abi.CNN_OFFSET, size, 1)
    abi.NODE.pack_into(payload, abi.CNN_OFFSET + 8, 0, size, 0, 1, 6, 0, 0, 0,
                       1, 1, 1, 0, 13, descriptor_bits())
    struct.pack_into("<h", payload, abi.DATA_OFFSET, word)
    return bytes(payload)


def replay_packet(kp, packet, header, head, word, sequence):
    reference = case_reference(word)
    result = decode_receive(kp, packet, header, [head], reference, expected_sequence=sequence)["burden"]
    fixed, decoded = result
    return {"fixed_word": int(fixed.item()), "decoded": float(decoded.item())}
