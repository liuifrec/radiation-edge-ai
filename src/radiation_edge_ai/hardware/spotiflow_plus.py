"""Offline PLUS 3.2.0 signed and floating output retrieval for the qualified KL720 transport."""

import ctypes
import hashlib
import json
import struct

import numpy as np

from radiation_edge_ai.audit_utils import sha256_file as sha


NEF_SHA = "dda7486fb693702ec784f3648d20ed195499bc5b038f5ecb72a9856d9a004ca1"
TYPED_SHA = "71f9fac7e8ca48fa2d860f7bdb079452dd19b21b5e767cce74684e05b2743799"
SETUP_TEXT_SHA = "9138b47141d295a7dd00171b023c5b0cbca5dfa4484b300a1ef953d7a97512ac"
IO_SHA = "e7798c0f6ad99adefd2ed8f293c86f4517bd5bf66ddf2cb483d5123016c1d913"
DLL_SHA = "90c46a76d538620f59481b8850c635deb925a42acc6047c6340ccdcb5b06e9e8"

# KL720 KDP2 v1 raw result ABI, from kdp2_inf_generic_raw.h and kp_struct.h.
# The outer packet reserves five 60-byte preprocessing records even for data inference.
STAMP = struct.Struct("<6I")
PRODUCT = struct.Struct("<4I")
PRODUCT_OFFSET = STAMP.size + 4 + 5 * 60
CNN_OFFSET = PRODUCT_OFFSET + PRODUCT.size
NODE = struct.Struct("<14I")
DATA_OFFSET = CNN_OFFSET + 8 + 40 * NODE.size


def verified(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Unrecognized source bytes in {path}")
    return path


def descriptors(candidate):
    typed = verified(candidate / "nef_typed_setup_descriptor/receipt.json", TYPED_SHA)
    model = candidate / "nef_compile/kneron_flow/model_modelid_32821"
    compiled = json.loads(verified(model / "setup.txt", SETUP_TEXT_SHA).read_bytes())
    io = json.loads(verified(model / "ioinfo.json", IO_SHA).read_bytes())
    audit = json.loads(typed.read_bytes())
    verified(candidate / "nef_compile/model.nef", NEF_SHA)
    if audit["nef_sha256"] != NEF_SHA or len(audit["rows"]) != 2:
        raise ValueError("Expected the named full-INT16 two-head NEF")
    heads = []
    for index, row in enumerate(audit["rows"]):
        node, desc = compiled["outputs"][index], io["output"][index]
        scale = float(row["scale_float32"])
        radix = int(row["radix"])
        channels = desc["onnx_shape"][1]
        if (node["name"] != row["name"] or desc["name"] != row["name"]
                or int(row["output_index"]) != index or int(row["quant_index"]) != 0
                or row["layout_enum"] != "6" or node["format"] != 6
                or desc["data_format"] != "8W1C16B" or desc["bitw"] != 16 or desc["ch_dim"] != 1
                or node["raw_shape"] != node["shape"] or node["shape"] != desc["onnx_shape"]
                or desc["scale"] != [scale] * channels or desc["radix"] != [radix] * channels
                or struct.pack(">f", scale).hex() != row["scale_ieee754_hex"]):
            raise ValueError("NEF binary descriptor disagrees with the compiled output contract")
        heads.append({"index": index, "name": row["name"], "shape": node["shape"], "scale": scale,
                      "radix": radix, "offset": node["buffer"]["entry"], "bytes": node["buffer"]["len"]})
    return heads


def raw_frame(kp, packet):
    frame = kp.GenericDataInferenceResult(buffer_size=len(packet))
    ctypes.memmove(frame.raw_result._get_element_buffer()._LP_raw_out_buffer, packet, len(packet))
    return frame


def retrieve_node(kp, frame, head, expected, ordering):
    fixed = kp.inference.generic_inference_retrieve_fixed_node(head["index"], frame, ordering)
    floating = kp.inference.generic_inference_retrieve_float_node(head["index"], frame, ordering)
    entries = fixed.quantization_parameters.v1.quantized_fixed_point_descriptor_list
    axis = fixed.quantization_parameters.v1.quantized_axis
    fixed_values, floats = fixed.ndarray, floating.ndarray
    if (fixed.name != "" or floating.name != "" or fixed.shape != head["shape"] or floating.shape != head["shape"]
            or axis != 1 or len(entries) != 1 or entries[0].scale.value != head["scale"]
            or entries[0].radix != head["radix"] or fixed_values.dtype != np.int16 or floats.dtype != np.float32):
        raise ValueError("PLUS returned an unexpected head, shape, dtype or quantization contract")
    if ordering == kp.ChannelOrdering.KP_CHANNEL_ORDERING_HWC:
        expected = expected.transpose(0, 2, 3, 1)
    reference = expected.astype(np.float32) / np.float32(head["scale"] * 2**head["radix"])
    # Signed zeros and every last bit of both float32 and fixed retrievals are checked.
    np.testing.assert_array_equal(fixed_values, expected)
    np.testing.assert_array_equal(floats.view(np.uint32), reference.view(np.uint32))
    metadata = {"words": int(expected.size), "dtype": str(floats.dtype), "raw_result_node_name": fixed.name,
            "quantized_axis": axis, "descriptor_count": len(entries), "fixed_factor": float(fixed.factor.item()),
            "minimum_signed_word": int(expected.min()), "maximum_signed_word": int(expected.max()),
            "fixed_sha256": hashlib.sha256(fixed_values.tobytes()).hexdigest(),
            "float_sha256": hashlib.sha256(floats.tobytes()).hexdigest(), "exact_words": True,
            "exact_descriptor_float32": True}
    return metadata, fixed_values, floats


def transport_words(head, data, info):
    stem = f"dma2seq_{head['index']}"
    if data.name != stem + ".seq.16B" or info.name != stem + ".info":
        raise ValueError("Expected vendor-selected 16-bit file and metadata for this head")
    scale_bits, = struct.unpack("<I", struct.pack("<f", head["scale"]))
    fields = [int(x) for x in info.read_text(encoding="ascii").splitlines()]
    if fields != head["shape"] + [head["radix"], scale_bits]:
        raise ValueError("CSIM descriptor differs from typed NEF shape, radix or binary32 scale")
    values = np.loadtxt(data, dtype=np.int64)
    if values.size != np.prod(head["shape"]) or values.min() < -32768 or values.max() > 32767:
        raise ValueError("Invalid signed 16-bit element count or range")
    return values.reshape(head["shape"]).astype("<i2")
