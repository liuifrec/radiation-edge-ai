"""KL720 ResNet18 scalar transport and three-channel packing checks, usable without USB."""

import hashlib
import struct

import numpy as np

from .multimodel_ptq import require

SCALE = 1.3206359148025513
SHAPE = (1, 3, 256, 256)
SMOKE = "BPSR1V2D_01501"
MAX_ABS = 1e-4


def input_buffer(tensor):
    require(tensor.dtype == np.float32 and tensor.shape == SHAPE and np.isfinite(tensor).all(),
            "Expected finite float32 [1,3,256,256]")
    hwc = tensor[0].transpose(1, 2, 0)
    codes = np.clip(np.rint(hwc * np.float32(32768)), -32768, 32767).astype("<i2").view("<u2")
    # 1W16C8BHL stores bits 7..1, then bits 15..8, in two groups of 16 channel lanes.
    lanes = np.zeros((SHAPE[2], SHAPE[3], 2, 16), dtype=np.uint8)
    lanes[:, :, 0, :3] = (codes >> 1) & 127
    lanes[:, :, 1, :3] = codes >> 8
    return lanes.tobytes()


def descriptor_bits():
    return struct.unpack("<I", struct.pack("<f", SCALE))[0]


def decode_transport(word):
    require(type(word) is int and -32768 <= word <= 32767, "Invalid transport scalar")
    return np.float32(word) / np.float32(SCALE * 8192)


def captured_scalar(folder, native_word, bie):
    info = (folder / "dma2seq_0.info").read_text(encoding="ascii").split()
    require(list(map(int, info)) == [1, 1, 1, 1, 13, descriptor_bits()], "CSIM transport descriptor disagrees")
    carrier = folder / "dma2seq_0.seq.16B"
    fields = carrier.read_text(encoding="ascii").split()
    require(len(fields) == 1, "CSIM must emit one signed transport code")
    word = int(fields[0])
    decoded = float(decode_transport(word))
    raw_bie = np.float32(native_word) / np.float32(4096) / np.float32(SCALE)
    dynasty_float = np.float32(format(float(raw_bie), ".8f"))
    require(-16384 <= native_word <= 16383 and float(dynasty_float) == bie,
            "BIE native code and frozen simulator output disagree")
    delta = abs(decoded - bie)
    return {"word": word, "decoded": decoded, "bie": bie, "bie_native_word": native_word,
            "double_native_code": word == native_word * 2, "abs_error": delta,
            "signed_le16_sha256": hashlib.sha256(struct.pack("<h", word)).hexdigest(),
            "passed": bool(delta <= MAX_ABS)}
