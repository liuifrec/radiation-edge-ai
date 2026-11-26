import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from radiation_edge_ai._kl720_worker import (
    normalize_retrieved_output,
    output_order,
    pack_kl720_input,
    record_raw_receive,
)


def test_spotiflow_chw_packing_and_descriptor_decoding():
    # The same Generic Data path covers both registered checkpoints. Test a
    # one-channel input with four-channel alignment and signed clipping.
    layout = SimpleNamespace(KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B=1)
    kp = SimpleNamespace(ModelTensorDataLayout=layout)
    quant = SimpleNamespace(radix=0, scale=1)
    node = SimpleNamespace(tensor_shape_info=SimpleNamespace(v1=SimpleNamespace(shape_npu=(1, 1, 128, 128))),
                           quantization_parameters=SimpleNamespace(v1=SimpleNamespace(quantized_fixed_point_descriptor_list=[quant])),
                           data_layout=1)
    x = np.zeros((1, 1, 128, 128), dtype=np.float32)
    x[0, 0, 0, :3] = [-150, 3, 140]
    raw, metadata = pack_kl720_input(kp, node, x)
    assert metadata["buffer_bytes"] == 128 * 128 * 4
    assert np.frombuffer(raw, np.int8)[:12].tolist() == [-128, 0, 0, 0, 3, 0, 0, 0, 127, 0, 0, 0]
    heads = np.zeros((4, 128, 128), np.float32)
    assert normalize_retrieved_output(heads, (1, 4, 128, 128)).shape == (1, 4, 128, 128)
    with pytest.raises(RuntimeError):
        normalize_retrieved_output(heads.transpose(1, 2, 0), (1, 4, 128, 128))


def test_raw_receive_binds_inference_number_and_preserves_allocation(tmp_path):
    payload = ctypes.create_string_buffer(b"\x00\xb7\x12", 3)
    pointer = SimpleNamespace(_LP_raw_out_buffer=ctypes.addressof(payload))
    result = SimpleNamespace(header=SimpleNamespace(get_member_variable_dict=lambda: {"inference_number": 4}),
                             raw_result=SimpleNamespace(buffer_size=3, _get_element_buffer=lambda: pointer))
    path = tmp_path / "usb.bin"
    with pytest.raises(RuntimeError, match="different"):
        record_raw_receive(result, path, 3)
    assert not path.exists()
    receipt = record_raw_receive(result, path, 4)
    assert path.read_bytes() == b"\x00\xb7\x12"
    assert receipt["allocation"]["size_bytes"] == 3


def test_kl720_v1_signed_word_input_preserves_low_bits_and_pads_each_row():
    layouts = SimpleNamespace(KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B=1, KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B=2,
                              KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B=3, KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B=4)
    node = SimpleNamespace(data_layout=4, tensor_shape_info=SimpleNamespace(v1=SimpleNamespace(shape_npu=(1, 1, 2, 5))),
                           quantization_parameters=SimpleNamespace(v1=SimpleNamespace(
                               quantized_fixed_point_descriptor_list=[SimpleNamespace(radix=1, scale=1)])))
    image = np.array([[-20000, -1.5, .25, .75, 20000], [2.5, 1, -1, 1.5, -2.5]], np.float32)
    raw, pack = pack_kl720_input(SimpleNamespace(ModelTensorDataLayout=layouts), node, image[None, None])
    expected = np.array([[-32768, -3, 0, 2, 32767, 0, 0, 0], [5, 2, -2, 3, -5, 0, 0, 0]], dtype="<i2")
    assert raw == expected.tobytes()
    assert pack["layout"] == "8W1C16B" and pack["buffer_bytes"] == 32


def test_distinct_descriptor_shapes_identify_independently_quantized_heads():
    shapes = [(1, 3, 128, 128), (1, 1, 128, 128)]
    assert output_order(shapes, [1, 3]) == ([1, 0], (1, 4, 128, 128))
    for wrong in (None, [1, 1], [4]):
        with pytest.raises(RuntimeError):
            output_order(shapes, wrong)


def test_signed_input_matches_validated_sdk_high_low_lane_order():
    layouts = SimpleNamespace(KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B=1, KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B=2,
                              KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B=3, KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8BHL=6)
    kp = SimpleNamespace(ModelTensorDataLayout=layouts)
    words = np.array([[4661, -1, 40000], [-40000, 3.5, 2.5]], np.float32)[None, None]
    quant = SimpleNamespace(v1=SimpleNamespace(quantized_fixed_point_descriptor_list=[SimpleNamespace(radix=0, scale=1.)]))
    info = SimpleNamespace(v1=SimpleNamespace(shape_npu=words.shape))
    node = SimpleNamespace(data_layout=6, tensor_shape_info=info, quantization_parameters=quant)
    data, metadata = pack_kl720_input(kp, node, words)
    expected = bytearray(6 * 32)
    for pixel, (lo, hi) in enumerate(((26, 18), (127, 255), (127, 127), (0, 128), (2, 0), (1, 0))):
        expected[32*pixel], expected[32*pixel+16] = lo, hi
    assert data == expected
    assert metadata['layout'] == '1W16C8BHL' and metadata['buffer_bytes'] == 192
    node.tensor_shape_info = SimpleNamespace(v2=SimpleNamespace(shape=words.shape))
    with pytest.raises(RuntimeError, match='KL720 V1'):
        pack_kl720_input(kp, node, words)
