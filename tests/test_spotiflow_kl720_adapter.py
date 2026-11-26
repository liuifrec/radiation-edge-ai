import numpy as np
import pytest

from radiation_edge_ai.hardware.spotiflow import decode_heads, join_heads, prepare_input


def test_other_padding_buckets_are_rejected():
    for shape in ((112, 128), (128, 129), (128, 128, 3)):
        with pytest.raises(ValueError, match="native"):
            prepare_input(np.zeros(shape, dtype=np.float32))


def test_separate_sdk_heads_reassemble_in_original_units_and_order():
    raw = np.arange(4 * 128 * 128, dtype=np.float32).reshape(1, 4, 128, 128)
    np.testing.assert_array_equal(join_heads(raw[:, :1], raw[:, 1:]), raw)
    with pytest.raises(ValueError, match="ordered"):
        join_heads(raw[:, 1:], raw[:, :1])


def test_odd_padding_and_vector_channel_order():
    pytest.importorskip("spotiflow")
    raw = np.zeros((1, 4, 128, 128), np.float32)
    raw[:, 0] = -10.0
    raw[:, 0, 11, 21] = 10.0
    raw[:, 3] = 2.0
    decoded = decode_heads(raw, (127, 127), 0.5)
    assert decoded["heatmap"].shape == (127, 127)
    np.testing.assert_array_equal(decoded["flow"][10, 20], [0, 0, 1])
    np.testing.assert_allclose(decoded["points"], [[10, 21]], atol=1e-5)
    np.testing.assert_array_equal(decoded["confidence"], decoded["heatmap"][[10], [20]])


def test_native_normalization_keeps_high_outlier_and_reflects_edges():
    pytest.importorskip("spotiflow")
    image = np.arange(127 * 127, dtype=np.float32).reshape(127, 127)
    image[-1, -1] = 1e8
    tensor = prepare_input(image)
    assert tensor.shape == (1, 1, 128, 128) and tensor.dtype == np.float32
    assert tensor[0, 0, -1, -1] > 1
    assert tensor[0, 0, 0, 0] == tensor[0, 0, 2, 2]


def test_refined_peaks_outside_original_field_are_removed_with_confidence():
    pytest.importorskip("spotiflow")
    raw = np.zeros((1, 4, 128, 128), np.float32)
    raw[:, 0] = -10
    raw[:, 0, 50, 0] = 10
    raw[:, 3] = -1
    values = decode_heads(raw, (128, 128), 0.5)
    assert values["points"].shape == (0, 2)
    assert values["confidence"].size == 0


def test_single_crop_boundaries_and_duplicate_mask_assignments():
    from radiation_edge_ai.hardware.spotiflow import _check_geometry, target_assignment

    for shape in ((113, 128), (128, 113), (114, 126)):
        _check_geometry(shape)
    mask = np.array([[1, 2, 2], [0, 1, 1]], dtype=np.int16)
    points = np.array([[0.0, 0.5], [0.0, 0.51], [0.0, -0.51], [1.0, 2.5]])
    pixels, selected, labels = target_assignment(mask, 2, points, len(points))
    assert pixels == [[0, 1], [0, 1], None, None]
    assert selected == [True, True, False, False] and labels == [2, 2, None, None]
