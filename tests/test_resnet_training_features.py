import numpy as np
import pytest

from radiation_edge_ai.nasa_bps.resnet_training_features import (
    MAP,
    code_range,
    pool_codes,
    scalar_effects,
    signed_word,
)


def test_signed_word_rejects_unsigned_or_wrong_scale():
    d = {"name": "burden", "onnx_shape": [1], "bitw": 8, "radix": [2], "scale": [2]}
    assert signed_word(b"\x80", d, -16) == -128
    assert signed_word(b"\x7f", d, 15.875) == 127
    with pytest.raises(ValueError, match="terminal word"):
        signed_word(b"\xff", d, 31.875)
    with pytest.raises(ValueError, match="terminal word"):
        signed_word(b"\x7f", dict(d, radix=[3]), 15.875)


@pytest.mark.parametrize("dtype", ["i1", "<i2"])
def test_clipped_negative_stem_activations_are_zero_after_relu(dtype):
    lo, hi = np.iinfo(dtype).min, np.iinfo(dtype).max
    codes = np.array([[[[lo, 0, 1, hi]], [[-4, 4, 1, 2]]]], dtype=dtype)
    native = codes * np.array([0.125, 0.01])[None, :, None, None]
    fp = native.copy()
    fp[0, 0, 0, 0] -= 2
    result = code_range(codes, native, fp, np.maximum(native, 0))
    assert (result["lower"], result["upper"], result["lower_relu_positive"]) == (1, 1, 0)
    assert result["reference_below"] == 1 and result["reference_above"] == 0
    with pytest.raises(ValueError, match="layout"):
        code_range(codes, native[..., ::-1], fp, native)


def test_signed_effects_reconstruct_observed_burden_with_zero_floor():
    fp = {"head": np.array([0.8]), "burden": np.array([0.8])}
    fixed = {
        "head": np.array([-0.1]),
        "burden": np.array([0]),
        "nonnegative": np.array([0]),
        "flatten": np.array([-0.02]),
        MAP: np.array([[[[0.1, 0.3]]]]),
    }
    suffix = {"stem_conv": 0.7, "stem_relu": 0.6, "stem_pool": 0.55}
    effects, pooled, spatial, pre = scalar_effects(
        fp, fixed, 0.75, suffix, lambda v: float(v.mean())
    )
    assert pooled == 0 and pre < 0 and spatial == pytest.approx(0.2)
    assert sum(effects.values()) == pytest.approx(-0.8)
    assert effects["floor"] == pytest.approx(0.1)


def test_pool_code_comparison_excludes_ties_out_of_range_and_agreement():
    # Field 1 gives a floor match, field 2 a nearest match, field 3 neither.
    codes = np.array([[[[0, 1, 3, 1, 127, 1]]]], dtype=np.int8)
    native = codes.astype(np.float32) / 2
    ideal = np.array([[[[0.6, 0.7, 0.9, 1.5, 200, 1]]]]) / 2
    result = pool_codes(codes, native, ideal)
    assert result["checked"] == 6 and result["in_range"] == 5
    assert result["distinguish"] == 3
    assert (
        result["distinguish_floor"],
        result["distinguish_nearest"],
        result["distinguish_neither"],
    ) == (1, 1, 1)
