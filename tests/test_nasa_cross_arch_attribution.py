import pytest

from radiation_edge_ai.nasa_bps.cross_arch_attribution import paired_records, summarize


def inputs(architecture="mobilenet_v2"):
    selected = [dict(sample_id="a", source_name="BALBCM1", array_sha256="8" * 64,
                     panel="confirmation")]
    members = {"a": dict(**selected[0], fold_female_role="train",
                         dataset_role="development_train_ptq_calibration")}
    rows = [dict(sample_id="a", source_name="BALBCM1", input_sha256="8" * 64,
                 panel="confirmation", original_fp32="2", pooled_head_fp32="1", bie="1.5")]
    return architecture, rows, selected, members


def test_signed_components_can_cancel_and_common_tensor_is_paired():
    mobile = paired_records(*inputs())
    resnet = paired_records(*inputs("resnet18"))
    result = summarize({"mobilenet_v2": mobile, "resnet18": resnet})
    common = result["summaries"]["mobilenet_v2"]["common_nonpilot"]
    assert result["common_nonpilot_ids"] == ["a"]
    assert common["upstream"]["bias"] == -1
    assert common["head_and_output"]["bias"] == .5
    assert common["total"]["mae"] == .5
    assert result["summaries"]["resnet18"]["confirmation_fp32_zero"] == {"n": 0}
    resnet["a"]["panel"] = "pilot"
    assert not summarize({"mobilenet_v2": mobile, "resnet18": resnet})["common_nonpilot_ids"]
    resnet["a"]["input_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="different training inputs"):
        summarize({"mobilenet_v2": mobile, "resnet18": resnet})


@pytest.mark.parametrize("change,value", [
    ("source_name", "BALBCF1"), ("input_sha256", "a" * 64), ("panel", "pilot"),
    ("bie", "nan"), ("pooled_head_fp32", "-.4"),
])
def test_mislabeled_or_invalid_attribution_rejected(change, value):
    args = inputs()
    args[1][0][change] = value
    with pytest.raises(ValueError):
        paired_records(*args)


def test_missing_duplicate_or_held_out_row_rejected():
    arch, rows, selected, members = inputs()
    with pytest.raises(ValueError, match="Incomplete"):
        paired_records(arch, [], selected, members)
    with pytest.raises(ValueError, match="Repeated"):
        paired_records(arch, rows * 2, selected, members)
    members["a"]["fold_female_role"] = "test"
    with pytest.raises(ValueError, match="training-source"):
        paired_records(arch, rows, selected, members)
