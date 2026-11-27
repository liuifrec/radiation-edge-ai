import hashlib
import json
from pathlib import Path

import pytest

from radiation_edge_ai.nasa_bps import training_precision as precision
from radiation_edge_ai.nasa_bps.training_precision_readout import (
    SOURCES,
    check_float_word,
    training_decision,
)


def result(mae, bias, per_source=None, size=100):
    by_source = {s: {"mae": v} for s, v in zip(SOURCES, per_source or [mae] * 4)}
    return {
        "status": "TRAINING_READOUT_COMPLETE",
        "bie": {"bytes": size, "sha256": str(size)},
        "summary": {"confirmation": {"mae": mae, "bias": bias, "by_source": by_source}},
    }


def test_recipe_identity_and_default_inherited_weight_width():
    policy = Path(__file__).resolve().parents[1] / "docs/NASA_MOBILENET_TRAINING_PRECISION_V1.json"
    assert hashlib.sha256(policy.read_bytes()).hexdigest() == precision.POLICY_SHA256
    assert json.loads(policy.read_bytes())["recipes"] == precision.RECIPES
    assert precision.RECIPES["datapath16"] == {"datapath_bitwidth_mode": "int16"}
    assert set(precision.RECIPES["neural16"]) == {
        "datapath_bitwidth_mode",
        "weight_bitwidth_mode",
        "model_in_bitwidth_mode",
        "model_out_bitwidth_mode",
    }


def test_no_advancement_from_better_average_alone():
    baseline = result(1, -0.4)
    first = result(0.51, -0.3, [0.2, 0.2, 0.2, 1.44])
    second = result(0.5, 0.6)
    decision = training_decision(baseline, {"datapath16": first, "neural16": second})
    assert decision["selected_configuration"] is None
    assert decision["candidates"]["datapath16"]["checks"]["source_mae_nonincrease"] is False
    assert decision["candidates"]["neural16"]["checks"]["lower_absolute_bias"] is False


def test_exact_tie_uses_size_then_recipe_and_never_allows_nef():
    baseline = result(1, -0.4)
    candidates = {"datapath16": result(0.1, 0, size=130), "neural16": result(0.1, 0, size=90)}
    decision = training_decision(baseline, candidates)
    assert decision["selected_configuration"] == "neural16" and not decision["nef_eligible"]
    candidates["neural16"] = candidates["datapath16"]
    assert training_decision(baseline, candidates)["selected_configuration"] == "datapath16"


def test_supported_incomplete_candidate_cannot_be_scored():
    with pytest.raises(ValueError, match="Incomplete"):
        training_decision(
            result(1, -0.4), {"datapath16": result(0.1, 0), "neural16": {"status": "FAILED"}}
        )


def test_precision_report_requires_effective_settings(tmp_path):
    report = {"docker_version": "kneron/toolchain:v0.33.1"}
    report.update({"kdp720/" + k + " bitwidth": "int8" for k in precision.SETTINGS})
    (tmp_path / "model_fx_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Effective precision"):
        precision.verify_report(tmp_path, tmp_path / "x.bie", "datapath16")


@pytest.mark.parametrize("word,width", [(-128, 8), (127, 8), (-32768, 16), (32767, 16)])
def test_actual_signed_output_width_and_boundary(word, width):
    check_float_word(word, word / 8, {"width": width, "scale": 1.0, "radix": 3})
    with pytest.raises(ValueError, match="Word exceeds"):
        check_float_word(2 ** (width - 1), 0, {"width": width, "scale": 1.0, "radix": 3})
