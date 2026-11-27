from copy import deepcopy

import pytest

from radiation_edge_ai.nasa_bps.multimodel_nef import passing_metrics
from radiation_edge_ai.nasa_bps.multimodel_bie_comparison import V1_LIMITS
from radiation_edge_ai.nasa_bps.precision_enrollment import SOURCES, TRAINING_PANEL, TRAINING_POLICY, select_binary


def summary(mae):
    return {"confirmation": {"n": 24, "mae": mae, "bias": -mae,
                             "by_source": {s: {"n": 6, "mae": mae, "bias": -mae} for s in SOURCES}}}


def selection():
    candidate = {"status": "SCREEN_PASS", "summary": summary(.02),
                 "bie": {"bytes": 8, "sha256": "a" * 64},
                 "checks": {"lower_mae": True, "lower_absolute_bias": True, "source_mae_nonincrease": True}}
    full = deepcopy(candidate)
    full["bie"].update(bytes=9, sha256="b" * 64)
    return dict(status="TRAINING_SELECTION_COMPLETE", nef_eligible=False, physical_execution=False,
                policy_sha256=TRAINING_POLICY, panel_sha256=TRAINING_PANEL, baseline=summary(.2),
                candidates={"datapath16": candidate, "neural16": full},
                selected_configuration="datapath16", selected_bie_sha256="a" * 64)


def test_passing_training_can_enroll_but_cannot_compile():
    record = selection()
    assert select_binary(record) == "datapath16"
    with pytest.raises(ValueError, match="eight independent"):
        passing_metrics(record, V1_LIMITS)


def test_failed_source_and_wrong_selected_hash_cannot_be_enrolled():
    s = selection()
    s["selected_bie_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="identity"):
        select_binary(s)
    s = selection()
    s["candidates"]["datapath16"]["summary"]["confirmation"]["by_source"]["BALBCM1"]["mae"] = .21
    with pytest.raises(ValueError, match="Inconsistent"):
        select_binary(s)


def test_winner_and_contenders_must_be_bound_to_frozen_screen():
    s = selection()
    s["selected_configuration"] = "neural16"
    with pytest.raises(ValueError, match="Incorrect training selection"):
        select_binary(s)
    s["panel_sha256"] = "7" * 64
    with pytest.raises(ValueError, match="Unexpected training screen"):
        select_binary(s)
