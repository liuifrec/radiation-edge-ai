import json
from pathlib import Path

import pytest

from radiation_edge_ai.hardware import deployment_catalogue as catalogue
from radiation_edge_ai.nasa_bps.multimodel_bie_comparison import V1_LIMITS
from radiation_edge_ai.nasa_bps.multimodel_nef import passing_metrics
from radiation_edge_ai.nasa_bps.multimodel_ptq import sha256


def test_precision_variants_are_individually_identified_and_simulation_only():
    models = {r["id"]: r for r in catalogue.report()["models"]}
    keys = ["nasa-mobilenetv2@datapath16", "nasa-mobilenetv2@neural16", "nasa-resnet18-female@neural16"]
    variants = [models[k] for k in keys]
    assert len({r["artifacts"]["bie"]["sha256"] for r in variants}) == 3
    for r in variants:
        assert r["track"] == "experimental"
        assert r["desktop_backends"] == []
        assert r["engineering"]["physical_kl720"] == "not_run"
        assert r["engineering"]["nef_compilation"] == "blocked_pending_development_fidelity"
        assert "nef" not in r["artifacts"]
    assert all(r["engineering"]["development_regression"] == "not_selected" for r in variants[:2])
    assert models["nasa-mobilenetv2"]["engineering"]["engineering_gates_passed"] == 7


def test_resnet_prospective_protocol_binds_the_new_binary_and_retains_every_limit():
    docs = Path(__file__).resolve().parents[1] / "docs"
    record = json.loads((docs / "NASA_RESNET18_NEURAL16_REGRESSION_V1.json").read_bytes())
    historic = json.loads((docs / "NASA_PTQ_ACCEPTANCE_V1.json").read_bytes())
    assert record["historical_policy_sha256"] == sha256(docs / "NASA_PTQ_ACCEPTANCE_V1.json")
    assert record["acceptance_document_sha256"] == sha256(docs / record["protocol"])
    assert record["standard_bie_sha256"] == historic["bie_sha256"] != record["bie_sha256"]
    assert record["maximums"] == historic["maximums"] == V1_LIMITS
    assert record["strong_effect_min_abs_fp32_delta"] == historic["strong_effect_min_abs_fp32_delta"]
    assert record["scope"] == "known_female_development_regression"
    assert record["nef_eligible_from_training"] is False
    assert [record[k] for k in ("nuclei","bags","contrasts")] == [2400,24,12]
    registered = catalogue.catalogue()["nasa-resnet18-female@neural16"]
    assert registered["artifacts"]["bie"][1] == record["bie_sha256"]
    assert registered["artifacts"]["training_readout"][1] == record["training_readout_sha256"]
    with pytest.raises(ValueError, match="eight independent"):
        passing_metrics(record, V1_LIMITS)
