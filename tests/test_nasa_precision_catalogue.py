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
    for r in variants[:2]:
        assert r["engineering"]["nef_compilation"] == "blocked_pending_development_fidelity"
        assert "nef" not in r["artifacts"]
        assert r["engineering"]["development_regression"] == "not_selected"
        assert r["scope"] == "Training-only precision experiment"
    assert models["nasa-mobilenetv2"]["engineering"]["engineering_gates_passed"] == 7


def test_neural16_catalogue_distinguishes_development_regression_and_scientific_status():
    result, = catalogue.report("nasa-resnet18-female@neural16")["models"]
    engineering = result["engineering"]
    assert engineering["development_regression"] == "passed"
    assert engineering["engineering_gates_passed"] == engineering["engineering_gates_total"] == 8
    assert (engineering["development_nuclei"], engineering["development_bags"],
            engineering["development_contrasts"]) == (2400, 24, 12)
    assert result["scope"] == "Known female-source development regression"
    assert result["scientific"] == "Independent biological evaluation pending"
    assert engineering["nef_compilation"] == "passed"
    assert (engineering["compiled_model_id"], engineering["compiled_model_version"],
            engineering["nef_bytes"]) == (32783, 1, 51392996)
    assert engineering["device_load_verified"] is False
    assert engineering["nef_simulation"] == "passed"
    assert engineering["nef_simulation_panel"] == "48 frozen training witnesses"
    assert engineering["max_abs_transport_vs_bie_foci_per_nucleus"] == 0.0
    assert engineering["measured_transport_words_equal_to_twice_native"] == 48
    assert result["artifacts"]["nef_simulation"]["sha256"] == (
        "1706ad42edcf401b055077534e6839e63597449f783a0422ab70ebce4d60d59b"
    )
    assert result["artifacts"]["nef"]["sha256"] == (
        "7f1843a62bb5244048f4df977051b6795f7a39320c6be890cc0fdd9a7f92d4ba"
    )
    assert result["artifacts"]["bie_comparison"]["sha256"] == (
        "b621663167bd60272b84e2e5a279251b28b4e79d3a28d621713768d671179736"
    )
    assert result["artifacts"]["development_audit"]["sha256"] == (
        "4f9dd6f061586840636de0aa4292944f880bef6325e4f0dcc0f90f7aa571bf20"
    )


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
