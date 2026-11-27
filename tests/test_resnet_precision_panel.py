import pytest

from radiation_edge_ai.nasa_bps.training_precision_readout import SOURCES, panel_inputs, summarize


def test_resnet_training_panel_distinguishes_historical_pilot_from_remaining_sources():
    rows = []
    for source in SOURCES:
        for bag in range(12):
            key = f"{source}-{bag}"
            rows.append(
                {
                    "sample_id": key,
                    "source_name": source,
                    "sample_name": key,
                    "nucleus_key": key,
                    "array_sha256": key,
                    "array_file": key,
                    "fold_female_role": "train",
                    "calibration_fold": "fold_female_role",
                    "dataset_role": "development_train_ptq_calibration",
                }
            )
    original = [{"sample_id": r["sample_id"], "source_fp32": "0.25"} for r in rows]
    policy = {"architecture": "resnet18", "pilot": rows[0]["sample_id"]}
    result = panel_inputs({"rows": rows}, rows, original, policy)
    assert sum(r["panel"] == "pilot" for r in result) == 1
    assert all(r["source_fp32"] == 0.25 for r in result)
    scores = summarize([dict(r, error=0.0) for r in result])
    assert scores["confirmation"]["n"] == 47
    assert scores["pilot"]["by_source"][SOURCES[-1]] == {"n": 0}
    rows[1]["sample_name"] = rows[0]["sample_name"]
    with pytest.raises(ValueError, match="bags or pilot"):
        panel_inputs({"rows": rows}, rows, original, policy)
