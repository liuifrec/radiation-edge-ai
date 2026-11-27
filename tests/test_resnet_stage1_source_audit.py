import pytest

from radiation_edge_ai.nasa_bps import resnet_stage1_source_audit as audit


def test_source_summary_excludes_the_pilot_and_pairs_each_block_with_its_own_stem():
    calibration, selected, trace = [], [], []
    for i in range(192):
        row = dict(sample_id=str(i), sample_name=f"bag{i}", source_name=sorted(audit.SOURCES)[i % 4],
                   array_sha256=str(i), dose_Gy="0.0", hr_post_exposure="4", particle_type="Fe",
                   cohort="development", fold_female_role="train")
        calibration.append(row.copy())
        if i < 48:
            selected.append(row.copy())
            trace.append(dict(row, fp32_head=0.0,
                              opt0_suffix_head=dict(stem_pool=3.0, stage1_block0=4.0),
                              opt1_suffix_head=dict(stem_pool=1.5, stage1_block0=2.0)))
    panel = dict(manifest_sha256=audit.CALIBRATION, rows=selected)
    observed = dict(panel_sha256=audit.SELECTION, calibration_manifest_sha256=audit.CALIBRATION,
                    historical_pilot="0", rows=trace)
    summaries = audit.source_summary(panel, observed, calibration)
    assert sum(s["nuclei"] for s in summaries) == 47
    assert all(s["mean_baseline_block0_increment"] == 1 for s in summaries)
    assert all(s["mean_opt1_minus_opt0_block0_increment"] == -0.5 for s in summaries)
    calibration[0]["fold_female_role"] = "test"
    with pytest.raises(ValueError, match="Training membership mismatch"):
        audit.source_summary(panel, observed, calibration)
