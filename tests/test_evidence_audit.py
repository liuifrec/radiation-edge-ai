from __future__ import annotations

from pathlib import Path

import pytest

from radiation_edge_ai.audit_utils import (
    enrich_contrast_sample_names,
    join_contrasts,
    peak_series_audit,
    read_json_object,
    refuse_nonempty_output_dir,
)


def make_row(source, branch, hour, nasa_sign=1, pred_sign=1):
    nasa = 1.0 * nasa_sign
    pred = 0.8 * pred_sign
    return {
        "source_name": source,
        "branch": branch,
        "timepoint_hr": str(hour),
        "sham_sample_name": f"{source}_{branch}_{hour}_sham",
        "exposed_sample_name": f"{source}_{branch}_{hour}_exp",
        "nasa_delta": str(nasa),
        "pred_delta": str(pred),
    }


def test_same_direction_counts_can_hide_different_failed_contrasts():
    keys = [
        ("S1", "Fe", 4),
        ("S1", "Fe", 24),
        ("S1", "Fe", 48),
        ("S1", "X-ray", 4),
        ("S1", "X-ray", 24),
        ("S2", "Fe", 4),
        ("S2", "Fe", 24),
        ("S2", "Fe", 48),
        ("S3", "X-ray", 4),
        ("S3", "X-ray", 24),
        ("S3", "X-ray", 48),
    ]
    fp32 = [make_row(*key, pred_sign=(-1 if key == ("S1", "Fe", 24) else 1)) for key in keys]
    bie = [dict(row) for row in fp32]
    physical = [make_row(*key, pred_sign=(-1 if key == ("S1", "Fe", 48) else 1)) for key in keys]
    joined, audit = join_contrasts(fp32, bie, physical)
    assert len(joined) == 11
    assert audit["actual_fp32_vs_physical_per_contrast_sign_identity"] is False
    assert len(audit["fp32_vs_physical_changed_contrasts"]) == 2
    # Both methods still have one NASA-direction error in the same source/time strata totals.
    assert sum(row["fp32_direction_agreement_with_nasa"] for row in joined) == 10
    assert sum(row["physical_direction_agreement_with_nasa"] for row in joined) == 10


def test_duplicate_missing_and_shuffled_keys():
    base = [make_row("S1", "Fe", 4), make_row("S1", "Fe", 24)]
    with pytest.raises(RuntimeError, match="Duplicate FP32 contrast"):
        join_contrasts(base + [dict(base[0])], base, base)
    with pytest.raises(RuntimeError, match="Contrast key sets differ"):
        join_contrasts(base, base[:-1], base)
    shuffled = list(reversed(base))
    joined, audit = join_contrasts(base, shuffled, shuffled)
    assert len(joined) == 2
    assert audit["actual_fp32_vs_physical_per_contrast_sign_identity"] is True


def test_peak_series_identity_and_ties():
    rows = []
    for hour, nasa, fp32, bie, physical in [
        (4, 1.0, 1.0, 1.0, 1.0),
        (24, 2.0, 2.0, 2.0, 2.0),
        (48, 1.0, 1.5, 1.0, 2.0),
    ]:
        rows.append(
            {
                "source_name": "S1",
                "branch": "Fe",
                "timepoint_hr": str(hour),
                "nasa_delta": nasa,
                "fp32_delta": fp32,
                "bie_delta": bie,
                "physical_delta": physical,
            }
        )
    audit = peak_series_audit(rows)
    assert len(audit) == 1
    row = audit[0]
    assert row["fp32_peak_hours"] == "24"
    assert row["physical_peak_hours"] == "24,48"
    assert row["physical_peak_tied"] is True
    assert row["fp32_vs_physical_peak_set_same"] == 0


def test_nonfinite_json_fails_closed(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text('{"x": NaN}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Non-finite JSON"):
        read_json_object(path)


def test_refuse_overwrite_nonempty_directory(tmp_path: Path):
    out = tmp_path / "result"
    out.mkdir()
    refuse_nonempty_output_dir(out)
    (out / "done.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        refuse_nonempty_output_dir(out)


def test_legacy_fp32_delta_is_enriched_from_verified_bags():
    delta = [
        {
            "source_name": "S1",
            "branch": "Fe",
            "timepoint_hr": "4",
            "exposed_dose_Gy": "0.82",
            "nasa_delta": "1.0",
            "pred_delta": "0.5",
        }
    ]
    bags = [
        {
            "sample_name": "S1_Fe_0_4",
            "source_name": "S1",
            "particle_type": "Fe",
            "dose_Gy": "0.0",
            "hr_post_exposure": "4",
        },
        {
            "sample_name": "S1_Fe_082_4",
            "source_name": "S1",
            "particle_type": "Fe",
            "dose_Gy": "0.82",
            "hr_post_exposure": "4",
        },
    ]
    rows = enrich_contrast_sample_names(
        delta, bags, exposed_dose_by_branch={"Fe": "0.82", "X-ray": "1.0"}, label="FP32"
    )
    assert rows[0]["sham_sample_name"] == "S1_Fe_0_4"
    assert rows[0]["exposed_sample_name"] == "S1_Fe_082_4"


def test_existing_contrast_sample_name_must_match_bag_identity():
    delta = [
        {
            "source_name": "S1",
            "branch": "X-ray",
            "timepoint_hr": "24",
            "sham_sample_name": "WRONG",
            "exposed_sample_name": "S1_X_1_24",
            "nasa_delta": "1.0",
            "pred_delta": "0.5",
        }
    ]
    bags = [
        {
            "sample_name": "S1_X_0_24",
            "source_name": "S1",
            "particle_type": "X-ray",
            "dose_Gy": "0",
            "hr_post_exposure": "24",
        },
        {
            "sample_name": "S1_X_1_24",
            "source_name": "S1",
            "particle_type": "X-ray",
            "dose_Gy": "1",
            "hr_post_exposure": "24",
        },
    ]
    with pytest.raises(RuntimeError, match="sham_sample_name mismatch"):
        enrich_contrast_sample_names(
            delta, bags, exposed_dose_by_branch={"Fe": "0.82", "X-ray": "1.0"}, label="physical"
        )
