import pytest

from radiation_edge_ai.nasa_bps.deployment_cohort import cohort_identity, require_cohort


def panel():
    rows = []
    for source in ("A", "B"):
        for particle, exposed in (("Fe", "0.82"), ("X-ray", "1.0")):
            for hour in (4, 24, 48):
                for dose in ("0.0", exposed):
                    bag = f"{source}-{particle}-{hour}-{dose}"
                    for index in range(100):
                        key = f"{bag}-{index}"
                        rows.append(dict(sample_id=key, nucleus_key="n" + key, sample_name=bag,
                                         source_name=source, strain=source, sex="Female",
                                         particle_type=particle, dose_Gy=dose, hr_post_exposure=hour))
    return rows


def test_order_and_numeric_spelling_do_not_change_frozen_membership():
    rows = panel()
    policy = cohort_identity(rows)
    for row in rows:
        row["hr_post_exposure"] = float(row["hr_post_exposure"])
        row["dose_Gy"] = float(row["dose_Gy"])
    assert require_cohort(reversed(rows), policy) == policy
    assert (policy["nuclei"], policy["bags"], policy["contrasts"]) == (2400, 24, 12)


def test_swapping_identities_or_contrast_doses_prevents_readout():
    rows = panel()
    policy = cohort_identity(rows)
    rows[0]["nucleus_key"], rows[100]["nucleus_key"] = rows[100]["nucleus_key"], rows[0]["nucleus_key"]
    with pytest.raises(ValueError, match="cohort_membership_sha256"):
        require_cohort(rows, policy)
    for row in rows[:200]:
        row["dose_Gy"] = "0.82" if row["dose_Gy"] == "0.0" else "0.0"
    assert cohort_identity(rows)["contrast_membership_sha256"] != policy["contrast_membership_sha256"]


def test_incomplete_or_duplicated_membership_fails():
    rows = panel()
    with pytest.raises(ValueError, match="Incomplete"):
        cohort_identity(rows[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        cohort_identity(rows + [rows[0]])
