import pytest

from radiation_edge_ai.nasa_bps.multimodel_bie_comparison import V1_LIMITS
from radiation_edge_ai.nasa_bps.multimodel_nef import passing_metrics


def receipt(values):
    decisions = {k: {"maximum": v, "actual": values[k], "pass": values[k] <= v} for k, v in V1_LIMITS.items()}
    passed = all(row["pass"] for row in decisions.values())
    return dict(metrics=values.copy(), decisions=decisions, all_declared_limits_pass=passed,
                eligible_for_nef_compilation=passed)


def test_compilation_accepts_equality_but_preserves_each_failed_limit():
    values = dict(V1_LIMITS)
    assert passing_metrics(receipt(values), V1_LIMITS) == []
    values["bag_paired_max_abs"] += 0.000001
    result = receipt(values)
    assert passing_metrics(result, V1_LIMITS) == ["bag_paired_max_abs"]
    result["eligible_for_nef_compilation"] = True
    with pytest.raises(ValueError, match="Inconsistent NEF"):
        passing_metrics(result, V1_LIMITS)


def test_missing_false_or_nonfinite_measurements_cannot_enable_compilation():
    result = receipt(V1_LIMITS)
    for invalid in (None, False, float("nan")):
        result["decisions"]["bag_paired_max_abs"]["actual"] = invalid
        with pytest.raises(ValueError, match="Invalid engineering measurement"):
            passing_metrics(result, V1_LIMITS)
    del result["decisions"]["bag_paired_max_abs"]
    with pytest.raises(ValueError, match="eight independent"):
        passing_metrics(result, V1_LIMITS)
