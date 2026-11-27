import numpy as np
import pytest

from radiation_edge_ai.nasa_bps import mobilenet_terminal_audit as audit


@pytest.mark.parametrize("original,code,expected", [
    (-100, -128, (-128, -128)), (100, 127, (127, 127)), (0, 0, (0, 0)),
    (1.25, 2, (2, 3)), (-1.25, -3, (-3, -2)),
])
def test_conservative_grid_saturates_without_assuming_rounding(original, code, expected):
    grid = audit.signed_grid(1.0, 1)
    value = grid[code + 128]
    row = audit.compare_scalar(grid, original, code, f"{value:.8f}", value)
    assert (row["lo_code"], row["hi_code"]) == expected
    assert not row["outside_code_interval"]


def test_signed_code_native_text_and_api_retrieval_are_distinct():
    grid = audit.signed_grid(3.0, 4)
    reported = float(np.float32("0.02083333"))
    row = audit.compare_scalar(grid, 0.2, 1, "0.02083333", reported)
    assert row["decoded"] != row["returned"] == row["text_roundtrip_float32"]
    assert row["outside_code_interval"] and (row["lo_code"], row["hi_code"]) == (9, 10)
    with pytest.raises(ValueError, match="eight-decimal text"):
        audit.compare_scalar(grid, 1.0, 2, "0.02083333", row["returned"])
    with pytest.raises(ValueError, match="API return"):
        audit.compare_scalar(grid, 1.0, 1, "0.02083333", 0.5)


def test_changed_archive_fails_before_readout(tmp_path):
    (tmp_path / "training_pilot").mkdir()
    (tmp_path / "training_pilot/training_pilot.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed training_pilot.json"):
        audit.audit(tmp_path, tmp_path, tmp_path / "result")
    assert not (tmp_path / "result").exists()
