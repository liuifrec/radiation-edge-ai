from __future__ import annotations

from radiation_edge_ai.cli import main


def test_assays_cli(capsys) -> None:
    assert main(["assays"]) == 0
    out = capsys.readouterr().out
    assert "nasa-53bp1-r1-v2" in out
    assert "dnai-fiber-v3" in out


def test_assays_cli_json(capsys) -> None:
    assert main(["assays", "--json"]) == 0
    out = capsys.readouterr().out
    assert '"assay_id": "dnai-fiber-v3"' in out
