from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from radiation_edge_ai.control import ControlPlaneError
from radiation_edge_ai.nasa_endpoint import (
    create_nasa_endpoint_report,
    verify_nasa_endpoint_report,
)

FIELDS = [
    "sample_id",
    "sample_name",
    "source_name",
    "strain",
    "sex",
    "particle_type",
    "dose_Gy",
    "hr_post_exposure",
    "physical_kl720_burden",
]


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _base_rows() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "N1",
            "sample_name": "SAMPLE_A",
            "source_name": "SRC1",
            "strain": "BALB/cByJ",
            "sex": "Female",
            "particle_type": "Fe",
            "dose_Gy": "0.0",
            "hr_post_exposure": "4",
            "physical_kl720_burden": "1.0",
        },
        {
            "sample_id": "N2",
            "sample_name": "SAMPLE_A",
            "source_name": "SRC1",
            "strain": "BALB/cByJ",
            "sex": "Female",
            "particle_type": "Fe",
            "dose_Gy": "0",
            "hr_post_exposure": "4.0",
            "physical_kl720_burden": "3.0",
        },
        {
            "sample_id": "N3",
            "sample_name": "SAMPLE_B",
            "source_name": "SRC2",
            "strain": "C57BL/6J",
            "sex": "Female",
            "particle_type": "X-ray",
            "dose_Gy": "1.0",
            "hr_post_exposure": "24",
            "physical_kl720_burden": "2.0",
        },
        {
            "sample_id": "N4",
            "sample_name": "SAMPLE_B",
            "source_name": "SRC2",
            "strain": "C57BL/6J",
            "sex": "Female",
            "particle_type": "X-ray",
            "dose_Gy": "1",
            "hr_post_exposure": "24.0",
            "physical_kl720_burden": "4.0",
        },
    ]


def test_create_and_verify_nasa_endpoint_report(tmp_path: Path) -> None:
    predictions = _write_predictions(
        tmp_path / "predictions.csv",
        _base_rows(),
    )

    report_path = create_nasa_endpoint_report(
        predictions_path=predictions,
        burden_column="physical_kl720_burden",
        output_dir=tmp_path / "aggregates",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"] == {
        "n_nuclei": 4,
        "n_samples": 2,
    }
    assert report["scientific_interpretation"][
        "biological_acceptance_evaluated"
    ] is False

    aggregate_path = Path(
        report["outputs"]["sample_aggregates"]["path"]
    )
    with aggregate_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert [row["sample_name"] for row in rows] == [
        "SAMPLE_A",
        "SAMPLE_B",
    ]
    assert [int(row["n_nuclei"]) for row in rows] == [2, 2]
    assert [float(row["mean_latent_burden"]) for row in rows] == [
        2.0,
        3.0,
    ]

    verification = verify_nasa_endpoint_report(report_path)
    assert verification["ok"] is True
    assert verification["fingerprint_ok"] is True
    assert verification["sample_aggregates_ok"] is True

    second = create_nasa_endpoint_report(
        predictions_path=predictions,
        burden_column="physical_kl720_burden",
        output_dir=tmp_path / "aggregates",
    )
    assert second == report_path


def test_mutated_aggregate_is_detected(tmp_path: Path) -> None:
    predictions = _write_predictions(
        tmp_path / "predictions.csv",
        _base_rows(),
    )
    report_path = create_nasa_endpoint_report(
        predictions_path=predictions,
        burden_column="physical_kl720_burden",
        output_dir=tmp_path / "aggregates",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    aggregate_path = Path(
        report["outputs"]["sample_aggregates"]["path"]
    )
    aggregate_path.write_text(
        aggregate_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    verification = verify_nasa_endpoint_report(report_path)
    assert verification["sample_aggregates_ok"] is False
    assert verification["ok"] is False


def test_duplicate_nucleus_identity_is_rejected(tmp_path: Path) -> None:
    rows = _base_rows()
    rows[1]["sample_id"] = rows[0]["sample_id"]

    predictions = _write_predictions(
        tmp_path / "predictions.csv",
        rows,
    )

    with pytest.raises(ControlPlaneError, match="Duplicate sample_id"):
        create_nasa_endpoint_report(
            predictions_path=predictions,
            burden_column="physical_kl720_burden",
            output_dir=tmp_path / "aggregates",
        )


def test_inconsistent_metadata_within_sample_is_rejected(
    tmp_path: Path,
) -> None:
    rows = _base_rows()
    rows[1]["particle_type"] = "X-ray"

    predictions = _write_predictions(
        tmp_path / "predictions.csv",
        rows,
    )

    with pytest.raises(ControlPlaneError, match="Inconsistent"):
        create_nasa_endpoint_report(
            predictions_path=predictions,
            burden_column="physical_kl720_burden",
            output_dir=tmp_path / "aggregates",
        )



def test_verify_target_recognizes_endpoint_report(tmp_path: Path) -> None:
    from radiation_edge_ai.execution import verify_target

    predictions = _write_predictions(
        tmp_path / "predictions.csv",
        _base_rows(),
    )
    report_path = create_nasa_endpoint_report(
        predictions_path=predictions,
        burden_column="physical_kl720_burden",
        output_dir=tmp_path / "aggregates",
    )

    report = verify_target(report_path)

    assert report["kind"] == "nasa_endpoint_report"
    assert report["n_nuclei"] == 4
    assert report["n_samples"] == 2
    assert report["ok"] is True


def test_cli_aggregate_and_verify_endpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from radiation_edge_ai.cli import main

    predictions = _write_predictions(
        tmp_path / "predictions.csv",
        _base_rows(),
    )
    output_dir = tmp_path / "aggregates"

    status = main(
        [
            "aggregate",
            "--assay",
            "nasa-53bp1-r1-v2",
            "--predictions",
            str(predictions),
            "--burden-column",
            "physical_kl720_burden",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert status == 0

    aggregate_stdout = capsys.readouterr().out.strip()
    report_path = Path(aggregate_stdout)
    assert report_path.is_file()

    status = main(["verify", str(report_path)])
    assert status == 0

    verify_stdout = capsys.readouterr().out
    assert "kind: nasa_endpoint_report" in verify_stdout
    assert "fingerprint: PASS" in verify_stdout
    assert "source predictions: PASS" in verify_stdout
    assert "sample aggregates: PASS" in verify_stdout
    assert "endpoint counts: 4 nuclei / 2 samples" in verify_stdout
    assert "verification: PASS" in verify_stdout
