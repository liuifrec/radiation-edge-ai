from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from radiation_edge_ai.control import ControlPlaneError
from radiation_edge_ai.measurement import (
    create_measurement_plan,
    verify_measurement_plan,
)

FIELDS = [
    "sample_id",
    "sample_name",
    "source_name",
    "particle_type",
    "dose_Gy",
    "hr_post_exposure",
    "physical_kl720_burden",
]


def _write_predictions(path: Path) -> Path:
    rows = [
        {
            "sample_id": "N1",
            "sample_name": "S1",
            "source_name": "SRC",
            "particle_type": "Fe",
            "dose_Gy": "0.0",
            "hr_post_exposure": "4",
            "physical_kl720_burden": "1.0",
        },
        {
            "sample_id": "N2",
            "sample_name": "S1",
            "source_name": "SRC",
            "particle_type": "Fe",
            "dose_Gy": "0",
            "hr_post_exposure": "4.0",
            "physical_kl720_burden": "2.0",
        },
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def test_create_verify_and_reuse_measurement_plan(
    tmp_path: Path,
) -> None:
    predictions = _write_predictions(
        tmp_path / "predictions.csv"
    )

    plan_path = create_measurement_plan(
        predictions_path=predictions,
        burden_column="physical_kl720_burden",
        output_dir=tmp_path / "plans",
    )

    verification = verify_measurement_plan(
        plan_path
    )

    assert verification["ok"] is True
    assert verification["fingerprint_ok"] is True
    assert verification["source_predictions_ok"] is True

    second = create_measurement_plan(
        predictions_path=predictions,
        burden_column="physical_kl720_burden",
        output_dir=tmp_path / "plans",
    )

    assert second == plan_path


def test_measurement_identity_is_path_independent(
    tmp_path: Path,
) -> None:
    first = _write_predictions(
        tmp_path / "first.csv"
    )
    second = tmp_path / "second.csv"
    shutil.copyfile(first, second)

    first_plan = create_measurement_plan(
        predictions_path=first,
        burden_column="physical_kl720_burden",
        output_dir=tmp_path / "plans-a",
    )
    second_plan = create_measurement_plan(
        predictions_path=second,
        burden_column="physical_kl720_burden",
        output_dir=tmp_path / "plans-b",
    )

    assert first_plan.parent.name == second_plan.parent.name


def test_source_mutation_is_detected(
    tmp_path: Path,
) -> None:
    predictions = _write_predictions(
        tmp_path / "predictions.csv"
    )

    plan_path = create_measurement_plan(
        predictions_path=predictions,
        burden_column="physical_kl720_burden",
        output_dir=tmp_path / "plans",
    )

    with predictions.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write("\n")

    verification = verify_measurement_plan(
        plan_path
    )

    assert verification["source_predictions_ok"] is False
    assert verification["ok"] is False


def test_missing_burden_column_is_rejected(
    tmp_path: Path,
) -> None:
    predictions = _write_predictions(
        tmp_path / "predictions.csv"
    )

    with pytest.raises(
        ControlPlaneError,
        match="missing required columns",
    ):
        create_measurement_plan(
            predictions_path=predictions,
            burden_column="does_not_exist",
            output_dir=tmp_path / "plans",
        )



def test_verify_target_recognizes_measurement_plan(
    tmp_path: Path,
) -> None:
    from radiation_edge_ai.execution import verify_target

    predictions = _write_predictions(
        tmp_path / "predictions.csv"
    )

    plan_path = create_measurement_plan(
        predictions_path=predictions,
        burden_column="physical_kl720_burden",
        output_dir=tmp_path / "plans",
    )

    report = verify_target(plan_path)

    assert report["kind"] == "measurement_plan"
    assert report["fingerprint_ok"] is True
    assert report["source_predictions_ok"] is True
    assert report["ok"] is True


def test_cli_measure_plan_and_verify(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from radiation_edge_ai.cli import main

    predictions = _write_predictions(
        tmp_path / "predictions.csv"
    )
    output_dir = tmp_path / "plans"

    status = main(
        [
            "measure-plan",
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

    plan_stdout = capsys.readouterr().out.strip()
    plan_path = Path(plan_stdout)
    assert plan_path.is_file()

    status = main(["verify", str(plan_path)])
    assert status == 0

    verify_stdout = capsys.readouterr().out
    assert "kind: measurement_plan" in verify_stdout
    assert "measurement_id: m1-" in verify_stdout
    assert "fingerprint: PASS" in verify_stdout
    assert "source predictions: PASS" in verify_stdout
    assert "verification: PASS" in verify_stdout



def test_execute_measurement_plan_and_verify_record(
    tmp_path: Path,
) -> None:
    from radiation_edge_ai.measurement import (
        execute_measurement_plan,
        verify_measurement_record,
    )

    predictions = _write_predictions(
        tmp_path / "predictions.csv"
    )

    plan_path = create_measurement_plan(
        predictions_path=predictions,
        burden_column="physical_kl720_burden",
        output_dir=tmp_path / "measurements",
    )

    record_path = execute_measurement_plan(plan_path)
    report = verify_measurement_record(record_path)

    assert report["ok"] is True
    assert report["plan_ok"] is True
    assert report["endpoint_report_ok"] is True
    assert report["sample_aggregates_ok"] is True
    assert report["cross_artifact_bindings_ok"] is True
    assert report["n_nuclei"] == 2
    assert report["n_samples"] == 1


def test_measurement_execution_is_idempotent(
    tmp_path: Path,
) -> None:
    from radiation_edge_ai.measurement import (
        execute_measurement_plan,
    )

    predictions = _write_predictions(
        tmp_path / "predictions.csv"
    )

    plan_path = create_measurement_plan(
        predictions_path=predictions,
        burden_column="physical_kl720_burden",
        output_dir=tmp_path / "measurements",
    )

    first = execute_measurement_plan(plan_path)
    second = execute_measurement_plan(plan_path)

    assert second == first


def test_verify_target_recognizes_measurement_record(
    tmp_path: Path,
) -> None:
    from radiation_edge_ai.execution import verify_target
    from radiation_edge_ai.measurement import (
        execute_measurement_plan,
    )

    predictions = _write_predictions(
        tmp_path / "predictions.csv"
    )

    plan_path = create_measurement_plan(
        predictions_path=predictions,
        burden_column="physical_kl720_burden",
        output_dir=tmp_path / "measurements",
    )

    record_path = execute_measurement_plan(plan_path)
    report = verify_target(record_path)

    assert report["kind"] == "measurement_record"
    assert report["ok"] is True


def test_cli_measure_and_verify_record(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from radiation_edge_ai.cli import main

    predictions = _write_predictions(
        tmp_path / "predictions.csv"
    )

    plan_path = create_measurement_plan(
        predictions_path=predictions,
        burden_column="physical_kl720_burden",
        output_dir=tmp_path / "measurements",
    )

    status = main(["measure", str(plan_path)])
    assert status == 0

    record_path = Path(
        capsys.readouterr().out.strip()
    )
    assert record_path.is_file()

    status = main(["verify", str(record_path)])
    assert status == 0

    output = capsys.readouterr().out

    assert "kind: measurement_record" in output
    assert "measurement_id: m1-" in output
    assert "fingerprint: PASS" in output
    assert "measurement plan: PASS" in output
    assert "endpoint report: PASS" in output
    assert "sample aggregates: PASS" in output
    assert "cross-artifact bindings: PASS" in output
    assert "measurement counts: 2 nuclei / 1 samples" in output
    assert "verification: PASS" in output
