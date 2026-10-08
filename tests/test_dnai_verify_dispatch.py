from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from radiation_edge_ai.control import (
    ControlPlaneError,
)
from radiation_edge_ai.dna_fiber import (
    field_record,
    field_record_v2,
    transaction,
    transaction_v2,
)
from radiation_edge_ai.execution import (
    verify_target,
)


def _write_record(
    path: Path,
    *,
    record_type: str,
    schema_version: object,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "record_type": record_type,
                "schema_version": schema_version,
            }
        ),
        encoding="utf-8",
    )

    return path


@pytest.mark.parametrize(
    (
        "record_type",
        "schema_version",
        "module",
        "attribute",
        "marker",
    ),
    [
        (
            "dnai_fiber_field_record",
            1,
            field_record,
            "verify_dnai_fiber_field_record",
            "field-v1",
        ),
        (
            "dnai_fiber_field_record",
            2,
            field_record_v2,
            "verify_dnai_fiber_physical_field_record",
            "field-v2",
        ),
        (
            "dnai_fiber_measurement_transaction",
            1,
            transaction,
            "verify_dnai_fiber_measurement_transaction",
            "transaction-v1",
        ),
        (
            "dnai_fiber_measurement_transaction",
            2,
            transaction_v2,
            "verify_dnai_fiber_physical_measurement_transaction",
            "transaction-v2",
        ),
    ],
)
def test_verify_target_dispatches_dnai_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_type: str,
    schema_version: int,
    module: Any,
    attribute: str,
    marker: str,
) -> None:
    target = _write_record(
        tmp_path / "record.json",
        record_type=record_type,
        schema_version=schema_version,
    )

    calls: dict[str, object] = {}

    def fake_verify(
        path: Path,
        *,
        check_artifacts: bool,
    ) -> dict[str, object]:
        calls["path"] = path
        calls["check_artifacts"] = (
            check_artifacts
        )

        return {
            "kind": marker,
            "ok": True,
        }

    monkeypatch.setattr(
        module,
        attribute,
        fake_verify,
    )

    result = verify_target(
        target,
        check_artifacts=False,
    )

    assert result == {
        "kind": marker,
        "ok": True,
    }

    assert calls == {
        "path": target.resolve(),
        "check_artifacts": False,
    }


@pytest.mark.parametrize(
    "record_type",
    [
        "dnai_fiber_field_record",
        "dnai_fiber_measurement_transaction",
    ],
)
def test_verify_target_rejects_unknown_dnai_schema(
    tmp_path: Path,
    record_type: str,
) -> None:
    target = _write_record(
        tmp_path / "record.json",
        record_type=record_type,
        schema_version=3,
    )

    with pytest.raises(
        ControlPlaneError,
        match="schema_version",
    ):
        verify_target(
            target,
            check_artifacts=False,
        )


@pytest.mark.parametrize(
    "record_type",
    [
        "dnai_fiber_field_record",
        "dnai_fiber_measurement_transaction",
    ],
)
def test_verify_target_rejects_boolean_dnai_schema(
    tmp_path: Path,
    record_type: str,
) -> None:
    target = _write_record(
        tmp_path / "record.json",
        record_type=record_type,
        schema_version=True,
    )

    with pytest.raises(
        ControlPlaneError,
        match="schema_version",
    ):
        verify_target(
            target,
            check_artifacts=False,
        )
