from __future__ import annotations

from pathlib import Path

from radiation_edge_ai.dna_fiber import (
    transaction,
)


def test_worker_source_provenance_accepts_crlf_checkout(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "_field_worker.py"
    )

    source.write_bytes(
        b"line1\nline2\n"
    )

    record = (
        transaction
        ._worker_source_artifact_record(
            source
        )
    )

    assert (
        transaction._artifact_bytes_ok(
            record
        )
        is True
    )

    source.write_bytes(
        b"line1\r\nline2\r\n"
    )

    assert (
        transaction._artifact_bytes_ok(
            record
        )
        is False
    )

    assert (
        transaction._worker_source_bytes_ok(
            record
        )
        is True
    )


def test_worker_source_provenance_rejects_content_change(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "_field_worker.py"
    )

    source.write_bytes(
        b"line1\nline2\n"
    )

    record = (
        transaction
        ._worker_source_artifact_record(
            source
        )
    )

    source.write_bytes(
        b"line1\r\nCHANGED\r\n"
    )

    assert (
        transaction._worker_source_bytes_ok(
            record
        )
        is False
    )
