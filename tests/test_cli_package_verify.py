from __future__ import annotations

from pathlib import Path

import pytest

from radiation_edge_ai import cli


def test_verify_dnai_package_human_output_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = (
        tmp_path
        / "package_manifest.json"
    )

    report = {
        "kind": "assay_result_package",
        "package_id": "drp1-test",
        "fingerprint_ok": True,
        "package_id_ok": True,
        "files_ok": True,
        "derivation_ok": True,
        "scientific_scope_ok": True,
        "counts_ok": True,
        "csv_binding_ok": True,
        "n_windows": 9,
        "n_fibers_valid": 14,
        "ok": True,
        "assay_id": "dnai-fiber-v3",
        "runtime_adapter": (
            "dnai-fiber-measurement-package-v1"
        ),
    }

    monkeypatch.setattr(
        cli,
        "verify_target",
        lambda *_args, **_kwargs: report,
    )

    exit_code = cli.main(
        [
            "verify",
            str(target),
        ]
    )

    output = (
        capsys.readouterr().out
    )

    assert exit_code == 0

    assert output.count(
        "kind: assay_result_package"
    ) == 1

    assert output.count(
        "package_id: drp1-test"
    ) == 1

    assert output.count(
        "fingerprint: PASS"
    ) == 1

    assert output.count(
        "package identity: PASS"
    ) == 1

    assert output.count(
        "files: PASS"
    ) == 1

    assert output.count(
        "derivation: PASS"
    ) == 1

    assert output.count(
        "scientific scope: PASS"
    ) == 1

    assert output.count(
        "counts: PASS"
    ) == 1

    assert output.count(
        "fiber table binding: PASS"
    ) == 1

    assert output.count(
        "result counts: 9 windows / 14 valid fibers"
    ) == 1

    assert output.count(
        "verification: PASS"
    ) == 1


def test_verify_nasa_package_human_output_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = (
        tmp_path
        / "package_manifest.json"
    )

    report = {
        "kind": "assay_result_package",
        "package_id": "rp1-test",
        "fingerprint_ok": True,
        "package_id_ok": True,
        "files_ok": True,
        "derivation_ok": True,
        "scientific_scope_ok": True,
        "counts_ok": True,
        "n_nuclei": 2058,
        "n_samples": 22,
        "ok": True,
        "assay_id": "nasa-53bp1-r1-v2",
        "runtime_adapter": (
            "nasa-sample-aggregate-package-v1"
        ),
    }

    monkeypatch.setattr(
        cli,
        "verify_target",
        lambda *_args, **_kwargs: report,
    )

    exit_code = cli.main(
        [
            "verify",
            str(target),
        ]
    )

    output = (
        capsys.readouterr().out
    )

    assert exit_code == 0

    assert output.count(
        "kind: assay_result_package"
    ) == 1

    assert output.count(
        "package_id: rp1-test"
    ) == 1

    assert output.count(
        "fingerprint: PASS"
    ) == 1

    assert output.count(
        "package identity: PASS"
    ) == 1

    assert output.count(
        "files: PASS"
    ) == 1

    assert output.count(
        "result counts: 2058 nuclei / 22 samples"
    ) == 1

    assert (
        "fiber table binding:"
        not in output
    )

    assert output.count(
        "verification: PASS"
    ) == 1
