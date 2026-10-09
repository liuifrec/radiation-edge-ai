"""Offline v0.12 multi-artifact verification tests; no hardware access."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from radiation_edge_ai import cli, verification_collection
from radiation_edge_ai.control import ControlPlaneError, create_run_plan, sha256_file


def _make_plan(tmp_path: Path) -> tuple[Path, Path]:
    input_path = tmp_path / "image.bin"
    model_path = tmp_path / "model.bin"
    metadata_path = tmp_path / "metadata.json"
    input_path.write_bytes(b"frozen test input")
    model_path.write_bytes(b"frozen dummy model")
    metadata_path.write_text(
        json.dumps(
            {
                "sample_id": "test-1",
                "pixel_size_um": 0.26,
                "channel_semantics": "DNA fiber fluorescence",
            }
        ),
        encoding="utf-8",
    )
    plan = create_run_plan(
        assay_id="dnai-fiber-v3",
        backend="cpu-onnx",
        input_path=input_path,
        model_path=model_path,
        metadata_path=metadata_path,
        output_dir=tmp_path / "runs",
    )
    return plan, input_path


def test_collection_verifies_real_run_plan_without_inference(tmp_path: Path) -> None:
    plan, _input_path = _make_plan(tmp_path)
    result = verification_collection.verify_collection([plan])
    assert result["report_type"] == "verification_collection_report"
    assert result["artifact_check_performed"] is True
    assert (result["n_targets"], result["n_pass"], result["n_fail"]) == (1, 1, 0)
    assert result["ok"] is True
    assert result["items"][0]["kind"] == "run_plan"
    assert result["items"][0]["source_sha256"] == sha256_file(plan)
    assert result["items"][0]["verification"]["ok"] is True


def test_collection_detects_tampered_source_artifact(tmp_path: Path) -> None:
    plan, image = _make_plan(tmp_path)
    before = sha256_file(plan)
    image.write_bytes(b"tampered image")
    result = verification_collection.verify_collection([plan])
    assert result["ok"] is False
    assert (result["n_pass"], result["n_fail"]) == (0, 1)
    assert result["items"][0]["kind"] == "run_plan"
    assert result["items"][0]["source_sha256"] == before
    assert result["items"][0]["verification"]["artifacts_ok"] is False


def test_collection_continues_after_unreadable_json(tmp_path: Path) -> None:
    plan, _ = _make_plan(tmp_path)
    unreadable = tmp_path / "bad.json"
    unreadable.write_text("{bad JSON", encoding="utf-8")
    result = verification_collection.verify_collection([unreadable, plan])
    assert result["ok"] is False
    assert (result["n_targets"], result["n_pass"], result["n_fail"]) == (2, 1, 1)
    assert result["items"][0]["kind"] == "unverified"
    assert result["items"][0]["ok"] is False
    assert "error" in result["items"][0]
    assert result["items"][1]["kind"] == "run_plan"
    assert result["items"][1]["ok"] is True


def test_collection_rejects_duplicate_normalized_target(tmp_path: Path) -> None:
    plan, _ = _make_plan(tmp_path)
    alias = plan.parent / "." / plan.name
    with pytest.raises(ControlPlaneError, match="Duplicate verification target"):
        verification_collection.verify_collection([plan, alias])


def test_collection_refuses_missing_or_nonboolean_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _ = _make_plan(tmp_path)
    monkeypatch.setattr(
        verification_collection,
        "verify_target",
        lambda *_args, **_kwargs: {"kind": "run_plan", "ok": 1},
    )
    result = verification_collection.verify_collection([plan])
    assert result["ok"] is False
    assert "non-boolean" in result["items"][0]["error"]


def test_collection_rejects_record_modified_mid_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _ = _make_plan(tmp_path)

    def changing_verify(target: Path, *, check_artifacts: bool) -> dict[str, object]:
        assert check_artifacts is True
        target.write_text('{},', encoding='utf-8')
        return {"kind": "run_plan", "ok": True}

    monkeypatch.setattr(verification_collection, "verify_target", changing_verify)
    result = verification_collection.verify_collection([plan])
    assert result["ok"] is False
    assert "changed during verification" in result["items"][0]["error"]


def test_cli_verify_many_json_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    plan, input_path = _make_plan(tmp_path)
    assert cli.main(["verify-many", "--json", str(plan)]) == 0
    clean = json.loads(capsys.readouterr().out)
    assert clean["n_pass"] == 1
    assert clean["n_fail"] == 0
    assert clean["items"][0]["kind"] == "run_plan"

    input_path.write_bytes(b"tampered")
    assert cli.main(["verify-many", "--json", str(plan)]) == 3
    broken = json.loads(capsys.readouterr().out)
    assert broken["ok"] is False
    assert broken["n_fail"] == 1


def test_cli_verify_many_human_and_duplicate_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    plan, _ = _make_plan(tmp_path)
    assert cli.main(["verify-many", str(plan)]) == 0
    text = capsys.readouterr().out
    assert "PASS: run_plan" in text
    assert "verification: PASS (1/1)" in text

    assert cli.main(["verify-many", str(plan), str(plan)]) == 2
    assert "Duplicate verification target" in capsys.readouterr().err
