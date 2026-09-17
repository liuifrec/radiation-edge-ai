from __future__ import annotations

import json
from pathlib import Path

import pytest

from radiation_edge_ai.control import (
    ControlPlaneError,
    create_run_plan,
    get_assay,
    list_assays,
    verify_run_plan,
)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fiber_metadata() -> dict[str, object]:
    return {
        "sample_id": "S1",
        "pixel_size_um": 0.26,
        "channel_semantics": "DNA fiber fluorescence",
    }


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_path = tmp_path / "input.bin"
    model_path = tmp_path / "model.bin"
    metadata_path = tmp_path / "metadata.json"
    input_path.write_bytes(b"image")
    model_path.write_bytes(b"model")
    _write_json(metadata_path, _fiber_metadata())
    return input_path, model_path, metadata_path


def test_assay_registry_is_two_assay_and_stable() -> None:
    assert [spec.assay_id for spec in list_assays()] == [
        "dnai-fiber-v3",
        "nasa-53bp1-r1-v2",
    ]
    assert "kl720" in get_assay("nasa-53bp1-r1-v2").approved_backends


def test_unknown_assay_fails() -> None:
    with pytest.raises(ControlPlaneError, match="Unknown assay"):
        get_assay("not-an-assay")


def test_plan_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    input_path, model_path, metadata_path = _artifacts(tmp_path)
    output = tmp_path / "runs"
    kwargs = {
        "assay_id": "dnai-fiber-v3",
        "backend": "kl720",
        "input_path": input_path,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "output_dir": output,
    }
    first = create_run_plan(**kwargs)
    second = create_run_plan(**kwargs)
    assert first == second
    plan = json.loads(first.read_text(encoding="utf-8"))
    assert plan["run_id"].startswith("r1-")
    assert plan["execution"]["performed"] is False
    assert plan["status"] == "planned"


def test_same_content_different_paths_has_same_run_id(tmp_path: Path) -> None:
    run_ids = []
    for name in ("a", "b"):
        root = tmp_path / name
        root.mkdir()
        input_path, model_path, metadata_path = _artifacts(root)
        plan_path = create_run_plan(
            assay_id="dnai-fiber-v3",
            backend="cpu-onnx",
            input_path=input_path,
            model_path=model_path,
            metadata_path=metadata_path,
            output_dir=root / "runs",
        )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        run_ids.append(plan["run_id"])
    assert run_ids[0] == run_ids[1]


def test_missing_required_metadata_fails(tmp_path: Path) -> None:
    input_path, model_path, metadata_path = _artifacts(tmp_path)
    _write_json(metadata_path, {"sample_id": "S1"})
    with pytest.raises(ControlPlaneError, match="missing required fields"):
        create_run_plan(
            assay_id="dnai-fiber-v3",
            backend="kl720",
            input_path=input_path,
            model_path=model_path,
            metadata_path=metadata_path,
            output_dir=tmp_path / "runs",
        )


def test_unapproved_backend_fails(tmp_path: Path) -> None:
    input_path, model_path, metadata_path = _artifacts(tmp_path)
    with pytest.raises(ControlPlaneError, match="not approved"):
        create_run_plan(
            assay_id="dnai-fiber-v3",
            backend="cuda",
            input_path=input_path,
            model_path=model_path,
            metadata_path=metadata_path,
            output_dir=tmp_path / "runs",
        )


def test_verify_detects_artifact_mutation(tmp_path: Path) -> None:
    input_path, model_path, metadata_path = _artifacts(tmp_path)
    plan_path = create_run_plan(
        assay_id="dnai-fiber-v3",
        backend="cpu-onnx",
        input_path=input_path,
        model_path=model_path,
        metadata_path=metadata_path,
        output_dir=tmp_path / "runs",
    )
    assert verify_run_plan(plan_path)["ok"] is True
    input_path.write_bytes(b"changed")
    report = verify_run_plan(plan_path)
    assert report["fingerprint_ok"] is True
    assert report["artifacts_ok"] is False
    assert report["ok"] is False


def test_verify_without_artifacts_checks_plan_only(tmp_path: Path) -> None:
    input_path, model_path, metadata_path = _artifacts(tmp_path)
    plan_path = create_run_plan(
        assay_id="dnai-fiber-v3",
        backend="cpu-onnx",
        input_path=input_path,
        model_path=model_path,
        metadata_path=metadata_path,
        output_dir=tmp_path / "runs",
    )
    input_path.write_bytes(b"changed")
    report = verify_run_plan(plan_path, check_artifacts=False)
    assert report["fingerprint_ok"] is True
    assert report["artifact_check_performed"] is False
    assert report["ok"] is True


def test_tampered_embedded_metadata_breaks_fingerprint(tmp_path: Path) -> None:
    input_path, model_path, metadata_path = _artifacts(tmp_path)
    plan_path = create_run_plan(
        assay_id="dnai-fiber-v3",
        backend="cpu-onnx",
        input_path=input_path,
        model_path=model_path,
        metadata_path=metadata_path,
        output_dir=tmp_path / "runs",
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["metadata"]["pixel_size_um"] = 999
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    report = verify_run_plan(plan_path, check_artifacts=False)
    assert report["fingerprint_ok"] is False
    assert report["ok"] is False
