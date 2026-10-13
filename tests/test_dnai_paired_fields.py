"""Read-only paired-field DNAi characterization, with synthetic frozen PNGs."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from radiation_edge_ai import cli, dnai_paired_fields as paired
from radiation_edge_ai.control import ControlPlaneError


def _artifact(path: Path) -> dict[str, object]:
    contents = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(contents).hexdigest(), "size_bytes": len(contents)}


def _make_field(path: Path, *, schema: int, segmentation: np.ndarray, input_hash: str = "a" * 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    seg_path = path.parent / "segmentation.png"
    Image.fromarray(segmentation).save(seg_path)
    probs = np.zeros((3, 1024, 1024), dtype=np.float32)
    for cls in range(3):
        probs[cls] = (segmentation == cls).astype(np.float32)
    prob_path = path.parent / "probabilities.npz"
    np.savez_compressed(prob_path, probabilities=probs)
    identity = {
        "sample_id": "1__tile_4",
        "key": "1__tile_4",
        "image_index": 0,
        "dnai_commit": "f" * 40,
        "validation_manifest_sha256": "b" * 64,
        "deployment_policy": {"tile": [1, 3, 512, 512], "overlap": 0.5, "stride": 256, "blend": "gaussian", "sigma_scale": 0.125, "pixel_size_um": 0.26},
        "windows": [
            {"window_index": i, "x": (i % 3) * 256, "y": (i // 3) * 256,
             "tensor_sha256": input_hash, "tensor_size_bytes": 2048}
            for i in range(9)
        ],
        "model_sha256": "f" * 64 if schema == 1 else "e" * 64,
    }
    if schema == 2:
        identity["backend"] = "kl720"
    field = {
        "schema_version": schema,
        "record_type": "dnai_fiber_field_record",
        "field_id": f"df{schema}-synthetic",
        "identity": identity,
        "counts": {"n_windows": 9, "n_fibers_all": 2, "n_fibers_valid": 1 if schema == 1 else 2},
        "measurements": {"mean_valid_ratio": 1.0 if schema == 1 else 1.2,
                         "median_valid_ratio": 1.0 if schema == 1 else 1.2,
                         "mean_valid_length_um": 12.0 if schema == 1 else 10.0,
                         "total_valid_length_um": 12.0 if schema == 1 else 20.0},
        "artifacts": {"segmentation": _artifact(seg_path), "stitched_probabilities": _artifact(prob_path)},
    }
    path.write_text(json.dumps(field), encoding="utf-8")
    return path


def _pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    left = np.zeros((1024, 1024), dtype=np.uint8)
    left[0:3, 0:3] = 1
    left[10:12, 10:12] = 2
    right = left.copy()
    right[10:12, 10:12] = 0
    right[20:22, 20:22] = 2
    fp = _make_field(tmp_path / "fp512" / "field.json", schema=1, segmentation=left)
    kl = _make_field(tmp_path / "kl720" / "field.json", schema=2, segmentation=right)
    calls = []

    def fake_verify(path: Path, *, check_artifacts: bool):
        assert check_artifacts is True
        calls.append(path)
        return {"kind": "dnai_fiber_field_record", "ok": True}

    monkeypatch.setattr(paired, "verify_target", fake_verify)
    return fp, kl, calls


def test_connected_components_diagonal_and_separation() -> None:
    mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
    labels, count = paired._label_components8(mask)
    assert count == 1
    assert labels[0, 0] == labels[1, 1] == labels[2, 2] == 1
    mask[1, 1] = 0
    _, count = paired._label_components8(mask)
    assert count == 2


def test_component_matching_uses_spatial_mask_iou_not_fiber_ids() -> None:
    a = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 2, 2]], dtype=np.uint8)
    b = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [3, 0, 0, 0]], dtype=np.uint8)
    metrics = paired._component_proxy_metrics(a, b)
    assert metrics["n_fp512_components"] == 2
    assert metrics["n_kl720_components"] == 2
    assert metrics["n_matched_proxies"] == 1
    assert metrics["f1"] == 0.5


def test_realistic_synthetic_pair_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fp, kl, calls = _pair(tmp_path, monkeypatch)
    original = (fp.read_bytes(), kl.read_bytes())
    report = paired.compare_dnai_fields(fp, kl)
    assert calls == [fp.resolve(), kl.resolve()]
    assert report["report_type"] == "dnai_fp512_kl720_paired_field_audit"
    assert report["shared_input"]["n_windows"] == 9
    assert report["segmentation_metrics"]["changed_pixels"] == 8
    assert report["segmentation_metrics"]["confusion_reference_rows_kl720_columns"][2][0] == 4
    assert report["component_proxy_metrics"]["n_matched_proxies"] == 1
    assert report["component_proxy_metrics"]["n_fp512_components"] == 2
    assert report["component_proxy_metrics"]["n_kl720_components"] == 2
    assert report["field_measurements"]["delta_kl720_minus_fp512"]["total_valid_length_um"] == 8.0
    assert report["scientific_scope"]["biological_fidelity_evaluated"] is False
    assert (fp.read_bytes(), kl.read_bytes()) == original


def test_mismatched_frozen_input_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fp, kl, _ = _pair(tmp_path, monkeypatch)
    value = json.loads(kl.read_text())
    value["identity"]["windows"][0]["tensor_sha256"] = "d" * 64
    kl.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ControlPlaneError, match="identical frozen input"):
        paired.compare_dnai_fields(fp, kl)


def test_tampered_image_fails_byte_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fp, kl, _ = _pair(tmp_path, monkeypatch)
    artifact = json.loads(fp.read_text())["artifacts"]["segmentation"]["path"]
    with open(artifact, "ab") as out:
        out.write(b"TAMPER")
    with pytest.raises(ControlPlaneError, match="byte verification"):
        paired.compare_dnai_fields(fp, kl)


def test_rejected_if_underlying_verifier_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fp, kl, _ = _pair(tmp_path, monkeypatch)
    monkeypatch.setattr(paired, "verify_target", lambda *_args, **_kwargs: {"kind": "dnai_fiber_field_record", "ok": False})
    with pytest.raises(ControlPlaneError, match="full verification"):
        paired.compare_dnai_fields(fp, kl)


def test_cli_pair_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fp, kl, _ = _pair(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "compare_dnai_fields", paired.compare_dnai_fields)
    assert cli.main(["compare-dnai-fields", "--fp512", str(fp), "--kl720", str(kl), "--json"]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["report_type"] == "dnai_fp512_kl720_paired_field_audit"
    assert obj["segmentation_metrics"]["changed_pixels"] == 8


def test_same_record_and_missing_file_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fp, _kl, _ = _pair(tmp_path, monkeypatch)
    with pytest.raises(ControlPlaneError, match="distinct"):
        paired.compare_dnai_fields(fp, fp)
    with pytest.raises(ControlPlaneError, match="Cannot read"):
        paired.compare_dnai_fields(tmp_path / "missing.json", fp)


def test_field_mutated_during_verification_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fp, kl, _ = _pair(tmp_path, monkeypatch)

    def tampering_verify(target: Path, *, check_artifacts: bool):
        assert check_artifacts is True
        target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return {"kind": "dnai_fiber_field_record", "ok": True}

    monkeypatch.setattr(paired, "verify_target", tampering_verify)
    with pytest.raises(ControlPlaneError, match="changed during comparison"):
        paired.compare_dnai_fields(fp, kl)
