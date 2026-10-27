"""Offline checks for fail-closed DNAi v0.15 output-only precision candidates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def load_quantizer(monkeypatch):
    """Kneron/ONNX are Docker-only: import with *empty* stubs, never simulate PTQ."""
    monkeypatch.setitem(sys.modules, "ktc", types.ModuleType("ktc"))
    monkeypatch.setitem(sys.modules, "onnx", types.ModuleType("onnx"))
    path = (
        Path(__file__).resolve().parents[1] / "src" / "radiation_edge_ai"
        / "dna_fiber" / "kneron_quantize_int8_512.py"
    )
    spec = importlib.util.spec_from_file_location("testable_v015_quantizer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args_base(**overrides):
    values = {
        "threads": 4, "datapath_range_method": "percentage",
        "percentage": 0.999, "percentage_16b": 0.999999,
        "percentile": 0.001, "outlier_factor": 1.0, "optimize": 0,
        "platform": "720", "model_id": 32769, "model_version": "8b28",
        "model_out_bitwidth_mode": None,
        "candidate_toolchain_image_id": None,
        "candidate_manifest_sha256": None,
        "candidate_preflight_only": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def candidate(tmp_path, mod, monkeypatch):
    onnx = tmp_path / "optimized.onnx"
    manifest = tmp_path / "calibration" / "manifest.json"
    manifest.parent.mkdir()
    onnx.write_bytes(b"synthetic optimized ONNX source")
    manifest.write_bytes(b'{"synthetic":true}')
    monkeypatch.setattr(mod, "V015_ONNX_SHA256", hashlib.sha256(onnx.read_bytes()).hexdigest())
    options = args_base(
        model_out_bitwidth_mode="int16", percentage=1.0,
        percentage_16b=1.0,
        candidate_toolchain_image_id=mod.V015_TOOLCHAIN_IMAGE_ID,
        candidate_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    output = tmp_path / "kneron" / mod.V015_OUTPUT_DIRECTORY
    return options, onnx, manifest, output


def test_legacy_analysis_call_stays_identical(monkeypatch, tmp_path):
    mod = load_quantizer(monkeypatch)
    args = args_base()
    expected = {
        "threads": 4, "datapath_range_method": "percentage",
        "percentage": 0.999, "percentage_16b": 0.999999,
        "percentile": 0.001, "outlier_factor": 1.0, "optimize": 0,
    }
    assert mod.build_analysis_options(args) == expected
    assert list(mod.build_analysis_options(args)) == list(expected)
    assert "model_out_bitwidth_mode" not in mod.build_analysis_options(args)
    assert mod.validate_int16_candidate(
        args, tmp_path / "unused.onnx", tmp_path / "unused.json", tmp_path / "unused"
    ) is None


def test_int16_candidate_changes_only_output_bitwidth(monkeypatch, tmp_path):
    mod = load_quantizer(monkeypatch)
    args, onnx, manifest, output = candidate(tmp_path, mod, monkeypatch)
    base = args_base(percentage=1.0, percentage_16b=1.0)
    expected = dict(mod.build_analysis_options(base))
    expected["model_out_bitwidth_mode"] = "int16"
    assert mod.build_analysis_options(args) == expected
    report = mod.validate_int16_candidate(args, onnx, manifest, output)
    assert report["experimental_not_deployed"] is True
    assert report["model_out_bitwidth_mode"] == "int16"
    assert not output.exists()  # No preflight side effects.


def test_int16_rejects_bias_tuning(monkeypatch, tmp_path):
    mod = load_quantizer(monkeypatch)
    args, onnx, manifest, output = candidate(tmp_path, mod, monkeypatch)
    args.optimize = 1
    with pytest.raises(SystemExit, match="change ONLY output bitwidth"):
        mod.validate_int16_candidate(args, onnx, manifest, output)
    assert not output.exists()


def test_int16_rejects_missing_or_wrong_model_and_manifest_hash(monkeypatch, tmp_path):
    mod = load_quantizer(monkeypatch)
    args, onnx, manifest, output = candidate(tmp_path, mod, monkeypatch)
    args.candidate_manifest_sha256 = "0" * 64
    with pytest.raises(SystemExit, match="manifest SHA256 mismatch"):
        mod.validate_int16_candidate(args, onnx, manifest, output)
    args.candidate_manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    onnx.write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="ONNX SHA256 mismatch"):
        mod.validate_int16_candidate(args, onnx, manifest, output)
    assert not output.exists()


def test_int16_rejects_reusing_candidate_or_historical_directory(monkeypatch, tmp_path):
    mod = load_quantizer(monkeypatch)
    args, onnx, manifest, output = candidate(tmp_path, mod, monkeypatch)
    output.mkdir(parents=True)
    with pytest.raises(SystemExit, match="must not exist"):
        mod.validate_int16_candidate(args, onnx, manifest, output)
    historic_child = (
        tmp_path / "kneron" / "int8_512_v3_pct100" / mod.V015_OUTPUT_DIRECTORY
    )
    with pytest.raises(SystemExit, match="new isolated"):
        mod.validate_int16_candidate(args, onnx, manifest, historic_child)


def test_int16_rejects_unverified_image_identity(monkeypatch, tmp_path):
    mod = load_quantizer(monkeypatch)
    args, onnx, manifest, output = candidate(tmp_path, mod, monkeypatch)
    args.candidate_toolchain_image_id = "sha256:" + "0" * 64
    with pytest.raises(SystemExit, match="verified immutable"):
        mod.validate_int16_candidate(args, onnx, manifest, output)
    args.candidate_toolchain_image_id = None
    args.model_out_bitwidth_mode = None
    with pytest.raises(SystemExit, match="Candidate-only"):
        mod.validate_int16_candidate(args, onnx, manifest, output)


def test_int16_calibration_rejects_legacy_panel_and_duplicates(monkeypatch):
    mod = load_quantizer(monkeypatch)
    records = [
        {"tensor": "tensors/c%03d.npy" % i, "source_sha256": ("%064x" % i)}
        for i in range(192)
    ]
    mod.validate_candidate_calibration(
        {"n_tensors": 192, "validation_panel_used_for_calibration": False}, records
    )
    with pytest.raises(SystemExit, match="exclude validation panel"):
        mod.validate_candidate_calibration(
            {"n_tensors": 192, "validation_panel_used_for_calibration": True}, records
        )
    with pytest.raises(SystemExit, match="Duplicate"):
        mod.validate_candidate_calibration(
            {"n_tensors": 192, "validation_panel_used_for_calibration": False},
            records[:-1] + [records[0]],
        )
    with pytest.raises(SystemExit, match="192 unique"):
        new_records = [dict(r) for r in records]
        new_records[-1]["source_sha256"] = new_records[0]["source_sha256"]
        mod.validate_candidate_calibration(
            {"n_tensors": 192, "validation_panel_used_for_calibration": False},
            new_records,
        )
