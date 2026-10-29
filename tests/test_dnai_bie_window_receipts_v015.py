"""Synthetic, hardware-free gates for v0.15 three-field BIE receipts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def load_runner(monkeypatch):
    ktc = types.ModuleType("ktc")
    monkeypatch.setitem(sys.modules, "ktc", ktc)
    path = (
        Path(__file__).resolve().parents[1]
        / "src/radiation_edge_ai/dna_fiber/kneron_run_bie_validation_512.py"
    )
    spec = importlib.util.spec_from_file_location("v015_bie_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, ktc


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_development(tmp_path, runner, monkeypatch, populate_tensors=False):
    model_root = tmp_path / "model"
    bie_root = model_root / "int8_512_v5_int16out"
    bie_root.mkdir(parents=True)
    onnx = model_root / "optimized.onnx"
    bie = bie_root / "unet_mobileone_s1_512x512_int16out.bie"
    onnx.write_bytes(b"synthetic onnx source")
    bie.write_bytes(b"synthetic int16 BIE")
    monkeypatch.setattr(runner, "V015_ONNX_SHA256", sha(onnx))
    monkeypatch.setattr(runner, "V015_BIE_SHA256", sha(bie))
    monkeypatch.setattr(runner, "EXPECTED_INPUT_SHAPE", (1, 3, 4, 4))
    monkeypatch.setattr(runner, "EXPECTED_OUTPUT_SHAPE", (1, 3, 4, 4))
    root = tmp_path / "validation"
    root.mkdir()
    records = []
    for i, sample in enumerate(sorted(runner.V015_SELECTED_SAMPLES)):
        for index in range(9):
            name = f"{i:02d}_{index:02d}_{sample}.npy"
            path = root / "inputs" / name
            if populate_tensors:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.zeros((1, 3, 4, 4), dtype=np.float32))
            records.append({
                "image_index": i,
                "window_index": index,
                "sample_id": sample,
                "key": f"field_{i}",
                "tensor": f"inputs/{name}",
                "x": (index % 3) * 256,
                "y": (index // 3) * 256,
            })
    manifest = {
        "dnai_commit": "fcf20c7d6eb385675ff7d07da4fdf471589ce0cf",
        "tile": [1, 3, 4, 4],
        "overlap": 0.5,
        "stride": 256,
        "blend": "gaussian",
        "windows_per_image": 9,
        "input_name": "input",
        "records": records,
    }
    manifest_path = root / "validation_windows_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "scratch" / runner.V015_OUTPUT_DIRECTORY
    opts = argparse.Namespace(
        v015_strict_development=True,
        sample_ids="1__tile_4,2__tile_9,16__tile_5",
        image_indices=None,
        limit=None,
        platform=720,
        validation_manifest=str(manifest_path),
    )
    return opts, onnx, bie, manifest_path, output, manifest, records


def test_strict_selection_sha_checks_and_no_directory_side_effects(monkeypatch, tmp_path):
    runner, _ = load_runner(monkeypatch)
    opts, onnx, bie, manifest_path, output, manifest, records = setup_development(
        tmp_path, runner, monkeypatch,
    )
    result = runner.validate_v015_development(
        opts, onnx, bie, output, manifest, records,
    )
    assert result["frozen_expected_windows"] == 27
    assert result["frozen_expected_samples"] == sorted(runner.V015_SELECTED_SAMPLES)
    assert result["v015_bie_sha256_verified"] == sha(bie)
    assert result["frozen_input_manifest_sha256"] == sha(manifest_path)
    assert not output.exists()
    bie.write_bytes(b"tampered candidate BIE")
    with pytest.raises(SystemExit, match="immutable SHA256 mismatch"):
        runner.validate_v015_development(
            opts, onnx, bie, output, manifest, records,
        )
    assert not output.exists()


def test_strict_rejects_wrong_selection_and_missing_window(monkeypatch, tmp_path):
    runner, _ = load_runner(monkeypatch)
    opts, onnx, bie, _, output, manifest, records = setup_development(
        tmp_path, runner, monkeypatch,
    )
    opts.sample_ids = "1__tile_4,2__tile_9"
    with pytest.raises(SystemExit, match="predeclared samples"):
        runner.validate_v015_development(opts, onnx, bie, output, manifest, records)
    opts.sample_ids = "1__tile_4,2__tile_9,16__tile_5"
    with pytest.raises(SystemExit, match="27 windows"):
        runner.validate_v015_development(opts, onnx, bie, output, manifest, records[:-1])


def test_strict_rejects_existing_output_and_path_escape(monkeypatch, tmp_path):
    runner, _ = load_runner(monkeypatch)
    opts, onnx, bie, _, output, manifest, records = setup_development(
        tmp_path, runner, monkeypatch,
    )
    output.mkdir(parents=True)
    with pytest.raises(SystemExit, match="new bie_validation_v015_3dev"):
        runner.validate_v015_development(opts, onnx, bie, output, manifest, records)
    output.rmdir()
    tampered = [dict(r) for r in records]
    tampered[0]["tensor"] = "../../outside.npy"
    with pytest.raises(SystemExit, match="escapes manifest root"):
        runner.validate_v015_development(opts, onnx, bie, output, manifest, tampered)


def test_strict_writes_27_current_input_and_simulator_output_hash_receipts(
    monkeypatch, tmp_path,
):
    runner, ktc = load_runner(monkeypatch)
    opts, onnx, bie, manifest_path, output, _, _ = setup_development(
        tmp_path, runner, monkeypatch, populate_tensors=True,
    )
    monkeypatch.setattr(ktc, "kneron_inference", lambda *args, **kwargs: [
        np.ones((1, 3, 4, 4), dtype=np.float32)
        if "bie_file" not in kwargs
        else np.zeros((1, 3, 4, 4), dtype=np.float32)
    ])
    monkeypatch.setattr(sys, "argv", [
        "script.py",
        "--onnx", str(onnx),
        "--bie", str(bie),
        "--validation-manifest", str(manifest_path),
        "--output-dir", str(output),
        "--sample-ids", opts.sample_ids,
        "--v015-strict-development",
    ])
    assert runner.main() == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["v015_strict_provenance"]["n_window_receipts"] == 27
    assert summary["n_windows"] == 27
    assert summary["bie_sha256"] == sha(bie)
    import csv
    with (output / "per_window.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 27
    for row in rows:
        assert row["input_tensor_sha256"] == sha(Path(row["input_tensor"]))
        assert row["floating_output_sha256"] == sha(Path(row["floating_output"]))
        assert row["bie_output_sha256"] == sha(Path(row["bie_output"]))
    assert len(list((output / "floating_outputs").glob("*.npy"))) == 27
    assert len(list((output / "bie_outputs").glob("*.npy"))) == 27
