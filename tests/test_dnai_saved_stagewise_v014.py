"""Offline, no-hardware checks for the saved DNAi stagewise diagnostic."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "windows"
    / "audit_dnai_saved_stagewise_v014.py"
)
SPEC = importlib.util.spec_from_file_location("saved_stagewise_v014", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stagewise_metric_telescopes_and_bie_physical_identical(monkeypatch):
    monkeypatch.setattr(mod, "TENSOR_SHAPE", (1, 3, 4, 4))
    cpu = np.zeros((1, 3, 4, 4), dtype=np.float32)
    cpu[:, 1] = 1.0
    floating = cpu.copy()
    floating[:, 1] = 0.55
    fixed = floating.copy()
    fixed[:, 0] = 0.80
    physical = fixed.copy()
    info = mod.analyze_window((cpu, floating, fixed, physical))
    assert info["cpu_physical_argmax_disagreement_pixels"] == 16
    assert info["stage_pairs"]["bie_to_physical"]["raw_array_identical"] is True
    assert info["stage_pairs"]["bie_to_physical"]["probability_mae_all"] == 0
    assert info["stage_pairs"]["cpu_to_kneron_floating"]["argmax_changed_pixels_all"] == 0
    assert info["stage_pairs"]["kneron_floating_to_bie"]["argmax_changed_pixels_all"] == 16
    assert info["max_stagewise_margin_telescope_error"] == 0


def test_stagewise_rejects_bie_physical_mismatch(monkeypatch):
    monkeypatch.setattr(mod, "TENSOR_SHAPE", (1, 3, 4, 4))
    cpu = np.zeros((1, 3, 4, 4), dtype=np.float32)
    bie = cpu.copy()
    physical = cpu.copy()
    physical[0, 1, 0, 0] = 1
    with pytest.raises(ValueError, match="not bitwise identical"):
        mod.analyze_window((cpu, cpu.copy(), bie, physical))


def synthetic_sources(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, Path]:
    monkeypatch.setattr(mod, "TENSOR_SHAPE", (1, 3, 4, 4))
    v3 = tmp_path / "kneron" / "int8_512_v3_pct100"
    inputs_root = tmp_path / "kneron" / "int8_512" / "validation_512_overlap50"
    bin_bie = v3 / "unet_mobileone_s1_512x512_int8.bie"
    bin_nef = v3 / "nef" / "unet_mobileone_s1_512x512_v3.nef"
    bin_bie.parent.mkdir(parents=True, exist_ok=True)
    bin_nef.parent.mkdir(parents=True, exist_ok=True)
    bin_bie.write_bytes(b"selected mock BIE")
    bin_nef.write_bytes(b"selected mock NEF")
    bie_hash, nef_hash = sha(bin_bie), sha(bin_nef)
    monkeypatch.setattr(mod, "EXPECTED_BIE", bie_hash)
    monkeypatch.setattr(mod, "EXPECTED_NEF", nef_hash)
    ptq = {
        "datapath_range_method": "percentage", "percentage": 1.0,
        "percentage_16b": 1.0, "percentile": 0.001,
        "outlier_factor": 1.0, "optimize": 0,
    }
    write_json(v3 / "int8_analysis_summary.json", {
        "bie_sha256": bie_hash, "onnx_sha256": mod.EXPECTED_ONNX,
        "validation_panel_used_for_calibration": False, "calibration_count": 192,
        "ptq_config": ptq,
    })
    write_json(v3 / "bie_validation" / "summary.json", {
        "bie_sha256": bie_hash, "onnx_sha256": mod.EXPECTED_ONNX,
        "selected_sample_ids": ["other_sample"],
    })
    write_json(v3 / "nef" / "nef_compile_summary.json", {
        "source_bie_sha256": bie_hash, "nef_sha256": nef_hash,
        "source_onnx_sha256": mod.EXPECTED_ONNX,
    })
    manifest_records = []
    field_inputs = []
    fp_outputs, kl_outputs = [], []
    for i in range(9):
        coords = {"window_index": i, "x": (i % 3) * 2, "y": (i // 3) * 2}
        inp = inputs_root / "inputs" / f"00_{i:02d}_1__tile_4.npy"
        inp.parent.mkdir(parents=True, exist_ok=True)
        np.save(inp, np.ones((1, 3, 4, 4), dtype=np.float32), allow_pickle=False)
        manifest_records.append({**coords, "sample_id": mod.SAMPLE, "tensor": f"inputs/{inp.name}"})
        field_inputs.append({**coords, "tensor_sha256": sha(inp)})
        base = np.zeros((1, 3, 4, 4), dtype=np.float32)
        base[:, 1] = 1
        floating = base.copy()
        floating[:, 1] = 0.9
        bie = floating.copy()
        bie[:, 0] = 1.3
        for stage, arr in (
            ("fp", base), ("kl", bie),
            ("floating_outputs", floating), ("bie_outputs", bie),
        ):
            dest = (
                tmp_path / "frozen" / stage / f"w{i}.npy"
                if stage in ("fp", "kl")
                else v3 / "bie_validation" / stage / f"00_{i:02d}_{mod.SAMPLE}.npy"
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.save(dest, arr, allow_pickle=False)
            if stage in ("fp", "kl"):
                output = {**coords, "output": {"path": str(dest), "sha256": sha(dest)}}
                (fp_outputs if stage == "fp" else kl_outputs).append(output)
    manifest = inputs_root / "validation_windows_manifest.json"
    write_json(manifest, {
        "tile": list(mod.TENSOR_SHAPE), "stride": 256, "overlap": 0.5,
        "records": manifest_records,
    })
    fp_json = tmp_path / "fp.json"
    kl_json = tmp_path / "kl.json"
    write_json(fp_json, {
        "identity": {"sample_id": mod.SAMPLE, "windows": field_inputs},
        "window_outputs": fp_outputs,
    })
    write_json(kl_json, {
        "identity": {"sample_id": mod.SAMPLE, "windows": field_inputs},
        "window_outputs": kl_outputs,
    })
    return fp_json, kl_json, v3, manifest


def test_synthetic_full_offline_audit(tmp_path, monkeypatch):
    inputs = synthetic_sources(tmp_path, monkeypatch)
    before = sha(inputs[0]), sha(inputs[1])
    report = mod.audit(*inputs)
    assert report["status"] == "descriptive_only_legacy_simulator_outputs"
    assert report["frozen_input_sha256s_verified"] == 9
    assert report["frozen_source_output_sha256s_verified"] == 18
    assert report["summary"]["bie_and_physical_byte_identical_windows"] == 9
    assert report["summary"]["cpu_to_physical"]["total_argmax_changed_pixels"] == 9 * 16
    assert report["summary"]["cpu_to_kneron_floating"]["total_argmax_changed_pixels"] == 0
    assert report["summary"]["kneron_floating_to_bie"]["total_argmax_changed_pixels"] == 9 * 16
    assert (sha(inputs[0]), sha(inputs[1])) == (before[0], before[1])


def test_synthetic_input_hash_tamper_fails_closed(tmp_path, monkeypatch):
    inputs = synthetic_sources(tmp_path, monkeypatch)
    path = inputs[3].parent / "inputs" / "00_06_1__tile_4.npy"
    with path.open("ab") as f:
        f.write(b"modified")
    with pytest.raises(ValueError, match="input tensor SHA256 mismatch"):
        mod.audit(*inputs)


def test_synthetic_bie_cache_tamper_fails_closed(tmp_path, monkeypatch):
    inputs = synthetic_sources(tmp_path, monkeypatch)
    path = inputs[2] / "bie_validation" / "bie_outputs" / "00_03_1__tile_4.npy"
    with path.open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="BIE/physical cache byte identity"):
        mod.audit(*inputs)


def test_missing_original_compile_record_fails_closed(tmp_path, monkeypatch):
    inputs = synthetic_sources(tmp_path, monkeypatch)
    (inputs[2] / "nef" / "nef_compile_summary.json").unlink()
    with pytest.raises(ValueError, match="exactly one saved NEF compile record"):
        mod.audit(*inputs)
