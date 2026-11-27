import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/windows/verify_dnai_hardware_baseline.py"
SPEC = importlib.util.spec_from_file_location("dnai_hardware_baseline_comparison", SCRIPT)
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def test_offline_command_is_explicit_and_never_uses_physical_runner(monkeypatch, tmp_path, capsys):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import run_dnai_hardware_baseline as launcher

    def forbid_usb(*args):
        pytest.fail("Offline mode called the physical runner")

    monkeypatch.setattr(launcher, "run", forbid_usb)
    monkeypatch.setattr(launcher, "preflight", lambda root: {"tile": [1, 3, 512, 512]})
    with pytest.raises(SystemExit):
        launcher.main(["--data-root", str(tmp_path)])
    assert launcher.main(["--data-root", str(tmp_path), "--preflight-only"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["usb_accessed"] is False and "historical_replay" not in result
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(FileNotFoundError):
        launcher.main(["--data-root", str(tmp_path), "--verify-archive"])
    assert list(tmp_path.iterdir()) == []


def test_window_comparison_detects_changes_and_invalid_arrays(tmp_path):
    hw, ref = (tmp_path / n for n in ("hw.npy", "ref.npy"))
    original = np.zeros((1, 3, 512, 512), dtype=np.float32)
    np.save(ref, original)
    np.save(hw, original)
    assert baseline.compare_window(hw, ref)["array_equal"]
    original[0, 1, 25, 26] = 0.125
    np.save(hw, original)
    assert baseline.compare_window(hw, ref) == {
        "array_equal": False,
        "max_abs_difference": 0.125,
        "different_values": 1,
    }
    original[0, 0, 0, 0] = np.nan
    np.save(hw, original)
    with pytest.raises(ValueError, match="Invalid physical logits"):
        baseline.compare_window(hw, ref)


def test_only_fresh_complete_correctly_bound_panel_rows_are_accepted(tmp_path):
    run, manifest = tmp_path / "scratch/run", tmp_path / "models/manifest.json"
    records, log = [], []
    for sample, image_index in zip(baseline.SAMPLES, (0, 1, 18)):
        for index in range(9):
            name = f"{image_index:02}_{index:02}_{sample}.npy"
            record = {
                "sample_id": sample,
                "key": sample,
                "image_index": image_index,
                "window_index": index,
                "tensor": "inputs/" + name,
            }
            records.append(record)
            log.append(
                {
                    **record,
                    "status": "ran",
                    "position": len(records),
                    "hardware_send_receive_ms": 100.0,
                    "input_tensor": str(manifest.parent / record["tensor"]),
                    "physical_output": str(run / "physical_outputs" / name),
                }
            )
    summary = {
        "complete": True,
        "nef_sha256": baseline.NEF,
        "selected_samples": list(baseline.SAMPLES),
        "model_id": 32769,
        "n_existing_skipped": 0,
        "n_ran": 27,
        "selected_windows": 27,
        "n_valid_outputs": 27,
        "missing_outputs": [],
        "validation_manifest": str(manifest),
        "physical_output_dir": str(run / "physical_outputs"),
        "per_window_csv": str(run / "per_window.csv"),
    }
    baseline.fresh_rows(summary, log, records, run, manifest)
    for key, value in (
        ("status", "existing"),
        ("hardware_send_receive_ms", "nan"),
        ("physical_output", str(tmp_path / "historic.npy")),
        ("window_index", 7),
    ):
        with pytest.raises(ValueError, match="physical execution row"):
            baseline.fresh_rows(
                summary, [dict(log[0], **{key: value}), *log[1:]], records, run, manifest
            )
    with pytest.raises(ValueError, match="fresh physical"):
        baseline.fresh_rows(dict(summary, n_existing_skipped=1), log, records, run, manifest)


def test_historical_directories_and_altered_references_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside frozen"):
        baseline.verify(tmp_path, tmp_path / "models" / "existing_hardware_panel")
    file = tmp_path / "ref"
    file.write_bytes(b"old")
    original = baseline.digest(file)
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen SHA256 mismatch"):
        baseline.checked(file, original)
