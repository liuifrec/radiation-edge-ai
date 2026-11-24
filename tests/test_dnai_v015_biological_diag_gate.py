"""Hardware-free validation of opt-in v5 biological diagnostic input rules."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


SAMPLES = ("1__tile_4", "2__tile_9", "16__tile_5")


def load_diag(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    dnafiber = types.ModuleType("dnafiber")
    dnafiber.__path__ = []
    postprocess = types.ModuleType("dnafiber.postprocess")
    postprocess.refine_segmentation = lambda x: x
    monkeypatch.setitem(sys.modules, "dnafiber", dnafiber)
    monkeypatch.setitem(sys.modules, "dnafiber.postprocess", postprocess)
    script = (
        Path(__file__).resolve().parents[1] / "src" / "radiation_edge_ai"
        / "dna_fiber" / "diagnose_int8_fiber_disagreements.py"
    )
    spec = importlib.util.spec_from_file_location("v015_biological_diag_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def make_fake_completed(tmp_path):
    root = tmp_path / "biological_fidelity_v015_3dev"
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps({"n_complete_images": 3}), encoding="utf-8"
    )
    with (root / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "key"])
        writer.writeheader()
        for i, sample in enumerate(SAMPLES):
            writer.writerow({"sample_id": sample, "key": f"f{i}"})
            sample_root = root / sample
            sample_root.mkdir()
            (sample_root / "floating_segmentation_class_ids.png").write_bytes(
                b"synthetic FP mask"
            )
            (sample_root / "bie_segmentation_class_ids.png").write_bytes(
                b"synthetic BIE mask"
            )
    return root


def test_v015_validates_all_three_biological_samples(monkeypatch, tmp_path):
    mod = load_diag(monkeypatch)
    root = make_fake_completed(tmp_path)
    for sample in SAMPLES:
        assert mod.validate_v015_bio_root(root, sample) == root / sample


def test_v015_rejects_unselected_sample_and_wrong_directory(monkeypatch, tmp_path):
    mod = load_diag(monkeypatch)
    root = make_fake_completed(tmp_path)
    with pytest.raises(SystemExit, match="predeclared"):
        mod.validate_v015_bio_root(root, "13__tile_14")
    with pytest.raises(SystemExit, match="biological_fidelity_v015_3dev"):
        mod.validate_v015_bio_root(root.parent, SAMPLES[0])


def test_v015_rejects_unfinished_masks(monkeypatch, tmp_path):
    mod = load_diag(monkeypatch)
    root = make_fake_completed(tmp_path)
    (root / SAMPLES[1] / "bie_segmentation_class_ids.png").unlink()
    with pytest.raises(SystemExit, match="masks are incomplete"):
        mod.validate_v015_bio_root(root, SAMPLES[1])


def test_v015_rejects_wrong_sample_set_or_partial_summary(monkeypatch, tmp_path):
    mod = load_diag(monkeypatch)
    root = make_fake_completed(tmp_path)
    (root / "summary.json").write_text(
        '{"n_complete_images":2}', encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="exactly three"):
        mod.validate_v015_bio_root(root, SAMPLES[0])
    (root / "summary.json").write_text(
        '{"n_complete_images":3}', encoding="utf-8"
    )
    with (root / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "key"])
        writer.writeheader()
        for sample in (SAMPLES[0], SAMPLES[0], SAMPLES[2]):
            writer.writerow({"sample_id": sample, "key": "wrong"})
    with pytest.raises(SystemExit, match="three frozen samples"):
        mod.validate_v015_bio_root(root, SAMPLES[0])
