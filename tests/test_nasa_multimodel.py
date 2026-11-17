"""Leakage/identity checks run without torch; numerical smoke checks are optional."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from radiation_edge_ai.nasa_bps import multimodel as mm


def grouped_rows(nuclei_per_bag=4):
    rows = []
    for source in range(6):
        for nucleus in range(nuclei_per_bag):
            key = f"source{source}_nucleus{nucleus}"
            rows.append({"sample_id": key, "nucleus_key": key,
                         "sample_name": f"sample{source}", "source_name": f"source{source}",
                         "fitc_filename": f"{key}_fitc.tif", "dapi_filename": f"{key}_dapi.tif",
                         **{fold: "test" if source // 2 == index else "train"
                            for index, fold in enumerate(mm.FOLD_FIELDS)}})
    return rows


def test_listing_does_not_import_training_or_hardware_dependencies():
    source = str(Path(mm.__file__).resolve().parents[2])
    program = f"import sys; sys.path.insert(0, {source!r})\n" + """
import builtins
original_import = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'cv2', 'onnx', 'onnxruntime', 'kp'}:
        raise AssertionError('optional dependency imported: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = restricted
from radiation_edge_ai.nasa_bps.multimodel import main
raise SystemExit(main(['list']))
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True,
                            text=True, check=True)
    document = json.loads(result.stdout)
    assert {row["architecture"] for row in document["models"]} == set(mm.MODEL_NAMES)
    assert all(row["initialization"]["pretrained_backbone"] is False
               for row in document["models"])


def test_source_grouping_and_train_only_calibration_are_deterministic():
    rows = grouped_rows()
    before = copy.deepcopy(rows)
    summary = mm.validate_grouping(rows)
    assert summary["sources"] == 6
    for fold in mm.FOLD_FIELDS:
        selected = mm.calibration_rows(rows, fold, seed=720, per_bag=2)
        assert len(selected) == 8
        assert selected == mm.calibration_rows(list(reversed(rows)), fold, seed=720, per_bag=2)
        assert all(row[fold] == "train" for row in selected)
        assert {row["source_name"] for row in selected}.isdisjoint(
            summary["folds"][fold]["test_sources"])
        assert all(row["dataset_role"] == "development_train_ptq_calibration"
                   for row in selected)
    assert rows == before


@pytest.mark.parametrize("mutation, message", [
    (lambda rows: rows[0].update(fold_female_role="train"), "Source leakage"),
    (lambda rows: rows[0].update(fold_female_role="validation"), "Unexpected"),
    (lambda rows: rows[1].update(nucleus_key=rows[0]["nucleus_key"]), "Duplicate"),
    (lambda rows: rows[0].update(fitc_filename="../outside.tif"), "plain image filename"),
])
def test_invalid_grouping_is_rejected(mutation, message):
    rows = grouped_rows()
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        mm.validate_grouping(rows)


def test_source_cannot_be_held_out_twice():
    rows = grouped_rows()
    for row in rows:
        if row["source_name"] == "source0":
            row["fold_male_1_role"] = "test"
    with pytest.raises(ValueError, match="exactly one"):
        mm.validate_grouping(rows)


def test_minibag_selection_is_without_replacement_and_epoch_specific():
    rows = grouped_rows(100)[:100]
    first = mm.select_bag_rows(rows, 16, seed=720, purpose="train_epoch_1")
    assert len(first) == len({row["nucleus_key"] for row in first}) == 16
    assert first == mm.select_bag_rows(list(reversed(rows)), 16, seed=720, purpose="train_epoch_1")
    assert first != mm.select_bag_rows(rows, 16, seed=720, purpose="train_epoch_2")
    with pytest.raises(ValueError, match="between"):
        mm.select_bag_rows(rows, 101, seed=720, purpose="train_epoch_1")


def test_new_image_snapshot_is_checked_strictly_when_receipts_are_supplied(tmp_path):
    rows = grouped_rows(1)[:1]
    for channel in ("fitc", "dapi"):
        (tmp_path / channel).mkdir()
        (tmp_path / channel / rows[0][f"{channel}_filename"]).write_bytes(channel.encode())
    measured = mm.verify_images(rows, tmp_path)
    assert len(measured) == 2
    assert measured == mm.verify_images(rows, tmp_path, measured)
    path = tmp_path / "fitc" / rows[0]["fitc_filename"]
    path.write_bytes(b"changed after snapshot")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        mm.verify_images(rows, tmp_path, measured)
    with pytest.raises(ValueError, match="Missing or mismatched"):
        mm.verify_images(rows, tmp_path, [])


def test_frozen_membership_hash_and_existing_output_directories_are_protected(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("different NASA membership\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        mm.preflight(SimpleNamespace(manifest=manifest))
    for function in (mm.train, mm.export):
        with pytest.raises(FileExistsError, match="overwrite"):
            function(SimpleNamespace(out=tmp_path))


@pytest.mark.parametrize("architecture", mm.MODEL_NAMES)
def test_actual_minibag_training_full_bag_evaluation_and_onnx_export(tmp_path, architecture):
    """Synthetic TIFFs exercise all three candidates; this is not NASA validation."""
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    if architecture != "r1countnet_v1":
        pytest.importorskip("torchvision")
    from radiation_edge_ai.nasa_bps import train_pilot_v1_r1_mil as v1
    from radiation_edge_ai.nasa_bps import train_r1_v2_contrast as v2

    torch.set_num_threads(1)
    torch.manual_seed(720)
    fold = mm.FOLD_FIELDS[0]
    image_root = tmp_path / "synthetic_images"
    for channel in ("fitc", "dapi"):
        (image_root / channel).mkdir(parents=True)
    bags = []
    rng = np.random.default_rng(720)
    for bag_index, dose in enumerate(("0.0", "0.82")):
        rows = []
        for index in range(3):
            key = f"synthetic_bag{bag_index}_nucleus{index}"
            row = {"sample_id": key, "nucleus_key": key, "sample_name": f"sample{bag_index}",
                   "source_name": "source_train", fold: "train",
                   "dataset_role": "development_train_ptq_calibration", "calibration_fold": fold}
            for channel in ("fitc", "dapi"):
                filename = f"{key}_{channel}.tif"
                pixels = rng.integers(0, 2000, (17 + index, 19), dtype=np.uint16)
                path = image_root / channel / filename
                assert cv2.imwrite(str(path), pixels)
                row[f"{channel}_filename"] = filename
                row[f"{channel}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(row)
        bags.append(v1.Bag(sample_name=f"sample{bag_index}", source_name="source_train",
                           strain="synthetic", sex="synthetic", particle_type="Fe", dose_Gy=dose,
                           hr_post_exposure="4", target_avg_nfoci=1.0 + bag_index,
                           target_avg_foci_no_outl=1.0 + bag_index, rows=rows,
                           fold_roles={fold: "train"}))
    assert mm.validate_geometry([row for bag in bags for row in bag.rows], image_root)[
        "all_pixels_finite"]
    model = mm.create_model(architecture)
    old_head = model.head.weight.detach().clone()
    cache = v1.NativeImageCache(image_root)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss = mm.train_one_epoch(v1, v2, model, [(bags[0], bags[1])], cache, optimizer,
                              torch.device("cpu"), seed=720, epoch=1, nuclei_per_bag=2)
    assert math_is_finite(loss.values())
    assert not torch.equal(old_head, model.head.weight)
    run_dir = tmp_path / "synthetic_run"
    run_dir.mkdir()
    parent = run_dir / "trained_before_bn.pt"
    before_bn = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save({"state_dict": before_bn}, parent)
    held_rows = [{**row, "sample_id": "held_out_" + row["sample_id"],
                  "source_name": "source_test", fold: "test",
                  "fitc_filename": "must_never_read_held_out.tif"} for row in bags[0].rows]
    held_bag = replace(bags[0], source_name="source_test", sample_name="held_out_sample",
                       rows=held_rows, fold_roles={fold: "test"})
    bn_audit, bn_rows = mm.recalibrate_batchnorm(v1, model, [held_bag, *reversed(bags)], cache,
                                               torch.device("cpu"), fold)
    assert bn_audit["sources"] == ["source_train"]
    assert bn_audit["mixed_batch_size"] == 2
    assert bn_audit["calibration_batches"] == 3
    assert len(bn_rows) == 6
    expected_ids = [bag.rows[index]["sample_id"] for index in range(3) for bag in bags]
    assert [row["sample_id"] for row in bn_rows] == expected_ids
    assert bn_audit["row_order_sha256"] == hashlib.sha256("\n".join(expected_ids).encode()).hexdigest()
    for name, parameter in model.named_parameters():
        assert torch.equal(before_bn[name], parameter.detach().cpu())
    assert any(not torch.equal(before_bn[name], model.state_dict()[name])
               for name in bn_audit["changed_running_state"])
    assert not any(module.training for module in model.modules())
    mm._write_csv(run_dir / "bn_calibration_manifest.csv", bn_rows)
    bn_audit["manifest"] = mm._identity(run_dir / "bn_calibration_manifest.csv")
    predictions, nuclei = mm.evaluate_bags_chunked(v1, model, bags, cache,
                                                  torch.device("cpu"), fold, chunk_size=2)
    assert [row["n_development_nuclei"] for row in predictions] == [3, 3]
    assert len(nuclei) == 6
    with torch.no_grad():
        for bag, row in zip(bags, predictions):
            x = v1.load_bag_tensor(bag, cache, augment=False, rng=rng)
            assert row["predicted_bag_mean"] == pytest.approx(float(model(x).mean()), abs=1e-6)

    checkpoint = run_dir / "model.pt"
    torch.save({"architecture": architecture, "fold": fold, "recipe_id": mm.RECIPE_ID,
                "manifest_sha256": mm.MANIFEST_SHA256, "state_dict": model.state_dict(),
                "parent_checkpoint_sha256": mm._identity(parent)["sha256"],
                "batchnorm_recalibration": bn_audit}, checkpoint)
    restored = mm.create_model(architecture).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)["state_dict"])
    with torch.no_grad():
        assert torch.equal(model(x), restored(x))
    calibration = [bag.rows[0] for bag in bags]
    mm._write_csv(run_dir / "calibration_manifest.csv", calibration)
    mm._write_json(run_dir / "run.json", {
        "status": "complete", "recipe": {"id": mm.RECIPE_ID}, "architecture": architecture,
        "fold": fold, "contract": mm.model_contract(architecture),
        "checkpoint": mm._identity(checkpoint),
        "parent_checkpoint": mm._identity(parent), "batchnorm_recalibration": bn_audit,
        "serialization_reload": {"passed": True},
        "calibration": mm._identity(run_dir / "calibration_manifest.csv"),
        "dataset": {"image_root": str(image_root)},
        "source_split": {"train_sources": ["source_train"], "test_sources": ["source_test"]},
        "implementation": {name: mm._identity(Path(mm.__file__).with_name(name)) for name in
                           ("model_factory.py", "train_pilot_v1_r1_mil.py")},
    })
    output = tmp_path / "export"
    result = mm.export(SimpleNamespace(run_dir=run_dir, out=output, image_root=None))
    assert result["parity_pass"] is True
    assert result["calibration_arrays"] == 2
    assert result["contract"]["input"]["deployment_shape"] == [1, 3, 256, 256]
    assert np.load(output / "calibration_nchw" / "0000.npy").shape == (1, 3, 256, 256)
    # Export must verify raw TIFF identities again, even if pixels remain readable.
    modified = image_root / "fitc" / calibration[0]["fitc_filename"]
    modified.write_bytes(modified.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        mm.export(SimpleNamespace(run_dir=run_dir, out=tmp_path / "reexport", image_root=None))
    assert not (tmp_path / "reexport").exists()
    # A complete-looking pre-BN run must not be exportable under this recipe.
    run = json.loads((run_dir / "run.json").read_text())
    del run["batchnorm_recalibration"]
    mm._write_json(run_dir / "run.json", run)
    with pytest.raises(ValueError, match="mandatory training-only BN"):
        mm.export(SimpleNamespace(run_dir=run_dir, out=tmp_path / "uncalibrated", image_root=None))


def math_is_finite(values):
    import math
    return all(math.isfinite(value) for value in values)


def test_train_orchestration_synthetic_fold_saves_repaired_checkpoint_and_comparison(tmp_path, monkeypatch):
    """Stub only frozen-data loading, while executing the actual training pipeline."""
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from radiation_edge_ai.nasa_bps import train_pilot_v1_r1_mil as v1
    from radiation_edge_ai.nasa_bps import train_r1_v2_contrast as v2

    torch.set_num_threads(1)
    image_root = tmp_path / "synthetic_images"
    for channel in ("fitc", "dapi"):
        (image_root / channel).mkdir(parents=True)
    rng = np.random.default_rng(42)
    bags, rows = [], []
    for source in range(6):
        roles = {fold: "test" if source // 2 == index else "train"
                 for index, fold in enumerate(mm.FOLD_FIELDS)}
        for branch, exposed_dose in (("Fe", "0.82"), ("X-ray", "1.0")):
            for dose in ("0.0", exposed_dose):
                for hour in ("4", "24", "48"):
                    sample = f"source{source}_{branch}_dose{dose}_hour{hour}"
                    bag_rows = []
                    for index in range(3):
                        key = f"{sample}_nucleus{index}"
                        row = {"sample_id": key, "nucleus_key": key, "sample_name": sample,
                               "source_name": f"source{source}", **roles}
                        for channel in ("fitc", "dapi"):
                            filename = f"{key}_{channel}.tif"
                            pixels = rng.integers(0, 2000, (17, 19), dtype=np.uint16)
                            assert cv2.imwrite(str(image_root / channel / filename), pixels)
                            row[f"{channel}_filename"] = filename
                        bag_rows.append(row)
                    target = 0.2 + float(dose) + float(hour) / 100 + source / 100
                    bags.append(v1.Bag(sample_name=sample, source_name=f"source{source}",
                                       strain="synthetic", sex="synthetic", particle_type=branch,
                                       dose_Gy=dose, hr_post_exposure=hour, target_avg_nfoci=target,
                                       target_avg_foci_no_outl=target, rows=bag_rows, fold_roles=roles))
                    rows.extend(bag_rows)
    manifest, reference = tmp_path / "synthetic_manifest.csv", tmp_path / "synthetic_reference.csv"
    mm._write_csv(manifest, rows)
    mm._write_csv(reference, [{"sample_name": bag.sample_name, "nasa_avg_nfoci": bag.target_avg_nfoci}
                              for bag in bags])
    snapshot = mm.verify_images(rows, image_root)
    report = {"synthetic_test_fixture": True, "manifest": mm._identity(manifest),
              "reference": mm._identity(reference), "image_root": str(image_root),
              **mm.validate_grouping(rows)}
    monkeypatch.setattr(mm, "preflight", lambda args: (report, rows, snapshot))
    monkeypatch.setattr(v2, "build_bags", lambda *args: bags)
    monkeypatch.setattr(v2, "validate_reference_freeze", lambda path: {})
    output = tmp_path / "synthetic_benchmark"
    fold = mm.FOLD_FIELDS[0]
    args = SimpleNamespace(out=output, device="cpu", manifest=manifest, reference=reference,
                           reference_freeze=tmp_path / "not_a_real_NASA_freeze.json",
                           image_root=image_root, inventory=None,
                           models=["r1countnet_v1"], folds=[fold], seed=720, epochs=1,
                           train_nuclei_per_bag=2, eval_chunk_size=2, calibration_nuclei_per_bag=1)
    assert mm.train(args)["status"] == "complete"
    run_dir = output / "r1countnet_v1" / fold
    run = json.loads((run_dir / "run.json").read_text())
    assert run["status"] == "complete"
    assert run["dataset"]["synthetic_test_fixture"] is True
    assert run["batchnorm_recalibration"]["mixed_batch_size"] == 48
    assert run["batchnorm_recalibration"]["calibration_batches"] == 3
    assert run["batchnorm_recalibration"]["sources"] == ["source2", "source3", "source4", "source5"]
    assert run["serialization_reload"]["passed"] is True
    assert run["image_snapshot_reverified_after_fold"] is True
    parent = torch.load(run_dir / "trained_before_bn.pt", weights_only=True)
    repaired = torch.load(run_dir / "model.pt", weights_only=True)
    assert repaired["parent_checkpoint_sha256"] == run["parent_checkpoint"]["sha256"]
    for name, tensor in parent["state_dict"].items():
        if name not in run["batchnorm_recalibration"]["changed_running_state"]:
            assert torch.equal(tensor, repaired["state_dict"][name])
    comparison = mm._read_csv(output / "comparison.csv")
    assert len(comparison) == 1
    assert comparison[0]["all_frozen_folds_complete"] == "False"
    assert comparison[0]["qualification"] == "development_candidate_only"
    assert set(run["metrics"]["per_source"]) == {"source0", "source1"}
    converted = mm.export(SimpleNamespace(run_dir=run_dir, out=tmp_path / "synthetic_onnx",
                                         image_root=None))
    assert converted["parity_pass"] is True
    assert converted["calibration_arrays"] == 48
