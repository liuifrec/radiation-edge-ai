"""Offline regression tests for the cross-platform training-only PTQ pilot."""

import csv
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from radiation_edge_ai.nasa_bps import multimodel_ptq as ptq


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    graph = tmp_path / "source.onnx"
    graph.write_bytes(b"original")
    optimized = tmp_path / "optimized.onnx"
    optimized.write_bytes(b"optimized")
    tensor_dir = tmp_path / "calibration_nchw"
    tensor_dir.mkdir()
    rows = []
    for index in range(192):
        name = "{:04d}.npy".format(index)
        path = tensor_dir / name
        path.write_bytes(str(index).encode())
        rows.append(dict(array_file="calibration_nchw\\" + name, array_sha256=ptq.sha256(path),
                         sample_id="n{}".format(index), sample_name="bag{}".format(index // 4),
                         source_name="s{}".format(index // 48), fold_female_role="train",
                         calibration_fold=ptq.FOLD, dataset_role="development_train_ptq_calibration"))
    arr = np.zeros((1, 3, 256, 256), dtype=np.float32)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: arr)
    args = SimpleNamespace(onnx=graph, optimized=optimized, calibration_root=tmp_path,
                           manifest=tmp_path / "calibration_manifest.csv", image_id=ptq.IMAGE_ID,
                           export=tmp_path / "export.json", screen=tmp_path / "screen.json",
                           out=tmp_path / "out")

    def pin_inputs():
        write_rows(args.manifest, rows)
        source = ptq.sha256(graph)
        export = dict(architecture=ptq.ARCH, fold=ptq.FOLD, parity_pass=True,
                      calibration_arrays=192, parity_atol=1e-5, parity_rtol=1e-4,
                      dataset_role="development_train_ptq_calibration",
                      training_run={"sha256": ptq.EXPECTED_TRAIN_SHA}, onnx={"sha256": source},
                      calibration={"sha256": ptq.sha256(args.manifest)},
                      contract={"input": {"deployment_shape": [1, 3, 256, 256],
                                          "channels": ["FITC_53BP1", "DAPI", "zeros"]},
                                "output": {"semantics": "latent_continuous_per_nucleus_53bp1_burden"}},
                      source_split={"train_sources": ["s0", "s1", "s2", "s3"],
                                    "test_sources": ["x", "y"], "train_bags": 48, "test_bags": 24})
        args.export.write_text(json.dumps(export))
        screen = dict(architecture=ptq.ARCH, fold=ptq.FOLD, model_id=ptq.MODEL_ID,
                      purpose="trained_fp32_exact_graph_feasibility_only", trained=True,
                      original_fp32_parity_pass=True, support_status="supported",
                      vendor_evaluation_completed=True, source_onnx_sha256=source,
                      export_receipt_sha256=ptq.sha256(args.export), platform="720",
                      optimized_onnx_sha256=ptq.sha256(optimized), toolchain_image_id=ptq.IMAGE_ID,
                      ptq_performed=False, nef_compiled=False)
        args.screen.write_text(json.dumps(screen))
        monkeypatch.setattr(ptq, "EXPECTED_SOURCE_SHA", source)
        monkeypatch.setattr(ptq, "EXPECTED_EXPORT_SHA", ptq.sha256(args.export))
        monkeypatch.setattr(ptq, "EXPECTED_SCREEN_SHA", ptq.sha256(args.screen))

    pin_inputs()
    return args, rows, pin_inputs


@pytest.mark.parametrize("name", ["../0001.npy", r"C:\calibration_nchw\0001.npy",
                                  "calibration_nchw/../0001.npy", "calibration_nchw//0001.npy",
                                  "calibration_nchw/./0001.npy", "/calibration_nchw/0001.npy"])
def test_bad_calibration_paths(tmp_path, name):
    with pytest.raises(ValueError, match="Invalid calibration array path"):
        ptq.calibration_path(tmp_path, name)


def test_legacy_and_portable_path_aliases_are_duplicates(pilot):
    args, rows, pin_inputs = pilot
    rows[1]["array_file"] = rows[0]["array_file"].replace("\\", "/")
    pin_inputs()
    with pytest.raises(ValueError, match="Duplicate calibration array"):
        ptq.preflight(args)


@pytest.mark.parametrize("separator", ["\\", "/"])
def test_read_real_array_preserves_manifest_bytes(tmp_path, separator):
    folder = tmp_path / "calibration_nchw"
    folder.mkdir()
    array = np.arange(8, dtype=np.float32)
    path = folder / "0001.npy"
    np.save(path, array)
    manifest = tmp_path / "manifest.csv"
    write_rows(manifest, [{"array_file": "calibration_nchw" + separator + path.name,
                           "array_sha256": ptq.sha256(path)}])
    raw = manifest.read_bytes()
    row, = ptq.checked_rows(manifest)
    resolved = ptq.calibration_path(tmp_path, row["array_file"])
    np.testing.assert_array_equal(ptq.read_calibration_array(resolved, row["array_sha256"]), array)
    assert manifest.read_bytes() == raw
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ptq.read_calibration_array(resolved, row["array_sha256"])


@pytest.mark.parametrize("content", ["a,a\r\n1,2\r\n", "a,b\r\n1\r\n", "a,b\r\n1,2,3\r\n"])
def test_ambiguous_csv_is_rejected(tmp_path, content):
    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(content.encode())
    with pytest.raises(ValueError, match="CSV"):
        ptq.checked_rows(manifest)


def test_aliased_calibration_file_rejected(pilot):
    args, rows, pin_inputs = pilot
    first, second = [ptq.calibration_path(args.calibration_root, row["array_file"]) for row in rows[:2]]
    second.unlink()
    os.link(first, second)
    rows[1]["array_sha256"] = rows[0]["array_sha256"]
    pin_inputs()
    with pytest.raises(ValueError, match="Duplicate calibration file identity"):
        ptq.preflight(args)


def test_heldout_source_is_rejected_before_quantization(pilot):
    args, rows, pin_inputs = pilot
    rows[0]["source_name"] = "x"
    pin_inputs()
    with pytest.raises(ValueError, match="train-only provenance"):
        ptq.preflight(args)


@pytest.mark.parametrize("shift", [0, 0.01])
def test_192_sample_parity_controls_vendor_gate(pilot, monkeypatch, shift):
    args, rows, _ = pilot

    class Session:
        def __init__(self, path, providers):
            assert providers == ["CPUExecutionProvider"]
            self.optimized = path == str(args.optimized)
            self.count = 0

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 3, 256, 256], type="tensor(float)")]

        def get_outputs(self):
            return [SimpleNamespace(name="burden", shape=[1], type="tensor(float)")]

        def run(self, outputs, inputs):
            self.count += 1
            return [np.array([1 + (shift if self.optimized and self.count == 1 else 0)],
                             dtype=np.float32)]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session))
    assert ptq.parity(args) == (1 if shift else 0)
    path = args.out / "fp32_optimized_parity.json"
    receipt = ptq.load_json(path)
    assert receipt["pass_count"] == 192 - bool(shift)
    assert receipt["fail_count"] == bool(shift)
    table = path.with_suffix(".csv")
    with table.open(newline="") as handle:
        assert [row["sample_id"] for row in csv.DictReader(handle)] == [r["sample_id"] for r in rows]
    if shift:
        with pytest.raises(ValueError, match="status"):
            ptq.verify_parity_receipt(path, args)
    else:
        ptq.verify_parity_receipt(path, args)
        table.write_text("changed")
        with pytest.raises(ValueError, match="predictions_sha256"):
            ptq.verify_parity_receipt(path, args)
        forged = [dict(sample, source_fp32="1.0", optimized_fp32="1.125", absolute_error="0.125",
                       parity_pass="True") for sample in rows]
        write_rows(table, forged)
        receipt.update(predictions_sha256=ptq.sha256(table), max_absolute_error=0.125)
        path.write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match="Invalid optimized FP32 parity row"):
            ptq.verify_parity_receipt(path, args)
