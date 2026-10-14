"""Offline synthetic tests for verified DNAi object correspondence; no NPU/DNAi required."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from radiation_edge_ai import dnai_fiber_instances as inst
from radiation_edge_ai.control import ControlPlaneError


@dataclass
class FakeFiber:
    index: int
    length: float
    ratio: float
    edges: dict[tuple[int, int], float]

    def bbox_iou(self, candidate: "FakeFiber") -> float:
        return self.edges.get((self.index, candidate.index), 0.0)


def _example():
    edges = {(0, 10): 0.9, (1, 11): 0.7}
    fp = [FakeFiber(i, length, ratio, edges) for i, length, ratio in (
        (0, 10, 1.0), (1, 20, 2.0), (2, 30, 3.0)
    )]
    kl = [FakeFiber(i, length, ratio, edges) for i, length, ratio in (
        (10, 12, 1.2), (11, 19, 2.1)
    )]
    return fp, kl


def test_spatial_matching_preserves_local_ids_and_reconciles_length() -> None:
    fp, kl = _example()
    report = inst._matching_report(fp, kl, [3, 5, 7], [71, 73], 0.26)
    primary = report["primary"]
    assert primary["summary"]["n_matched"] == 2
    assert primary["summary"]["n_reference_unmatched"] == 1
    assert primary["summary"]["n_candidate_unmatched"] == 0
    assert [(pair["fp512_local_fiber_id"], pair["kl720_local_fiber_id"])
            for pair in primary["matched_pairs"]] == [(3, 71), (5, 73)]
    decomp = primary["length_decomposition_um"]
    assert decomp["matched_kl720_minus_fp512"] == pytest.approx(0.26)
    assert decomp["unmatched_fp512"] == pytest.approx(7.8)
    assert decomp["unmatched_kl720"] == pytest.approx(0)
    assert decomp["kl720_minus_fp512_total"] == pytest.approx(-7.54)
    assert primary["matched_length_mae_um"] == pytest.approx(0.39)
    assert primary["matched_ratio_mae"] == pytest.approx(0.15)
    assert report["maximum_cardinality_sensitivity"]["same_pair_indices"] is True
    assert decomp["reconciled"] is True


def test_empty_fiber_fields_do_not_claim_error_estimates() -> None:
    report = inst._matching_report([], [], [], [], 0.26)
    first = report["primary"]
    assert first["summary"]["n_matched"] == 0
    assert first["summary"]["f1"] == 1.0
    assert first["matched_length_mae_um"] is None
    assert first["matched_ratio_mae"] is None
    assert first["length_decomposition_um"]["kl720_minus_fp512_total"] == 0.0


def test_nonfinite_ratio_ignored_but_length_retained() -> None:
    fp, kl = _example()
    fp[0].ratio = float("nan")
    report = inst._matching_report(fp, kl, [3, 5, 7], [71, 73], 0.26)
    one = report["primary"]
    assert one["n_ratio_comparable"] == 1
    assert one["matched_ratio_mae"] == pytest.approx(0.1)
    assert one["matched_pairs"][0]["fp512_ratio"] is None
    assert one["matched_pairs"][0]["delta_ratio"] is None
    json.dumps(report, allow_nan=False)


def test_sensitivity_can_change_cardinality_without_retuning() -> None:
    edges = {(0, 10): 0.9, (0, 11): 0.8, (1, 10): 0.8}
    fp = [FakeFiber(i, 10, 1.0, edges) for i in (0, 1)]
    kl = [FakeFiber(j, 10, 1.0, edges) for j in (10, 11)]
    report = inst._matching_report(fp, kl, [1, 2], [11, 12], 0.26)
    assert report["primary"]["summary"]["n_matched"] == 1
    assert report["maximum_cardinality_sensitivity"]["summary"]["n_matched"] == 2
    assert report["maximum_cardinality_sensitivity"]["same_cardinality"] is False
    assert report["matching_definition"]["threshold"] == 0.5


@pytest.mark.parametrize("pixels,ids", [
    (0.0, ([3, 5, 7], [71, 73])),
    (float("nan"), ([3, 5, 7], [71, 73])),
    (0.26, ([3, 3, 7], [71, 73])),
    (0.26, ([3], [71, 73])),
    (0.26, ([3, 5, 7], [71, "73"])),
])
def test_invalid_matching_inputs_fail_closed(pixels, ids) -> None:
    fp, kl = _example()
    with pytest.raises(ControlPlaneError):
        inst._matching_report(fp, kl, ids[0], ids[1], pixels)


def _identity(sha="a" * 64):
    return {
        "sample_id": "1__tile_4", "key": "1/tile_4.png", "image_index": 0,
        "dnai_commit": "f" * 40, "validation_manifest_sha256": "b" * 64,
        "deployment_policy": {"pixel_size_um": 0.26},
        "windows": [
            {"window_index": i, "x": i % 3 * 256, "y": i // 3 * 256,
             "tensor_sha256": sha, "tensor_size_bytes": 300} for i in range(9)
        ],
    }


def _make_two_records(tmp_path: Path) -> tuple[Path, Path]:
    fp = tmp_path / "df1.json"
    kl = tmp_path / "df2.json"
    for path, schema in ((fp, 1), (kl, 2)):
        path.write_text(json.dumps({
            "record_type": "dnai_fiber_field_record", "schema_version": schema,
            "field_id": f"df{schema}-fake", "identity": _identity(),
            "runtime": {"dnafiber_version": "synthetic"},
        }), encoding="utf-8")
    return fp, kl


def test_verified_pair_checks_full_gate_and_shared_inputs(tmp_path, monkeypatch) -> None:
    fp, kl = _make_two_records(tmp_path)
    calls = []

    def yes(path, *, check_artifacts):
        calls.append((path, check_artifacts))
        return {"kind": "dnai_fiber_field_record", "ok": True}

    monkeypatch.setattr(inst, "verify_target", yes)
    records, hashes, identity = inst._verified_field_pair(fp, kl)
    assert [r["field_id"] for r in records] == ["df1-fake", "df2-fake"]
    assert calls == [(fp.resolve(), True), (kl.resolve(), True)]
    assert hashes == [hashlib.sha256(fp.read_bytes()).hexdigest(), hashlib.sha256(kl.read_bytes()).hexdigest()]
    assert identity["sample_id"] == "1__tile_4"


def test_verified_pair_rejects_unverified_source(tmp_path, monkeypatch) -> None:
    fp, kl = _make_two_records(tmp_path)
    monkeypatch.setattr(inst, "verify_target", lambda *_a, **_kw: {"kind": "dnai_fiber_field_record", "ok": False})
    with pytest.raises(ControlPlaneError, match="full verification"):
        inst._verified_field_pair(fp, kl)


def test_verified_pair_rejects_input_digest_difference(tmp_path, monkeypatch) -> None:
    fp, kl = _make_two_records(tmp_path)
    payload = json.loads(kl.read_text(encoding="utf-8"))
    payload["identity"]["windows"][0]["tensor_sha256"] = "z" * 64
    kl.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(inst, "verify_target", lambda *_a, **_kw: {"kind": "dnai_fiber_field_record", "ok": True})
    with pytest.raises(ControlPlaneError, match="identical frozen input"):
        inst._verified_field_pair(fp, kl)


def test_verified_pair_detects_source_modified_mid_audit(tmp_path, monkeypatch) -> None:
    fp, kl = _make_two_records(tmp_path)

    def malicious(path, *, check_artifacts):
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return {"kind": "dnai_fiber_field_record", "ok": True}

    monkeypatch.setattr(inst, "verify_target", malicious)
    with pytest.raises(ControlPlaneError, match="changed during verification"):
        inst._verified_field_pair(fp, kl)


def test_worker_wrapper_uses_explicit_scientific_python(tmp_path, monkeypatch) -> None:
    interp = tmp_path / "science-python.exe"
    interp.write_bytes(b"stub")
    fp = tmp_path / "fp.json"
    kl = tmp_path / "kl.json"
    fp.write_text("{}", encoding="utf-8")
    kl.write_text("{}", encoding="utf-8")
    result_obj = {
        "schema_version": 1, "report_type": "dnai_fp512_kl720_valid_fiber_instance_audit",
        "scientific_scope": {"source_artifacts_verified": True},
        "source_fp512": {"path": str(fp.resolve())},
        "source_kl720": {"path": str(kl.resolve())},
    }
    invocations = []

    def fake_run(cmd, **kwargs):
        invocations.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(result_obj), stderr="")

    monkeypatch.setattr(inst.subprocess, "run", fake_run)
    report = inst.run_scientific_fiber_worker(fp, kl, scientific_python=interp)
    assert report == result_obj
    assert len(invocations) == 1
    cmd, kwargs = invocations[0]
    assert cmd[:4] == [str(interp.resolve()), "-B", "-m", "radiation_edge_ai.dnai_fiber_instances"]
    assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert kwargs["env"]["PYTHONPATH"].split(inst.os.pathsep)[0].endswith("src")


def test_worker_wrapper_rejects_science_error_and_non_json(tmp_path, monkeypatch) -> None:
    interp = tmp_path / "science.exe"
    interp.write_bytes(b"stub")
    monkeypatch.setattr(inst.subprocess, "run", lambda *_a, **_k:
                        SimpleNamespace(returncode=2, stdout="", stderr="source failure"))
    with pytest.raises(ControlPlaneError, match="source failure"):
        inst.run_scientific_fiber_worker(tmp_path / "fp", tmp_path / "kl", scientific_python=interp)
    monkeypatch.setattr(inst.subprocess, "run", lambda *_a, **_k:
                        SimpleNamespace(returncode=0, stdout="not-json", stderr=""))
    with pytest.raises(ControlPlaneError, match="invalid JSON"):
        inst.run_scientific_fiber_worker(tmp_path / "fp", tmp_path / "kl", scientific_python=interp)


def test_orchestration_reports_source_hashes_and_scope(tmp_path, monkeypatch) -> None:
    fp, kl = _make_two_records(tmp_path)
    records = [json.loads(fp.read_text(encoding="utf-8")), json.loads(kl.read_text(encoding="utf-8"))]
    source_hashes = [hashlib.sha256(fp.read_bytes()).hexdigest(), hashlib.sha256(kl.read_bytes()).hexdigest()]
    fp_fibers, kl_fibers = _example()
    reproduced = [
        {"all_count": 4, "valid_count": 3, "csv_reproduced": True,
         "fiber_ids": [3, 5, 7], "fibers": fp_fibers, "pixel_size_um": 0.26},
        {"all_count": 5, "valid_count": 2, "csv_reproduced": True,
         "fiber_ids": [71, 73], "fibers": kl_fibers, "pixel_size_um": 0.26},
    ]
    monkeypatch.setattr(inst, "_verified_field_pair", lambda *_p: (records, source_hashes, _identity()))
    monkeypatch.setattr(inst.importlib.metadata, "version", lambda name: "synthetic")
    items = iter(reproduced)
    monkeypatch.setattr(inst, "_reconstruct_verified_field", lambda *args: next(items))
    result = inst.compare_dnai_fiber_instances(fp, kl)
    assert result["primary"]["summary"]["n_matched"] == 2
    assert result["matched_input_sha256"]
    assert result["scientific_scope"]["model_inference_performed"] is False
    assert result["scientific_scope"]["unmatched_means_biologically_absent"] is False
    json.dumps(result, allow_nan=False)


def test_orchestration_rejects_modified_file_after_reconstruction(tmp_path, monkeypatch) -> None:
    fp, kl = _make_two_records(tmp_path)
    records = [json.loads(fp.read_text(encoding="utf-8")), json.loads(kl.read_text(encoding="utf-8"))]
    source_hashes = [hashlib.sha256(fp.read_bytes()).hexdigest(), hashlib.sha256(kl.read_bytes()).hexdigest()]
    monkeypatch.setattr(inst, "_verified_field_pair", lambda *_p: (records, source_hashes, _identity()))
    monkeypatch.setattr(inst.importlib.metadata, "version", lambda name: "synthetic")

    def alter(field, version):
        fp.write_text(fp.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return {"all_count": 0, "valid_count": 0, "csv_reproduced": True,
                "fiber_ids": [], "fibers": [], "pixel_size_um": 0.26}

    monkeypatch.setattr(inst, "_reconstruct_verified_field", alter)
    with pytest.raises(ControlPlaneError, match="changed during instance matching"):
        inst.compare_dnai_fiber_instances(fp, kl)


def test_reconstruction_replays_csv_under_frozen_version(tmp_path, monkeypatch) -> None:
    import csv
    import io
    import sys
    import types

    import numpy as np
    from PIL import Image

    seg_path = tmp_path / "segmentation.png"
    segmentation = np.zeros((1024, 1024), dtype=np.uint8)
    segmentation[10:13, 12:15] = 1
    Image.fromarray(segmentation).save(seg_path)
    fake = FakeFiber(0, length=10.0, ratio=1.5, edges={})

    class FakeSeries:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return list(self.values)

    class FakeTable:
        def __init__(self, records):
            self.records = records
            self.columns = list(records[0]) if records else []

        def __getitem__(self, key):
            return FakeSeries([row[key] for row in self.records])

        def reset_index(self, *, drop):
            assert drop is True
            return self

    original = FakeTable([{"Fiber ID": 2, "First analog (um)": 2.6,
                           "Second analog (um)": 1.3, "Ratio": 1.5}])
    csv_path = tmp_path / "fibers.csv"
    csv_path.write_text("Fiber ID,First analog (um),Second analog (um),Ratio\n2,2.6,1.3,1.5\n",
                        encoding="utf-8")

    class FakeCollection:
        def __len__(self):
            return 3

        def valid_copy(self):
            return self

        def __iter__(self):
            return iter([fake])

        def to_df(self, *, pixel_size, img_name):
            assert pixel_size == 0.26
            assert img_name == "1/tile_4.png"
            return original

    class FakeAll(FakeCollection):
        def valid_copy(self):
            return FakeCollection()

    # The production worker imports DNAi, OpenCV and pandas dynamically.
    # CI substitutes them here; no scientific dependency or GPU is required.
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_UNCHANGED = -1
    cv2.imdecode = lambda array, flags: np.asarray(Image.open(io.BytesIO(array.tobytes())))
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    pandas = types.ModuleType("pandas")

    def read_csv(stream, *, encoding):
        assert encoding == "utf-8-sig"
        decoded = io.TextIOWrapper(stream, encoding=encoding)
        rows = list(csv.DictReader(decoded))
        return FakeTable([{
            key: int(value) if key == "Fiber ID" else float(value)
            for key, value in row.items()
        } for row in rows])

    def assert_frame_equal(actual, expected, **kwargs):
        assert actual.records == expected.records

    pandas.read_csv = read_csv
    pandas.testing = types.SimpleNamespace(assert_frame_equal=assert_frame_equal)
    monkeypatch.setitem(sys.modules, "pandas", pandas)

    dnafiber = types.ModuleType("dnafiber")
    dnafiber.__path__ = []
    postprocess = types.ModuleType("dnafiber.postprocess")
    postprocess.refine_segmentation = lambda array: FakeAll()
    monkeypatch.setitem(sys.modules, "dnafiber", dnafiber)
    monkeypatch.setitem(sys.modules, "dnafiber.postprocess", postprocess)

    def binding(path):
        raw = path.read_bytes()
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}

    field = {
        "runtime": {"dnafiber_version": "synthetic"},
        "identity": {"key": "1/tile_4.png", "deployment_policy": {"pixel_size_um": 0.26}},
        "counts": {"n_fibers_all": 3, "n_fibers_valid": 1},
        "measurements": {"total_valid_length_um": 2.6},
        "artifacts": {"segmentation": binding(seg_path), "valid_fibers_csv": binding(csv_path)},
    }
    report = inst._reconstruct_verified_field(field, "synthetic")
    assert report["all_count"] == 3
    assert report["valid_count"] == 1
    assert report["fiber_ids"] == [2]
    assert report["csv_reproduced"] is True

    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "TAMPER", encoding="utf-8")
    with pytest.raises(ControlPlaneError, match="byte verification"):
        inst._reconstruct_verified_field(field, "synthetic")


def test_cli_instance_audit_json_and_human_mode(monkeypatch, capsys) -> None:
    from radiation_edge_ai import cli

    summary = {
        "schema_version": 1,
        "report_type": "dnai_fp512_kl720_valid_fiber_instance_audit",
        "source_fp512": {"field_id": "df1-test"},
        "source_kl720": {"field_id": "df2-test"},
        "shared_input": {"sample_id": "field-one"},
        "primary": {
            "summary": {"n_matched": 11, "n_reference_unmatched": 3,
                        "n_candidate_unmatched": 1, "f1": 0.8461538461538461},
            "matched_length_mae_um": 0.3187430143,
            "matched_ratio_mae": 0.0696598154,
            "length_decomposition_um": {
                "kl720_minus_fp512_total": -42.920334726568484,
                "matched_kl720_minus_fp512": 1.2107821048680236,
                "unmatched_fp512": 45.95111683143651,
                "unmatched_kl720": 1.82,
            },
        },
    }
    calls = []

    def fake_worker(fp, kl, *, scientific_python):
        calls.append((fp, kl, scientific_python))
        return summary

    monkeypatch.setattr(cli, "run_scientific_fiber_worker", fake_worker)
    cmd = [
        "compare-dnai-fibers", "--fp512", "floating.json", "--kl720", "physical.json",
        "--scientific-python", "science.exe",
    ]
    assert cli.main(cmd + ["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["primary"]["summary"]["n_matched"] == 11
    assert cli.main(cmd) == 0
    out = capsys.readouterr().out
    assert "11" in out and "-42.920" in out
    assert len(calls) == 2


def test_worker_wrapper_rejects_malformed_success_payload(tmp_path, monkeypatch) -> None:
    interpreter = tmp_path / "science.exe"
    interpreter.write_bytes(b"stub")
    for malformed in (
        {"schema_version": 1, "report_type": "dnai_fp512_kl720_valid_fiber_instance_audit",
         "scientific_scope": None, "source_fp512": "not-an-object"},
        ["not-an-object"],
    ):
        monkeypatch.setattr(inst.subprocess, "run", lambda *_a, **_kw:
                            SimpleNamespace(returncode=0, stdout=json.dumps(malformed), stderr=""))
        with pytest.raises(ControlPlaneError, match="incompatible audit report"):
            inst.run_scientific_fiber_worker(tmp_path / "fp", tmp_path / "kl",
                                             scientific_python=interpreter)
