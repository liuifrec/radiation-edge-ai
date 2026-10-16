"""Synthetic, offline DNAi failure-mode audit tests; no DNAi or KL720 hardware."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from radiation_edge_ai import dnai_failure_modes as failure
from radiation_edge_ai.control import ControlPlaneError


def _probs(raw: np.ndarray) -> np.ndarray:
    probabilities = np.full((3,) + raw.shape, 0.03, dtype=np.float32)
    for class_id in (0, 1, 2):
        probabilities[class_id][raw == class_id] = 0.94
    return probabilities


def _masks() -> tuple[np.ndarray, np.ndarray]:
    fp = np.zeros((64, 64), dtype=np.uint8)
    fp[12, 0:4] = 1
    fp[12, 4:8] = 2
    kl = fp.copy()
    kl[12, 0:4] = 0
    return fp, kl


@dataclass
class FakeFiber:
    bbox: tuple[int, int, int, int]
    data: np.ndarray
    length: float = 12
    ratio: float = 1.0

    def bbox_iou(self, other: "FakeFiber") -> float:
        x, y, w, h = self.bbox
        X, Y, W, H = other.bbox
        intersection = max(0, min(x + w, X + W) - max(x, X)) * max(
            0, min(y + h, Y + H) - max(y, Y)
        )
        return intersection / (w * h + W * H - intersection)


def _fiber(bbox=(0, 10, 8, 5)) -> FakeFiber:
    data = np.zeros((bbox[3], bbox[2]), dtype=np.uint8)
    data[2, 0:4] = 1
    data[2, 4:8] = 2
    return FakeFiber(bbox, data)


def _measure(
    *,
    side: str = "fp512",
    matched_row: int | None = None,
    fp_raw: np.ndarray | None = None,
    kl_raw: np.ndarray | None = None,
    fiber: FakeFiber | None = None,
) -> dict:
    baseline, physical = _masks()
    fp_raw = baseline if fp_raw is None else fp_raw
    kl_raw = physical if kl_raw is None else kl_raw
    return failure._measure_fiber(
        fiber or _fiber(), side=side, row=0, local_fiber_id=17,
        matched_row=matched_row,
        fp_final=failure._dilate_three_class_argmax(fp_raw),
        kl_final=failure._dilate_three_class_argmax(kl_raw),
        fp_raw=fp_raw, kl_raw=kl_raw,
        fp_probs=_probs(fp_raw), kl_probs=_probs(kl_raw),
    )


def test_upstream_grayscale_dilation_green_can_replace_red() -> None:
    raw = np.zeros((7, 7), dtype=np.uint8)
    raw[3, 2] = 1
    raw[3, 3] = 2
    final = failure._dilate_three_class_argmax(raw)
    assert final[3, 2] == 2  # Not independent class-specific dilation
    assert final[3, 1] == 1
    assert final[2, 3] == 2
    assert final[0, 0] == 0
    raw_again = failure._replay_saved_morphology(_probs(raw), final, "synthetic")
    assert np.array_equal(raw_again, raw)


def test_morphology_gate_rejects_one_changed_pixel() -> None:
    raw, _ = _masks()
    final = failure._dilate_three_class_argmax(raw)
    corrupted = final.copy()
    corrupted[40, 40] = 1
    with pytest.raises(ControlPlaneError, match="1 disagreeing pixels"):
        failure._replay_saved_morphology(_probs(raw), corrupted, "FP512")
    with pytest.raises(ControlPlaneError, match="probability contract"):
        failure._replay_saved_morphology(_probs(raw).astype(np.float64), final, "FP512")


def test_border_fiber_red_disappearance_green_retention_and_direct_margins() -> None:
    report = _measure()
    assert report["bbox_xywh"] == [0, 10, 8, 5]
    assert report["padded_roi_xyxy_exclusive"] == [0, 6, 12, 19]
    assert report["boundary_distance_px"] == 0
    assert report["within_32px_of_image_boundary"] is True
    assert report["primary_matched"] is False
    assert report["fiber_colored_support_pixels"] == 8
    red, green = report["per_class"]
    assert red["source_fiber_support_pixels"] == 4
    assert red["source_raw_same_class_on_support"] == 4
    assert red["target_raw_same_class_on_support"] == 0
    assert red["target_final_same_class_on_support"] == 0
    assert red["target_final_class_absent_on_support"] is True
    assert red["target_final_class_absent_in_padded_roi"] is True
    assert red["eight_connected_segmentation_component_proxy"]["target_raw"]["roi_components"] == 0
    assert green["target_final_same_class_on_support"] == 4
    assert green["target_final_class_absent_on_support"] is False
    assert report["target_raw_foreground_retention"] == 0.5
    assert report["target_final_foreground_retention"] == 0.625
    margin = red["probability_margins"]["against_background"]
    assert margin["fp512"]["mean"] == pytest.approx(0.91)
    assert margin["kl720"]["mean"] == pytest.approx(-0.91)
    assert margin["delta_kl720_minus_fp512_mean"] == pytest.approx(-1.82)
    assert red["probability_margins"]["against_strongest_competitor"]["kl720"][
        "fraction_nonpositive"
    ] == 1.0
    json.dumps(report, allow_nan=False)


def test_raw_final_dilation_can_change_class_retention_not_prove_fiber() -> None:
    fp_raw, kl_raw = _masks()
    kl_raw[12, 1] = 1  # One residual red pixel expands under dilation
    report = _measure(fp_raw=fp_raw, kl_raw=kl_raw)
    red = report["per_class"][0]
    assert red["target_raw_same_class_on_support"] == 1
    assert red["target_final_same_class_on_support"] == 3
    assert red["raw_to_dilated_target_class_pixel_delta_on_support"] == 2
    assert red["target_final_class_absent_on_support"] is False


def test_fragmentation_is_segmentation_proxy_with_source_overlap() -> None:
    support = np.zeros((5, 9), dtype=bool)
    support[2, 1:8] = True
    target = np.zeros((5, 9), dtype=bool)
    target[2, 1:3] = True
    target[2, 6:8] = True
    metrics = failure._components_touching_source(target, support)
    assert metrics == {"roi_components": 2, "components_touching_source_support": 2}
    target[1, 3:6] = True  # Diagonally adjacent pixels bridge the components
    assert failure._components_touching_source(target, support)["roi_components"] == 1


def test_inverted_source_orientation_and_no_fabricated_margin() -> None:
    fp_raw = np.zeros((128, 128), dtype=np.uint8)
    kl_raw = fp_raw.copy()
    kl_raw[57, 50:54] = 1
    kl_raw[57, 54:58] = 2
    fiber = _fiber((50, 55, 8, 5))
    report = _measure(side="kl720", fp_raw=fp_raw, kl_raw=kl_raw, fiber=fiber)
    assert report["source_side"] == "kl720"
    assert report["target_side"] == "fp512"
    assert report["boundary_distance_px"] == 50
    assert report["within_32px_of_image_boundary"] is False
    assert all(x["target_final_class_absent_on_support"] for x in report["per_class"])
    assert report["per_class"][0]["probability_margins"]["against_background"][
        "delta_kl720_minus_fp512_mean"
    ] == pytest.approx(1.82)


def test_invalid_bbox_support_and_shapes_fail_closed() -> None:
    baseline, kl = _masks()
    invalid = _fiber((0, 10, 8, 5))
    invalid.bbox = (-1, 10, 8, 5)
    with pytest.raises(ControlPlaneError, match="field geometry"):
        _measure(fiber=invalid)
    invalid.bbox = (0, 10, 8, 4)
    with pytest.raises(ControlPlaneError, match="field geometry"):
        _measure(fiber=invalid)
    with pytest.raises(ControlPlaneError, match="shapes do not agree"):
        failure._measure_fiber(
            _fiber(), side="fp512", row=0, local_fiber_id=0,
            matched_row=None, fp_final=baseline, kl_final=kl[:10],
            fp_raw=baseline, kl_raw=kl,
            fp_probs=_probs(baseline), kl_probs=_probs(kl),
        )


def test_boundary_association_keeps_denominators_and_no_significance() -> None:
    rows = [
        {"boundary_distance_px": distance, "primary_matched": matched, "per_class": []}
        for distance, matched in [(0, False), (2, False), (32, False),
                                   (33, True), (50, True)]
    ]
    unmatched = failure._boundary_group([row for row in rows if not row["primary_matched"]])
    matched = failure._boundary_group([row for row in rows if row["primary_matched"]])
    assert unmatched == {"n": 3, "n_within_32px": 3, "distances_px": [0, 2, 32]}
    assert matched == {"n": 2, "n_within_32px": 0, "distances_px": [33, 50]}
    a = _measure()
    assert failure._disappearance_summary([a]) == {
        "unmatched_source_classes_tested": 2,
        "n_zero_target_class_on_support": 1,
        "n_zero_target_class_in_padded_roi": 1,
    }


def test_end_to_end_reads_bound_arrays_and_preserves_json(tmp_path, monkeypatch) -> None:
    from radiation_edge_ai import dnai_fiber_instances as inst
    from radiation_edge_ai import dnai_paired_fields as paired

    fp, kl = tmp_path / "df1.json", tmp_path / "df2.json"
    fp.write_text('{"schema_version":1}', encoding="utf-8")
    kl.write_text('{"schema_version":2}', encoding="utf-8")
    original_bytes = (fp.read_bytes(), kl.read_bytes())
    original_hashes = [hashlib.sha256(value).hexdigest() for value in original_bytes]
    identity = {
        "sample_id": "1__tile_4", "key": "tile_4", "image_index": 0,
        "dnai_commit": "fcf20c7d6eb385675ff7d07da4fdf471589ce0cf",
        "validation_manifest_sha256": "a" * 64,
    }
    fields = [
        {"field_id": "df1-test", "runtime": {"dnafiber_version": "synthetic"}},
        {"field_id": "df2-test", "runtime": {"dnafiber_version": "synthetic"}},
    ]
    monkeypatch.setattr(inst, "_verified_field_pair", lambda *_p:
                        (fields, original_hashes, identity))
    monkeypatch.setattr(failure.importlib.metadata, "version", lambda *_a: "synthetic")
    source_fiber = _fiber()
    target_fiber = _fiber()
    records = [
        {"fibers": [source_fiber], "fiber_ids": [17],
         "pixel_size_um": 0.26, "csv_reproduced": True},
        {"fibers": [target_fiber], "fiber_ids": [19],
         "pixel_size_um": 0.26, "csv_reproduced": True},
    ]
    monkeypatch.setattr(inst, "_reconstruct_verified_field",
                        lambda field, version: records[fields.index(field)])
    fp_raw, kl_raw = _masks()
    segmentations = {
        id(fields[0]): failure._dilate_three_class_argmax(fp_raw),
        id(fields[1]): failure._dilate_three_class_argmax(kl_raw),
    }
    monkeypatch.setattr(failure, "_read_saved_segmentation",
                        lambda field: segmentations[id(field)])
    probs = iter((_probs(fp_raw), _probs(kl_raw)))
    monkeypatch.setattr(paired, "_verified_bytes", lambda *_a: b"frozen")
    monkeypatch.setattr(paired, "_load_probabilities", lambda *_a: next(probs))
    checked = []
    monkeypatch.setattr(inst, "verify_target",
                        lambda path, *, check_artifacts:
                        checked.append((path, check_artifacts))
                        or {"kind": "dnai_fiber_field_record", "ok": True})
    report = failure.audit_dnai_failure_modes(fp, kl)
    assert report["report_type"] == failure.REPORT_TYPE
    assert report["source_fp512"]["sha256"] == original_hashes[0]
    assert report["historical_instance_matching"]["primary"]["summary"]["n_matched"] == 1
    assert report["morphology_reproduced_exactly"] == {"fp512": True, "kl720": True}
    assert report["scientific_scope"]["model_inference_performed"] is False
    assert report["scientific_scope"]["quantization_or_stitching_cause_established"] is False
    assert checked == [(fp.resolve(), True), (kl.resolve(), True)]
    assert (fp.read_bytes(), kl.read_bytes()) == original_bytes
    assert sorted(p.name for p in tmp_path.iterdir()) == ["df1.json", "df2.json"]
    json.dumps(report, allow_nan=False)


def test_final_full_artifact_verification_fails_closed(tmp_path, monkeypatch) -> None:
    from radiation_edge_ai import dnai_fiber_instances as inst

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("{}", encoding="utf-8")
    b.write_text("{}", encoding="utf-8")
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (a, b)]
    # If full validation fails before reconstruction, the existing gate stops first.
    monkeypatch.setattr(inst, "_verified_field_pair", lambda *_p:
                        (_ for _ in ()).throw(ControlPlaneError("verification failed")))
    with pytest.raises(ControlPlaneError, match="verification failed"):
        failure.audit_dnai_failure_modes(a, b)
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in (a, b)] == before


def test_worker_enforces_scientific_python_and_verified_morphology(tmp_path, monkeypatch) -> None:
    interpreter = tmp_path / "science.exe"
    interpreter.write_bytes(b"stub")
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("{}", encoding="utf-8")
    b.write_text("{}", encoding="utf-8")
    report = {
        "schema_version": 1,
        "report_type": failure.REPORT_TYPE,
        "source_fp512": {"path": str(a.resolve())},
        "source_kl720": {"path": str(b.resolve())},
        "morphology_reproduced_exactly": {"fp512": True, "kl720": True},
        "scientific_scope": {"sources_and_artifacts_verified_before_and_after": True},
    }
    captured = []

    def process(cmd, **kw):
        captured.append((cmd, kw))
        return SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr="")

    monkeypatch.setattr(failure.subprocess, "run", process)
    assert failure.run_scientific_failure_mode_worker(
        a, b, scientific_python=interpreter
    ) == report
    assert captured[0][0][:4] == [
        str(interpreter.resolve()), "-B", "-m", "radiation_edge_ai.dnai_failure_modes"
    ]
    assert captured[0][1]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    report["morphology_reproduced_exactly"]["kl720"] = False
    with pytest.raises(ControlPlaneError, match="incompatible report"):
        failure.run_scientific_failure_mode_worker(a, b, scientific_python=interpreter)


def test_worker_failure_does_not_claim_success(tmp_path, monkeypatch) -> None:
    python = tmp_path / "scientific.exe"
    python.write_bytes(b"stub")
    monkeypatch.setattr(
        failure.subprocess, "run",
        lambda *_a, **_kw: SimpleNamespace(
            returncode=2, stderr="morphology reproduction error", stdout=""
        ),
    )
    with pytest.raises(ControlPlaneError, match="morphology reproduction error"):
        failure.run_scientific_failure_mode_worker(
            tmp_path / "a", tmp_path / "b", scientific_python=python
        )
