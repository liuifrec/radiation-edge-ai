"""Offline DNAi saved-window replay tests; never load a model or contact KL720."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from radiation_edge_ai import dnai_window_origin as origin
from radiation_edge_ai.control import ControlPlaneError


def _probabilities(bg: float, red: float, green: float, shape=(4, 4)) -> np.ndarray:
    assert bg + red + green == pytest.approx(1.0)
    p = np.empty((3,) + shape, dtype=np.float32)
    p[0] = bg
    p[1] = red
    p[2] = green
    return p


def _tiny_windows():
    # Two horizontally overlapping 4x4 windows covering one 4x6 field.
    fp = [
        {"window_index": 0, "x": 0, "y": 0,
         "probabilities": _probabilities(.6, .3, .1)},
        {"window_index": 1, "x": 2, "y": 0,
         "probabilities": _probabilities(.6, .3, .1)},
    ]
    kl = [
        {"window_index": 0, "x": 0, "y": 0,
         "probabilities": _probabilities(.7, .2, .1)},
        {"window_index": 1, "x": 2, "y": 0,
         "probabilities": _probabilities(.5, .4, .1)},
    ]
    weights = np.ones((4, 4), dtype=np.float32)

    def stitch(entries):
        total = np.zeros((3, 4, 6), dtype=np.float64)
        denom = np.zeros((4, 6), dtype=np.float64)
        for w in entries:
            x = w["x"]
            total[:, :, x:x + 4] += w["probabilities"] * weights[None]
            denom[:, x:x + 4] += weights
        return (total / denom).astype(np.float32)

    return fp, kl, stitch(fp), stitch(kl), weights


def test_pre_stitch_margin_decomposition_is_exact_and_not_causal() -> None:
    fp, kl, fp_saved, kl_saved, weights = _tiny_windows()
    yy = np.array([1, 1, 1, 1], dtype=np.int32)
    xx = np.array([1, 2, 3, 4], dtype=np.int32)
    result = origin._fixed_support_margin(
        fp, kl, fp_saved, kl_saved, weights,
        yy, xx, 1, seam_x=[2, 4], seam_y=[],
    )
    assert result["source_class_support_pixels"] == 4
    assert result["n_support_pixels_covered_by_1_window"] == 2
    assert result["n_support_pixels_covered_by_2_windows"] == 2
    assert result["n_support_pixels_covered_by_4_windows"] == 0
    assert result["n_multiwindow_pixels_with_opposite_delta_signs"] == 2
    assert result["saved_stitched_margin_delta_mean"] == pytest.approx(0.0, abs=1e-7)
    assert result["weighted_pre_stitch_margin_delta_mean"] == pytest.approx(0.0, abs=1e-7)
    assert result["maximum_absolute_margin_decomposition_residual"] < 1e-7
    assert [w["n_class_support_pixels_in_window"] for w in result["per_window"]] == [3, 3]
    assert [w["mean_pre_stitch_margin_delta"] for w in result["per_window"]] == [
        pytest.approx(-.2), pytest.approx(+.2)
    ]
    assert sum(w["mean_weighted_contribution_over_all_class_support_pixels"]
               for w in result["per_window"]) == pytest.approx(0.0, abs=1e-7)
    json.dumps(result, allow_nan=False)


def test_single_coverage_region_inherits_window_probability_shift() -> None:
    fp, kl, fp_saved, kl_saved, weights = _tiny_windows()
    result = origin._fixed_support_margin(
        fp, kl, fp_saved, kl_saved, weights,
        np.array([1]), np.array([1]), 1,
        seam_x=[2], seam_y=[],
    )
    assert result["n_support_pixels_covered_by_1_window"] == 1
    assert result["saved_stitched_margin_delta_mean"] == pytest.approx(-.2)
    assert len(result["per_window"]) == 1
    assert result["per_window"][0]["window_index"] == 0
    assert result["per_window"][0]["mean_pre_stitch_margin_delta"] == pytest.approx(-.2)


def test_stitched_margin_replay_tamper_fails_closed() -> None:
    fp, kl, fp_saved, kl_saved, weights = _tiny_windows()
    kl_saved = kl_saved.copy()
    kl_saved[1, 1, 1] += .03
    with pytest.raises(ControlPlaneError, match="does not reconcile"):
        origin._fixed_support_margin(
            fp, kl, fp_saved, kl_saved, weights,
            np.array([1]), np.array([1]), 1, seam_x=[2], seam_y=[],
        )


def test_paired_window_swapping_and_uncovered_support_rejected() -> None:
    fp, kl, fp_saved, kl_saved, weights = _tiny_windows()
    wrong = list(kl)
    wrong[0] = dict(kl[0], x=3)
    with pytest.raises(ControlPlaneError, match="positions/indices differ"):
        origin._fixed_support_margin(
            fp, wrong, fp_saved, kl_saved, weights,
            np.array([1]), np.array([1]), 1, seam_x=[2], seam_y=[],
        )
    with pytest.raises(ControlPlaneError, match="not covered"):
        origin._fixed_support_margin(
            fp[:1], kl[:1], fp_saved, kl_saved, weights,
            np.array([1]), np.array([5]), 1, seam_x=[2], seam_y=[],
        )


def test_window_seams_exclude_image_edges() -> None:
    windows = [{"x": i, "y": j} for i in (0, 256, 512) for j in (0, 256, 512)]
    assert origin._internal_window_edges(windows, "x") == [256, 512, 768]
    assert origin._internal_window_edges(windows, "y") == [256, 512, 768]
    with pytest.raises(ControlPlaneError, match="coordinate axis"):
        origin._internal_window_edges(windows, "z")


def test_frozen_softmax_tensor_contract_and_float32() -> None:
    logits = np.zeros((1, 3, 512, 512), dtype=np.float32)
    logits[:, 1] = 1.0
    probs = origin._softmax_frozen(logits)
    assert probs.shape == logits.shape
    assert probs.dtype == np.float32
    assert probs[0, 1, 0, 0] > probs[0, 0, 0, 0]
    assert probs[:, :, 0, 0].sum() == pytest.approx(1, abs=1e-7)
    with pytest.raises(ControlPlaneError, match="logits must be"):
        origin._softmax_frozen(logits.astype(np.float64))
    with pytest.raises(ControlPlaneError, match="logits must be"):
        origin._softmax_frozen(logits[:, :, :4, :4])
    logits[0, 0, 0, 0] = np.nan
    with pytest.raises(ControlPlaneError, match="logits must be"):
        origin._softmax_frozen(logits)


def test_frozen_window_bound_sha256_and_decoder_fail_closed(tmp_path: Path) -> None:
    logits = np.zeros((1, 3, 512, 512), dtype=np.float32)
    path = tmp_path / "window.npy"
    with path.open("wb") as handle:
        np.save(handle, logits, allow_pickle=False)
    data = path.read_bytes()
    meta = {
        "output": {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                   "size_bytes": len(data)}
    }
    assert np.array_equal(origin._verified_window_output(meta), logits)
    path.write_bytes(data + b"TAMPER")
    with pytest.raises(ControlPlaneError, match="SHA256/size"):
        origin._verified_window_output(meta)
    meta["output"]["size_bytes"] = len(data) + 6
    meta["output"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ControlPlaneError, match="Cannot decode frozen window"):
        # A valid digest does not make a malformed raw tensor safe.
        path.write_bytes(b"not a numpy tensor")
        newdata = path.read_bytes()
        meta["output"]["size_bytes"] = len(newdata)
        meta["output"]["sha256"] = hashlib.sha256(newdata).hexdigest()
        origin._verified_window_output(meta)


def test_stitch_replay_verified_all_nine_outputs(tmp_path: Path, monkeypatch) -> None:
    from radiation_edge_ai import dnai_paired_fields as paired

    logits = np.zeros((1, 3, 512, 512), dtype=np.float32)
    logits[:, 2] = .3
    saved_window = tmp_path / "frozen_window.npy"
    np.save(saved_window, logits, allow_pickle=False)
    raw = saved_window.read_bytes()
    binding = {
        "path": str(saved_window), "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    coords = [(y, x) for y in (0, 256, 512) for x in (0, 256, 512)]
    windows = [
        {"window_index": i, "x": x, "y": y, "input": {
            "sha256": "a" * 64, "size_bytes": 2048,
        }, "output": dict(binding)}
        for i, (y, x) in enumerate(coords)
    ]
    identity = {"windows": [
        {"window_index": i, "x": x, "y": y,
         "tensor_sha256": "a" * 64, "tensor_size_bytes": 2048}
        for i, (y, x) in enumerate(coords)
    ]}
    field = {"identity": identity, "window_outputs": windows}
    probabilities = origin._softmax_frozen(logits)[0]
    saved = np.broadcast_to(probabilities[:, 0:1, 0:1], (3, 1024, 1024)).copy()
    packed = io.BytesIO()
    np.savez_compressed(packed, probabilities=saved)
    monkeypatch.setattr(paired, "_verified_bytes", lambda *_args: packed.getvalue())
    result = origin._replay_field_windows(field, np.ones((512, 512), dtype=np.float32))
    assert result["bitwise_identical_to_saved"] is True
    assert result["max_abs_replay_error"] == 0
    assert len(result["windows"]) == 9
    assert result["windows"][0]["x"] == 0
    assert result["windows"][8]["x"] == 512

    # Even fully bound logits are not accepted if the stitched NPZ differs.
    saved[1, 17, 18] += .01
    packed = io.BytesIO()
    np.savez_compressed(packed, probabilities=saved)
    with pytest.raises(ControlPlaneError, match="does not reproduce"):
        origin._replay_field_windows(field, np.ones((512, 512), dtype=np.float32))


def test_full_precision_stitch_replay_uses_float32_product_order(
    tmp_path: Path, monkeypatch,
) -> None:
    """Original workers multiply float32 then accumulate in float64."""
    from radiation_edge_ai import dnai_paired_fields as paired

    rng = np.random.default_rng(11)
    importance = np.linspace(.019, .943, 512 * 512, dtype=np.float32).reshape(512, 512)
    numerator = np.zeros((3, 1024, 1024), dtype=np.float64)
    denominator = np.zeros((1024, 1024), dtype=np.float64)
    outputs, identity_windows = [], []
    positions = [(y, x) for y in (0, 256, 512) for x in (0, 256, 512)]
    for i, (y, x) in enumerate(positions):
        logits = rng.standard_normal((1, 3, 512, 512)).astype(np.float32)
        source = tmp_path / f"window_{i}.npy"
        np.save(source, logits, allow_pickle=False)
        raw = source.read_bytes()
        probabilities = origin._softmax_frozen(logits)[0]
        numerator[:, y:y + 512, x:x + 512] += probabilities * importance[None]
        denominator[y:y + 512, x:x + 512] += importance
        outputs.append({
            "window_index": i, "y": y, "x": x,
            "input": {"sha256": "a" * 64, "size_bytes": 2048},
            "output": {"path": str(source),
                       "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)},
        })
        identity_windows.append({
            "window_index": i, "y": y, "x": x,
            "tensor_sha256": "a" * 64, "tensor_size_bytes": 2048,
        })
    expected = (numerator / denominator).astype(np.float32)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, probabilities=expected)
    monkeypatch.setattr(paired, "_verified_bytes", lambda *_a: buffer.getvalue())
    result = origin._replay_field_windows(
        {"identity": {"windows": identity_windows}, "window_outputs": outputs},
        importance,
    )
    assert result["bitwise_identical_to_saved"] is True
    assert result["max_abs_replay_error"] == 0.0


def test_scientific_worker_wrapper_and_failure(tmp_path: Path, monkeypatch) -> None:
    python = tmp_path / "science.exe"
    python.write_bytes(b"stub")
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("{}", encoding="utf-8")
    b.write_text("{}", encoding="utf-8")
    fake = {
        "schema_version": 1, "report_type": origin.REPORT_TYPE,
        "source_fp512": {"path": str(a.resolve())},
        "source_kl720": {"path": str(b.resolve())},
        "frozen_replay": {"both_replays_passed": True},
        "scientific_scope": {"frozen_full_artifact_verification_before_and_after": True},
    }
    called = []

    def fake_run(command, **options):
        called.append((command, options))
        return SimpleNamespace(returncode=0, stdout=json.dumps(fake), stderr="")

    monkeypatch.setattr(origin.subprocess, "run", fake_run)
    assert origin.run_scientific_window_origin_worker(
        a, b, scientific_python=python
    ) == fake
    assert called[0][0][:4] == [
        str(python.resolve()), "-B", "-m", "radiation_edge_ai.dnai_window_origin"
    ]
    assert called[0][1]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    fake["frozen_replay"]["both_replays_passed"] = False
    with pytest.raises(ControlPlaneError, match="incompatible report"):
        origin.run_scientific_window_origin_worker(a, b, scientific_python=python)
    monkeypatch.setattr(origin.subprocess, "run", lambda *_a, **_kw:
                        SimpleNamespace(returncode=2, stderr="bad frozen digest", stdout=""))
    with pytest.raises(ControlPlaneError, match="bad frozen digest"):
        origin.run_scientific_window_origin_worker(a, b, scientific_python=python)


def test_cli_window_origin_displays_json_and_human(monkeypatch, capsys) -> None:
    from radiation_edge_ai import cli

    example = {
        "schema_version": 1, "report_type": origin.REPORT_TYPE,
        "frozen_replay": {
            "fp512_max_absolute_probability_error": 0.0,
            "kl720_max_absolute_probability_error": 1e-8,
        },
        "historical_instance_matching": {"primary": {"summary": {"n_matched": 11}}},
        "per_fiber": {"fp512_source": [
            {"historical_primary_matched": False, "local_fiber_id": 1, "per_class": [{
                "class_id": 1, "saved_stitched_margin_delta_mean": -.8,
                "n_support_pixels_covered_by_1_window": 7,
                "n_support_pixels_covered_by_2_windows": 5,
                "n_support_pixels_covered_by_4_windows": 0,
            }]},
        ], "kl720_source": []},
    }
    monkeypatch.setattr(cli, "run_scientific_window_origin_worker",
                        lambda *_a, **_k: example)
    cmd = [
        "audit-dnai-window-origin", "--fp512", "df1.json",
        "--kl720", "df2.json", "--scientific-python", "dnai311.exe",
    ]
    assert cli.main(cmd + ["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["report_type"] == origin.REPORT_TYPE
    assert cli.main(cmd) == 0
    human = capsys.readouterr().out
    assert "matches: 11" in human
    assert "class=1" in human
    assert "No inference" in human
