import csv
import io
import runpy
import struct
import zipfile
from pathlib import Path

import pytest

from radiation_edge_ai.nasa_bps import multimodel_bie_comparison as scorer
from radiation_edge_ai.nasa_bps import neural16_audit as audit


@pytest.mark.parametrize("word", [-16384, -1, 0, 1, 3760, 16383])
def test_signed_native_dump_including_vendor_decimal_roundtrip(word):
    decoded = audit.fp32(word / 4096 / audit.SCALE)
    text = format(decoded, ".8f")
    received = audit.fp32(float(text))
    assert audit.native_value(struct.pack("<h", word), str(word).encode(), text.encode()) == (word, received)
    if abs(word) == 1:
        assert decoded != received
    if word:
        with pytest.raises(ValueError, match="radix-12"):
            audit.native_value(struct.pack("<h", word), str(word).encode(), format(decoded / 2, ".8f").encode())


@pytest.mark.parametrize("binary,text,actual", [(b"\x01", b"1", b"0.00000000"),
                                               (b"\x00\x40", b"16384", b"3.02884388"),
                                               (b"\xff\xff", b"65535", b"0.00000000"),
                                               (b"\x00\x00", b"0", b"NaN")])
def test_reject_invalid_native_scalar(binary, text, actual):
    with pytest.raises(ValueError):
        audit.native_value(binary, text, actual)


def sample_archive(path, change):
    inputs, predictions = [], []
    with zipfile.ZipFile(path, "x") as z:
        for i, code in enumerate([0, 3760]):
            source = {k: f"{k}_{i}" for k in ("sample_id", "sample_name", "source_name", "nucleus_key",
                                              "fold", "array_file", "array_sha256")}
            inputs.append(source)
            dump = format(audit.fp32(code / 4096 / audit.SCALE), ".8f")
            predictions.append(dict(source, row_in_chunk=str(i), chunk=path.stem, bie_fixed_code=str(code),
                                    bie_burden=str(audit.fp32(float(dump)))))
            prefix = f"res_kdp720_test/in_{i:06d}.txt/mode_720_piano/layer_output_burden_"
            z.writestr(prefix + "fx.bin", struct.pack("<h", 7520 if change == "word" and i else code))
            z.writestr(prefix + "fx.txt", str(code))
            z.writestr(prefix + "fl.txt", dump)
            z.writestr(prefix + "output_size.csv", "1\n1\n1\n" + ("2" if change == "shape" else "1"))
            for stream in ("stdout", "stderr"):
                z.writestr(f"console/{i+1}/{stream}", "")
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=predictions[0])
        writer.writeheader()
        writer.writerows(predictions)
        z.writestr("predictions.csv", buf.getvalue())
        z.writestr("jobs.tsv", "Seq\tExitval\tSignal\tJobRuntime\n1\t0\t0\t2.0\n2\t0\t%d\t1.5\n"
                   % (1 if change == "signal" else 0))
    return inputs, predictions


@pytest.mark.parametrize("change", ["none", "word", "signal", "order", "shape"])
def test_native_archive_checks_observed_words_job_exits_and_mapping(tmp_path, change):
    path = tmp_path / "0000_0001.zip"
    inputs, predictions = sample_archive(path, change)
    if change == "none":
        observed, timing = audit.read_archive(path, predictions, inputs)
        assert observed == [(0, 0.), (3760, 0.6950960159301758)] and timing == [2., 1.5]
    else:
        if change == "order":
            inputs.reverse()
        with pytest.raises(ValueError):
            audit.read_archive(path, predictions, inputs)


def test_independent_full_panel_rebuild_agrees_with_original_scorer():
    inputs, fixed, fp32, truth, manifest, observations = [], [], [], [], [], []
    fold = "fold_female_role"
    for source in ("BALBCF1", "C57BLF1"):
        for branch, dose in (("Fe", .82), ("X-ray", 1.)):
            for hour in (4, 24, 48):
                for d in (0, dose):
                    name = f"{source}_{branch}_{hour}_{d}"
                    label = 1.25 if d else 1.
                    burden = .99 if d else 1.
                    meta = dict(source_name=source, sample_name=name, dose_Gy=d, particle_type=branch,
                                hr_post_exposure=hour, strain=source, sex="female", fold=fold)
                    truth.append(dict(meta, nasa_avg_nfoci=label, n_development_nuclei=100, predicted_bag_mean=label))
                    for i in range(100):
                        ident = dict(meta, sample_id=f"{name}_{i}", nucleus_key=f"{name}_{i}_key", array_sha256="f"*64)
                        inputs.append(dict(ident, original_fp32=label, optimized_fp32=label))
                        manifest.append(dict(ident, cohort="development", **{fold: "test"}))
                        fp32.append(dict(ident, predicted_continuous_burden=label))
                        fixed.append(dict(ident, bie_burden=burden))
                        observations.append((0, burden))
    _, _, _, _, official = scorer.score_rows(fp32, fixed, truth, manifest, fold, inputs, .20)
    bags, deltas = audit.build_means(inputs, observations, truth)
    independent = audit.metrics(bags, deltas, .20)
    assert independent.keys() == scorer.V1_LIMITS.keys()
    assert independent == pytest.approx({k: official[k] for k in independent}, abs=1e-14)
    assert independent["strong_effect_sign_reversals"] == 12 and independent["four_hour_nonpositive"] == 4


def test_int8_and_int16_comparison_requires_identical_frozen_float_inputs():
    plot_script = Path(__file__).resolve().parents[1] / "scripts/analysis/report_resnet_precision_regression.py"
    compare = runpy.run_path(str(plot_script))["compare_tables"]
    row = dict(sample_id="1", nucleus_key="i", array_sha256="a"*64, sample_name="bag", source_name="source",
               fp32="0.5", optimized_fp32="0.49", bie="0.4")
    paired, = compare([row], [dict(row, bie="0.48")], "nucleus")
    assert paired["standard_int8_minus_fp32"] == pytest.approx(-0.1)
    assert paired["full_neural_int16_minus_fp32"] == pytest.approx(-0.02)
    with pytest.raises(ValueError, match="Mismatched FP32"):
        compare([row], [dict(row, fp32="0.51")], "nucleus")

