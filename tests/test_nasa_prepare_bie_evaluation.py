import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from radiation_edge_ai.nasa_bps import prepare_multimodel_bie_evaluation as prep


def test_fp32_sessions_read_exact_staged_tensor(tmp_path):
    (tmp_path / "tensors").mkdir()
    native = np.ones((2, 11, 7), dtype=np.float32)
    packed = np.zeros((3, 256, 256), dtype=np.float32)
    packed[:2, 122:133, 124:131] = native
    row = dict(sample_id="N1", nucleus_key="K1", sample_name="B1", source_name="S1",
               strain="C57BL", sex="female", particle_type="Fe", dose_Gy="0.82", hr_post_exposure="4")
    sessions = []

    def predict(output_names, inputs):
        assert output_names == ["burden"]
        sessions.append(inputs["input"])
        return [np.array([0.25 * len(sessions)], dtype=np.float32)]

    fake = SimpleNamespace(run=predict)
    result = prep.stage_row(tmp_path, 7, row, native, fake, fake, np,
                            SimpleNamespace(pad_and_pack=lambda pixels: packed))
    path = tmp_path / result["array_file"]
    saved = np.load(path, allow_pickle=False)
    assert path.name == "0007.npy"
    assert result["array_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["original_fp32"] == 0.25 and result["optimized_fp32"] == 0.5
    assert (result["native_height"], result["native_width"]) == (11, 7)
    assert all(np.array_equal(a, saved) for a in sessions)
    with pytest.raises(FileExistsError):
        prep.stage_row(tmp_path, 7, row, native, fake, fake, np,
                       SimpleNamespace(pad_and_pack=lambda pixels: packed))


@pytest.fixture
def cohort():
    manifest, calibration, nuclei, bags, reference = [], [], [], [], []
    sources = ("BALBCF1", "C57BLF1", "M1", "M2", "M3", "M4")
    for i, src in enumerate(sources):
        member = i < 2
        for b in range(12):
            bag = dict(source_name=src, sample_name=f"{src}_{b}", strain="strain", sex="sex",
                       particle_type="Fe", dose_Gy="0.82", hr_post_exposure="4")
            if member:
                bags.append(dict(bag, fold=prep.ptq.FOLD, n_development_nuclei="100", nasa_avg_nfoci="2.5"))
            reference.append(dict(bag, nasa_avg_nfoci="2.5"))
            for n in range(100):
                key = f"{src}-{b}-{n}"
                row = dict(bag, sample_id=key, nucleus_key=key, fitc_filename=key+".tif",
                           dapi_filename=key+".tif", cohort="development")
                row.update({f: "test" if j == i // 2 else "train" for j, f in enumerate(prep.mm.FOLD_FIELDS)})
                manifest.append(row)
                if member:
                    nuclei.append(dict(row, fold=prep.ptq.FOLD))
                elif n < 4:
                    calibration.append(dict(row))
    split = dict(test_sources=list(sources[:2]), train_sources=list(sources[2:]), train_bags=48, test_bags=24)
    return manifest, calibration, nuclei, bags, reference, split


def test_preparation_selects_source_held_out_members_in_frozen_prediction_order(cohort):
    cohort[2].reverse()
    rows = prep.select_held_out(*cohort)
    assert [r["sample_id"] for r in rows] == [r["sample_id"] for r in cohort[2]]
    assert len(rows) == 2400


def test_preparation_rejects_calibration_overlap_and_frozen_target_drift(cohort):
    calibration = cohort[1]
    calibration[0]["sample_id"] = cohort[2][0]["sample_id"]
    with pytest.raises(ValueError, match="overlaps"):
        prep.select_held_out(*cohort)
    calibration[0]["sample_id"] = calibration[0]["nucleus_key"]
    cohort[4][0]["nasa_avg_nfoci"] = "2.51"
    with pytest.raises(ValueError, match="NASA target"):
        prep.select_held_out(*cohort)


def test_maps_legacy_archive_to_relocated_data(tmp_path):
    assert prep.recorded_data_path(tmp_path, r"D:\radiation-edge-ai-data\data\train.csv") == (
        tmp_path / "data" / "train.csv"
    )


@pytest.mark.parametrize("path", [r"D:\radiation-edge-ai-data\..\outside",
                                  r"D:\other\file", r"D:\radiation-edge-ai-data\data\sample:stream",
                                  r"\\server\share\input.csv", r"D:\radiation-edge-ai-data\AUX.txt",
                                  r"D:\radiation-edge-ai-data\sample."])
def test_rejects_external_recorded_data_path(tmp_path, path):
    with pytest.raises(ValueError):
        prep.recorded_data_path(tmp_path, path)
