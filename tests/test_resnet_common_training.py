import importlib.util
from pathlib import Path

import pytest


def test_common_training_panel_binds_role_and_exact_input_identity():
    path = Path(__file__).resolve().parents[1] / "scripts/windows/replay_resnet_common_training.py"
    spec = importlib.util.spec_from_file_location("resnet_common_capture", path)
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    sources = ["BALBCM1", "BALBCM2", "C57BLM1", "C57BLM3"]
    calibration = [
        dict(sample_id=str(i), source_name=sources[i % 4], array_sha256=str(i),
             fold_female_role="train", dataset_role="development_train_ptq_calibration",
             cohort="development", calibration_fold="fold_female_role")
        for i in range(192)
    ]
    panel = {"rows": [dict(r, panel="confirmation") for r in calibration[:24]]}
    assert capture.training_rows(panel, calibration) == panel["rows"]
    panel["rows"][0]["array_sha256"] = "wrong"
    with pytest.raises(ValueError, match="Mismatched training case"):
        capture.training_rows(panel, calibration)
    panel["rows"][0]["array_sha256"] = calibration[0]["array_sha256"]
    panel["rows"][0]["fold_female_role"] = calibration[0]["fold_female_role"] = "test"
    with pytest.raises(ValueError, match="Mismatched training case"):
        capture.training_rows(panel, calibration)
