import csv
import json
import os
import zipfile

import numpy as np
import pytest

from radiation_edge_ai.nasa_bps import multimodel_bie_inference as worker


def test_prepared_tensor_uses_verified_uncast_float32(tmp_path):
    folder = tmp_path / "tensors"
    folder.mkdir()
    path = folder / "0001.npy"
    array = np.zeros(worker.SHAPE, np.float32)
    array[:, 1, 10:100, 30:130] = np.float32(0.1234567)
    np.save(path, array)
    row = {"array_file": "tensors/0001.npy", "array_sha256": worker.sha256(path)}
    assert worker.tensor(tmp_path, row).tobytes() == array.tobytes()
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        worker.tensor(tmp_path, row)
    row["array_file"] = "../tensors/0001.npy"
    with pytest.raises(ValueError, match="tensor path"):
        worker.tensor(tmp_path, row)


def test_stale_or_incomplete_native_process_logs_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("PARALLEL", "--jobs 1")

    def fx(model, platform, inputs, input_nodes, out_node_shape, d_ioinfo, p_working):
        assert platform == 720 and set(inputs) == {"input"}
        assert os.environ["PARALLEL"].startswith("--jobs 3 --results")
        (p_working / "jobs.tsv").write_text("Seq\tExitval\tSignal\n1\t0\t0\n1\t0\t0\n")
        return {"burden": [[1], [2]]}, {"burden": [[1], [2]]}

    with pytest.raises(ValueError, match="vendor job log"):
        worker.native_batch(fx, "dummy.bie", (["input"], ["burden"], {"burden": (1,)}, {}),
                            [None, None], tmp_path, 3)
    assert os.environ["PARALLEL"] == "--jobs 1"


@pytest.mark.parametrize("result,n,fixed", [({"burden": [[float("nan")]]}, 1, False),
                                           ({"burden": [[2.125]]}, 1, True),
                                           ({"burden": [[1, 2]]}, 1, False),
                                           ({"other": [[1]]}, 1, False),
                                           ({"burden": [[1]]}, 2, False)])
def test_scalar_output_contract(result, n, fixed):
    with pytest.raises(ValueError):
        worker.scalar_outputs(result, n, fixed)


def test_preparation_is_complete_identity_bound_and_development_only(tmp_path):
    rows = []
    for i in range(2400):
        rows.append(dict(fold=worker.FOLD, sample_id=str(i), nucleus_key="n" + str(i),
                         source_name="BALBCF1" if i < 1200 else "C57BLF1",
                         sample_name="bag" + str(i // 100), array_file="tensors/{:04d}.npy".format(i)))
    path = tmp_path / "inputs.csv"
    prep = tmp_path / "preparation.json"
    policy = worker.V1_POLICY_SHA

    def write():
        with path.open("w", newline="", encoding="utf-8") as stream:
            table = csv.DictWriter(stream, rows[0].keys())
            table.writeheader()
            table.writerows(rows)
        prep.write_text(json.dumps(dict(worker.EXPECTED, status="PREPARATION_COMPLETE",
                                        policy_sha256=policy, evaluated_nuclei=2400,
                                        input_manifest_sha256=worker.sha256(path))))

    write()
    actual, _ = worker.load_inputs(tmp_path, worker.sha256(prep), policy)
    assert [r["sample_id"] for r in actual] == [r["sample_id"] for r in rows]
    rows[0]["source_name"] = "training_source"
    write()
    with pytest.raises(ValueError, match="held-out bags|held-out panel"):
        worker.load_inputs(tmp_path, worker.sha256(prep), policy)


def test_archives_raw_output_and_logs_and_omits_large_tensors(tmp_path):
    native = tmp_path / "res_candidate" / "in_000002.txt" / "mode_720_piano"
    native.mkdir(parents=True)
    (native / "layer_output_burden_fx.txt").write_bytes(b"62\n")
    (native / "layer_input_input_fl.txt").write_bytes(b"input dump")
    console = tmp_path / "console" / "3"
    console.mkdir(parents=True)
    (console / "stderr").write_bytes(b"raw stderr")
    out = tmp_path / "slice.zip"
    with zipfile.ZipFile(out, "x") as archive:
        worker.capture_chunk(tmp_path, archive)
    with zipfile.ZipFile(out) as archive:
        assert archive.read("res_candidate/in_000002.txt/mode_720_piano/layer_output_burden_fx.txt") == b"62\n"
        assert archive.read("console/3/stderr") == b"raw stderr"
        assert not any("layer_input_" in name for name in archive.namelist())


def test_every_returned_scalar_matches_numeric_native_input_ordinal(tmp_path):
    codes = list(range(24))
    decoded = [float(np.float32(i / 20)) for i in codes]
    for i in codes:
        folder = tmp_path / "res_kdp720_x" / "in_{:06d}.txt".format(i) / "mode_720_piano"
        folder.mkdir(parents=True)
        (folder / "layer_output_burden_fl.txt").write_text(str(i / 20))
        (folder / "layer_output_burden_fx.txt").write_text(str(i))
        log = tmp_path / "console" / str(i + 1)
        log.mkdir(parents=True)
        (log / "stdout").touch()
        (log / "stderr").touch()
    worker.verify_native_dumps(tmp_path, decoded, codes)
    decoded[2], decoded[12] = decoded[12], decoded[2]
    with pytest.raises(ValueError, match="ordering mismatch"):
        worker.verify_native_dumps(tmp_path, decoded, codes)
