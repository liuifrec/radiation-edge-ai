import io
import zipfile

import numpy as np
import pytest

from radiation_edge_ai.nasa_bps import neural16_native as native
from radiation_edge_ai.nasa_bps import neural16_regression as study


def scalar():
    return {"node": "/Gather", "width": 15, "radix": 12, "scale": 1.3206359148025513,
            "transport": {"bitw": 16, "radix": [13], "name": "burden", "data_format": "8W1C16B",
                          "scale": [1.3206359148025513], "shape": [1, 1, 1, 1], "onnx_shape": [1]}}


@pytest.mark.parametrize("field,value", [("radix", 13), ("width", 16), ("node", "other")])
def test_selected_scalar_contract_rejects_native_substitution(field, value):
    desc = scalar()
    study.verify_descriptor(desc)
    desc[field] = value
    with pytest.raises(ValueError):
        study.verify_descriptor(desc)


def test_selected_scalar_contract_checks_transport_separately():
    desc = scalar()
    desc["transport"]["radix"] = [12]
    with pytest.raises(ValueError):
        study.verify_descriptor(desc)


def archived(word=3760, raw=None, wrong_float=False):
    row = dict(fold="fold_female_role", sample_id="x", nucleus_key="source:x", sample_name="b",
               source_name="s", array_sha256="f"*64, array_file="tensors/0000.npy")
    head = np.float32(word) / np.float32(4096) / np.float32(scalar()["scale"])
    value = float(np.float32(format(head, ".8f")))
    table = native.rows_csv([dict(row, chunk="0000_0000", row_in_chunk=0, bie_burden=value,
                                 bie_fixed_code=word)])
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        root = "res_kdp720_synthetic/in_000000.txt/mode_720_piano/layer_output_burden_"
        z.writestr(root + "fx.bin", int(word if raw is None else raw).to_bytes(2, "little", signed=True))
        z.writestr(root + "fx.txt", str(word))
        z.writestr(root + "fl.txt", format(value / 2 if wrong_float else value, ".8f"))
        z.writestr("predictions.csv", table)
    return stream, row


@pytest.mark.parametrize("code", [-16384, -1, 0, 3760, 16383])
def test_archive_replays_native_signed_domain_and_order(code):
    stream, row = archived(code)
    with zipfile.ZipFile(stream) as z:
        result, = native.check_archive(z, [row], scalar(), "0000_0000")
        assert int(result["bie_fixed_code"]) == code
        with pytest.raises(ValueError, match="misalignment"):
            native.check_archive(z, [dict(row, sample_id="y")], scalar(), "0000_0000")


@pytest.mark.parametrize("extra", [{"raw": 7520}, {"wrong_float": True}, {"word": 16384}])
def test_native_archive_rejects_transport_bytes_values_and_out_of_range(extra):
    stream, row = archived(**extra)
    with zipfile.ZipFile(stream) as z, pytest.raises(ValueError):
        native.check_archive(z, [row], scalar(), "0000_0000")


def test_changed_policy_is_refused_before_loading(tmp_path):
    candidate = tmp_path / "policy.json"
    candidate.write_text("{}", encoding="utf8")
    with pytest.raises(ValueError, match="Unrecognized frozen"):
        study.verify_policy(candidate)

