import importlib.util
import json
import os
import tempfile

_HERE = os.path.dirname(__file__)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, name + ".py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_primary_code():
    prep = _load("prep_unlabeled")
    assert prep.primary_code("Itzoc_Comid_JBM566-MPH564_x_2023-05-04-d.wav") == "JBM566"
    assert prep.primary_code("Omitl_Narra_MGx512-CRF402_y.wav") == "MGx512"
    assert prep.primary_code("no_codes_here.wav") is None


def test_sanitize():
    prep = _load("prep_unlabeled")
    assert prep.sanitize("a-b.c d") == "a_b_c_d"


def test_val_test_consultants():
    prep = _load("prep_unlabeled")
    splits = {
        "Hidalgo/Transcriptions/Itzoc_Comid_TRA111-MPH564_x_2023-05-04-d.trs": "hidalgo-train",
        "Hidalgo/Transcriptions/Itzoc_Comid_VAL222-MPH564_y_2023-05-05-a.trs": "hidalgo-val",
        "Hidalgo/Transcriptions/Itzoc_Comid_TST333-MPH564_z_2023-05-06-b.trs": "hidalgo-test",
    }
    with tempfile.TemporaryDirectory() as td:
        splits_file = os.path.join(td, "splits.json")
        with open(splits_file, "w") as f:
            json.dump(splits, f)
        excl = prep.val_test_consultants(splits_file)
    assert excl == {"VAL222", "TST333"}
    assert "TRA111" not in excl
