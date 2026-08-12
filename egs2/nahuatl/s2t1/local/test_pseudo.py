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


def test_char_cer():
    tt = _load("tune_threshold")
    assert tt.char_cer("abcd", "abcd") == 0.0
    assert abs(tt.char_cer("abcd", "abXd") - 0.25) < 1e-9
    assert abs(tt.char_cer("a b", "ab") - (1/3)) < 1e-9   # deleting the space is 1 error over 3 ref tokens


def test_strip_special():
    tt = _load("tune_threshold")
    assert tt.strip_special("<nah_hid><asr><notimestamps> hola") == "hola"


def test_pick_threshold_keeps_high_confidence():
    tt = _load("tune_threshold")
    rows = [(-0.1, 0.05), (-0.2, 0.10), (-0.9, 0.60), (-1.0, 0.70)]
    thr, frac, est = tt.pick_threshold(rows, target_cer=0.15)
    assert thr == -0.2 and abs(est - 0.075) < 1e-9  # keeps first two
