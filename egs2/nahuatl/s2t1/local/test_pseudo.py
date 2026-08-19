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


def test_all_codes():
    prep = _load("prep_unlabeled")
    assert prep.all_codes("Itzoc_Comid_JBM566-MPH564_x_2023-05-04-d.wav") == [
        "JBM566",
        "MPH564",
    ]
    assert prep.all_codes("no_codes.wav") == []


def test_secondary_speaker_excluded():
    prep = _load("prep_unlabeled")
    # ACT470 is the secondary (interviewer) speaker code in this filename.
    fname = "Itzoc_Comid_JBM566-ACT470_x_2023-06-01-a.wav"
    exclude = {"ACT470"}
    codes = prep.all_codes(fname)
    assert any(c in exclude for c in codes) is True
    # sanity: ACT470 is not the primary code, so the old primary-only check
    # would have missed this recording.
    assert prep.primary_code(fname) != "ACT470"


def test_labeled_uids():
    prep = _load("prep_unlabeled")
    splits = {
        "Hidalgo/Transcriptions/Itzoc_Comid_TRA111-MPH564_x_2023-05-04-a.trs": "hidalgo-train",
        "Hidalgo/Transcriptions/Itzoc_Comid_VAL222-MPH564_y_2023-05-05-b.eaf": "hidalgo-val",
    }
    with tempfile.TemporaryDirectory() as td:
        splits_file = os.path.join(td, "splits.json")
        with open(splits_file, "w") as f:
            json.dump(splits, f)
        uids = prep.labeled_uids(splits_file)
    assert uids == {"2023-05-04-a", "2023-05-05-b"}


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


def test_is_degenerate():
    fp = _load("filter_pseudo")
    assert fp.is_degenerate("") is True
    assert fp.is_degenerate("a a a a a a a a") is True
    assert fp.is_degenerate("nika chiwa se tapowalistli") is False


def test_parse_score_tensor_format():
    # s2t_inference writes str(hyp.score) = "tensor(-14.1931)", not a bare float
    tt = _load("tune_threshold")
    fp = _load("filter_pseudo")
    assert abs(tt.parse_score("tensor(-14.1931)") - (-14.1931)) < 1e-6
    assert abs(tt.parse_score("-3.5") - (-3.5)) < 1e-9          # bare float too
    assert abs(fp.parse_score("tensor(-3.2029)") - (-3.2029)) < 1e-6
    # norm_score divides parsed score by token count
    assert abs(tt.norm_score("tensor(-8.0)", 4) - (-2.0)) < 1e-9
