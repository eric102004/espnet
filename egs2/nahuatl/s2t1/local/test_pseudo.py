import importlib.util
import os

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
