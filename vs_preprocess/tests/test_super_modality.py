# tests/test_super_modality.py
import pytest

from dataset import get_super_modality, SUPER_MODALITY_MAP


# --- Task 1: Super-modality mapping tests ---

def test_t1c_variants():
    assert get_super_modality("t1+_thick_ax") == "T1C"
    assert get_super_modality("t1+_thick_cor") == "T1C"
    assert get_super_modality("t1+_thin") == "T1C"


def test_t2_variants():
    assert get_super_modality("t2_thick") == "T2"
    assert get_super_modality("t2_thin") == "T2"


def test_unknown_modality():
    assert get_super_modality("flair") == "FLAIR"


def test_map_has_all_known():
    assert set(SUPER_MODALITY_MAP.keys()) == {
        "t1+_thick_ax", "t1+_thick_cor", "t1+_thin",
        "t2_thick", "t2_thin",
    }
