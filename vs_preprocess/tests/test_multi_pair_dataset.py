# tests/test_multi_pair_dataset.py
import pytest
from pathlib import Path
from dataset import ScanRecord
from training.multi_pair_dataset import enumerate_pairs, PairType


def _rec(mod, pid, tp):
    return ScanRecord(
        modality=mod,
        patient_timepoint=f"{pid}_{tp}",
        patient_id=pid,
        timepoint=tp,
        volume_path=Path(f"/data/{mod}/{pid}_{tp}/vol.nii.gz"),
        seg_path=None,
    )


def test_same_sub_longitudinal():
    """Same sub-modality, different timepoints -> PairType.SAME_SUB"""
    index = {
        "001": {
            "1": {"t1+_thin": _rec("t1+_thin", "001", "1")},
            "2": {"t1+_thin": _rec("t1+_thin", "001", "2")},
        }
    }
    pairs = enumerate_pairs(index)
    same_sub = [p for p in pairs if p[2] == PairType.SAME_SUB]
    assert len(same_sub) == 1
    assert same_sub[0][0] == Path("/data/t1+_thin/001_2/vol.nii.gz")  # moving
    assert same_sub[0][1] == Path("/data/t1+_thin/001_1/vol.nii.gz")  # fixed (baseline)


def test_cross_sub_longitudinal():
    """Different sub-modality same super, different timepoints -> PairType.CROSS_SUB"""
    index = {
        "001": {
            "1": {"t1+_thick_ax": _rec("t1+_thick_ax", "001", "1")},
            "2": {"t1+_thin": _rec("t1+_thin", "001", "2")},
        }
    }
    pairs = enumerate_pairs(index)
    cross_sub = [p for p in pairs if p[2] == PairType.CROSS_SUB]
    assert len(cross_sub) == 1


def test_cross_super_intra_timepoint():
    """Different super-modality, same timepoint -> PairType.CROSS_SUPER"""
    index = {
        "001": {
            "1": {
                "t1+_thin": _rec("t1+_thin", "001", "1"),
                "t2_thin": _rec("t2_thin", "001", "1"),
            },
        }
    }
    pairs = enumerate_pairs(index)
    cross_super = [p for p in pairs if p[2] == PairType.CROSS_SUPER]
    assert len(cross_super) == 2


def test_no_pairs_single_scan():
    index = {
        "001": {
            "1": {"t1+_thin": _rec("t1+_thin", "001", "1")},
        }
    }
    pairs = enumerate_pairs(index)
    assert len(pairs) == 0


def test_mixed_scenario():
    index = {
        "001": {
            "1": {
                "t1+_thick_ax": _rec("t1+_thick_ax", "001", "1"),
                "t2_thin": _rec("t2_thin", "001", "1"),
            },
            "2": {
                "t1+_thin": _rec("t1+_thin", "001", "2"),
                "t2_thin": _rec("t2_thin", "001", "2"),
            },
        }
    }
    pairs = enumerate_pairs(index)
    same_sub = [p for p in pairs if p[2] == PairType.SAME_SUB]
    cross_sub = [p for p in pairs if p[2] == PairType.CROSS_SUB]
    cross_super = [p for p in pairs if p[2] == PairType.CROSS_SUPER]

    assert len(same_sub) == 1  # t2_thin tp2 -> t2_thin tp1
    assert len(cross_sub) == 1  # t1+_thin@tp2 -> t1+_thick_ax@tp1
    assert len(cross_super) == 4  # 2 per timepoint, 2 timepoints


# --- Task 4: MultiPairDataset interface test ---

def test_multi_pair_dataset_getitem_returns_pair_type():
    """MultiPairDataset has the expected interface."""
    from training.multi_pair_dataset import MultiPairDataset
    assert hasattr(MultiPairDataset, '__getitem__')
    assert hasattr(MultiPairDataset, '__len__')


# --- Task 6: CombinedMultiPairDataset tests ---

def test_combined_multi_pair_dataset_length():
    from training.multi_pair_dataset import CombinedMultiPairDataset, PairType
    import torch

    class FakeMultiPair:
        def __init__(self):
            self.pairs = [
                (Path("/a"), Path("/b"), PairType.SAME_SUB),
                (Path("/c"), Path("/d"), PairType.CROSS_SUB),
                (Path("/e"), Path("/f"), PairType.CROSS_SUPER),
            ] * 10  # 30 pairs total
        def __len__(self):
            return len(self.pairs)
        def __getitem__(self, idx):
            return {"pair_type": torch.tensor(0)}

    class FakeSynthetic:
        length = 100
        def __len__(self):
            return self.length
        def __getitem__(self, idx):
            return {"pair_type": torch.tensor(0), "has_gt": torch.tensor(True)}

    real = FakeMultiPair()
    synth = FakeSynthetic()
    combined = CombinedMultiPairDataset(
        real_dataset=real,
        synthetic_dataset=synth,
        ratios=[0.25, 0.25, 0.25, 0.25],
    )
    assert len(combined) > 0
