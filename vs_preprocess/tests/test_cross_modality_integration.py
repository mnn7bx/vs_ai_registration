# tests/test_cross_modality_integration.py
"""
Integration test: verify the full pipeline from data discovery
through pair enumeration with real VS data.

Requires: data at /home/kwa/Documents/vs_ai_registration/data/unprocessed_raw_studies/
"""
import pytest
from pathlib import Path
from dataset import discover_scans, group_by_patient_timepoint, get_super_modality
from training.multi_pair_dataset import enumerate_pairs, PairType

DATA_DIR = Path("/home/kwa/Documents/vs_ai_registration/data/unprocessed_raw_studies")

@pytest.mark.skipif(not DATA_DIR.exists(), reason="VS data not available")
class TestCrossModalityIntegration:

    def test_discover_all_modalities(self):
        records = discover_scans(DATA_DIR)
        modalities = set(r.modality for r in records)
        assert "t1+_thick_ax" in modalities
        assert "t2_thin" in modalities
        assert len(records) > 1000

    def test_group_by_patient_timepoint(self):
        records = discover_scans(DATA_DIR)
        index = group_by_patient_timepoint(records)
        # Should have hundreds of patients
        assert len(index) > 100
        # At least some patients should have multiple timepoints
        multi_tp = sum(1 for tps in index.values() if len(tps) > 1)
        assert multi_tp > 50

    def test_enumerate_pairs_counts(self):
        records = discover_scans(DATA_DIR)
        index = group_by_patient_timepoint(records)
        pairs = enumerate_pairs(index)

        same_sub = sum(1 for p in pairs if p[2] == PairType.SAME_SUB)
        cross_sub = sum(1 for p in pairs if p[2] == PairType.CROSS_SUB)
        cross_super = sum(1 for p in pairs if p[2] == PairType.CROSS_SUPER)

        print(f"Same-sub: {same_sub}, Cross-sub: {cross_sub}, Cross-super: {cross_super}")
        assert same_sub > 0
        assert cross_sub > 0
        assert cross_super > 0
        assert same_sub + cross_sub + cross_super == len(pairs)

    def test_all_pairs_have_valid_paths(self):
        records = discover_scans(DATA_DIR)
        index = group_by_patient_timepoint(records)
        pairs = enumerate_pairs(index)

        for mov_path, fix_path, pt in pairs[:100]:  # check first 100
            assert mov_path.exists(), f"Missing: {mov_path}"
            assert fix_path.exists(), f"Missing: {fix_path}"
            assert mov_path != fix_path
