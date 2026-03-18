"""
Multi-pair dataset for cross-modality and longitudinal registration.

Enumerates three pair types from a patient-timepoint index:
  SAME_SUB:    same sub-modality, different timepoints (NCC)
  CROSS_SUB:   different sub-modality same super-modality, different timepoints (MI)
  CROSS_SUPER: different super-modality, same timepoint (MI)
"""

from __future__ import annotations

import enum
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import ScanRecord, get_super_modality


class PairType(enum.IntEnum):
    SAME_SUB = 0     # NCC
    CROSS_SUB = 1    # MI
    CROSS_SUPER = 2  # MI


# Type alias for the patient-timepoint index
PatientIndex = Dict[str, Dict[str, Dict[str, ScanRecord]]]


def enumerate_pairs(
    index: PatientIndex,
) -> List[Tuple[Path, Path, PairType]]:
    """
    Enumerate all training pairs from the patient-timepoint index.

    Returns list of (moving_path, fixed_path, pair_type).

    Longitudinal pairs: non-baseline -> baseline (earliest timepoint).
    Cross-super pairs: both directions (A->B and B->A).
    """
    pairs: List[Tuple[Path, Path, PairType]] = []

    for pid, timepoints in index.items():
        sorted_tps = sorted(timepoints.keys())
        if len(sorted_tps) < 1:
            continue

        baseline_tp = sorted_tps[0]

        # --- Longitudinal pairs (across timepoints) ---
        for tp in sorted_tps[1:]:
            tp_mods = timepoints[tp]
            bl_mods = timepoints[baseline_tp]

            for sub_mod, record in tp_mods.items():
                super_mod = get_super_modality(sub_mod)

                # Check baseline for same sub-modality
                if sub_mod in bl_mods:
                    # Same sub-modality longitudinal
                    pairs.append((
                        record.volume_path,
                        bl_mods[sub_mod].volume_path,
                        PairType.SAME_SUB,
                    ))
                else:
                    # Look for any sub-modality in baseline with same super-modality
                    for bl_sub, bl_rec in bl_mods.items():
                        if get_super_modality(bl_sub) == super_mod:
                            pairs.append((
                                record.volume_path,
                                bl_rec.volume_path,
                                PairType.CROSS_SUB,
                            ))

        # --- Cross-super-modality pairs (intra-timepoint) ---
        for tp, mods in timepoints.items():
            sub_list = list(mods.items())
            for i, (sub_a, rec_a) in enumerate(sub_list):
                for sub_b, rec_b in sub_list[i + 1:]:
                    if get_super_modality(sub_a) != get_super_modality(sub_b):
                        # Both directions
                        pairs.append((rec_a.volume_path, rec_b.volume_path, PairType.CROSS_SUPER))
                        pairs.append((rec_b.volume_path, rec_a.volume_path, PairType.CROSS_SUPER))

    return pairs
