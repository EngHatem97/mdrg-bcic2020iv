"""
Regenerate the machine-readable artefacts that accompany the paper:

    artifacts/channel_to_region.json   Table 2
    artifacts/mdrg_block_layout.csv    Table 3

    python src/make_artifacts.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdrg_blocks import (REGION_MAP, UNASSIGNED_CHANNELS, block_table,
                         mdrg_block_indices, D_TOTAL)
try:
    from paths import ARTIFACT_DIR
except Exception:   # no dataset path configured yet: artefacts need none
    ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    region_path = os.path.join(ARTIFACT_DIR, "channel_to_region.json")
    with open(region_path, "w", encoding="utf-8") as f:
        json.dump({
            "regions": REGION_MAP,
            "unassigned_channels": UNASSIGNED_CHANNELS,
            "n_assigned": sum(len(v) for v in REGION_MAP.values()),
            "n_recorded": 64,
            "note": ("The corrected Stage-2 extractor assigns all 64 recorded channels "
                     "to the 18 regions. An earlier extractor left FT9, FT10, TP7 and TP8 "
                     "unassigned and covered only 60 channels; feature matrices from that "
                     "version are not interchangeable with these."),
        }, f, indent=2)
    print("Wrote", region_path)

    layout_path = os.path.join(ARTIFACT_DIR, "mdrg_block_layout.csv")
    pd.DataFrame(block_table()).to_csv(layout_path, index=False)
    print("Wrote", layout_path)

    sizes = {k: len(v) for k, v in mdrg_block_indices().items()}
    print("Verified block sizes:", sizes, "total", sum(sizes.values()), "==", D_TOTAL)


if __name__ == "__main__":
    main()
