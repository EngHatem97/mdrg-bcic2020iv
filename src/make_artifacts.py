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
from paths import ARTIFACT_DIR


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    region_path = os.path.join(ARTIFACT_DIR, "channel_to_region.json")
    with open(region_path, "w", encoding="utf-8") as f:
        json.dump({
            "regions": REGION_MAP,
            "unassigned_channels": UNASSIGNED_CHANNELS,
            "n_assigned": sum(len(v) for v in REGION_MAP.values()),
            "n_recorded": 64,
            "note": ("The 18 regions are disjoint and cover 60 of the 64 recorded "
                     "channels. The four unassigned channels feed the raw EEG branch, "
                     "which uses all 64, but not the MDRG branch."),
        }, f, indent=2)
    print("Wrote", region_path)

    layout_path = os.path.join(ARTIFACT_DIR, "mdrg_block_layout.csv")
    pd.DataFrame(block_table()).to_csv(layout_path, index=False)
    print("Wrote", layout_path)

    sizes = {k: len(v) for k, v in mdrg_block_indices().items()}
    print("Verified block sizes:", sizes, "total", sum(sizes.values()), "==", D_TOTAL)


if __name__ == "__main__":
    main()
