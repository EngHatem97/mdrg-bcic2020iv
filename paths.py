"""
Central path resolution. Replaces the hard-coded Google Drive paths in the
original Colab scripts.

Edit configs/config.json once; every script reads it from here.
"""
import json, os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_CFG  = os.path.join(_REPO, "configs", "config.json")

with open(_CFG, "r", encoding="utf-8") as _f:
    CONFIG = json.load(_f)

ROOT = os.path.expanduser(CONFIG["paths"]["dataset_root"])

if not os.path.isdir(ROOT):
    raise FileNotFoundError(
        f"dataset_root does not exist: {ROOT}\n"
        f"Edit {_CFG} and point 'dataset_root' at your copy of BCIC2020-IV Track 3."
    )

REPO_ROOT    = _REPO
ARTIFACT_DIR = os.path.join(_REPO, "artifacts")
RESULT_DIR   = os.path.join(_REPO, "results")
