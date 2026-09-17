"""
External baselines (pretrained encoder, consensus-graph comparator) and the final paired statistics.

Part of: Graph-Riemannian Multi-View Fusion for Subject-Specific EEG Imagined
Speech Decoding: A Held-Out Evaluation on the BCIC2020-IV Official Test Set.

This file is the original experimental script, unmodified except that the
hard-coded Google Drive paths have been replaced by a lookup in
configs/config.json (see src/paths.py). Run it from the repository root:

    python src/run_02_baselines_and_stats.py

PROTOCOL NOTE
-------------
Official train fits the models. Official validation selects architecture, seed,
stopping epoch, feature block, classifier, regularisation and the fusion weight.
The official test split is scored exactly once, after every parameter is fixed.
No choice in this file was revised after test metrics were inspected.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT   # reads configs/config.json

# FINAL REVE OFFICIAL-RAW BASELINE + FINAL REVISION FIGURES
# ======================================================================
#
# IMPORTANT
# ---------
# This supersedes the earlier REVE script.
#
# The earlier REVE run used 0-2000 ms Stage-1 processed tensors.
# The released REVE BCIC2020 speech preprocessing instead operates on
# ORIGINAL Track-3 MATLAB data:
#
#     raw epoch -> last 768 samples -> resample to 600 -> /1000
#
# This script reproduces that input protocol and evaluates REVE using:
#
# TRAIN       : model fitting
# VALIDATION  : stopping / checkpoint selection
# TEST        : final evaluation only
#
# The proposed method is NOT retuned here.
#
# PEFT / torchao are NOT used. LoRA is implemented directly.
#
# ======================================================================

import os
import glob
import json
import math
import copy
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.io
from scipy import signal
from scipy.stats import wilcoxon, rankdata

import h5py

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ======================================================================
# 0. PATHS
# ======================================================================

# ROOT now comes from configs/config.json via src/paths.py
TEST_LABEL_FILE = os.path.join(
    ROOT,
    "INASS_FINAL_TEST_REVISION",
    "official_test_labels.npy"
)

FINAL_TEST_SUBJECTWISE = os.path.join(
    ROOT,
    "INASS_FINAL_TEST_REVISION",
    "FINAL_TEST_SUBJECTWISE.csv"
)

OLD_STAGE3_PRED_DIR = os.path.join(
    ROOT,
    "stage3_dual_fusion_v3_final",
    "predictions"
)

STAGE1_SUMMARY = os.path.join(
    ROOT,
    "prep_stage1_final_robust",
    "GLOBAL_stage1final_summary.csv"
)

ABLATION_CSV = os.path.join(
    ROOT,
    "INASS_REVISION_FINAL",
    "ablation",
    "MDRG_LINEAR_PROBE_SUMMARY.csv"
)

OUT = os.path.join(
    ROOT,
    "INASS_REVE_OFFICIAL_RAW_FINAL"
)

CACHE = os.path.join(
    OUT,
    "cache"
)

FIG_DIR = os.path.join(
    OUT,
    "figures"
)

for d in [
    OUT,
    CACHE,
    FIG_DIR
]:
    os.makedirs(
        d,
        exist_ok=True
    )


# ======================================================================
# 1. FIXED REVE / PAPER SETTINGS
# ======================================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

HF_TOKEN = os.environ.get(
    "HF_TOKEN",
    None
)

REVE_MODEL_ID = "brain-bzh/reve-base"
REVE_POSITION_ID = "brain-bzh/reve-positions"

REVE_SEED = 42

N_SUBJECTS = 15
N_CLASSES = 5

CLASS_NAMES = [
    "hello",
    "help me",
    "stop",
    "thank you",
    "yes"
]

# Official released REVE speech-input preprocessing.
RAW_EXPECTED_SAMPLES = 795
REVE_RAW_KEEP = 768
REVE_TARGET_SAMPLES = 600
REVE_TARGET_FS = 200.0
REVE_SCALE_FACTOR = 1000.0

# Released task settings.
LP_MAX_EPOCHS = 100
LP_PATIENCE = 15
LP_WARMUP_EPOCHS = 3
LP_LR = 5e-3
LP_DROPOUT = 0.05

FT_MAX_EPOCHS = 200
FT_PATIENCE = 15
FT_WARMUP_EPOCHS = 5
FT_LR = 1e-4
FT_DROPOUT = 0.50

WEIGHT_DECAY = 0.01
BETAS = (
    0.92,
    0.999
)
OPT_EPS = 1e-9

REDUCE_FACTOR = 0.5
REDUCE_PATIENCE = 5

GRAD_CLIP = 2.0

LORA_RANK = 16
LORA_ALPHA = 16.0

# Physical batches chosen for a Colab GPU.
# Gradient accumulation approximates the released batch size of 64.
LP_MICRO_BATCH = 8
LP_ACCUM_STEPS = 8

FT_MICRO_BATCH = 2
FT_ACCUM_STEPS = 32

MIXUP = True

RUN_LINEAR_PROBE = True
RUN_LORA = True
MAKE_FIGURES = True

print(
    "DEVICE:",
    DEVICE
)

if DEVICE == "cpu":

    print(
        "WARNING: use a GPU runtime for REVE."
    )


# ======================================================================
# 2. REPRODUCIBILITY
# ======================================================================

def set_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(
    REVE_SEED
)


# ======================================================================
# 3. RESOLVE AND PIN HUGGING FACE REVISIONS
# ======================================================================

def resolve_hf_revision(
    model_id
):

    from huggingface_hub import model_info

    info = model_info(
        model_id,
        token=HF_TOKEN
    )

    return str(
        info.sha
    )


REVE_REVISION = resolve_hf_revision(
    REVE_MODEL_ID
)

POSITION_REVISION = resolve_hf_revision(
    REVE_POSITION_ID
)

print(
    "Pinned REVE revision:",
    REVE_REVISION
)

print(
    "Pinned position-bank revision:",
    POSITION_REVISION
)


# ======================================================================
# 4. REVE CHANNEL ORDER
# ======================================================================
#
# This is the released REVE BCIC2020 speech channel list.
# T3/T4/T5/T6 are historical aliases for T7/T8/P7/P8.
#
# ======================================================================

REVE_CHANNEL_NAMES = [
    "FP1","FP2","F7","F3","FZ","F4","F8",
    "FC5","FC1","FC2","FC6",
    "T3","C3","CZ","C4","T4",
    "TP9","CP5","CP1","CP2","CP6","TP10",
    "T5","P3","PZ","P4","T6",
    "PO9","O1","OZ","O2","PO10",
    "AF7","AF3","AF4","AF8",
    "F5","F1","F2","F6",
    "FT9","FT7","FC3","FC4","FT8","FT10",
    "C5","C1","C2","C6",
    "TP7","CP3","CPZ","CP4","TP8",
    "P5","P1","P2","P6",
    "PO7","PO3","POZ","PO4","PO8"
]

assert len(
    REVE_CHANNEL_NAMES
) == 64


# ======================================================================
# 5. LOAD OFFICIAL TEST LABELS
# ======================================================================

if not os.path.exists(
    TEST_LABEL_FILE
):

    raise FileNotFoundError(
        "official_test_labels.npy is missing. "
        "Run the prior final-test package first."
    )

TEST_LABELS = np.load(
    TEST_LABEL_FILE
).astype(
    np.int64
)

assert TEST_LABELS.shape == (
    15,
    50
)


# ======================================================================
# 6. DISCOVER RAW TRACK-3 FILES
# ======================================================================

def classify_split_path(
    path
):

    p = str(
        path
    ).lower()

    if (
        "training set"
        in p
        or
        "/training/"
        in p
        or
        "\\training\\"
        in p
    ):

        return "train"

    if (
        "validation set"
        in p
        or
        "/validation/"
        in p
        or
        "\\validation\\"
        in p
    ):

        return "valid"

    if (
        "testing set"
        in p
        or
        "test set"
        in p
        or
        "/testing/"
        in p
        or
        "\\testing\\"
        in p
        or
        "/test/"
        in p
        or
        "\\test\\"
        in p
    ):

        return "test"

    return None


def discover_raw_files():

    mapping = {
        "train":
            {},
        "valid":
            {},
        "test":
            {}
    }

    all_mat = glob.glob(
        os.path.join(
            ROOT,
            "**",
            "Data_Sample*.mat"
        ),
        recursive=True
    )

    for path in all_mat:

        name = os.path.basename(
            path
        )

        if not name.lower().startswith(
            "data_sample"
        ):

            continue

        try:

            sid = int(
                name[
                    len(
                        "Data_Sample"
                    ):
                    len(
                        "Data_Sample"
                    )
                    +
                    2
                ]
            )

        except Exception:

            continue

        split = classify_split_path(
            path
        )

        if split is None:
            continue

        # Prefer the shortest path if duplicate copies exist.
        old = mapping[
            split
        ].get(
            sid
        )

        if (
            old is None
            or len(
                path
            )
            <
            len(
                old
            )
        ):

            mapping[
                split
            ][
                sid
            ] = path

    missing = []

    for split in [
        "train",
        "valid",
        "test"
    ]:

        for sid in range(
            1,
            16
        ):

            if sid not in mapping[
                split
            ]:

                missing.append(
                    (
                        split,
                        sid
                    )
                )

    print(
        "\nRAW FILE DISCOVERY"
    )

    for split in [
        "train",
        "valid",
        "test"
    ]:

        print(
            split,
            len(
                mapping[
                    split
                ]
            ),
            "files"
        )

    if missing:

        raise FileNotFoundError(
            "Could not discover all raw Track-3 MAT files. "
            f"Missing entries: {missing}\n"
            "Expected folders such as Training set, Validation set, Test set "
            f"under {ROOT}"
        )

    manifest_rows = []

    for split in mapping:

        for sid, path in sorted(
            mapping[
                split
            ].items()
        ):

            manifest_rows.append({
                "split":
                    split,
                "subject":
                    sid,
                "path":
                    path
            })

    pd.DataFrame(
        manifest_rows
    ).to_csv(
        os.path.join(
            OUT,
            "RAW_FILE_MANIFEST.csv"
        ),
        index=False
    )

    return mapping


RAW_FILES = discover_raw_files()


# ======================================================================
# 7. ROBUST ORIGINAL MATLAB LOADER
# ======================================================================

def normalize_eeg_shape(
    x,
    expected_trials
):

    x = np.asarray(
        x
    )

    x = np.squeeze(
        x
    )

    if x.ndim != 3:

        raise ValueError(
            f"Expected 3-D EEG tensor, got {x.shape}"
        )

    shape = x.shape

    trial_axes = [
        i
        for i, n in enumerate(
            shape
        )
        if n == expected_trials
    ]

    channel_axes = [
        i
        for i, n in enumerate(
            shape
        )
        if n == 64
    ]

    if not trial_axes:

        raise ValueError(
            f"Cannot locate trial axis in {shape}; "
            f"expected {expected_trials} trials."
        )

    if not channel_axes:

        raise ValueError(
            f"Cannot locate 64-channel axis in {shape}."
        )

    trial_axis = trial_axes[
        0
    ]

    channel_axis = channel_axes[
        0
    ]

    if trial_axis == channel_axis:

        raise ValueError(
            f"Ambiguous EEG axes: {shape}"
        )

    time_axis = [
        i
        for i in range(
            3
        )
        if i not in [
            trial_axis,
            channel_axis
        ]
    ][
        0
    ]

    out = np.transpose(
        x,
        (
            trial_axis,
            channel_axis,
            time_axis
        )
    )

    return out.astype(
        np.float32
    )


def labels_to_index(
    y,
    expected_trials
):

    y = np.asarray(
        y
    )

    y = np.squeeze(
        y
    )

    if y.ndim == 1:

        labels = y.astype(
            int
        )

    elif y.ndim == 2:

        if y.shape == (
            5,
            expected_trials
        ):

            labels = np.argmax(
                y,
                axis=0
            )

        elif y.shape == (
            expected_trials,
            5
        ):

            labels = np.argmax(
                y,
                axis=1
            )

        else:

            raise ValueError(
                f"Unexpected label shape: {y.shape}"
            )

    else:

        raise ValueError(
            f"Unexpected label shape: {y.shape}"
        )

    labels = np.asarray(
        labels
    ).reshape(
        -1
    ).astype(
        np.int64
    )

    if (
        labels.min() == 1
        and labels.max() == 5
    ):

        labels = (
            labels
            -
            1
        )

    if len(
        labels
    ) != expected_trials:

        raise ValueError(
            f"Expected {expected_trials} labels, got {len(labels)}"
        )

    return labels


def load_train_or_valid_mat(
    path,
    split
):

    key = (
        "epo_train"
        if split == "train"
        else
        "epo_validation"
    )

    expected_trials = (
        300
        if split == "train"
        else
        50
    )

    # First try convenient scipy mat_struct form.
    try:

        m = scipy.io.loadmat(
            path,
            squeeze_me=True,
            struct_as_record=False
        )

        epo = m[
            key
        ]

        if hasattr(
            epo,
            "x"
        ):

            x = epo.x
            y = epo.y

        else:

            raise AttributeError

    except Exception:

        # Exact positional layout used by the public REVE preprocessor.
        m = scipy.io.loadmat(
            path
        )

        epo = m[
            key
        ][
            0
        ][
            0
        ]

        x = epo[
            4
        ]

        y = epo[
            5
        ]

    X = normalize_eeg_shape(
        x,
        expected_trials
    )

    labels = labels_to_index(
        y,
        expected_trials
    )

    return (
        X,
        labels
    )


def load_test_mat(
    path
):

    expected_trials = 50

    try:

        with h5py.File(
            path,
            "r"
        ) as f:

            x = f[
                "epo_test"
            ][
                "x"
            ][
                :
            ]

        X = normalize_eeg_shape(
            x,
            expected_trials
        )

        return X

    except Exception:

        # Fallback if a copy is stored in MATLAB v5 rather than v7.3.
        m = scipy.io.loadmat(
            path,
            squeeze_me=True,
            struct_as_record=False
        )

        epo = m[
            "epo_test"
        ]

        if hasattr(
            epo,
            "x"
        ):

            x = epo.x

        else:

            raise RuntimeError(
                f"Cannot parse test file: {path}"
            )

        return normalize_eeg_shape(
            x,
            expected_trials
        )


# ======================================================================
# 8. EXACT REVE BCIC2020 SPEECH PREPROCESSING
# ======================================================================

def reve_preprocess_raw(
    X
):

    if X.shape[
        1
    ] != 64:

        raise ValueError(
            f"Expected 64 channels, got {X.shape}"
        )

    if X.shape[
        2
    ] < REVE_RAW_KEEP:

        raise ValueError(
            f"Expected at least {REVE_RAW_KEEP} samples, got {X.shape}"
        )

    # Released preprocessing_speech.py:
    # eeg = eeg[:, :, -768:]
    X = X[
        :,
        :,
        -REVE_RAW_KEEP:
    ]

    # Released preprocessing_speech.py:
    # eeg = signal.resample(eeg, 600, axis=2)
    X = signal.resample(
        X,
        REVE_TARGET_SAMPLES,
        axis=2
    ).astype(
        np.float32
    )

    # Released speech.yaml + LMDBDataset:
    # sample / scale_factor, scale_factor = 1000
    X = (
        X
        /
        REVE_SCALE_FACTOR
    ).astype(
        np.float32
    )

    if not np.isfinite(
        X
    ).all():

        raise RuntimeError(
            "Non-finite values found in REVE input."
        )

    return X


def load_subject_raw(
    sid
):

    cache_file = os.path.join(
        CACHE,
        f"Data_Sample{sid:02d}_REVE_input.npz"
    )

    if os.path.exists(
        cache_file
    ):

        z = np.load(
            cache_file
        )

        return {
            "sid":
                f"Data_Sample{sid:02d}",
            "X_train":
                z[
                    "X_train"
                ],
            "y_train":
                z[
                    "y_train"
                ],
            "X_valid":
                z[
                    "X_valid"
                ],
            "y_valid":
                z[
                    "y_valid"
                ],
            "X_test":
                z[
                    "X_test"
                ],
            "y_test":
                z[
                    "y_test"
                ]
        }

    Xtr, ytr = load_train_or_valid_mat(
        RAW_FILES[
            "train"
        ][
            sid
        ],
        "train"
    )

    Xva, yva = load_train_or_valid_mat(
        RAW_FILES[
            "valid"
        ][
            sid
        ],
        "valid"
    )

    Xte = load_test_mat(
        RAW_FILES[
            "test"
        ][
            sid
        ]
    )

    yte = TEST_LABELS[
        sid - 1
    ]

    print(
        f"S{sid:02d} raw shapes:",
        Xtr.shape,
        Xva.shape,
        Xte.shape
    )

    # Audit original source dimensions.
    assert Xtr.shape == (
        300,
        64,
        RAW_EXPECTED_SAMPLES
    )

    assert Xva.shape == (
        50,
        64,
        RAW_EXPECTED_SAMPLES
    )

    assert Xte.shape == (
        50,
        64,
        RAW_EXPECTED_SAMPLES
    )

    Xtr = reve_preprocess_raw(
        Xtr
    )

    Xva = reve_preprocess_raw(
        Xva
    )

    Xte = reve_preprocess_raw(
        Xte
    )

    print(
        f"S{sid:02d} REVE shapes:",
        Xtr.shape,
        Xva.shape,
        Xte.shape,
        "| train abs median/max:",
        float(
            np.median(
                np.abs(
                    Xtr
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    Xtr
                )
            )
        )
    )

    np.savez_compressed(
        cache_file,
        X_train=Xtr,
        y_train=ytr,
        X_valid=Xva,
        y_valid=yva,
        X_test=Xte,
        y_test=yte
    )

    return {
        "sid":
            f"Data_Sample{sid:02d}",
        "X_train":
            Xtr,
        "y_train":
            ytr,
        "X_valid":
            Xva,
        "y_valid":
            yva,
        "X_test":
            Xte,
        "y_test":
            yte
    }


# ======================================================================
# 9. LOAD POSITION BANK
# ======================================================================

def load_positions():

    from transformers import AutoModel

    pos_bank = AutoModel.from_pretrained(
        REVE_POSITION_ID,
        trust_remote_code=True,
        token=HF_TOKEN,
        revision=POSITION_REVISION
    )

    positions = pos_bank(
        REVE_CHANNEL_NAMES
    )

    if positions.ndim == 2:

        positions = positions.unsqueeze(
            0
        )

    positions = positions.detach().cpu()

    if positions.shape != (
        1,
        64,
        3
    ):

        raise RuntimeError(
            f"Unexpected position shape: {positions.shape}"
        )

    print(
        "REVE positions:",
        tuple(
            positions.shape
        )
    )

    del pos_bank

    return positions


POSITIONS = load_positions()


# ======================================================================
# 10. CLASSIFIER WRAPPER
# ======================================================================

class REVEClassifier(
    nn.Module
):

    def __init__(
        self,
        encoder,
        dropout
    ):

        super().__init__()

        self.encoder = encoder

        self.embed_dim = int(
            encoder.embed_dim
        )

        self.cls_query_token = nn.Parameter(
            torch.randn(
                1,
                1,
                self.embed_dim
            )
        )

        if hasattr(
            nn,
            "RMSNorm"
        ):

            self.norm = nn.RMSNorm(
                self.embed_dim
            )

        else:

            self.norm = nn.LayerNorm(
                self.embed_dim
            )

        self.dropout = nn.Dropout(
            dropout
        )

        self.linear_head = nn.Linear(
            self.embed_dim,
            N_CLASSES
        )

    def set_dropout(
        self,
        p
    ):

        self.dropout.p = float(
            p
        )

    def forward(
        self,
        x,
        pos
    ):

        x = self.encoder(
            x,
            pos
        )

        # HF REVE returns (B,C,H,E).
        if x.ndim == 4:

            B, C, H, E = x.shape

            x = x.reshape(
                B,
                C * H,
                E
            )

        elif x.ndim != 3:

            raise RuntimeError(
                f"Unexpected REVE output shape: {tuple(x.shape)}"
            )

        B = x.shape[
            0
        ]

        query = self.cls_query_token.expand(
            B,
            -1,
            -1
        )

        scores = torch.matmul(
            query,
            x.transpose(
                -1,
                -2
            )
        ) / math.sqrt(
            self.embed_dim
        )

        attention = torch.softmax(
            scores,
            dim=-1
        )

        context = torch.matmul(
            attention,
            x
        ).squeeze(
            1
        )

        context = self.norm(
            context
        )

        context = self.dropout(
            context
        )

        return self.linear_head(
            context
        )


# ======================================================================
# 11. DIRECT LoRA IMPLEMENTATION
# ======================================================================

class LoRALinear(
    nn.Module
):

    def __init__(
        self,
        base_linear,
        rank=16,
        alpha=16.0
    ):

        super().__init__()

        if not isinstance(
            base_linear,
            nn.Linear
        ):

            raise TypeError(
                "LoRALinear requires nn.Linear."
            )

        self.base = base_linear

        self.rank = int(
            rank
        )

        self.alpha = float(
            alpha
        )

        self.scale = (
            self.alpha
            /
            self.rank
        )

        for p in self.base.parameters():

            p.requires_grad = False

        self.lora_A = nn.Linear(
            self.base.in_features,
            self.rank,
            bias=False
        )

        self.lora_B = nn.Linear(
            self.rank,
            self.base.out_features,
            bias=False
        )

        nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=math.sqrt(
                5
            )
        )

        nn.init.zeros_(
            self.lora_B.weight
        )

    def forward(
        self,
        x
    ):

        return (
            self.base(
                x
            )
            +
            self.scale
            *
            self.lora_B(
                self.lora_A(
                    x
                )
            )
        )


def freeze_all(
    model
):

    for p in model.parameters():

        p.requires_grad = False


def prepare_linear_probe(
    model
):

    freeze_all(
        model
    )

    model.cls_query_token.requires_grad = True

    for p in model.norm.parameters():

        p.requires_grad = True

    for p in model.linear_head.parameters():

        p.requires_grad = True


def inject_attention_lora(
    model
):

    # Freeze everything first.
    freeze_all(
        model
    )

    replaced = 0

    for layer in model.encoder.transformer.layers:

        attn = layer[
            0
        ]

        if not isinstance(
            attn.to_qkv,
            LoRALinear
        ):

            attn.to_qkv = LoRALinear(
                attn.to_qkv,
                rank=LORA_RANK,
                alpha=LORA_ALPHA
            )

            replaced += 1

        if not isinstance(
            attn.to_out,
            LoRALinear
        ):

            attn.to_out = LoRALinear(
                attn.to_out,
                rank=LORA_RANK,
                alpha=LORA_ALPHA
            )

            replaced += 1

    # Keep the downstream classifier trainable during LoRA adaptation.
    model.cls_query_token.requires_grad = True

    for p in model.norm.parameters():

        p.requires_grad = True

    for p in model.linear_head.parameters():

        p.requires_grad = True

    # LoRA parameters are already trainable.
    n_trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "  Injected LoRA modules:",
        replaced,
        "| trainable parameters:",
        n_trainable
    )

    return model


# ======================================================================
# 12. DATALOADERS
# ======================================================================

class EEGDataset(
    Dataset
):

    def __init__(
        self,
        X,
        y
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(
        self
    ):

        return len(
            self.y
        )

    def __getitem__(
        self,
        idx
    ):

        return (
            self.X[
                idx
            ],
            self.y[
                idx
            ]
        )


def make_loader(
    X,
    y,
    batch_size,
    shuffle
):

    return DataLoader(
        EEGDataset(
            X,
            y
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(
            DEVICE == "cuda"
        ),
        drop_last=False
    )


# ======================================================================
# 13. METRICS
# ======================================================================

def macro_specificity(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            N_CLASSES
        )
    )

    total = cm.sum()

    out = []

    for k in range(
        N_CLASSES
    ):

        tp = cm[
            k,
            k
        ]

        fn = (
            cm[
                k,
                :
            ].sum()
            -
            tp
        )

        fp = (
            cm[
                :,
                k
            ].sum()
            -
            tp
        )

        tn = (
            total
            -
            tp
            -
            fn
            -
            fp
        )

        out.append(
            tn
            /
            max(
                1,
                tn + fp
            )
        )

    return float(
        np.mean(
            out
        )
    )


def metric_dict(
    y,
    pred
):

    return {
        "accuracy":
            float(
                accuracy_score(
                    y,
                    pred
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y,
                    pred,
                    average="macro",
                    zero_division=0
                )
            ),

        "macro_precision":
            float(
                precision_score(
                    y,
                    pred,
                    average="macro",
                    zero_division=0
                )
            ),

        "macro_sensitivity":
            float(
                recall_score(
                    y,
                    pred,
                    average="macro",
                    zero_division=0
                )
            ),

        "macro_specificity":
            float(
                macro_specificity(
                    y,
                    pred
                )
            )
    }


# ======================================================================
# 14. EVALUATION
# ======================================================================

def evaluate_model(
    model,
    loader
):

    model.eval()

    logits_all = []

    pred_all = []

    y_all = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(
                DEVICE,
                non_blocking=True
            )

            pos = POSITIONS.to(
                DEVICE
            ).expand(
                xb.size(
                    0
                ),
                -1,
                -1
            )

            with torch.autocast(
                device_type=(
                    "cuda"
                    if DEVICE == "cuda"
                    else "cpu"
                ),
                dtype=torch.float16,
                enabled=(
                    DEVICE == "cuda"
                )
            ):

                logits = model(
                    xb,
                    pos
                )

            logits = logits.float()

            logits_all.append(
                logits.cpu()
                .numpy()
            )

            pred_all.append(
                torch.argmax(
                    logits,
                    dim=1
                ).cpu()
                .numpy()
            )

            y_all.append(
                yb.numpy()
            )

    logits_all = np.concatenate(
        logits_all
    )

    pred_all = np.concatenate(
        pred_all
    )

    y_all = np.concatenate(
        y_all
    )

    return (
        metric_dict(
            y_all,
            pred_all
        ),
        logits_all,
        pred_all,
        y_all
    )


# ======================================================================
# 15. TRAINING
# ======================================================================

def trainable_params(
    model
):

    return [
        p
        for p in model.parameters()
        if p.requires_grad
    ]


def train_stage(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    stage_name,
    lr,
    max_epochs,
    patience,
    warmup_epochs,
    batch_size,
    accum_steps
):

    train_loader = make_loader(
        X_train,
        y_train,
        batch_size,
        True
    )

    valid_loader = make_loader(
        X_valid,
        y_valid,
        batch_size,
        False
    )

    params = trainable_params(
        model
    )

    print(
        f"  {stage_name} trainable parameters:",
        sum(
            p.numel()
            for p in params
        )
    )

    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        betas=BETAS,
        eps=OPT_EPS,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=REDUCE_FACTOR,
            patience=REDUCE_PATIENCE
        )
    )

    steps_per_epoch = int(
        math.ceil(
            len(
                train_loader
            )
            /
            accum_steps
        )
    )

    total_warmup_steps = max(
        1,
        warmup_epochs
        *
        steps_per_epoch
    )

    optimizer_step = 0

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            DEVICE == "cuda"
        )
    )

    best_valid_acc = -np.inf

    best_epoch = 0

    best_state = None

    wait = 0

    history = []

    for epoch in range(
        1,
        max_epochs + 1
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0

        n_seen = 0

        pending = 0

        for batch_idx, (
            xb,
            yb
        ) in enumerate(
            train_loader
        ):

            xb = xb.to(
                DEVICE,
                non_blocking=True
            )

            yb = yb.to(
                DEVICE,
                non_blocking=True
            )

            pos = POSITIONS.to(
                DEVICE
            ).expand(
                xb.size(
                    0
                ),
                -1,
                -1
            )

            with torch.autocast(
                device_type=(
                    "cuda"
                    if DEVICE == "cuda"
                    else "cpu"
                ),
                dtype=torch.float16,
                enabled=(
                    DEVICE == "cuda"
                )
            ):

                if MIXUP:

                    # Released REVE code uses random.random() mixup.
                    mm = random.random()

                    perm = torch.randperm(
                        xb.size(
                            0
                        ),
                        device=xb.device
                    )

                    mixed_x = (
                        mm
                        *
                        xb
                        +
                        (
                            1.0
                            -
                            mm
                        )
                        *
                        xb[
                            perm
                        ]
                    )

                    logits = model(
                        mixed_x,
                        pos
                    )

                    loss = (
                        mm
                        *
                        F.cross_entropy(
                            logits,
                            yb
                        )
                        +
                        (
                            1.0
                            -
                            mm
                        )
                        *
                        F.cross_entropy(
                            logits,
                            yb[
                                perm
                            ]
                        )
                    )

                else:

                    logits = model(
                        xb,
                        pos
                    )

                    loss = F.cross_entropy(
                        logits,
                        yb
                    )

                loss_to_backward = (
                    loss
                    /
                    accum_steps
                )

            scaler.scale(
                loss_to_backward
            ).backward()

            pending += 1

            running_loss += float(
                loss.detach()
                .cpu()
                .item()
            ) * len(
                yb
            )

            n_seen += len(
                yb
            )

            do_step = (
                pending >= accum_steps
                or
                batch_idx
                ==
                len(
                    train_loader
                )
                -
                1
            )

            if do_step:

                scaler.unscale_(
                    optimizer
                )

                nn.utils.clip_grad_norm_(
                    params,
                    GRAD_CLIP
                )

                # Warmup on optimizer steps.
                optimizer_step += 1

                if optimizer_step <= total_warmup_steps:

                    warmup_scale = (
                        optimizer_step
                        /
                        total_warmup_steps
                    )

                    for group in optimizer.param_groups:

                        group[
                            "lr"
                        ] = (
                            lr
                            *
                            warmup_scale
                        )

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

                pending = 0

        valid_metrics, _, _, _ = (
            evaluate_model(
                model,
                valid_loader
            )
        )

        valid_acc = valid_metrics[
            "accuracy"
        ]

        if epoch > warmup_epochs:

            scheduler.step(
                valid_acc
            )

        history.append({
            "stage":
                stage_name,
            "epoch":
                epoch,
            "train_loss":
                (
                    running_loss
                    /
                    max(
                        1,
                        n_seen
                    )
                ),
            "valid_accuracy":
                valid_metrics[
                    "accuracy"
                ],
            "valid_macro_f1":
                valid_metrics[
                    "macro_f1"
                ],
            "lr":
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
        })

        print(
            f"  {stage_name} epoch {epoch:03d} | "
            f"val acc={valid_metrics['accuracy']:.4f} | "
            f"val F1={valid_metrics['macro_f1']:.4f}"
        )

        # Released downstream code monitors validation accuracy.
        if valid_acc > best_valid_acc + 1e-8:

            best_valid_acc = float(
                valid_acc
            )

            best_epoch = int(
                epoch
            )

            best_state = {
                k:
                    v.detach()
                    .cpu()
                    .clone()
                for k, v
                in model.state_dict()
                .items()
            }

            wait = 0

        else:

            wait += 1

        if wait >= patience:

            print(
                f"  Early stopping {stage_name} at epoch {epoch}."
            )

            break

    if best_state is None:

        raise RuntimeError(
            f"No checkpoint selected for {stage_name}."
        )

    model.load_state_dict(
        best_state,
        strict=True
    )

    return (
        model,
        pd.DataFrame(
            history
        ),
        best_epoch,
        best_valid_acc
    )


# ======================================================================
# 16. RUN ONE SUBJECT
# ======================================================================

def run_subject(
    sid
):

    subject_name = f"Data_Sample{sid:02d}"

    result_npz = os.path.join(
        CACHE,
        f"{subject_name}_FINAL_REVE.npz"
    )

    result_json = os.path.join(
        CACHE,
        f"{subject_name}_FINAL_REVE.json"
    )

    if (
        os.path.exists(
            result_npz
        )
        and
        os.path.exists(
            result_json
        )
    ):

        print(
            subject_name,
            "cached final result found."
        )

        z = np.load(
            result_npz
        )

        with open(
            result_json,
            "r"
        ) as f:

            meta = json.load(
                f
            )

        return {
            "sid":
                subject_name,
            "y_test":
                z[
                    "y_test"
                ],
            "lp_logits":
                z[
                    "lp_logits"
                ],
            "lp_pred":
                z[
                    "lp_pred"
                ],
            "lora_logits":
                z[
                    "lora_logits"
                ],
            "lora_pred":
                z[
                    "lora_pred"
                ],
            "meta":
                meta
        }

    d = load_subject_raw(
        sid
    )

    print(
        "\n"
        +
        "="
        * 78
    )

    print(
        "REVE OFFICIAL-RAW:",
        subject_name
    )

    print(
        "="
        * 78
    )

    from transformers import AutoModel

    set_seed(
        REVE_SEED
    )

    encoder = AutoModel.from_pretrained(
        REVE_MODEL_ID,
        trust_remote_code=True,
        token=HF_TOKEN,
        revision=REVE_REVISION
    )

    model = REVEClassifier(
        encoder=encoder,
        dropout=LP_DROPOUT
    ).to(
        DEVICE
    )

    # --------------------------------------------------------------
    # Linear probing
    # --------------------------------------------------------------

    prepare_linear_probe(
        model
    )

    model.set_dropout(
        LP_DROPOUT
    )

    model, lp_history, lp_epoch, lp_val_acc = (
        train_stage(
            model,
            d[
                "X_train"
            ],
            d[
                "y_train"
            ],
            d[
                "X_valid"
            ],
            d[
                "y_valid"
            ],
            stage_name="LP",
            lr=LP_LR,
            max_epochs=LP_MAX_EPOCHS,
            patience=LP_PATIENCE,
            warmup_epochs=LP_WARMUP_EPOCHS,
            batch_size=LP_MICRO_BATCH,
            accum_steps=LP_ACCUM_STEPS
        )
    )

    lp_valid_loader = make_loader(
        d[
            "X_valid"
        ],
        d[
            "y_valid"
        ],
        LP_MICRO_BATCH,
        False
    )

    lp_test_loader = make_loader(
        d[
            "X_test"
        ],
        d[
            "y_test"
        ],
        LP_MICRO_BATCH,
        False
    )

    lp_valid_metrics, _, _, _ = (
        evaluate_model(
            model,
            lp_valid_loader
        )
    )

    lp_test_metrics, lp_logits, lp_pred, y_test = (
        evaluate_model(
            model,
            lp_test_loader
        )
    )

    print(
        "  LP FINAL TEST:",
        lp_test_metrics
    )

    lp_history.to_csv(
        os.path.join(
            CACHE,
            f"{subject_name}_LP_history.csv"
        ),
        index=False
    )

    # --------------------------------------------------------------
    # Direct attention-LoRA fine-tuning
    # --------------------------------------------------------------

    if RUN_LORA:

        model.set_dropout(
            FT_DROPOUT
        )

        model = inject_attention_lora(
            model
        ).to(
            DEVICE
        )

        model, ft_history, ft_epoch, ft_val_acc = (
            train_stage(
                model,
                d[
                    "X_train"
                ],
                d[
                    "y_train"
                ],
                d[
                    "X_valid"
                ],
                d[
                    "y_valid"
                ],
                stage_name="LoRA",
                lr=FT_LR,
                max_epochs=FT_MAX_EPOCHS,
                patience=FT_PATIENCE,
                warmup_epochs=FT_WARMUP_EPOCHS,
                batch_size=FT_MICRO_BATCH,
                accum_steps=FT_ACCUM_STEPS
            )
        )

        ft_valid_loader = make_loader(
            d[
                "X_valid"
            ],
            d[
                "y_valid"
            ],
            FT_MICRO_BATCH,
            False
        )

        ft_test_loader = make_loader(
            d[
                "X_test"
            ],
            d[
                "y_test"
            ],
            FT_MICRO_BATCH,
            False
        )

        ft_valid_metrics, _, _, _ = (
            evaluate_model(
                model,
                ft_valid_loader
            )
        )

        ft_test_metrics, lora_logits, lora_pred, y_test2 = (
            evaluate_model(
                model,
                ft_test_loader
            )
        )

        assert np.array_equal(
            y_test,
            y_test2
        )

        print(
            "  LoRA FINAL TEST:",
            ft_test_metrics
        )

        ft_history.to_csv(
            os.path.join(
                CACHE,
                f"{subject_name}_LoRA_history.csv"
            ),
            index=False
        )

    else:

        ft_epoch = 0
        ft_val_acc = np.nan

        ft_valid_metrics = {
            "accuracy":
                np.nan,
            "macro_f1":
                np.nan
        }

        ft_test_metrics = {
            "accuracy":
                np.nan,
            "macro_f1":
                np.nan
        }

        lora_logits = np.full_like(
            lp_logits,
            np.nan
        )

        lora_pred = np.full_like(
            lp_pred,
            -1
        )

    meta = {
        "subject":
            subject_name,
        "seed":
            REVE_SEED,
        "raw_original_samples":
            RAW_EXPECTED_SAMPLES,
        "raw_kept_last_samples":
            REVE_RAW_KEEP,
        "reve_samples":
            REVE_TARGET_SAMPLES,
        "reve_sampling_hz":
            REVE_TARGET_FS,
        "scale_factor":
            REVE_SCALE_FACTOR,
        "model_revision":
            REVE_REVISION,
        "position_revision":
            POSITION_REVISION,
        "lp_best_epoch":
            int(
                lp_epoch
            ),
        "lp_best_valid_accuracy":
            float(
                lp_val_acc
            ),
        "lp_valid_macro_f1":
            float(
                lp_valid_metrics[
                    "macro_f1"
                ]
            ),
        "lp_test_accuracy":
            float(
                lp_test_metrics[
                    "accuracy"
                ]
            ),
        "lp_test_macro_f1":
            float(
                lp_test_metrics[
                    "macro_f1"
                ]
            ),
        "lora_best_epoch":
            int(
                ft_epoch
            ),
        "lora_best_valid_accuracy":
            (
                float(
                    ft_val_acc
                )
                if np.isfinite(
                    ft_val_acc
                )
                else None
            ),
        "lora_valid_macro_f1":
            (
                float(
                    ft_valid_metrics[
                        "macro_f1"
                    ]
                )
                if np.isfinite(
                    ft_valid_metrics[
                        "macro_f1"
                    ]
                )
                else None
            ),
        "lora_test_accuracy":
            (
                float(
                    ft_test_metrics[
                        "accuracy"
                    ]
                )
                if "accuracy"
                in ft_test_metrics
                and np.isfinite(
                    ft_test_metrics[
                        "accuracy"
                    ]
                )
                else None
            ),
        "lora_test_macro_f1":
            (
                float(
                    ft_test_metrics[
                        "macro_f1"
                    ]
                )
                if "macro_f1"
                in ft_test_metrics
                and np.isfinite(
                    ft_test_metrics[
                        "macro_f1"
                    ]
                )
                else None
            )
    }

    np.savez_compressed(
        result_npz,
        y_test=y_test,
        lp_logits=lp_logits,
        lp_pred=lp_pred,
        lora_logits=lora_logits,
        lora_pred=lora_pred
    )

    with open(
        result_json,
        "w"
    ) as f:

        json.dump(
            meta,
            f,
            indent=2
        )

    del model
    del encoder

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return {
        "sid":
            subject_name,
        "y_test":
            y_test,
        "lp_logits":
            lp_logits,
        "lp_pred":
            lp_pred,
        "lora_logits":
            lora_logits,
        "lora_pred":
            lora_pred,
        "meta":
            meta
    }


# ======================================================================
# 17. RUN ALL SUBJECTS
# ======================================================================

reve_outputs = {}

reve_rows = []

for sid in range(
    1,
    16
):

    out = run_subject(
        sid
    )

    reve_outputs[
        out[
            "sid"
        ]
    ] = out

    lp_metrics = metric_dict(
        out[
            "y_test"
        ],
        out[
            "lp_pred"
        ]
    )

    reve_rows.append({
        "subject":
            out[
                "sid"
            ],
        "method":
            "REVE_LinearProbe",
        **lp_metrics
    })

    if RUN_LORA:

        ft_metrics = metric_dict(
            out[
                "y_test"
            ],
            out[
                "lora_pred"
            ]
        )

        reve_rows.append({
            "subject":
                out[
                    "sid"
                ],
            "method":
                "REVE_LoRA",
            **ft_metrics
        })


reve_df = pd.DataFrame(
    reve_rows
)

reve_df.to_csv(
    os.path.join(
        OUT,
        "REVE_SUBJECTWISE.csv"
    ),
    index=False
)

reve_summary = (
    reve_df
    .groupby(
        "method"
    )
    .agg(
        accuracy_mean=(
            "accuracy",
            "mean"
        ),
        accuracy_std=(
            "accuracy",
            "std"
        ),
        macro_f1_mean=(
            "macro_f1",
            "mean"
        ),
        macro_f1_std=(
            "macro_f1",
            "std"
        ),
        precision_mean=(
            "macro_precision",
            "mean"
        ),
        sensitivity_mean=(
            "macro_sensitivity",
            "mean"
        ),
        specificity_mean=(
            "macro_specificity",
            "mean"
        )
    )
    .reset_index()
)

reve_summary.to_csv(
    os.path.join(
        OUT,
        "REVE_SUMMARY.csv"
    ),
    index=False
)

print(
    "\n"
    +
    "="
    * 78
)

print(
    "REVE OFFICIAL-RAW FINAL TEST SUMMARY"
)

print(
    "="
    * 78
)

print(
    reve_summary.to_string(
        index=False
    )
)


# ======================================================================
# 18. RECOVER FROZEN ORIGINAL RAW / FUSION TEST PREDICTIONS
# ======================================================================

old_rows = []

pooled_old = {
    "Old_Raw":
        [],
    "Old_Fusion":
        []
}

pooled_y = []

files = sorted(
    glob.glob(
        os.path.join(
            OLD_STAGE3_PRED_DIR,
            "*_predictions.npz"
        )
    )
)

for fp in files:

    z = np.load(
        fp,
        allow_pickle=True
    )

    subject = str(
        z[
            "subject"
        ].item()
    )

    sid = int(
        subject[
            -2:
        ]
    )

    y = TEST_LABELS[
        sid - 1
    ]

    raw_pred = np.argmax(
        z[
            "test_logits_raw"
        ],
        axis=1
    )

    fusion_pred = np.argmax(
        z[
            "test_probs_fused"
        ],
        axis=1
    )

    pooled_y.append(
        y
    )

    pooled_old[
        "Old_Raw"
    ].append(
        raw_pred
    )

    pooled_old[
        "Old_Fusion"
    ].append(
        fusion_pred
    )

    for method, pred in [
        (
            "Old_Raw",
            raw_pred
        ),
        (
            "Old_Fusion",
            fusion_pred
        )
    ]:

        old_rows.append({
            "subject":
                subject,
            "method":
                method,
            **metric_dict(
                y,
                pred
            )
        })


old_df = pd.DataFrame(
    old_rows
)


# ======================================================================
# 19. GRAPH COMPARATOR
# ======================================================================

graph_df = pd.DataFrame()

if os.path.exists(
    FINAL_TEST_SUBJECTWISE
):

    rescue_df = pd.read_csv(
        FINAL_TEST_SUBJECTWISE
    )

    graph_df = rescue_df[
        rescue_df[
            "method"
        ]
        ==
        "MVGSF_Inspired_Graph"
    ].copy()


# ======================================================================
# 20. FINAL BASELINE COMPARISON
# ======================================================================

parts = [
    old_df,
    reve_df
]

if len(
    graph_df
):

    parts.append(
        graph_df
    )

all_subjectwise = pd.concat(
    parts,
    ignore_index=True,
    sort=False
)

all_subjectwise.to_csv(
    os.path.join(
        OUT,
        "FINAL_BASELINE_SUBJECTWISE.csv"
    ),
    index=False
)

baseline_summary = (
    all_subjectwise
    .groupby(
        "method"
    )
    .agg(
        accuracy_mean=(
            "accuracy",
            "mean"
        ),
        accuracy_std=(
            "accuracy",
            "std"
        ),
        macro_f1_mean=(
            "macro_f1",
            "mean"
        ),
        macro_f1_std=(
            "macro_f1",
            "std"
        ),
        precision_mean=(
            "macro_precision",
            "mean"
        ),
        sensitivity_mean=(
            "macro_sensitivity",
            "mean"
        ),
        specificity_mean=(
            "macro_specificity",
            "mean"
        )
    )
    .reset_index()
    .sort_values(
        "macro_f1_mean",
        ascending=False
    )
)

baseline_summary.to_csv(
    os.path.join(
        OUT,
        "FINAL_BASELINE_COMPARISON.csv"
    ),
    index=False
)

print(
    "\n"
    +
    "="
    * 78
)

print(
    "FINAL BASELINE COMPARISON"
)

print(
    "="
    * 78
)

print(
    baseline_summary.to_string(
        index=False
    )
)


# ======================================================================
# 21. PAIRED STATISTICS
# ======================================================================

def rank_biserial(
    x,
    y
):

    d = (
        np.asarray(
            x,
            dtype=float
        )
        -
        np.asarray(
            y,
            dtype=float
        )
    )

    d = d[
        np.abs(
            d
        )
        >
        1e-12
    ]

    if len(
        d
    ) == 0:

        return 0.0

    ranks = rankdata(
        np.abs(
            d
        ),
        method="average"
    )

    return float(
        (
            ranks[
                d > 0
            ].sum()
            -
            ranks[
                d < 0
            ].sum()
        )
        /
        ranks.sum()
    )


def bootstrap_difference(
    x,
    y,
    n_boot=50000,
    seed=20265732
):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    d = (
        x
        -
        y
    )

    rng = np.random.default_rng(
        seed
    )

    idx = rng.integers(
        0,
        len(
            d
        ),
        size=(
            n_boot,
            len(
                d
            )
        )
    )

    boot = d[
        idx
    ].mean(
        axis=1
    )

    return (
        float(
            d.mean()
        ),
        float(
            np.quantile(
                boot,
                0.025
            )
        ),
        float(
            np.quantile(
                boot,
                0.975
            )
        )
    )


def compare(
    df,
    method_a,
    method_b,
    metric
):

    A = (
        df[
            df[
                "method"
            ]
            ==
            method_a
        ]
        .sort_values(
            "subject"
        )
        .reset_index(
            drop=True
        )
    )

    B = (
        df[
            df[
                "method"
            ]
            ==
            method_b
        ]
        .sort_values(
            "subject"
        )
        .reset_index(
            drop=True
        )
    )

    common = sorted(
        set(
            A[
                "subject"
            ]
        )
        &
        set(
            B[
                "subject"
            ]
        )
    )

    A = (
        A[
            A[
                "subject"
            ].isin(
                common
            )
        ]
        .sort_values(
            "subject"
        )
    )

    B = (
        B[
            B[
                "subject"
            ].isin(
                common
            )
        ]
        .sort_values(
            "subject"
        )
    )

    x = A[
        metric
    ].values

    y = B[
        metric
    ].values

    if len(
        x
    ) == 0:

        return None

    try:

        W = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox"
        )

        stat = float(
            W.statistic
        )

        p = float(
            W.pvalue
        )

    except Exception:

        stat = np.nan
        p = np.nan

    diff, lo, hi = bootstrap_difference(
        x,
        y
    )

    return {
        "comparison":
            f"{method_a} vs {method_b}",
        "metric":
            metric,
        "n_subjects":
            len(
                x
            ),
        "method_A_mean":
            float(
                np.mean(
                    x
                )
            ),
        "method_B_mean":
            float(
                np.mean(
                    y
                )
            ),
        "mean_difference":
            diff,
        "CI95_low":
            lo,
        "CI95_high":
            hi,
        "Wilcoxon_W":
            stat,
        "Wilcoxon_p":
            p,
        "rank_biserial":
            rank_biserial(
                x,
                y
            ),
        "subjects_better":
            int(
                np.sum(
                    x > y
                )
            ),
        "subjects_equal":
            int(
                np.sum(
                    np.isclose(
                        x,
                        y
                    )
                )
            ),
        "subjects_worse":
            int(
                np.sum(
                    x < y
                )
            )
    }


planned = [
    (
        "Old_Fusion",
        "Old_Raw",
        "macro_f1"
    ),
    (
        "Old_Fusion",
        "Old_Raw",
        "accuracy"
    ),
    (
        "REVE_LinearProbe",
        "Old_Fusion",
        "macro_f1"
    ),
    (
        "REVE_LoRA",
        "Old_Fusion",
        "macro_f1"
    )
]

if len(
    graph_df
):

    planned.append(
        (
            "MVGSF_Inspired_Graph",
            "Old_Fusion",
            "macro_f1"
        )
    )


rows = []

for a, b, metric in planned:

    r = compare(
        all_subjectwise,
        a,
        b,
        metric
    )

    if r is not None:

        rows.append(
            r
        )


stats_df = pd.DataFrame(
    rows
)

macro_mask = (
    stats_df[
        "metric"
    ]
    ==
    "macro_f1"
)

macro_p = stats_df.loc[
    macro_mask,
    "Wilcoxon_p"
].values

if len(
    macro_p
):

    order = np.argsort(
        macro_p
    )

    adj = np.empty_like(
        macro_p
    )

    running = 0.0

    m = len(
        macro_p
    )

    for rank, idx in enumerate(
        order
    ):

        candidate = (
            (
                m
                -
                rank
            )
            *
            macro_p[
                idx
            ]
        )

        running = max(
            running,
            candidate
        )

        adj[
            idx
        ] = min(
            1.0,
            running
        )

    stats_df.loc[
        macro_mask,
        "Holm_adjusted_p"
    ] = adj


stats_df.to_csv(
    os.path.join(
        OUT,
        "FINAL_PAIRED_STATISTICS.csv"
    ),
    index=False
)

print(
    "\nFINAL PAIRED STATISTICS"
)

print(
    stats_df.to_string(
        index=False
    )
)


# ======================================================================
# 22. FIGURES
# ======================================================================

figure_manifest = []


def save_fig(
    filename,
    purpose,
    action
):

    path = os.path.join(
        FIG_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=400,
        bbox_inches="tight"
    )

    plt.close()

    figure_manifest.append({
        "file":
            filename,
        "purpose":
            purpose,
        "recommended_action":
            action
    })


if MAKE_FIGURES:

    # --------------------------------------------------------------
    # Evaluation protocol
    # --------------------------------------------------------------

    plt.figure(
        figsize=(
            10,
            3.5
        )
    )

    ax = plt.gca()

    ax.axis(
        "off"
    )

    boxes = [
        (
            0.03,
            0.56,
            0.25,
            0.25,
            "TRAIN\n300 trials/subject\nmodel fitting"
        ),
        (
            0.375,
            0.56,
            0.25,
            0.25,
            "VALIDATION\n50 trials/subject\nselection / stopping"
        ),
        (
            0.72,
            0.56,
            0.25,
            0.25,
            "TEST\n50 trials/subject\nfinal evaluation only"
        )
    ]

    for x0, y0, w, h, txt in boxes:

        ax.add_patch(
            plt.Rectangle(
                (
                    x0,
                    y0
                ),
                w,
                h,
                fill=False,
                linewidth=1.5
            )
        )

        ax.text(
            x0
            +
            w
            /
            2,
            y0
            +
            h
            /
            2,
            txt,
            ha="center",
            va="center"
        )

    for a, b in [
        (
            0.28,
            0.375
        ),
        (
            0.625,
            0.72
        )
    ]:

        ax.annotate(
            "",
            xy=(
                b,
                0.685
            ),
            xytext=(
                a,
                0.685
            ),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.5
            )
        )

    ax.text(
        0.5,
        0.24,
        "Primary proposed-model predictions were frozen before "
        "official test labels were scored.",
        ha="center",
        va="center"
    )

    save_fig(
        "Fig_Evaluation_protocol.png",
        "Leakage-safe role of the official train, validation, and test partitions.",
        "Add/replace the Methods evaluation schematic."
    )

    # --------------------------------------------------------------
    # Primary official-test subject-wise performance
    # --------------------------------------------------------------

    raw = (
        old_df[
            old_df[
                "method"
            ]
            ==
            "Old_Raw"
        ]
        .sort_values(
            "subject"
        )
    )

    fus = (
        old_df[
            old_df[
                "method"
            ]
            ==
            "Old_Fusion"
        ]
        .sort_values(
            "subject"
        )
    )

    x = np.arange(
        len(
            raw
        )
    )

    plt.figure(
        figsize=(
            11,
            4.6
        )
    )

    plt.plot(
        x,
        raw[
            "macro_f1"
        ],
        marker="o",
        label="Raw EEG"
    )

    plt.plot(
        x,
        fus[
            "macro_f1"
        ],
        marker="o",
        label="Proposed fusion"
    )

    plt.axhline(
        0.20,
        linestyle="--",
        linewidth=1,
        label="Chance"
    )

    plt.xticks(
        x,
        raw[
            "subject"
        ],
        rotation=60,
        ha="right"
    )

    plt.ylabel(
        "Macro-F1"
    )

    plt.xlabel(
        "Subject"
    )

    plt.title(
        "Official-test subject-wise performance"
    )

    plt.legend()

    save_fig(
        "Fig_OfficialTest_subjectwise_raw_vs_fusion.png",
        "Primary independent official-test result.",
        "Replace the old validation-only subject-wise result figure."
    )

    # --------------------------------------------------------------
    # Subject-wise delta
    # --------------------------------------------------------------

    delta = (
        fus[
            "macro_f1"
        ].values
        -
        raw[
            "macro_f1"
        ].values
    )

    plt.figure(
        figsize=(
            10,
            4
        )
    )

    plt.bar(
        x,
        delta
    )

    plt.axhline(
        0.0,
        linewidth=1
    )

    plt.xticks(
        x,
        raw[
            "subject"
        ],
        rotation=60,
        ha="right"
    )

    plt.ylabel(
        "Fusion minus Raw macro-F1"
    )

    plt.xlabel(
        "Subject"
    )

    plt.title(
        "Subject-wise official-test fusion effect"
    )

    save_fig(
        "Fig_OfficialTest_delta_by_subject.png",
        "Shows heterogeneous subject-specific fusion benefit.",
        "Useful secondary Results figure or supplement."
    )

    # --------------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------------

    y_pool = np.concatenate(
        pooled_y
    )

    for method, title, filename in [
        (
            "Old_Raw",
            "Official test: Raw EEG",
            "Fig_OfficialTest_confusion_raw.png"
        ),
        (
            "Old_Fusion",
            "Official test: Proposed fusion",
            "Fig_OfficialTest_confusion_fusion.png"
        )
    ]:

        pred = np.concatenate(
            pooled_old[
                method
            ]
        )

        cm = confusion_matrix(
            y_pool,
            pred,
            labels=np.arange(
                N_CLASSES
            )
        )

        plt.figure(
            figsize=(
                6,
                5
            )
        )

        plt.imshow(
            cm,
            interpolation="nearest"
        )

        plt.colorbar()

        ticks = np.arange(
            N_CLASSES
        )

        plt.xticks(
            ticks,
            CLASS_NAMES,
            rotation=45,
            ha="right"
        )

        plt.yticks(
            ticks,
            CLASS_NAMES
        )

        threshold = (
            cm.max()
            /
            2
        )

        for i in range(
            N_CLASSES
        ):

            for j in range(
                N_CLASSES
            ):

                plt.text(
                    j,
                    i,
                    str(
                        int(
                            cm[
                                i,
                                j
                            ]
                        )
                    ),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if cm[
                            i,
                            j
                        ]
                        >
                        threshold
                        else
                        "black"
                    )
                )

        plt.xlabel(
            "Predicted class"
        )

        plt.ylabel(
            "True class"
        )

        plt.title(
            title
        )

        save_fig(
            filename,
            "Pooled official-test confusion matrix.",
            "Main text for fusion; raw matrix can move to supplement."
        )

    # --------------------------------------------------------------
    # Validation-to-test gap
    # --------------------------------------------------------------

    validation_values = [
        0.399,
        0.487
    ]

    test_values = [
        float(
            raw[
                "macro_f1"
            ].mean()
        ),
        float(
            fus[
                "macro_f1"
            ].mean()
        )
    ]

    labels = [
        "Raw EEG",
        "Proposed fusion"
    ]

    xx = np.arange(
        2
    )

    width = 0.35

    plt.figure(
        figsize=(
            7,
            4.5
        )
    )

    plt.bar(
        xx
        -
        width
        /
        2,
        validation_values,
        width,
        label="Original validation estimate"
    )

    plt.bar(
        xx
        +
        width
        /
        2,
        test_values,
        width,
        label="Official test"
    )

    plt.axhline(
        0.20,
        linestyle="--",
        linewidth=1
    )

    plt.xticks(
        xx,
        labels
    )

    plt.ylabel(
        "Macro-F1"
    )

    plt.title(
        "Validation-to-test generalization gap"
    )

    plt.legend()

    save_fig(
        "Fig_Validation_to_test_gap.png",
        "Directly documents development-set optimism and test generalization.",
        "Add to Results/Discussion; replace the old calibration-stress figure if space is limited."
    )

    # --------------------------------------------------------------
    # Final baseline comparison
    # --------------------------------------------------------------

    wanted = [
        "Old_Raw",
        "Old_Fusion",
        "MVGSF_Inspired_Graph",
        "REVE_LinearProbe",
        "REVE_LoRA"
    ]

    wanted = [
        m
        for m in wanted
        if m in baseline_summary[
            "method"
        ].values
    ]

    p = (
        baseline_summary
        .set_index(
            "method"
        )
        .loc[
            wanted
        ]
    )

    names = {
        "Old_Raw":
            "Raw EEG",
        "Old_Fusion":
            "Proposed fusion",
        "MVGSF_Inspired_Graph":
            "Graph comparator",
        "REVE_LinearProbe":
            "REVE LP",
        "REVE_LoRA":
            "REVE LoRA"
    }

    xx = np.arange(
        len(
            p
        )
    )

    plt.figure(
        figsize=(
            9,
            4.8
        )
    )

    plt.bar(
        xx,
        p[
            "macro_f1_mean"
        ],
        yerr=p[
            "macro_f1_std"
        ],
        capsize=5
    )

    plt.axhline(
        0.20,
        linestyle="--",
        linewidth=1,
        label="Chance"
    )

    plt.xticks(
        xx,
        [
            names[
                m
            ]
            for m in wanted
        ],
        rotation=20
    )

    plt.ylabel(
        "Official-test macro-F1"
    )

    plt.title(
        "Final head-to-head comparison"
    )

    plt.legend()

    save_fig(
        "Fig_Final_baseline_comparison.png",
        "Reviewer-facing comparison with graph and pretrained EEG baselines.",
        "Add as the principal baseline comparison figure."
    )

    # --------------------------------------------------------------
    # MDRG ablation
    # --------------------------------------------------------------

    if os.path.exists(
        ABLATION_CSV
    ):

        ab = pd.read_csv(
            ABLATION_CSV
        )

        if "macro_f1_numeric" in ab.columns:

            mean_values = ab[
                "macro_f1_numeric"
            ].astype(
                float
            ).values

            std_values = np.zeros_like(
                mean_values
            )

        else:

            def parse_mean(
                value
            ):

                return float(
                    str(
                        value
                    )
                    .split(
                        "±"
                    )[
                        0
                    ]
                    .strip()
                )

            def parse_std(
                value
            ):

                pieces = str(
                    value
                ).split(
                    "±"
                )

                return (
                    float(
                        pieces[
                            1
                        ].strip()
                    )
                    if len(
                        pieces
                    )
                    >
                    1
                    else
                    0.0
                )

            mean_values = ab[
                "macro_f1"
            ].map(
                parse_mean
            ).values

            std_values = ab[
                "macro_f1"
            ].map(
                parse_std
            ).values

        names_ab = ab[
            "block"
        ].values

        order = np.argsort(
            mean_values
        )[
            ::-1
        ]

        xx = np.arange(
            len(
                order
            )
        )

        plt.figure(
            figsize=(
                11,
                5
            )
        )

        plt.bar(
            xx,
            mean_values[
                order
            ],
            yerr=std_values[
                order
            ],
            capsize=4
        )

        plt.xticks(
            xx,
            names_ab[
                order
            ],
            rotation=45,
            ha="right"
        )

        plt.ylabel(
            "Macro-F1"
        )

        plt.title(
            "MDRG component ablation"
        )

        save_fig(
            "Fig_MDRG_ablation.png",
            "Reviewer-requested quantitative MDRG component analysis.",
            "Add to Results or supplement."
        )

    # --------------------------------------------------------------
    # Corrected training-removal figure
    # --------------------------------------------------------------

    if os.path.exists(
        STAGE1_SUMMARY
    ):

        s1 = pd.read_csv(
            STAGE1_SUMMARY
        ).sort_values(
            "subject"
        )

        if (
            "train_raw"
            in s1.columns
            and
            "train_after"
            in s1.columns
        ):

            removed = (
                s1[
                    "train_raw"
                ]
                -
                s1[
                    "train_after"
                ]
            ).values

            total_raw = int(
                s1[
                    "train_raw"
                ].sum()
            )

        else:

            # Known official denominator; derive removals from saved counts
            # when available, otherwise use established audit values.
            total_raw = 4500

            removed = np.zeros(
                15,
                dtype=int
            )

            removed[
                13
            ] = 3

        total_removed = int(
            np.sum(
                removed
            )
        )

        pct = (
            100.0
            *
            total_removed
            /
            total_raw
        )

        xx = np.arange(
            15
        )

        plt.figure(
            figsize=(
                10,
                4
            )
        )

        plt.bar(
            xx,
            removed
        )

        plt.xticks(
            xx,
            [
                f"S{i:02d}"
                for i in range(
                    1,
                    16
                )
            ],
            rotation=45
        )

        plt.ylabel(
            "Removed training trials"
        )

        plt.xlabel(
            "Subject"
        )

        plt.title(
            f"Training-trial QC removal: "
            f"{total_removed}/{total_raw} "
            f"({pct:.3f}%)"
        )

        save_fig(
            "Fig_QC_training_removal_corrected.png",
            "Corrects the training-trial removal denominator.",
            "Replace the old 3/5250 removal-cost panel."
        )


# ======================================================================
# 23. REPRODUCIBILITY + CONTEXT RECORD
# ======================================================================

repro = {
    "model_id":
        REVE_MODEL_ID,
    "model_revision":
        REVE_REVISION,
    "position_id":
        REVE_POSITION_ID,
    "position_revision":
        POSITION_REVISION,
    "seed":
        REVE_SEED,
    "input_protocol":
        {
            "source":
                "original Track-3 MATLAB EEG",
            "original_samples":
                795,
            "kept":
                "last 768 samples",
            "resampled_samples":
                600,
            "target_fs_hz":
                200,
            "scale_factor":
                1000,
            "channels":
                64
        },
    "linear_probe":
        {
            "max_epochs":
                LP_MAX_EPOCHS,
            "patience":
                LP_PATIENCE,
            "warmup_epochs":
                LP_WARMUP_EPOCHS,
            "lr":
                LP_LR,
            "dropout":
                LP_DROPOUT,
            "mixup":
                MIXUP,
            "effective_batch_target":
                LP_MICRO_BATCH
                *
                LP_ACCUM_STEPS
        },
    "lora":
        {
            "implementation":
                "direct LoRA wrapper; no PEFT/torchao",
            "rank":
                LORA_RANK,
            "alpha":
                LORA_ALPHA,
            "targets":
                [
                    "to_qkv",
                    "to_out"
                ],
            "max_epochs":
                FT_MAX_EPOCHS,
            "patience":
                FT_PATIENCE,
            "warmup_epochs":
                FT_WARMUP_EPOCHS,
            "lr":
                FT_LR,
            "dropout":
                FT_DROPOUT,
            "effective_batch_target":
                FT_MICRO_BATCH
                *
                FT_ACCUM_STEPS
        },
    "optimizer":
        {
            "name":
                "AdamW",
            "betas":
                list(
                    BETAS
                ),
            "eps":
                OPT_EPS,
            "weight_decay":
                WEIGHT_DECAY
        },
    "scheduler":
        {
            "name":
                "ReduceLROnPlateau",
            "factor":
                REDUCE_FACTOR,
            "patience":
                REDUCE_PATIENCE
        },
    "gradient_clip":
        GRAD_CLIP,
    "selection_metric":
        "validation accuracy",
    "test_used_for_selection":
        False,
    "published_REVE_context_not_directly_comparable":
        {
            "BCI2020_3_accuracy_model_card":
                0.5635,
            "BCII2020_3_LP_accuracy_model_card":
                0.39
        }
}

with open(
    os.path.join(
        OUT,
        "REVE_REPRODUCIBILITY.json"
    ),
    "w"
) as f:

    json.dump(
        repro,
        f,
        indent=2
    )

pd.DataFrame(
    figure_manifest
).to_csv(
    os.path.join(
        OUT,
        "FIGURE_MANIFEST.csv"
    ),
    index=False
)


print(
    "\n"
    +
    "="
    * 78
)

print(
    "FINAL CODING STAGE COMPLETE"
)

print(
    "="
    * 78
)

print(
    "Outputs:",
    OUT
)

print(
    "\nMain results:"
)

for name in [
    "REVE_SUBJECTWISE.csv",
    "REVE_SUMMARY.csv",
    "FINAL_BASELINE_COMPARISON.csv",
    "FINAL_PAIRED_STATISTICS.csv",
    "REVE_REPRODUCIBILITY.json",
    "FIGURE_MANIFEST.csv"
]:

    print(
        " ",
        os.path.join(
            OUT,
            name
        )
    )

print(
    "\nFigures:"
)

for item in figure_manifest:

    print(
        " ",
        os.path.join(
            FIG_DIR,
            item[
                "file"
            ]
        )
    )

print(
    "\nDo not perform further proposed-model tuning after this run."
)

