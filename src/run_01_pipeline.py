"""
Stage 1 to 3: preprocessing, MDRG features, branch training, selection and fusion.

Part of: Graph-Riemannian Multi-View Fusion for Subject-Specific EEG Imagined
Speech Decoding: A Held-Out Evaluation on the BCIC2020-IV Official Test Set.

This file is the original experimental script, unmodified except that the
hard-coded Google Drive paths have been replaced by a lookup in
configs/config.json (see src/paths.py). Run it from the repository root:

    python src/run_01_pipeline.py

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

# ======================================================================
# INASS FINAL REVISION + PERFORMANCE RESCUE PIPELINE
# ======================================================================
#
# FINAL DATA PROTOCOL
# -------------------
# Official TRAIN      : model fitting
# Official VALIDATION : architecture, seed, C/gamma, calibration,
#                       and fusion selection
# Official TEST       : untouched FINAL evaluation only
#
# IMPORTANT:
# Do not modify choices after inspecting FINAL TEST metrics.
#
# ======================================================================

import os
import glob
import json
import math
import time
import random
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import signal
from scipy.special import softmax

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.metrics import pairwise_distances
from scipy.stats import wilcoxon

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")


# ======================================================================
# 0. PATHS
# ======================================================================

# ROOT now comes from configs/config.json via src/paths.py
STAGE1_ROOT = os.path.join(
    ROOT,
    "prep_stage1_final_robust"
)

STAGE1_SUMMARY = os.path.join(
    STAGE1_ROOT,
    "GLOBAL_stage1final_summary.csv"
)

MDRG_DIR = os.path.join(
    ROOT,
    "INASS_REVISION_FINAL",
    "stage2_MDRG_corrected",
    "npz"
)

OLD_STAGE3_DIR = os.path.join(
    ROOT,
    "stage3_dual_fusion_v3_final",
    "predictions"
)

OUT = os.path.join(
    ROOT,
    "INASS_FINAL_TEST_REVISION"
)

CACHE = os.path.join(
    OUT,
    "cache"
)

os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)


# ======================================================================
# 1. SWITCHES
# ======================================================================

RUN_OLD_PREDICTION_AUDIT = False

RUN_DEEP_RESCUE = True

RUN_MDRG_RESCUE = True

RUN_FUSION = True

RUN_CONSENSUS_GRAPH_BASELINE = True

# Requires Hugging Face access and is much faster on GPU.
RUN_REVE = False


# ======================================================================
# 2. REPRODUCIBILITY
# ======================================================================

GLOBAL_SEED = 20265732

DEEP_SEEDS = [
    0,
    1,
    2
]

N_CLASSES = 5

CLASS_NAMES = [
    "hello",
    "help me",
    "stop",
    "thank you",
    "yes"
]

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "DEVICE:",
    DEVICE
)


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(
    GLOBAL_SEED
)


# ======================================================================
# 3. OFFICIAL TEST ANSWER SHEET
# ======================================================================
#
# Public OSF version-1 URL indexed in the Track-3 file manifest:
#
# Track3_Answer Sheet_Test.xlsx
#
# SHA256:
# 7010974ad753b961684b1d9cb1cd55a5279904523afe5b9a3f051e9f53e6b6c7
#
# ======================================================================

ANSWER_URL = (
    "https://osf.io/download/n3a2p/?version=1"
)

ANSWER_SHA256 = (
    "7010974ad753b961684b1d9cb1cd55a5279904523afe5b9a3f051e9f53e6b6c7"
)

ANSWER_FILE = os.path.join(
    CACHE,
    "Track3_Answer_Sheet_Test.xlsx"
)


def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def download_answer_sheet():

    if (
        os.path.exists(
            ANSWER_FILE
        )
        and sha256_file(
            ANSWER_FILE
        )
        == ANSWER_SHA256
    ):

        print(
            "Test answer sheet already present and checksum verified."
        )

        return ANSWER_FILE

    import requests

    print(
        "Downloading official Track-3 test answer sheet..."
    )

    response = requests.get(
        ANSWER_URL,
        timeout=60
    )

    response.raise_for_status()

    with open(
        ANSWER_FILE,
        "wb"
    ) as f:

        f.write(
            response.content
        )

    actual = sha256_file(
        ANSWER_FILE
    )

    if actual != ANSWER_SHA256:

        raise RuntimeError(
            "\nChecksum mismatch for test answer sheet.\n"
            f"Expected: {ANSWER_SHA256}\n"
            f"Actual:   {actual}\n"
            "Do not use this file."
        )

    print(
        "Answer sheet checksum verified."
    )

    return ANSWER_FILE


def load_test_labels():

    path = download_answer_sheet()

    # This reproduces the layout used by the public BCIC2020
    # preprocessing code. The Track3 sheet has participant pairs
    # and rows 4-53 contain 50 true labels per participant.

    df = pd.read_excel(
        path,
        sheet_name="Track3"
    )

    values = (
        df.head(
            53
        )
        .values
    )

    labels = (
        values[
            2:,
            1:
        ][
            :,
            1:30:2
        ]
        .T
    )

    labels = np.asarray(
        labels,
        dtype=int
    )

    if labels.shape != (
        15,
        50
    ):

        raise RuntimeError(
            f"Unexpected test-label shape: {labels.shape}"
        )

    # Source event codes are 1...5.

    if (
        labels.min() == 1
        and labels.max() == 5
    ):

        labels = (
            labels
            - 1
        )

    if not (
        labels.min() == 0
        and labels.max() == 4
    ):

        raise RuntimeError(
            "Unexpected test labels."
        )

    print(
        "Loaded test labels:",
        labels.shape
    )

    print(
        "Per-subject class counts:"
    )

    for s in range(
        15
    ):

        print(
            f"S{s+1:02d}:",
            np.bincount(
                labels[s],
                minlength=5
            ).tolist()
        )

    np.save(
        os.path.join(
            OUT,
            "official_test_labels.npy"
        ),
        labels
    )

    return labels


TEST_LABELS = load_test_labels()


# ======================================================================
# 4. DATA LOADING
# ======================================================================

stage1_df = (
    pd.read_csv(
        STAGE1_SUMMARY
    )
    .sort_values(
        "subject"
    )
    .reset_index(
        drop=True
    )
)


def subject_number(
    subject_name
):

    return int(
        str(
            subject_name
        )[
            -2:
        ]
    )


def load_subject_data(
    row
):

    sid = str(
        row[
            "subject"
        ]
    )

    sidx = (
        subject_number(
            sid
        )
        - 1
    )

    z1 = np.load(
        str(
            row[
                "npz_path"
            ]
        ),
        allow_pickle=True
    )

    mdrg_path = os.path.join(
        MDRG_DIR,
        f"{sid}_MDRG_corrected.npz"
    )

    z2 = np.load(
        mdrg_path,
        allow_pickle=True
    )

    data = {

        "sid":
            sid,

        "X_train":
            z1[
                "X_train"
            ].astype(
                np.float32
            ),

        "y_train":
            z1[
                "y_train"
            ].astype(
                np.int64
            ),

        "X_valid":
            z1[
                "X_valid"
            ].astype(
                np.float32
            ),

        "y_valid":
            z1[
                "y_valid"
            ].astype(
                np.int64
            ),

        "X_test":
            z1[
                "X_test"
            ].astype(
                np.float32
            ),

        "y_test":
            TEST_LABELS[
                sidx
            ].astype(
                np.int64
            ),

        "F_train":
            z2[
                "X_train"
            ].astype(
                np.float32
            ),

        "F_valid":
            z2[
                "X_valid"
            ].astype(
                np.float32
            ),

        "F_test":
            z2[
                "X_test"
            ].astype(
                np.float32
            ),

        "stage1_npz":
            str(
                row[
                    "npz_path"
                ]
            )
    }

    assert (
        data[
            "X_train"
        ].shape[0]
        ==
        data[
            "F_train"
        ].shape[0]
    )

    assert (
        data[
            "X_valid"
        ].shape[0]
        ==
        data[
            "F_valid"
        ].shape[0]
    )

    assert (
        data[
            "X_test"
        ].shape[0]
        ==
        data[
            "F_test"
        ].shape[0]
        ==
        50
    )

    return data


# ======================================================================
# 5. METRICS
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

    specs = []

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
            - tp
        )

        fp = (
            cm[
                :,
                k
            ].sum()
            - tp
        )

        tn = (
            total
            - tp
            - fn
            - fp
        )

        specs.append(
            tn
            / max(
                1,
                tn + fp
            )
        )

    return float(
        np.mean(
            specs
        )
    )


def metrics(
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
            macro_specificity(
                y,
                pred
            )
    }


# ======================================================================
# 6. AUDIT EXISTING STAGE-3 TEST PREDICTIONS
# ======================================================================

def first_key(
    z,
    names
):

    for name in names:

        if name in z.files:

            return name

    return None


def old_prediction_audit():

    files = sorted(
        glob.glob(
            os.path.join(
                OLD_STAGE3_DIR,
                "*_predictions.npz"
            )
        )
    )

    if not files:

        print(
            "No old prediction files found."
        )

        return None

    rows = []

    print(
        "\nOLD STAGE-3 TEST-PREDICTION AUDIT"
    )

    for fp in files:

        z = np.load(
            fp,
            allow_pickle=True
        )

        print(
            "\n",
            os.path.basename(
                fp
            )
        )

        print(
            "keys:",
            z.files
        )

        sid_key = first_key(
            z,
            [
                "subject",
                "sid"
            ]
        )

        if sid_key is None:
            continue

        sid = str(
            np.asarray(
                z[
                    sid_key
                ]
            ).item()
        )

        sidx = (
            subject_number(
                sid
            )
            - 1
        )

        y = TEST_LABELS[
            sidx
        ]

        # Raw

        raw_pred = None

        k = first_key(
            z,
            [
                "test_pred_raw",
                "test_preds_raw"
            ]
        )

        if k is not None:

            raw_pred = np.asarray(
                z[k]
            ).astype(int)

        else:

            k = first_key(
                z,
                [
                    "test_logits_raw"
                ]
            )

            if k is not None:

                raw_pred = np.argmax(
                    z[k],
                    axis=1
                )

        # Fused

        fused_pred = None

        k = first_key(
            z,
            [
                "test_pred_fused",
                "test_preds_fused"
            ]
        )

        if k is not None:

            fused_pred = np.asarray(
                z[k]
            ).astype(int)

        else:

            k = first_key(
                z,
                [
                    "test_probs_fused",
                    "test_probability_fused"
                ]
            )

            if k is not None:

                fused_pred = np.argmax(
                    z[k],
                    axis=1
                )

        for method, pred in [
            (
                "Old_Raw",
                raw_pred
            ),
            (
                "Old_Fusion",
                fused_pred
            )
        ]:

            if (
                pred is not None
                and len(
                    pred
                )
                == 50
            ):

                m = metrics(
                    y,
                    pred
                )

                rows.append({
                    "subject":
                        sid,

                    "method":
                        method,

                    **m
                })

    if rows:

        out = pd.DataFrame(
            rows
        )

        out.to_csv(
            os.path.join(
                OUT,
                "old_stage3_test_audit.csv"
            ),
            index=False
        )

        print(
            "\nOld prediction test results:"
        )

        print(
            out.groupby(
                "method"
            )[
                [
                    "accuracy",
                    "macro_f1"
                ]
            ].agg(
                [
                    "mean",
                    "std"
                ]
            )
        )

        return out

    return None


if RUN_OLD_PREDICTION_AUDIT:

    old_audit = old_prediction_audit()


# ======================================================================
# 7. DEEP MODELS
# ======================================================================

class EEGNetLite(
    nn.Module
):

    def __init__(
        self,
        n_ch=64,
        n_classes=5,
        dropout=0.35
    ):

        super().__init__()

        self.temporal = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=(
                    1,
                    33
                ),
                padding=(
                    0,
                    16
                ),
                bias=False
            ),

            nn.BatchNorm2d(
                16
            )
        )

        self.spatial = nn.Sequential(

            nn.Conv2d(
                16,
                32,
                kernel_size=(
                    n_ch,
                    1
                ),
                groups=16,
                bias=False
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ELU(),

            nn.AvgPool2d(
                (
                    1,
                    4
                )
            ),

            nn.Dropout(
                dropout
            )
        )

        self.sep = nn.Sequential(

            nn.Conv2d(
                32,
                32,
                kernel_size=(
                    1,
                    17
                ),
                padding=(
                    0,
                    8
                ),
                groups=32,
                bias=False
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=(
                    1,
                    1
                ),
                bias=False
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ELU(),

            nn.AvgPool2d(
                (
                    1,
                    4
                )
            ),

            nn.Dropout(
                dropout
            )
        )

        self.head = nn.Sequential(

            nn.AdaptiveAvgPool2d(
                (
                    1,
                    1
                )
            ),

            nn.Flatten(),

            nn.Linear(
                64,
                n_classes
            )
        )

    def forward(
        self,
        x
    ):

        x = x.unsqueeze(
            1
        )

        x = self.temporal(
            x
        )

        x = self.spatial(
            x
        )

        x = self.sep(
            x
        )

        return self.head(
            x
        )


class EEGConformerLite(
    nn.Module
):

    def __init__(
        self,
        n_ch=64,
        n_classes=5,
        emb=64,
        dropout=0.30
    ):

        super().__init__()

        self.temporal = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=(
                    1,
                    25
                ),
                padding=(
                    0,
                    12
                ),
                bias=False
            ),

            nn.BatchNorm2d(
                32
            )
        )

        self.spatial = nn.Sequential(

            nn.Conv2d(
                32,
                64,
                kernel_size=(
                    n_ch,
                    1
                ),
                groups=32,
                bias=False
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ELU(),

            nn.AvgPool2d(
                (
                    1,
                    4
                )
            ),

            nn.Dropout(
                dropout
            )
        )

        self.project = nn.Linear(
            64,
            emb
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=emb,
                nhead=4,
                dim_feedforward=emb * 3,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=2
            )
        )

        self.norm = nn.LayerNorm(
            emb
        )

        self.head = nn.Linear(
            emb,
            n_classes
        )

    def forward(
        self,
        x
    ):

        x = x.unsqueeze(
            1
        )

        x = self.temporal(
            x
        )

        x = self.spatial(
            x
        )

        # B, 64, 1, T
        x = x.squeeze(
            2
        ).transpose(
            1,
            2
        )

        x = self.project(
            x
        )

        x = self.transformer(
            x
        )

        x = self.norm(
            x.mean(
                dim=1
            )
        )

        return self.head(
            x
        )


# ======================================================================
# 8. DEEP TRAINING
# ======================================================================

BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1.5e-4
MAX_EPOCHS = 90
PATIENCE = 15
LABEL_SMOOTH = 0.05
MIXUP_ALPHA = 0.25


def make_loader(
    X,
    y,
    shuffle
):

    ds = TensorDataset(

        torch.tensor(
            X,
            dtype=torch.float32
        ),

        torch.tensor(
            y,
            dtype=torch.long
        )
    )

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )


def soft_ce(
    logits,
    target
):

    return -(
        target
        *
        F.log_softmax(
            logits,
            dim=1
        )
    ).sum(
        dim=1
    ).mean()


def train_with_validation(
    model,
    Xtr,
    ytr,
    Xva,
    yva,
    seed
):

    set_seed(
        seed
    )

    model = model.to(
        DEVICE
    )

    tr_loader = make_loader(
        Xtr,
        ytr,
        True
    )

    va_loader = make_loader(
        Xva,
        yva,
        False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS
        )
    )

    best_state = None
    best_f1 = -1
    best_epoch = 1
    wait = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        for xb, yb in tr_loader:

            xb = xb.to(
                DEVICE
            )

            yb = yb.to(
                DEVICE
            )

            target = F.one_hot(
                yb,
                N_CLASSES
            ).float()

            target = (
                (
                    1
                    -
                    LABEL_SMOOTH
                )
                * target
                +
                LABEL_SMOOTH
                / N_CLASSES
            )

            lam = np.random.beta(
                MIXUP_ALPHA,
                MIXUP_ALPHA
            )

            perm = torch.randperm(
                xb.size(
                    0
                ),
                device=xb.device
            )

            xb_mix = (
                lam
                * xb
                +
                (
                    1
                    -
                    lam
                )
                * xb[
                    perm
                ]
            )

            y_mix = (
                lam
                * target
                +
                (
                    1
                    -
                    lam
                )
                * target[
                    perm
                ]
            )

            logits = model(
                xb_mix
            )

            loss = soft_ce(
                logits,
                y_mix
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

        scheduler.step()

        pred = predict_logits(
            model,
            Xva
        ).argmax(
            axis=1
        )

        score = f1_score(
            yva,
            pred,
            average="macro",
            zero_division=0
        )

        if score > best_f1 + 1e-8:

            best_f1 = float(
                score
            )

            best_epoch = epoch

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

        if wait >= PATIENCE:
            break

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_f1,
        best_epoch
    )


def fit_fixed_epochs(
    model,
    X,
    y,
    seed,
    epochs
):

    set_seed(
        seed
    )

    model = model.to(
        DEVICE
    )

    loader = make_loader(
        X,
        y,
        True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(
                1,
                epochs
            )
        )
    )

    for _ in range(
        epochs
    ):

        model.train()

        for xb, yb in loader:

            xb = xb.to(
                DEVICE
            )

            yb = yb.to(
                DEVICE
            )

            target = F.one_hot(
                yb,
                N_CLASSES
            ).float()

            target = (
                (
                    1
                    -
                    LABEL_SMOOTH
                )
                * target
                +
                LABEL_SMOOTH
                / N_CLASSES
            )

            lam = np.random.beta(
                MIXUP_ALPHA,
                MIXUP_ALPHA
            )

            perm = torch.randperm(
                xb.size(
                    0
                ),
                device=xb.device
            )

            logits = model(
                lam
                * xb
                +
                (
                    1
                    -
                    lam
                )
                * xb[
                    perm
                ]
            )

            target_mix = (
                lam
                * target
                +
                (
                    1
                    -
                    lam
                )
                * target[
                    perm
                ]
            )

            loss = soft_ce(
                logits,
                target_mix
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

        scheduler.step()

    return model


def predict_logits(
    model,
    X
):

    model.eval()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X,
                dtype=torch.float32
            )
        ),
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    out = []

    with torch.no_grad():

        for batch in loader:

            xb = batch[
                0
            ].to(
                DEVICE
            )

            out.append(
                model(
                    xb
                ).cpu().numpy()
            )

    return np.concatenate(
        out,
        axis=0
    )


def make_deep_model(
    name
):

    if name == "EEGNetLite":

        return EEGNetLite()

    if name == "EEGConformerLite":

        return EEGConformerLite()

    raise ValueError(
        name
    )


# ======================================================================
# 9. MDRG BLOCKS
# ======================================================================

STATIC = []

RIEM = []

GRAPH = []

for block in range(
    8
):

    base = (
        block
        * 198
    )

    STATIC += list(
        range(
            base,
            base + 18
        )
    )

    RIEM += list(
        range(
            base + 18,
            base + 189
        )
    )

    GRAPH += list(
        range(
            base + 189,
            base + 198
        )
    )

DYNAMIC = list(
    range(
        1584,
        1656
    )
)

MDRG_BLOCKS = {

    "Full":
        list(
            range(
                1656
            )
        ),

    "Riemannian":
        RIEM,

    "RiemannianGraph":
        sorted(
            RIEM
            +
            GRAPH
        ),

    "SpectralDynamic":
        sorted(
            STATIC
            +
            DYNAMIC
        )
}


# ======================================================================
# 10. STRONG MDRG MODEL SELECTION
# ======================================================================

def fit_feature_candidate(
    Ftr,
    ytr,
    Fva,
    yva,          # <-- ADDED
    block,
    model_type,
    parameter
):

    idx = MDRG_BLOCKS[
        block
    ]

    scaler = StandardScaler()

    Xtr = scaler.fit_transform(
        Ftr[
            :,
            idx
        ]
    )

    Xva = scaler.transform(
        Fva[
            :,
            idx
        ]
    )

    if model_type == "LR":

        clf = LogisticRegression(
            C=parameter,
            solver="lbfgs",
            max_iter=5000,
            random_state=GLOBAL_SEED
        )

    elif model_type == "RBF_SVM":

        clf = SVC(
            C=parameter,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=GLOBAL_SEED
        )

    else:

        raise ValueError(
            model_type
        )

    clf.fit(
        Xtr,
        ytr
    )

    p = clf.predict_proba(
        Xva
    )

    pred = np.argmax(
        p,
        axis=1
    )

    score = f1_score(
        yva,       # <-- NOW DEFINED
        pred,
        average="macro",
        zero_division=0
    )

    return (
        scaler,
        clf,
        p,
        float(
            score
        )
    )


def select_feature_model(
    d
):

    candidates = []

    for block in [
        "Full",
        "Riemannian",
        "RiemannianGraph",
        "SpectralDynamic"
    ]:

        for C in [
            0.01,
            0.1,
            1.0,
            10.0
        ]:

            candidates.append(
                (
                    block,
                    "LR",
                    C
                )
            )

        for C in [
            0.1,
            1.0,
            10.0
        ]:

            candidates.append(
                (
                    block,
                    "RBF_SVM",
                    C
                )
            )

    best = None

    for (
        block,
        model_type,
        parameter
    ) in candidates:

        scaler, clf, p, score = (
            fit_feature_candidate(
                d[
                    "F_train"
                ],
                d[
                    "y_train"
                ],
                d[
                    "F_valid"
                ],
                d[
                    "y_valid"
                ],          # <-- ADDED
                block,
                model_type,
                parameter
            )
        )

        candidate = {

            "block":
                block,

            "model_type":
                model_type,

            "parameter":
                parameter,

            "validation_f1":
                score,

            "scaler":
                scaler,

            "model":
                clf,

            "valid_prob":
                p
        }

        if (
            best is None
            or score
            >
            best[
                "validation_f1"
            ]
        ):

            best = candidate

    # Train-only selected model test probability.

    idx = MDRG_BLOCKS[
        best[
            "block"
        ]
    ]

    Xtest = (
        best[
            "scaler"
        ]
        .transform(
            d[
                "F_test"
            ][
                :,
                idx
            ]
        )
    )

    best[
        "test_prob_train_only"
    ] = (
        best[
            "model"
        ]
        .predict_proba(
            Xtest
        )
    )

    # Refit selected configuration on TRAIN+VALID,
    # then evaluate untouched TEST.

    Fdev = np.concatenate(
        [
            d[
                "F_train"
            ],
            d[
                "F_valid"
            ]
        ],
        axis=0
    )

    ydev = np.concatenate(
        [
            d[
                "y_train"
            ],
            d[
                "y_valid"
            ]
        ],
        axis=0
    )

    scaler = StandardScaler()

    Xdev = scaler.fit_transform(
        Fdev[
            :,
            idx
        ]
    )

    Xtest = scaler.transform(
        d[
            "F_test"
        ][
            :,
            idx
        ]
    )

    if best[
        "model_type"
    ] == "LR":

        final_model = LogisticRegression(
            C=best[
                "parameter"
            ],
            solver="lbfgs",
            max_iter=5000,
            random_state=GLOBAL_SEED
        )

    else:

        final_model = SVC(
            C=best[
                "parameter"
            ],
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=GLOBAL_SEED
        )

    final_model.fit(
        Xdev,
        ydev
    )

    best[
        "test_prob_refit"
    ] = final_model.predict_proba(
        Xtest
    )

    return best


# ======================================================================
# 11. RAW DEEP MODEL SELECTION
# ======================================================================

def select_deep_model(
    d
):

    best = None

    for architecture in [
        "EEGNetLite",
        "EEGConformerLite"
    ]:

        for seed in DEEP_SEEDS:

            model = make_deep_model(
                architecture
            )

            model, score, epoch = (
                train_with_validation(
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
                    seed
                )
            )

            valid_logits = predict_logits(
                model,
                d[
                    "X_valid"
                ]
            )

            test_logits = predict_logits(
                model,
                d[
                    "X_test"
                ]
            )

            candidate = {

                "architecture":
                    architecture,

                "seed":
                    seed,

                "epoch":
                    epoch,

                "validation_f1":
                    score,

                "valid_prob":
                    softmax(
                        valid_logits,
                        axis=1
                    ),

                "test_prob_train_only":
                    softmax(
                        test_logits,
                        axis=1
                    )
            }

            if (
                best is None
                or score
                >
                best[
                    "validation_f1"
                ]
            ):

                best = candidate

            del model

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

    # Standard refit on all labelled development data
    # using architecture/seed/epoch chosen without TEST.

    Xdev = np.concatenate(
        [
            d[
                "X_train"
            ],
            d[
                "X_valid"
            ]
        ],
        axis=0
    )

    ydev = np.concatenate(
        [
            d[
                "y_train"
            ],
            d[
                "y_valid"
            ]
        ],
        axis=0
    )

    final_model = make_deep_model(
        best[
            "architecture"
        ]
    )

    final_model = fit_fixed_epochs(
        final_model,
        Xdev,
        ydev,
        best[
            "seed"
        ],
        best[
            "epoch"
        ]
    )

    logits = predict_logits(
        final_model,
        d[
            "X_test"
        ]
    )

    best[
        "test_prob_refit"
    ] = softmax(
        logits,
        axis=1
    )

    return best


# ======================================================================
# 12. GLOBAL VALIDATION-SELECTED FUSION
# ======================================================================

FUSION_GRID = np.linspace(
    0.0,
    1.0,
    41
)


def choose_fusion_weight(
    y_valid,
    raw_prob,
    feat_prob
):

    best = {
        "weight": 0.5,
        "f1": -1.0
    }

    for w in FUSION_GRID:

        p = (
            w
            * raw_prob
            +
            (
                1.0
                -
                w
            )
            * feat_prob
        )

        pred = np.argmax(
            p,
            axis=1
        )

        score = f1_score(
            y_valid,
            pred,
            average="macro",
            zero_division=0
        )

        if score > best[
            "f1"
        ] + 1e-12:

            best = {
                "weight":
                    float(
                        w
                    ),

                "f1":
                    float(
                        score
                    )
            }

    return best


# ======================================================================
# 13. MVGSF-INSPIRED CONSENSUS-GRAPH BASELINE
#
# This is a controlled leakage-safe adaptation of the multi-view
# consensus-graph principle. It does NOT claim to be a bit-for-bit
# reproduction of the authors' MATLAB program because the public
# implementation selects parameters using target labels.
#
# Validation labels select graph parameters.
# Test labels are never used by the graph learner.
#
# ======================================================================

def view_arrays(
    F
):

    return [
        F[
            :,
            sorted(
                STATIC
                +
                DYNAMIC
            )
        ],

        F[
            :,
            RIEM
        ],

        F[
            :,
            GRAPH
        ]
    ]


def simplex_weight_grid(
    step=0.25
):

    vals = np.arange(
        0,
        1.0001,
        step
    )

    out = []

    for a in vals:

        for b in vals:

            c = (
                1.0
                -
                a
                -
                b
            )

            if c < -1e-9:
                continue

            if abs(
                round(
                    c / step
                )
                * step
                -
                c
            ) < 1e-7:

                out.append(
                    (
                        float(
                            a
                        ),
                        float(
                            b
                        ),
                        float(
                            max(
                                0,
                                c
                            )
                        )
                    )
                )

    return out


GRAPH_WEIGHTS = simplex_weight_grid(
    0.25
)

GRAPH_GAMMAS = [
    0.25,
    0.5,
    1.0,
    2.0
]

GRAPH_K = [
    10,
    20
]


def prepare_graph_views(
    train_views,
    target_views
):

    train_out = []

    target_out = []

    for Xtr, Xtar in zip(
        train_views,
        target_views
    ):

        scaler = StandardScaler()

        A = scaler.fit_transform(
            Xtr
        )

        B = scaler.transform(
            Xtar
        )

        train_out.append(
            A
        )

        target_out.append(
            B
        )

    return (
        train_out,
        target_out
    )


def consensus_graph_predict(
    train_views,
    y_train,
    target_views,
    weights,
    gamma,
    k_neighbors,
    alpha=0.99,
    max_iter=200
):

    train_views, target_views = (
        prepare_graph_views(
            train_views,
            target_views
        )
    )

    combined_views = [
        np.concatenate(
            [
                tr,
                tar
            ],
            axis=0
        )
        for tr, tar
        in zip(
            train_views,
            target_views
        )
    ]

    n_l = len(
        y_train
    )

    n = combined_views[
        0
    ].shape[
        0
    ]

    K = np.zeros(
        (
            n,
            n
        ),
        dtype=np.float64
    )

    for w, X in zip(
        weights,
        combined_views
    ):

        if w <= 0:
            continue

        D = pairwise_distances(
            X,
            metric="sqeuclidean"
        )

        nz = D[
            D > 0
        ]

        scale = (
            np.median(
                nz
            )
            if nz.size
            else 1.0
        )

        D = D / max(
            scale,
            1e-12
        )

        K += (
            w
            * np.exp(
                -gamma
                * D
            )
        )

    np.fill_diagonal(
        K,
        0.0
    )

    # kNN sparsification

    S = np.zeros_like(
        K
    )

    for i in range(
        n
    ):

        idx = np.argpartition(
            K[
                i
            ],
            -k_neighbors
        )[
            -k_neighbors:
        ]

        S[
            i,
            idx
        ] = K[
            i,
            idx
        ]

    S = np.maximum(
        S,
        S.T
    )

    row_sum = S.sum(
        axis=1,
        keepdims=True
    )

    S = S / np.maximum(
        row_sum,
        1e-12
    )

    Y = np.zeros(
        (
            n,
            N_CLASSES
        ),
        dtype=np.float64
    )

    Y[
        np.arange(
            n_l
        ),
        y_train
    ] = 1.0

    Fmat = Y.copy()

    Fmat[
        n_l:
    ] = (
        1.0
        /
        N_CLASSES
    )

    for _ in range(
        max_iter
    ):

        Fnew = (
            alpha
            * S.dot(
                Fmat
            )
            +
            (
                1
                -
                alpha
            )
            * Y
        )

        # Clamp known labels

        Fnew[
            :n_l
        ] = Y[
            :n_l
        ]

        delta = np.max(
            np.abs(
                Fnew
                -
                Fmat
            )
        )

        Fmat = Fnew

        if delta < 1e-7:
            break

    prob = Fmat[
        n_l:
    ]

    prob = prob / np.maximum(
        prob.sum(
            axis=1,
            keepdims=True
        ),
        1e-12
    )

    return prob


def run_graph_baseline(
    d
):

    train_views = view_arrays(
        d[
            "F_train"
        ]
    )

    valid_views = view_arrays(
        d[
            "F_valid"
        ]
    )

    best = None

    for weights in GRAPH_WEIGHTS:

        for gamma in GRAPH_GAMMAS:

            for k in GRAPH_K:

                p = consensus_graph_predict(
                    train_views,
                    d[
                        "y_train"
                    ],
                    valid_views,
                    weights,
                    gamma,
                    k
                )

                score = f1_score(
                    d[
                        "y_valid"
                    ],
                    np.argmax(
                        p,
                        axis=1
                    ),
                    average="macro",
                    zero_division=0
                )

                candidate = {

                    "weights":
                        weights,

                    "gamma":
                        gamma,

                    "k":
                        k,

                    "validation_f1":
                        float(
                            score
                        )
                }

                if (
                    best is None
                    or score
                    >
                    best[
                        "validation_f1"
                    ]
                ):

                    best = candidate

    development_views = [

        np.concatenate(
            [
                a,
                b
            ],
            axis=0
        )

        for a, b
        in zip(
            train_views,
            valid_views
        )
    ]

    test_views = view_arrays(
        d[
            "F_test"
        ]
    )

    ydev = np.concatenate(
        [
            d[
                "y_train"
            ],
            d[
                "y_valid"
            ]
        ]
    )

    test_prob = consensus_graph_predict(
        development_views,
        ydev,
        test_views,
        best[
            "weights"
        ],
        best[
            "gamma"
        ],
        best[
            "k"
        ]
    )

    best[
        "test_prob"
    ] = test_prob

    return best


# ======================================================================
# 14. OPTIONAL REVE FROZEN-ENCODER BASELINE
# ======================================================================

REVE_CHANNELS = [
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
    "TP7","CP3","CPz","CP4","TP8",
    "P5","P1","P2","P6",
    "PO7","PO3","POz","PO4","PO8"
]


def restore_pre_normalization(
    X,
    npz
):

    mu = np.asarray(
        npz[
            "norm_mu"
        ]
    )

    sd = np.asarray(
        npz[
            "norm_sd"
        ]
    )

    mu = mu.reshape(
        1,
        64,
        1
    )

    sd = sd.reshape(
        1,
        64,
        1
    )

    return (
        X
        * sd
        +
        mu
    )


def prepare_reve_input(
    X
):

    # Common paper window remains 0-2 s.
    # REVE expects 200 Hz. Convert 513 samples at 256 Hz
    # to exactly 400 samples.

    X200 = signal.resample(
        X,
        400,
        axis=2
    ).astype(
        np.float32
    )

    # REVE's BCIC2020 loader divides source microvolt values
    # by 1000.

    X200 = (
        X200
        /
        1000.0
    )

    return X200


def extract_reve_embeddings(
    d
):

    cache_file = os.path.join(
        CACHE,
        f"{d['sid']}_REVE_embeddings.npz"
    )

    if os.path.exists(
        cache_file
    ):

        z = np.load(
            cache_file
        )

        return (
            z[
                "train"
            ],
            z[
                "valid"
            ],
            z[
                "test"
            ]
        )

    from transformers import AutoModel

    token = os.environ.get(
        "HF_TOKEN",
        None
    )

    pos_bank = AutoModel.from_pretrained(
        "brain-bzh/reve-positions",
        trust_remote_code=True,
        token=token
    )

    model = AutoModel.from_pretrained(
        "brain-bzh/reve-base",
        trust_remote_code=True,
        token=token
    ).to(
        DEVICE
    )

    model.eval()

    z1 = np.load(
        d[
            "stage1_npz"
        ],
        allow_pickle=True
    )

    arrays = []

    for key in [
        "X_train",
        "X_valid",
        "X_test"
    ]:

        X = restore_pre_normalization(
            d[
                key
            ],
            z1
        )

        arrays.append(
            prepare_reve_input(
                X
            )
        )

    positions = pos_bank(
        REVE_CHANNELS
    )

    if positions.ndim == 2:

        positions = positions.unsqueeze(
            0
        )

    outputs = []

    for X in arrays:

        emb = []

        for start in range(
            0,
            len(
                X
            ),
            4
        ):

            xb = torch.tensor(
                X[
                    start:
                    start + 4
                ],
                dtype=torch.float32,
                device=DEVICE
            )

            pos = positions.to(
                DEVICE
            )

            if pos.size(
                0
            ) == 1:

                pos = pos.expand(
                    xb.size(
                        0
                    ),
                    -1,
                    -1
                )

            with torch.no_grad():

                out = model(
                    xb,
                    pos
                )

                pooled = (
                    model.attention_pooling(
                        out
                    )
                )

            emb.append(
                pooled
                .cpu()
                .numpy()
            )

        outputs.append(
            np.concatenate(
                emb,
                axis=0
            )
        )

    np.savez_compressed(
        cache_file,
        train=outputs[
            0
        ],
        valid=outputs[
            1
        ],
        test=outputs[
            2
        ]
    )

    del model
    del pos_bank

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return tuple(
        outputs
    )


def run_reve_probe(
    d
):

    Etr, Eva, Ete = (
        extract_reve_embeddings(
            d
        )
    )

    best = None

    for C in [
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0
    ]:

        scaler = StandardScaler()

        Xtr = scaler.fit_transform(
            Etr
        )

        Xva = scaler.transform(
            Eva
        )

        clf = LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=5000,
            random_state=GLOBAL_SEED
        )

        clf.fit(
            Xtr,
            d[
                "y_train"
            ]
        )

        p = clf.predict_proba(
            Xva
        )

        score = f1_score(
            d[
                "y_valid"
            ],
            np.argmax(
                p,
                axis=1
            ),
            average="macro",
            zero_division=0
        )

        if (
            best is None
            or score
            >
            best[
                "validation_f1"
            ]
        ):

            best = {
                "C":
                    C,

                "validation_f1":
                    float(
                        score
                    )
            }

    Edev = np.concatenate(
        [
            Etr,
            Eva
        ],
        axis=0
    )

    ydev = np.concatenate(
        [
            d[
                "y_train"
            ],
            d[
                "y_valid"
            ]
        ]
    )

    scaler = StandardScaler()

    Xdev = scaler.fit_transform(
        Edev
    )

    Xte = scaler.transform(
        Ete
    )

    clf = LogisticRegression(
        C=best[
            "C"
        ],
        solver="lbfgs",
        max_iter=5000,
        random_state=GLOBAL_SEED
    )

    clf.fit(
        Xdev,
        ydev
    )

    best[
        "test_prob"
    ] = clf.predict_proba(
        Xte
    )

    return best


# ======================================================================
# 15. MAIN SUBJECT LOOP
# ======================================================================

subject_rows = []

pair_records = []

graph_rows = []

reve_rows = []

for _, row in stage1_df.iterrows():

    d = load_subject_data(
        row
    )

    sid = d[
        "sid"
    ]

    print(
        "\n"
        +
        "="
        * 78
    )

    print(
        "FINAL DEVELOPMENT:",
        sid
    )

    print(
        "="
        * 78
    )

    raw_best = None

    feat_best = None

    # --------------------------------------------------------------
    # Deep raw rescue
    # --------------------------------------------------------------

    if RUN_DEEP_RESCUE:

        raw_best = select_deep_model(
            d
        )

        print(
            "Best raw:",
            raw_best[
                "architecture"
            ],
            "seed=",
            raw_best[
                "seed"
            ],
            "epoch=",
            raw_best[
                "epoch"
            ],
            "VALID F1=",
            round(
                raw_best[
                    "validation_f1"
                ],
                4
            )
        )

    # --------------------------------------------------------------
    # MDRG rescue
    # --------------------------------------------------------------

    if RUN_MDRG_RESCUE:

        feat_best = select_feature_model(
            d
        )

        print(
            "Best MDRG:",
            feat_best[
                "block"
            ],
            feat_best[
                "model_type"
            ],
            feat_best[
                "parameter"
            ],
            "VALID F1=",
            round(
                feat_best[
                    "validation_f1"
                ],
                4
            )
        )

    # --------------------------------------------------------------
    # Final test methods
    # --------------------------------------------------------------

    methods = {}

    if raw_best is not None:

        methods[
            "Raw_Selected_TrainOnly"
        ] = raw_best[
            "test_prob_train_only"
        ]

        methods[
            "Raw_Selected_Refit350"
        ] = raw_best[
            "test_prob_refit"
        ]

    if feat_best is not None:

        methods[
            "MDRG_Selected_TrainOnly"
        ] = feat_best[
            "test_prob_train_only"
        ]

        methods[
            "MDRG_Selected_Refit350"
        ] = feat_best[
            "test_prob_refit"
        ]

    # --------------------------------------------------------------
    # Validation-selected fusion
    # --------------------------------------------------------------

    if (
        RUN_FUSION
        and raw_best is not None
        and feat_best is not None
    ):

        fusion = choose_fusion_weight(
            d[
                "y_valid"
            ],
            raw_best[
                "valid_prob"
            ],
            feat_best[
                "valid_prob"
            ]
        )

        print(
            "Fusion weight raw=",
            fusion[
                "weight"
            ],
            "VALID F1=",
            round(
                fusion[
                    "f1"
                ],
                4
            )
        )

        # Cleanest strict fusion:
        # both component models were fitted only on official TRAIN,
        # and VALID selected only the fusion weight.

        methods[
            "GlobalFusion_TrainOnly"
        ] = (
            fusion[
                "weight"
            ]
            *
            raw_best[
                "test_prob_train_only"
            ]
            +
            (
                1.0
                -
                fusion[
                    "weight"
                ]
            )
            *
            feat_best[
                "test_prob_train_only"
            ]
        )

        # Pre-specified refit transfer:
        # the SAME validation-selected fusion weight is transferred
        # unchanged to models refit on TRAIN+VALID.

        methods[
            "GlobalFusion_Refit350"
        ] = (
            fusion[
                "weight"
            ]
            *
            raw_best[
                "test_prob_refit"
            ]
            +
            (
                1.0
                -
                fusion[
                    "weight"
                ]
            )
            *
            feat_best[
                "test_prob_refit"
            ]
        )

        methods[
            "EqualFusion_Refit350"
        ] = (
            0.5
            *
            raw_best[
                "test_prob_refit"
            ]
            +
            0.5
            *
            feat_best[
                "test_prob_refit"
            ]
        )

    # --------------------------------------------------------------
    # Consensus graph baseline
    # --------------------------------------------------------------

    if RUN_CONSENSUS_GRAPH_BASELINE:

        graph_best = run_graph_baseline(
            d
        )

        print(
            "Graph baseline:",
            graph_best[
                "weights"
            ],
            "gamma=",
            graph_best[
                "gamma"
            ],
            "k=",
            graph_best[
                "k"
            ],
            "VALID F1=",
            round(
                graph_best[
                    "validation_f1"
                ],
                4
            )
        )

        methods[
            "MVGSF_Inspired_Graph"
        ] = graph_best[
            "test_prob"
        ]

        graph_rows.append({
            "subject":
                sid,

            "weights":
                str(
                    graph_best[
                        "weights"
                    ]
                ),

            "gamma":
                graph_best[
                    "gamma"
                ],

            "k":
                graph_best[
                    "k"
                ],

            "validation_f1":
                graph_best[
                    "validation_f1"
                ]
        })

    # --------------------------------------------------------------
    # REVE
    # --------------------------------------------------------------

    if RUN_REVE:

        try:

            reve_best = run_reve_probe(
                d
            )

            print(
                "REVE C=",
                reve_best[
                    "C"
                ],
                "VALID F1=",
                round(
                    reve_best[
                        "validation_f1"
                    ],
                    4
                )
            )

            methods[
                "REVE_LinearProbe"
            ] = reve_best[
                "test_prob"
            ]

            reve_rows.append({
                "subject":
                    sid,

                "C":
                    reve_best[
                        "C"
                    ],

                "validation_f1":
                    reve_best[
                        "validation_f1"
                    ]
            })

        except Exception as e:

            print(
                "REVE skipped for",
                sid,
                "because:",
                repr(
                    e
                )
            )

    # --------------------------------------------------------------
    # FINAL TEST SCORING
    #
    # This is the first point where TEST labels enter.
    # --------------------------------------------------------------

    for method, prob in methods.items():

        pred = np.argmax(
            prob,
            axis=1
        )

        m = metrics(
            d[
                "y_test"
            ],
            pred
        )

        subject_rows.append({
            "subject":
                sid,

            "method":
                method,

            **m
        })

        np.savez_compressed(
            os.path.join(
                OUT,
                f"{sid}_{method}_FINAL_TEST.npz"
            ),

            y_test=d[
                "y_test"
            ],

            probability=prob,

            prediction=pred
        )


# ======================================================================
# 16. FINAL TEST SUMMARY
# ======================================================================

results = pd.DataFrame(
    subject_rows
)

results.to_csv(
    os.path.join(
        OUT,
        "FINAL_TEST_SUBJECTWISE.csv"
    ),
    index=False
)

summary = (
    results
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

summary.to_csv(
    os.path.join(
        OUT,
        "FINAL_TEST_SUMMARY.csv"
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
    "FINAL UNTOUCHED TEST SUMMARY"
)

print(
    "="
    * 78
)

print(
    summary.to_string(
        index=False
    )
)


# ======================================================================
# 17. PAIRED TEST-SET STATISTICS
# ======================================================================

def rank_biserial(
    x,
    y
):

    d = (
        np.asarray(
            x
        )
        -
        np.asarray(
            y
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

    ranks = (
        pd.Series(
            np.abs(
                d
            )
        )
        .rank(
            method="average"
        )
        .values
    )

    w_plus = ranks[
        d > 0
    ].sum()

    w_minus = ranks[
        d < 0
    ].sum()

    return float(
        (
            w_plus
            -
            w_minus
        )
        /
        ranks.sum()
    )


def bootstrap_difference(
    x,
    y,
    n=20000
):

    x = np.asarray(
        x
    )

    y = np.asarray(
        y
    )

    d = (
        x
        -
        y
    )

    rng = np.random.default_rng(
        GLOBAL_SEED
    )

    boot = []

    for _ in range(
        n
    ):

        idx = rng.integers(
            0,
            len(
                d
            ),
            len(
                d
            )
        )

        boot.append(
            d[
                idx
            ].mean()
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


methods = sorted(
    results[
        "method"
    ].unique()
)

reference = (
    "Raw_Selected_Refit350"
    if "Raw_Selected_Refit350"
    in methods
    else methods[
        0
    ]
)

ref = (
    results[
        results[
            "method"
        ]
        ==
        reference
    ]
    .sort_values(
        "subject"
    )[
        "macro_f1"
    ]
    .values
)

stat_rows = []

for method in methods:

    if method == reference:
        continue

    vals = (
        results[
            results[
                "method"
            ]
            ==
            method
        ]
        .sort_values(
            "subject"
        )[
            "macro_f1"
        ]
        .values
    )

    if len(
        vals
    ) != len(
        ref
    ):
        continue

    try:

        p = float(
            wilcoxon(
                vals,
                ref
            ).pvalue
        )

    except Exception:

        p = np.nan

    diff, lo, hi = bootstrap_difference(
        vals,
        ref
    )

    stat_rows.append({
        "comparison":
            f"{method} vs {reference}",

        "mean_macro_f1_difference":
            diff,

        "CI95_low":
            lo,

        "CI95_high":
            hi,

        "Wilcoxon_p":
            p,

        "rank_biserial":
            rank_biserial(
                vals,
                ref
            ),

        "subjects_better":
            int(
                np.sum(
                    vals
                    >
                    ref
                )
            ),

        "subjects_equal":
            int(
                np.sum(
                    np.isclose(
                        vals,
                        ref
                    )
                )
            ),

        "subjects_worse":
            int(
                np.sum(
                    vals
                    <
                    ref
                )
            )
    })

stats = pd.DataFrame(
    stat_rows
)

stats.to_csv(
    os.path.join(
        OUT,
        "FINAL_TEST_PAIRED_STATS.csv"
    ),
    index=False
)

print(
    "\nFINAL TEST PAIRED STATISTICS"
)

print(
    stats.to_string(
        index=False
    )
)


# ======================================================================
# 18. SAVE GRAPH / REVE SELECTION LOGS
# ======================================================================

if graph_rows:

    pd.DataFrame(
        graph_rows
    ).to_csv(
        os.path.join(
            OUT,
            "MVGSF_INSPIRED_SELECTION.csv"
        ),
        index=False
    )

if reve_rows:

    pd.DataFrame(
        reve_rows
    ).to_csv(
        os.path.join(
            OUT,
            "REVE_SELECTION.csv"
        ),
        index=False
    )


# ======================================================================
# 19. FINAL RULE
# ======================================================================

print(
    "\n"
    +
    "="
    * 78
)

print(
    "FINAL TEST EVALUATION COMPLETE"
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
    "\nDo not alter model/hyperparameter choices after inspecting "
    "these final test results."
)

