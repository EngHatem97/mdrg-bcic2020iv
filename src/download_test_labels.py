"""
Retrieve and verify the official BCIC2020-IV Track-3 test answer sheet.

The answer sheet is distributed by the competition organisers through their
public Open Science Framework project (OSF pq7vb, doi 10.17605/OSF.IO/PQ7VB,
"2020 International BCI Competition", CC BY 4.0). It is NOT redistributed in
this repository. This script downloads it and verifies it by SHA-256 before
any label is used.

    python src/download_test_labels.py

Writes official_test_labels.npy, shape (15, 50), integer labels 0..4.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, ARTIFACT_DIR

ANSWER_URL    = "https://osf.io/download/n3a2p/?version=1"
ANSWER_SHA256 = "7010974ad753b961684b1d9cb1cd55a5279904523afe5b9a3f051e9f53e6b6c7"
CACHE_DIR     = os.path.join(ROOT, "cache")
ANSWER_FILE   = os.path.join(CACHE_DIR, "Track3_Answer_Sheet_Test.xlsx")
N_SUBJECTS, N_TEST_TRIALS, N_CLASSES = 15, 50, 5


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_answer_sheet():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(ANSWER_FILE) and sha256_file(ANSWER_FILE) == ANSWER_SHA256:
        print("Answer sheet already present, checksum verified.")
        return ANSWER_FILE

    import requests
    print("Downloading official Track-3 test answer sheet from OSF ...")
    response = requests.get(ANSWER_URL, timeout=60)
    response.raise_for_status()
    with open(ANSWER_FILE, "wb") as f:
        f.write(response.content)

    actual = sha256_file(ANSWER_FILE)
    if actual != ANSWER_SHA256:
        raise RuntimeError(
            "Checksum mismatch for the test answer sheet.\n"
            f"  expected {ANSWER_SHA256}\n  actual   {actual}\n"
            "Do not use this file."
        )
    print("Checksum verified.")
    return ANSWER_FILE


def load_test_labels():
    path = download_answer_sheet()
    df = pd.read_excel(path, sheet_name="Track3")
    values = df.head(53).values
    labels = values[2:, 1:][:, 1:30:2].T
    labels = np.asarray(labels, dtype=int)

    if labels.shape != (N_SUBJECTS, N_TEST_TRIALS):
        raise RuntimeError(f"Unexpected test-label shape: {labels.shape}")

    if labels.min() == 1 and labels.max() == N_CLASSES:     # source codes are 1..5
        labels = labels - 1
    if not (labels.min() == 0 and labels.max() == N_CLASSES - 1):
        raise RuntimeError("Unexpected test label values.")

    # Alignment check. The organisers' published split is 10 test trials per
    # class per participant; anything else means the sheet was parsed wrongly.
    for s in range(N_SUBJECTS):
        counts = np.bincount(labels[s], minlength=N_CLASSES)
        if not np.all(counts == N_TEST_TRIALS // N_CLASSES):
            raise RuntimeError(
                f"Subject {s + 1} has class counts {counts.tolist()}, expected "
                f"{[N_TEST_TRIALS // N_CLASSES] * N_CLASSES}. Parsing is misaligned."
            )
    print(f"Loaded and validated test labels: {labels.shape}, "
          f"{N_TEST_TRIALS // N_CLASSES} trials per class per participant.")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    out = os.path.join(ARTIFACT_DIR, "official_test_labels.npy")
    np.save(out, labels)
    print("Wrote", out)
    return labels


if __name__ == "__main__":
    load_test_labels()
