"""
Statistics for Tables 6, 7 and 8 and the numbers quoted in Sections 4.3, 4.4 and 5.2,
computed from the result CSVs alone. No data, no labels and no GPU are needed.

    python src/stats_final_v2.py

Inputs  (results/):  FINAL_v2_SUBJECTWISE.csv, REVE_SUBJECTWISE.csv,
                     FINAL_v2_ABLATION_SUBJECTWISE.csv (optional: Table 7 and the search-width contrast)
Outputs (results/):  FINAL_v2_PAIRED_STATISTICS_FAMILY5.csv (Table 6), FINAL_v2_GENERALISATION_GAP.csv (Table 8)

The bootstrap, rank-biserial and Holm routines are copied unchanged from
run_final_v2.py, with the same seed, so every interval matches that run.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from paths import RESULT_DIR
except Exception:                     # allow use without a dataset path configured
    RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

GLOBAL_SEED = 20265732
TIE = 1e-12


def boot_d(a, b, n=10000, seed=GLOBAL_SEED):
    rng = np.random.default_rng(seed); d = np.asarray(a, float) - np.asarray(b, float)
    s = rng.choice(d, (n, len(d)), True).mean(1)
    return float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def rb(a, b):
    d = np.asarray(a, float) - np.asarray(b, float); d = d[np.abs(d) > TIE]
    if d.size == 0:
        return 0.0
    r = pd.Series(np.abs(d)).rank().values
    return float((r[d > 0].sum() - r[d < 0].sum()) / (d.size * (d.size + 1) / 2))


def holm(ps):
    idx = np.argsort(ps); out = np.empty(len(ps)); run = 0
    for i, j in enumerate(idx):
        run = max(run, (len(ps) - i) * ps[j]); out[j] = min(1.0, run)
    return out


def contrast(name, metric, a, b):
    m, lo, hi = boot_d(a, b)
    w = wilcoxon(a, b)
    d = np.asarray(a, float) - np.asarray(b, float)
    return dict(comparison=name, metric=metric, mean_difference=m, CI95_low=lo, CI95_high=hi,
                W=float(w.statistic), Wilcoxon_p=float(w.pvalue), rank_biserial=rb(a, b),
                subjects_better=int((d > TIE).sum()), subjects_tied=int((np.abs(d) <= TIE).sum()),
                subjects_worse=int((d < -TIE).sum()))


def main():
    sub = pd.read_csv(os.path.join(RESULT_DIR, "FINAL_v2_SUBJECTWISE.csv"))
    reve = pd.read_csv(os.path.join(RESULT_DIR, "REVE_SUBJECTWISE.csv"))
    col = lambda m, k: sub[f"{m}|{k}"].values if f"{m}|{k}" in sub else None
    # accept either the pipeline's column names or the short aliases
    F = {m: sub[f"{m}|macro_f1"].values for m in ["Proposed fusion", "Raw EEG", "MDRG branch", "Graph comparator"]}
    A = {m: sub[f"{m}|accuracy"].values for m in ["Proposed fusion", "Raw EEG"]}
    for m in ["REVE_LinearProbe", "REVE_LoRA"]:
        r = reve[reve.method == m].set_index("subject").loc[sub.subject]
        F[m] = r.macro_f1.values

    rows = [contrast(f"Proposed fusion vs {m}", "macro_f1", F["Proposed fusion"], F[m])
            for m in ["Raw EEG", "MDRG branch", "Graph comparator", "REVE_LinearProbe", "REVE_LoRA"]]
    for r, a in zip(rows, holm(np.array([r["Wilcoxon_p"] for r in rows]))):
        r["Holm_adjusted_p_family5"] = float(a)
    extra = [
        contrast("Proposed fusion vs Raw EEG", "accuracy", A["Proposed fusion"], A["Raw EEG"]),
        contrast("Proposed fusion vs Raw EEG (VALIDATION split)", "macro_f1",
                 sub.fusion_valid_f1.values, sub.raw_valid_f1.values),
        contrast("Fusion reduction vs Raw reduction (validation minus test)", "macro_f1",
                 sub.fusion_valid_f1.values - F["Proposed fusion"], sub.raw_valid_f1.values - F["Raw EEG"]),
    ]
    abl_path = os.path.join(RESULT_DIR, "FINAL_v2_ABLATION_SUBJECTWISE.csv")
    if os.path.exists(abl_path):
        ab = pd.read_csv(abl_path)
        if "accuracy" in ab:            # Table 7: mean and SD per variant
            t7 = ab.groupby("block").agg(dims=("dimensions", "first") if "dimensions" in ab else ("macro_f1", "size"),
                                         acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                                         f1_mean=("macro_f1", "mean"), f1_sd=("macro_f1", "std"))
            print("Table 7 (mean and SD per variant)\n", t7.sort_values("f1_mean", ascending=False).round(3).to_string(), "\n")
        full = ab[ab.block == "Full_MDRG"].set_index("subject").loc[sub.subject].macro_f1.values
        extra.append(contrast("Full descriptor, 4-candidate search vs MDRG branch, 28-candidate search",
                              "macro_f1", full, F["MDRG branch"]))
    for r in extra:
        r["Holm_adjusted_p_family5"] = np.nan
    T = pd.DataFrame(rows + extra)
    T.to_csv(os.path.join(RESULT_DIR, "FINAL_v2_PAIRED_STATISTICS_FAMILY5.csv"), index=False)

    gap = []
    for name, v, t in [("Proposed fusion", "fusion_valid_f1", "Proposed fusion"), ("Raw EEG", "raw_valid_f1", "Raw EEG"),
                       ("Graph comparator", "graph_valid_f1", "Graph comparator"), ("MDRG branch", "mdrg_valid_f1", "MDRG branch")]:
        vv, tt = sub[v].values, F[t]
        gap.append(dict(system=name, validation_macro_f1_mean=round(vv.mean(), 4),
                        validation_macro_f1_std=round(vv.std(ddof=1), 4),
                        test_macro_f1_mean=round(tt.mean(), 4), test_macro_f1_std=round(tt.std(ddof=1), 4),
                        absolute_drop=round((vv - tt).mean(), 4),
                        relative_drop_pct=round(100 * (vv - tt).mean() / vv.mean(), 1),
                        participants_declining=int((vv - tt > 0).sum())))
    G = pd.DataFrame(gap)
    G.to_csv(os.path.join(RESULT_DIR, "FINAL_v2_GENERALISATION_GAP.csv"), index=False)

    M = pd.DataFrame(F, index=sub.subject)
    top = M.max(axis=1)
    wins = {c: int((np.abs(M[c] - top) <= TIE).sum()) for c in M}
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print("Table 6\n", T.round(4).to_string(index=False))
    print("\nTable 8\n", G.to_string(index=False))
    print("\nHighest-scoring system per participant (joint leaders counted once each):", wins)
    print("Median within-participant range across the six systems: %.4f" % (M.max(axis=1) - M.min(axis=1)).median())
    print("Selected fusion weight: median %.3f, participants with weight 1: %d, with weight >= 0.8: %d"
          % (sub.fusion_w.median(), int((sub.fusion_w == 1).sum()), int((sub.fusion_w >= 0.8).sum())))
    print("Raw-branch architecture:", sub.raw_arch.value_counts().to_dict())
    print("MDRG block selected:", sub.mdrg_block.value_counts().to_dict())


if __name__ == "__main__":
    main()
