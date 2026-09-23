"""
Draw Figures 1 and 4 to 7 of the IJIES paper at their printed size.

Every figure is drawn at the full text width of the IJIES A4 two-column layout
(17 cm = 6.7 in) and inserted at 100% scale, so the 10 pt text set here prints
at 10 pt. Do not rescale the images in Word: shrinking them shrinks the text
below the editor's 10 pt minimum. No panel letters, no outer frames.

    python src/make_figures_v2.py --results results --out figures

Reads only files under results/.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

WIDTH = 6.7            # inches, full text width
FS = 10                # points, the editor's minimum
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "figure.dpi": 150, "savefig.dpi": 600, "figure.constrained_layout.use": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "hatch.linewidth": 0.8,
})

INK = "#1a1a1a"        # primary text
INK2 = "#52514e"       # secondary text
GRID = "#d9d8d4"
RAW = "#2a78d6"        # categorical slot 1 (validated with the orange below)
FUS = "#eb6834"        # categorical slot 2
OTHER = "#8a8983"      # neutral context series
CHANCE = 0.20


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _boot_ci(x, n=10000, seed=20265732):
    rng = np.random.default_rng(seed)
    m = rng.choice(np.asarray(x, float), size=(n, len(x)), replace=True).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


# ------------------------------------------------------------------ Fig. 1
def fig1(out):
    fig, ax = plt.subplots(figsize=(WIDTH, 4.6), layout="none")
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995)
    ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")

    def box(cx, cy, w, h, title, body, fill="#f3f2ef"):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=fill, ec=INK, lw=0.9))
        ax.text(cx, cy + h * 0.18, title, ha="center", va="center", fontweight="bold", color=INK)
        ax.text(cx, cy - h * 0.20, body, ha="center", va="center", color=INK2)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                     lw=0.9, color=INK, shrinkA=0, shrinkB=0))

    # row 1: data path
    xs, w1 = [12.5, 37.5, 62.5, 87.5], 23
    r1 = [("BCIC2020-IV", "15 participants"), ("Loading, QC", "train-only limits"),
          ("Conditioning", "1-70 Hz, CAR"), ("Trial tensor", "64 x 513 samples")]
    for x, (t, b) in zip(xs, r1):
        box(x, 62, w1, 10, t, b)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + w1 / 2 + 0.4, 62, b - w1 / 2 - 0.4, 62)

    # row 2: two branches fed by the trial tensor
    ax.plot([87.5, 87.5], [56.6, 52.5], color=INK, lw=0.9)
    ax.plot([27, 87.5], [52.5, 52.5], color=INK, lw=0.9)
    arrow(27, 52.5, 27, 48.4); arrow(73, 52.5, 73, 48.4)
    box(27, 43, 44, 10, "Raw EEG branch", "EEGNetLite or EEGConformerLite")
    box(73, 43, 44, 10, "MDRG branch", "1656-D descriptor, shallow readout")

    # row 3: fusion
    arrow(27, 37.6, 44, 31.6); arrow(73, 37.6, 56, 31.6)
    box(50, 27, 44, 9, "Decision-level fusion", "one weight on a 41-point grid")

    # row 4: split roles
    ax.text(50, 17.2, "Per participant: role of each official split", ha="center",
            va="center", color=INK2)
    r4 = [("Training", "300 trials: fit", "#e8f0fb"), ("Validation", "50 trials: select", "#e8f0fb"),
          ("Refit", "350 trials", "#e8f0fb"), ("Test", "50 trials: score once", "#fdece4")]
    for x, (t, b, f) in zip(xs, r4):
        box(x, 7, w1, 10, t, b, fill=f)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + w1 / 2 + 0.4, 7, b - w1 / 2 - 0.4, 7)
    fig.savefig(os.path.join(out, "fig1_pipeline.png"))
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 4
def fig4(res, out):
    gap = pd.read_csv(os.path.join(res, "FINAL_v2_GENERALISATION_GAP.csv"))
    sub = pd.read_csv(os.path.join(res, "FINAL_v2_SUBJECTWISE.csv"))
    reve = pd.read_csv(os.path.join(res, "REVE_SUBJECTWISE.csv"))

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(WIDTH, 6.4),
                                   gridspec_kw={"height_ratios": [1.0, 1.15]})
    # top: validation -> test for the four systems built here
    gap = gap.sort_values("test_macro_f1_mean", ascending=False).reset_index(drop=True)
    ys = gap.test_macro_f1_mean.values.astype(float)
    lab = 0.305 - 0.026 * np.arange(len(ys))
    col = {"Proposed fusion": FUS, "Raw EEG": RAW}
    name = {"Proposed fusion": "Proposed fusion", "Raw EEG": "Raw EEG branch",
            "Graph comparator": "Graph comparator", "MDRG branch": "MDRG branch"}
    for i, r in gap.iterrows():
        c = col.get(r["system"], OTHER)
        axT.plot([0, 1], [r.validation_macro_f1_mean, r.test_macro_f1_mean], "-o",
                 color=c, lw=2.0, ms=6, zorder=3)
        axT.plot([1.02, 1.12], [ys[i], lab[i]], color=c, lw=0.8)
        axT.text(1.15, lab[i], f"{name[r['system']]}   −{r.relative_drop_pct:.1f}%",
                 va="center", ha="left", color=INK)
    axT.axhline(CHANCE, ls="--", lw=0.9, color=INK2)
    axT.text(-0.08, CHANCE + 0.006, "chance", color=INK2, ha="left")
    axT.set_xlim(-0.1, 2.05)
    axT.set_xticks([0, 1]); axT.set_xticklabels(["Validation split", "Test split"])
    axT.set_ylabel("Macro-F1")
    axT.set_ylim(0.18, 0.46)
    _grid(axT)

    # bottom: six systems on the test split
    series = [
        ("Encoder,\nlow-rank", reve[reve.method == "REVE_LoRA"].macro_f1.values, OTHER),
        ("Encoder,\nlinear probe", reve[reve.method == "REVE_LinearProbe"].macro_f1.values, OTHER),
        ("Proposed\nfusion", sub["Proposed fusion|macro_f1"].values, FUS),
        ("Raw EEG\nbranch", sub["Raw EEG|macro_f1"].values, RAW),
        ("MDRG\nbranch", sub["MDRG branch|macro_f1"].values, OTHER),
        ("Graph\ncomparator", sub["Graph comparator|macro_f1"].values, OTHER),
    ]
    rng = np.random.default_rng(7)
    for i, (k, v, c) in enumerate(series):
        axB.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=22, color=c, alpha=0.55,
                    edgecolor="white", linewidth=0.5, zorder=3)
        lo, hi = _boot_ci(v)
        m = float(np.mean(v))
        axB.plot([i - 0.28, i + 0.28], [m, m], color=INK, lw=2.0, zorder=4)
        axB.plot([i, i], [lo, hi], color=INK, lw=1.2, zorder=4)
    axB.axhline(CHANCE, ls="--", lw=0.9, color=INK2)
    axB.set_xticks(range(len(series))); axB.set_xticklabels([s[0] for s in series])
    axB.set_xlim(-0.6, len(series) - 0.4)
    axB.set_ylabel("Test macro-F1")
    _grid(axB)
    fig.savefig(os.path.join(out, "fig4_generalisation_and_headtohead.png"))
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 5
def fig5(res, out):
    sub = pd.read_csv(os.path.join(res, "FINAL_v2_SUBJECTWISE.csv"))
    n = len(sub); x = np.arange(n)
    labels = [f"S{i + 1:02d}" for i in range(n)]
    d = sub["Proposed fusion|macro_f1"].values - sub["Raw EEG|macro_f1"].values

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDTH, 5.9),
                                 gridspec_kw={"height_ratios": [1.25, 1.0]})
    a1.plot(x, sub["Raw EEG|macro_f1"], "-o", color=RAW, lw=1.6, ms=6, label="Raw EEG branch", zorder=3)
    a1.plot(x, sub["Proposed fusion|macro_f1"], "-s", color=FUS, lw=1.6, ms=6, label="Proposed fusion", zorder=3)
    a1.axhline(CHANCE, ls="--", lw=0.9, color=INK2)
    a1.set_xticks(x); a1.set_xticklabels(labels, rotation=90)
    a1.set_xlim(-0.6, n - 0.4); a1.set_ylabel("Test macro-F1")
    a1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    _grid(a1)

    tied = np.isclose(d, 0.0, atol=1e-9)
    for i in range(n):
        if tied[i]:
            continue
        if d[i] > 0:
            a2.bar(i, d[i], width=0.62, color="#4d4c49", zorder=3)
        else:
            a2.bar(i, d[i], width=0.62, color="white", edgecolor="#4d4c49", hatch="////",
                   lw=0.9, zorder=3)
    a2.plot(x[tied], np.zeros(tied.sum()), "o", mfc="white", mec=INK, ms=7, mew=1.2, zorder=4)
    a2.axhline(0, color=INK, lw=0.8)
    a2.axhline(d.mean(), ls="--", lw=1.0, color=INK2)
    a2.text(10.5, d.mean() + 0.012, f"mean {d.mean():+.3f}", ha="center", color=INK2)
    a2.set_xticks(x); a2.set_xticklabels(labels, rotation=90)
    a2.set_xlim(-0.6, n - 0.4); a2.set_ylabel("Fusion minus raw")
    _grid(a2)
    fig.savefig(os.path.join(out, "fig5_subject_level.png"))
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 6
def fig6(res, out):
    cm = json.load(open(os.path.join(res, "FINAL_v2_CONFUSION.json")))
    cls = cm["class_names"]
    raw = np.array(cm["Raw EEG"], float); fus = np.array(cm["Proposed fusion"], float)

    fig = plt.figure(figsize=(WIDTH, 6.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.06], height_ratios=[1.35, 0.8])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    for ax, M, title in [(axes[0], raw, "Raw EEG branch"), (axes[1], fus, "Proposed fusion")]:
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=60)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center",
                        color="white" if M[i, j] > 38 else INK)
        ax.set_xticks(range(5)); ax.set_xticklabels(cls, rotation=45, ha="right")
        ax.set_yticks(range(5)); ax.set_yticklabels(cls if ax is axes[0] else [""] * 5)
        ax.set_xlabel("Predicted command")
        if ax is axes[0]:
            ax.set_ylabel("True command")
        ax.set_title(f"{title}: {int(np.trace(M))} of {int(M.sum())} correct")
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, cax=cax); cb.set_label("Trials"); cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[1, :2])
    delta = (np.diag(fus) - np.diag(raw)) / 150 * 100
    for i, v in enumerate(delta):
        if v >= 0:
            ax.barh(i, v, height=0.6, color="#4d4c49", zorder=3)
        else:
            ax.barh(i, v, height=0.6, color="white", edgecolor="#4d4c49", hatch="////", lw=0.9, zorder=3)
        ax.text(v + (0.25 if v >= 0 else -0.25), i, f"{v:+.1f}".replace("-", "\u2212"), va="center",
                ha="left" if v >= 0 else "right", color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(5)); ax.set_yticklabels(cls); ax.invert_yaxis()
    ax.set_xlim(-4, 9)
    ax.set_xlabel("Change in recall, fusion minus raw (percentage points)")
    _grid(ax, "x")
    fig.savefig(os.path.join(out, "fig6_confusion.png"))
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 7
def fig7(res, out):
    ab = pd.read_csv(os.path.join(res, "FINAL_v2_ABLATION_SUMMARY.csv"))
    st = pd.read_csv(os.path.join(res, "FINAL_v2_ABLATION_STATISTICS.csv"))
    holm = {r.comparison.split(" vs ")[1]: r.Holm_adjusted_p for _, r in st.iterrows()}
    pretty = {"Riemannian": "Tangent-space only", "Full_MDRG": "Full MDRG",
              "Riemannian_Graph": "Tangent-space + graph",
              "Static_Spectral": "Regional log-bandpower only",
              "Spectral_Dynamic_Riemannian": "Spectral + dynamic + tangent",
              "Spectral_Dynamic_Graph": "Spectral + dynamic + graph",
              "Spectral_Dynamic": "Spectral + dynamic", "Graph": "Graph spectra only",
              "Dynamic": "Dynamic contrast only"}
    ab = ab.sort_values("macro_f1_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(WIDTH, 4.2))
    for i, r in ab.iterrows():
        full = r.block == "Full_MDRG"
        sig = holm.get(r.block, 1.0) < 0.05
        ax.plot([r.ci_low, r.ci_high], [i, i], color=INK if (full or sig) else OTHER, lw=1.8,
                solid_capstyle="round", zorder=3)
        ax.plot(r.macro_f1_mean, i, "D" if full else "o", ms=7 if full else 6,
                color=INK if (full or sig) else OTHER, zorder=4)
        if sig:
            ax.text(r.ci_high + 0.004, i, f"Holm p = {holm[r.block]:.3f}", va="center", color=INK)
    ax.axvline(CHANCE, ls="--", lw=0.9, color=INK2)
    ax.text(CHANCE + 0.002, len(ab) - 0.35, "chance", color=INK2, va="center")
    ax.set_yticks(range(len(ab)))
    ax.set_yticklabels([f"{pretty[r.block]} ({int(r.dimensions)})" for _, r in ab.iterrows()])
    ax.set_ylim(-0.6, len(ab) - 0.1)
    ax.set_xlabel("Test macro-F1, mean and 95% bootstrap interval")
    _grid(ax, "x")
    fig.savefig(os.path.join(out, "fig7_ablation.png"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig1(a.out); fig4(a.results, a.out); fig5(a.results, a.out)
    fig6(a.results, a.out); fig7(a.results, a.out)
    print("wrote five figures to", a.out)
