"""
Deterministic column layout of the 1656-dimensional MDRG trial vector.

Assembly order used by the Stage-2 extractor:

    for band in [theta, alpha, beta, low-gamma]:
        for window in [early, late]:
            regional log-bandpower        18
            tangent-space covariance     171
            graph spectral descriptors     9      -> 198 per (band, window)
    then
    for band in [theta, alpha, beta, low-gamma]:
        late-minus-early log-bandpower    18      -> 72

    8 x 198 + 72 = 1656

Because the blocks are contiguous and non-overlapping, any descriptor family
can be isolated by column slicing of the stored feature matrices, with no
feature recomputed. This is what makes the ablation in the paper an exact
decomposition of the representation rather than an approximation of it.

Reproduces Table 3 of the paper.
"""
import itertools
import numpy as np

R_REGIONS   = 18
N_BANDS     = 4
N_WINDOWS   = 2
TANGENT_DIM = R_REGIONS * (R_REGIONS + 1) // 2      # 171
GRAPH_DIM   = 6 + 3                                 # 6 Laplacian eigenvalues + 3 coupling stats
PER_BW      = R_REGIONS + TANGENT_DIM + GRAPH_DIM   # 198
BW_TOTAL    = PER_BW * N_BANDS * N_WINDOWS          # 1584
DYN_START   = BW_TOTAL
D_TOTAL     = BW_TOTAL + N_BANDS * R_REGIONS        # 1656

BAND_NAMES   = ["theta", "alpha", "beta", "lgamma"]
WINDOW_NAMES = ["early", "late"]


def mdrg_block_indices():
    """{family: np.ndarray of column indices}. The four arrays tile 0..1655."""
    logp, tangent, graph = [], [], []
    for k in range(N_BANDS * N_WINDOWS):
        base = PER_BW * k
        logp    += list(range(base,                               base + R_REGIONS))
        tangent += list(range(base + R_REGIONS,                   base + R_REGIONS + TANGENT_DIM))
        graph   += list(range(base + R_REGIONS + TANGENT_DIM,     base + PER_BW))
    out = {
        "logP":    np.array(logp,    dtype=int),    # 144
        "tangent": np.array(tangent, dtype=int),    # 1368
        "graph":   np.array(graph,   dtype=int),    # 72
        "dynamic": np.array(range(DYN_START, D_TOTAL), dtype=int),   # 72
    }
    assert [len(v) for v in out.values()] == [144, 1368, 72, 72]
    assert len(np.unique(np.concatenate(list(out.values())))) == D_TOTAL
    return out


def ablation_variants():
    """The nine variants reported in Table 7 and Fig. 7, as column-index arrays."""
    B = mdrg_block_indices()
    cat = lambda *ks: np.sort(np.concatenate([B[k] for k in ks]))
    return {
        "Full_MDRG":                    cat("logP", "dynamic", "tangent", "graph"),
        "Riemannian_Graph":             cat("tangent", "graph"),
        "Spectral_Dynamic_Riemannian":  cat("logP", "dynamic", "tangent"),
        "Riemannian":                   B["tangent"],
        "Static_Spectral":              B["logP"],
        "Spectral_Dynamic":             cat("logP", "dynamic"),
        "Dynamic":                      B["dynamic"],
        "Spectral_Dynamic_Graph":       cat("logP", "dynamic", "graph"),
        "Graph":                        B["graph"],
    }


def block_table():
    """Table 3 as a list of dicts, one row per contiguous block."""
    rows = []
    for k, (b, w) in enumerate(itertools.product(BAND_NAMES, WINDOW_NAMES)):
        base = PER_BW * k
        rows += [
            dict(block=k, band=b, window=w, family="logP",
                 start=base, end=base + R_REGIONS - 1, dims=R_REGIONS),
            dict(block=k, band=b, window=w, family="tangent",
                 start=base + R_REGIONS, end=base + R_REGIONS + TANGENT_DIM - 1, dims=TANGENT_DIM),
            dict(block=k, band=b, window=w, family="graph",
                 start=base + R_REGIONS + TANGENT_DIM, end=base + PER_BW - 1, dims=GRAPH_DIM),
        ]
    for i, b in enumerate(BAND_NAMES):
        rows.append(dict(block="dynamic", band=b, window="late-early", family="dynamic",
                         start=DYN_START + i * R_REGIONS,
                         end=DYN_START + (i + 1) * R_REGIONS - 1, dims=R_REGIONS))
    return rows


# Table 2 of the paper. The corrected Stage-2 extractor assigns all 64 recorded
# channels to the 18 regions; the per-run audit reports 64 unique channels
# represented, with no missing and no unknown electrodes.
#
# NOTE ON HISTORY: an earlier extractor left FT9, FT10, TP7 and TP8 unassigned
# and therefore covered only 60 channels. Feature matrices produced by that
# earlier version are not interchangeable with these.
REGION_MAP = {
    "L_Prefrontal":     ["Fp1", "AF3", "AF7"],
    "R_Prefrontal":     ["Fp2", "AF4", "AF8"],
    "L_Frontal":        ["F3", "F5", "F7"],
    "M_Frontal":        ["Fz", "F1", "F2"],
    "R_Frontal":        ["F4", "F6", "F8"],
    "L_Frontocentral":  ["FC1", "FC3", "FC5"],
    "R_Frontocentral":  ["FC2", "FC4", "FC6"],
    "L_Temporal":       ["FT9", "FT7", "T7", "TP7", "TP9"],
    "R_Temporal":       ["FT10", "FT8", "T8", "TP8", "TP10"],
    "Central":          ["C1", "C2", "C3", "C4", "Cz", "C5", "C6"],
    "L_Centroparietal": ["CP1", "CP3", "CP5"],
    "M_Centroparietal": ["CPz", "Pz"],
    "R_Centroparietal": ["CP2", "CP4", "CP6"],
    "L_Parietal":       ["P1", "P3", "P5", "P7"],
    "R_Parietal":       ["P2", "P4", "P6", "P8"],
    "L_Occipital":      ["O1", "PO3", "PO7", "PO9"],
    "M_Occipital":      ["Oz", "POz"],
    "R_Occipital":      ["O2", "PO4", "PO8", "PO10"],
}

# Every recorded channel is assigned. Kept as an empty list so that code and
# tests that referenced it continue to work.
UNASSIGNED_CHANNELS = []


if __name__ == "__main__":
    B = mdrg_block_indices()
    print("MDRG block sizes:", {k: len(v) for k, v in B.items()},
          "-> total", sum(len(v) for v in B.values()))
    print("Ablation variants:", {k: len(v) for k, v in ablation_variants().items()})
    chs = [c for v in REGION_MAP.values() for c in v]
    print("Channels assigned:", len(set(chs)), "of 64; duplicates:", len(chs) - len(set(chs)),
          "; unassigned:", UNASSIGNED_CHANNELS or "none")
