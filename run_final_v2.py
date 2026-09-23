"""
The run that produced every number in Tables 5 to 8 (except the two encoder rows) and Figs. 4 to 7 of

  H. T. M. Duhair, M. bin Mat Ibrahim, J. A. J. Alsayaydeh and M. Farid,
  "Graph-Riemannian Multi-View Fusion for Subject-Specific EEG Imagined Speech
  Decoding: A Held-Out Evaluation on the BCIC2020-IV Official Test Set", IJIES.

Identical in every computational step to the script executed on a Colab T4 GPU
on 17 September 2026 (INASS_FINAL_V2); only the Google Drive paths were replaced
by configs/config.json. Run from the repository root:

    python src/download_test_labels.py     # once: fetches and verifies the answer sheet
    python src/run_final_v2.py             # about 30 minutes on a GPU
    python src/stats_final_v2.py           # Tables 6 and 8, from the CSVs alone
    python src/make_figures_v2.py          # Figs. 1 and 4 to 7

PROTOCOL
  official TRAIN      fits every candidate
  official VALIDATION selects architecture, seed, stopping epoch, feature block,
                      classifier, regularisation constant, graph parameters and
                      the fusion weight
  refit               the selected configuration on TRAIN + VALIDATION
  official TEST       scored once

The test labels (TL, yte) are passed only to metrics() and to the per-subject
.npz that stores class posteriors for the confusion matrices. No fitting,
selection or fusion-weight call receives them. Search for "yte" to check.
The per-subject .npz files contain the labels and are excluded by .gitignore.
"""
import os, json, glob, time, random, warnings, itertools
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy.stats import wilcoxon

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, ARTIFACT_DIR, RESULT_DIR   # configs/config.json

# ----------------------------------------------------------------------------
# 0. PATHS  (all resolved from configs/config.json through src/paths.py)
# ----------------------------------------------------------------------------
STAGE1_DIR  = os.path.join(ROOT, "prep_stage1_final_robust")
STAGE1_CSV  = os.path.join(STAGE1_DIR, "GLOBAL_stage1final_summary.csv")
MDRG_DIR    = os.path.join(ROOT, "INASS_REVISION_FINAL", "stage2_MDRG_corrected", "npz")
TEST_LABELS = os.path.join(ARTIFACT_DIR, "official_test_labels.npy")   # src/download_test_labels.py
REVE_CSV    = os.path.join(RESULT_DIR, "REVE_SUBJECTWISE.csv")          # src/run_02_baselines_and_stats.py
OUT         = RESULT_DIR
PART        = os.path.join(RESULT_DIR, "per_subject")
os.makedirs(PART, exist_ok=True)
if not os.path.exists(TEST_LABELS):
    raise FileNotFoundError("Run  python src/download_test_labels.py  first.")

N_CLASSES  = 5
CLASS_NAMES = ["hello", "help me", "stop", "thank you", "yes"]
GLOBAL_SEED = 20265732
DEEP_SEEDS  = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}\noutput: {OUT}\n")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed(GLOBAL_SEED)

# ----------------------------------------------------------------------------
# 1. MDRG BLOCK LAYOUT  (corrected 64-channel extractor, D = 1656)
# ----------------------------------------------------------------------------
PER_BW, R_REG, TAN, GRA = 198, 18, 171, 9
STATIC, RIEM, GRAPH = [], [], []
for k in range(8):
    b = PER_BW * k
    STATIC += list(range(b, b + R_REG))
    RIEM   += list(range(b + R_REG, b + R_REG + TAN))
    GRAPH  += list(range(b + R_REG + TAN, b + PER_BW))
DYNAMIC = list(range(1584, 1656))

BLOCKS = {
    "Full_MDRG":                   sorted(STATIC + DYNAMIC + RIEM + GRAPH),
    "Riemannian_Graph":            sorted(RIEM + GRAPH),
    "Spectral_Dynamic_Riemannian": sorted(STATIC + DYNAMIC + RIEM),
    "Riemannian":                  sorted(RIEM),
    "Static_Spectral":             sorted(STATIC),
    "Spectral_Dynamic":            sorted(STATIC + DYNAMIC),
    "Dynamic":                     sorted(DYNAMIC),
    "Spectral_Dynamic_Graph":      sorted(STATIC + DYNAMIC + GRAPH),
    "Graph":                       sorted(GRAPH),
}
assert len(BLOCKS["Full_MDRG"]) == 1656
SELECT_BLOCKS = ["Full_MDRG", "Riemannian", "Riemannian_Graph", "Spectral_Dynamic"]

# ----------------------------------------------------------------------------
# 2. METRICS
# ----------------------------------------------------------------------------
def macro_spec(y, p):
    cm = confusion_matrix(y, p, labels=np.arange(N_CLASSES)); tot = cm.sum(); out = []
    for k in range(N_CLASSES):
        tp = cm[k, k]; fn = cm[k, :].sum() - tp; fp = cm[:, k].sum() - tp
        tn = tot - tp - fn - fp; out.append(tn / max(1, tn + fp))
    return float(np.mean(out))

def metrics(y, p):
    return dict(accuracy=float(accuracy_score(y, p)),
                macro_f1=float(f1_score(y, p, average="macro", zero_division=0)),
                macro_precision=float(precision_score(y, p, average="macro", zero_division=0)),
                macro_sensitivity=float(recall_score(y, p, average="macro", zero_division=0)),
                macro_specificity=macro_spec(y, p))

def mf1(y, p): return float(f1_score(y, p, average="macro", zero_division=0))

# ----------------------------------------------------------------------------
# 3. DEEP MODELS  (unchanged from your pipeline)
# ----------------------------------------------------------------------------
class EEGNetLite(nn.Module):
    def __init__(s, n_ch=64, n_classes=5, dropout=0.35):
        super().__init__()
        s.temporal = nn.Sequential(nn.Conv2d(1,16,(1,33),padding=(0,16),bias=False), nn.BatchNorm2d(16))
        s.spatial  = nn.Sequential(nn.Conv2d(16,32,(n_ch,1),groups=16,bias=False), nn.BatchNorm2d(32),
                                   nn.ELU(), nn.AvgPool2d((1,4)), nn.Dropout(dropout))
        s.sep      = nn.Sequential(nn.Conv2d(32,32,(1,17),padding=(0,8),groups=32,bias=False),
                                   nn.Conv2d(32,64,(1,1),bias=False), nn.BatchNorm2d(64),
                                   nn.ELU(), nn.AvgPool2d((1,4)), nn.Dropout(dropout))
        s.head     = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(64,n_classes))
    def forward(s,x): return s.head(s.sep(s.spatial(s.temporal(x.unsqueeze(1)))))

class EEGConformerLite(nn.Module):
    def __init__(s, n_ch=64, n_classes=5, emb=64, dropout=0.30):
        super().__init__()
        s.temporal = nn.Sequential(nn.Conv2d(1,32,(1,25),padding=(0,12),bias=False), nn.BatchNorm2d(32))
        s.spatial  = nn.Sequential(nn.Conv2d(32,64,(n_ch,1),groups=32,bias=False), nn.BatchNorm2d(64),
                                   nn.ELU(), nn.AvgPool2d((1,4)), nn.Dropout(dropout))
        s.project  = nn.Linear(64, emb)
        lay = nn.TransformerEncoderLayer(d_model=emb, nhead=4, dim_feedforward=emb*3,
                                         dropout=dropout, activation="gelu",
                                         batch_first=True, norm_first=True)
        s.transformer = nn.TransformerEncoder(lay, num_layers=2)
        s.norm = nn.LayerNorm(emb); s.head = nn.Linear(emb, n_classes)
    def forward(s,x):
        x = s.spatial(s.temporal(x.unsqueeze(1))).squeeze(2).transpose(1,2)
        return s.head(s.norm(s.transformer(s.project(x)).mean(dim=1)))

BATCH, LR, WD, MAXEP, PATIENCE = 32, 1e-3, 1.5e-4, 90, 15
SMOOTH, MIXUP = 0.05, 0.25

def loader(X, y, shuf):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                    torch.tensor(y, dtype=torch.long)),
                      batch_size=BATCH, shuffle=shuf)

def soft_ce(lg, tg): return -(tg * F.log_softmax(lg, 1)).sum(1).mean()

def predict_logits(model, X):
    model.eval(); out = []
    dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=64)
    with torch.no_grad():
        for (xb,) in dl: out.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(out, 0)

def _epoch(model, dl, opt):
    model.train()
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        tg = F.one_hot(yb, N_CLASSES).float()
        tg = (1 - SMOOTH) * tg + SMOOTH / N_CLASSES
        lam = np.random.beta(MIXUP, MIXUP); pm = torch.randperm(xb.size(0), device=xb.device)
        loss = soft_ce(model(lam * xb + (1 - lam) * xb[pm]), lam * tg + (1 - lam) * tg[pm])
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

def train_select(model, Xtr, ytr, Xva, yva, seed):
    set_seed(seed); model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAXEP)
    dl = loader(Xtr, ytr, True); best, bep, bstate, bad = -1, 1, None, 0
    for ep in range(1, MAXEP + 1):
        _epoch(model, dl, opt); sch.step()
        s = mf1(yva, predict_logits(model, Xva).argmax(1))
        if s > best + 1e-8:
            best, bep, bad = s, ep, 0
            bstate = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(bstate)
    return model, float(best), bep

def fit_fixed(model, X, y, seed, epochs):
    set_seed(seed); model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    dl = loader(X, y, True)
    for _ in range(epochs): _epoch(model, dl, opt); sch.step()
    return model

# ----------------------------------------------------------------------------
# 4. GRAPH COMPARATOR  (multi-view consensus graph, MVGSF-inspired)
# ----------------------------------------------------------------------------
def simplex(step=0.25):
    v = np.arange(0, 1.0001, step); out = []
    for a in v:
        for b in v:
            c = 1.0 - a - b
            if c >= -1e-9 and abs(round(c/step)*step - c) < 1e-7:
                out.append((float(a), float(b), float(max(0, c))))
    return out
G_W, G_GAM, G_K = simplex(0.25), [0.25, 0.5, 1.0, 2.0], [10, 20]

def views(Fm): return [Fm[:, sorted(STATIC + DYNAMIC)], Fm[:, RIEM], Fm[:, GRAPH]]

def rbf(A, B, gamma):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) / max(1, A.shape[1])
    return np.exp(-gamma * d2)

def graph_predict(dev_v, ydev, tgt_v, w, gamma, k):
    K = None
    for wi, Dv, Tv in zip(w, dev_v, tgt_v):
        if wi <= 0: continue
        sc = StandardScaler().fit(Dv)
        Kv = rbf(sc.transform(Tv), sc.transform(Dv), gamma)
        K = Kv * wi if K is None else K + Kv * wi
    if K is None: K = np.zeros((tgt_v[0].shape[0], dev_v[0].shape[0]))
    P = np.zeros((K.shape[0], N_CLASSES))
    kk = min(k, K.shape[1])
    for i in range(K.shape[0]):
        idx = np.argsort(-K[i])[:kk]
        for j in idx: P[i, ydev[j]] += K[i, j]
    s = P.sum(1, keepdims=True); s[s == 0] = 1
    return P / s

# ----------------------------------------------------------------------------
# 5. LINEAR / KERNEL READOUT for MDRG blocks
# ----------------------------------------------------------------------------
def fit_probe(Ftr, ytr, Fev, cols, kind, C):
    sc = StandardScaler()
    A = sc.fit_transform(Ftr[:, cols]); B = sc.transform(Fev[:, cols])
    clf = (LogisticRegression(C=C, max_iter=5000, random_state=GLOBAL_SEED) if kind == "LR"
           else SVC(C=C, kernel="rbf", gamma="scale", probability=True, random_state=GLOBAL_SEED))
    clf.fit(A, ytr)
    return clf.predict_proba(B)

# ----------------------------------------------------------------------------
# 6. PER-SUBJECT RUN
# ----------------------------------------------------------------------------
df1 = pd.read_csv(STAGE1_CSV).sort_values("subject").reset_index(drop=True)
TL  = np.load(TEST_LABELS)
print(f"subjects: {len(df1)} | test labels: {TL.shape}\n")

def run_subject(row):
    sid = str(row["subject"]); si = int(sid[-2:]) - 1
    z1 = np.load(os.path.join(STAGE1_DIR, "npz", os.path.basename(str(row["npz_path"]))), allow_pickle=True)
    z2 = np.load(f"{MDRG_DIR}/{sid}_MDRG_corrected.npz", allow_pickle=True)

    Xtr, ytr = z1["X_train"].astype(np.float32), z1["y_train"].astype(np.int64)
    Xva, yva = z1["X_valid"].astype(np.float32), z1["y_valid"].astype(np.int64)
    Xte      = z1["X_test"].astype(np.float32)
    Ftr, Fva, Fte = (z2["X_train"].astype(np.float32), z2["X_valid"].astype(np.float32),
                     z2["X_test"].astype(np.float32))
    yte = TL[si].astype(np.int64)
    assert Ftr.shape[1] == 1656 and Xte.shape[0] == Fte.shape[0] == 50

    Xdev = np.concatenate([Xtr, Xva]); Fdev = np.concatenate([Ftr, Fva])
    ydev = np.concatenate([ytr, yva])
    res = {"subject": sid}

    # --- raw branch: 2 architectures x 3 seeds, selected on validation --------
    best = None
    for arch in ["EEGNetLite", "EEGConformerLite"]:
        for sd in DEEP_SEEDS:
            m = EEGNetLite() if arch == "EEGNetLite" else EEGConformerLite()
            m, s, ep = train_select(m, Xtr, ytr, Xva, yva, sd)
            if best is None or s > best["s"]:
                best = dict(arch=arch, seed=sd, ep=ep, s=s,
                            vprob=torch.softmax(torch.tensor(predict_logits(m, Xva)), 1).numpy())
            del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    fm = EEGNetLite() if best["arch"] == "EEGNetLite" else EEGConformerLite()
    fm = fit_fixed(fm, Xdev, ydev, best["seed"], best["ep"])
    raw_test = torch.softmax(torch.tensor(predict_logits(fm, Xte)), 1).numpy()
    raw_val  = best["vprob"]
    res.update(raw_arch=best["arch"], raw_seed=best["seed"], raw_epoch=best["ep"],
               raw_valid_f1=best["s"])
    del fm; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- MDRG branch: block x classifier x C, selected on validation ----------
    bb = None
    for blk in SELECT_BLOCKS:
        for kind, Cs in [("LR", [0.01, 0.1, 1.0, 10.0]), ("RBF_SVM", [0.1, 1.0, 10.0])]:
            for C in Cs:
                p = fit_probe(Ftr, ytr, Fva, BLOCKS[blk], kind, C)
                s = mf1(yva, p.argmax(1))
                if bb is None or s > bb["s"]:
                    bb = dict(blk=blk, kind=kind, C=C, s=s, vprob=p)
    mdrg_test = fit_probe(Fdev, ydev, Fte, BLOCKS[bb["blk"]], bb["kind"], bb["C"])
    mdrg_val  = bb["vprob"]
    res.update(mdrg_block=bb["blk"], mdrg_clf=bb["kind"], mdrg_C=bb["C"], mdrg_valid_f1=bb["s"])

    # --- fusion: one global weight on a 41-point grid, selected on validation -
    bw, bs = 0.5, -1
    for w in np.linspace(0, 1, 41):
        s = mf1(yva, (w * raw_val + (1 - w) * mdrg_val).argmax(1))
        if s > bs + 1e-12: bs, bw = s, float(w)
    fus_test = bw * raw_test + (1 - bw) * mdrg_test
    res.update(fusion_w=bw, fusion_valid_f1=bs)

    # --- graph comparator on the corrected MDRG views -------------------------
    gv_tr, gv_va, gv_dev, gv_te = views(Ftr), views(Fva), views(Fdev), views(Fte)
    gb = None
    for w in G_W:
        for g in G_GAM:
            for k in G_K:
                s = mf1(yva, graph_predict(gv_tr, ytr, gv_va, w, g, k).argmax(1))
                if gb is None or s > gb["s"]: gb = dict(w=w, g=g, k=k, s=s)
    graph_test = graph_predict(gv_dev, ydev, gv_te, gb["w"], gb["g"], gb["k"])
    res.update(graph_gamma=gb["g"], graph_k=gb["k"], graph_valid_f1=gb["s"])

    # --- ablation: 9 variants, common LR probe, C selected on validation ------
    abl = {}
    for name, cols in BLOCKS.items():
        ab = None
        for C in [0.01, 0.1, 1.0, 10.0]:
            s = mf1(yva, fit_probe(Ftr, ytr, Fva, cols, "LR", C).argmax(1))
            if ab is None or s > ab["s"]: ab = dict(C=C, s=s)
        p = fit_probe(Fdev, ydev, Fte, cols, "LR", ab["C"])
        abl[name] = metrics(yte, p.argmax(1)); abl[name]["dimensions"] = len(cols)

    preds = {"Raw EEG": raw_test, "MDRG branch": mdrg_test,
             "Proposed fusion": fus_test, "Graph comparator": graph_test}
    for k, p in preds.items():
        for mk, mv in metrics(yte, p.argmax(1)).items(): res[f"{k}|{mk}"] = mv

    np.savez_compressed(f"{PART}/{sid}.npz",
                        y_test=yte, **{f"prob_{k.replace(' ','_')}": v for k, v in preds.items()})
    return res, abl

# ----------------------------------------------------------------------------
# 7. MAIN LOOP (resumable)
# ----------------------------------------------------------------------------
rows, ablr = [], []
for _, row in df1.iterrows():
    sid = str(row["subject"]); cache = f"{PART}/{sid}.json"
    if os.path.exists(cache):
        d = json.load(open(cache)); rows.append(d["res"])
        ablr += [dict(subject=sid, block=b, **m) for b, m in d["abl"].items()]
        print(f"[skip] {sid} already done"); continue
    t0 = time.time(); r, a = run_subject(row)
    json.dump({"res": r, "abl": a}, open(cache, "w"), indent=2)
    rows.append(r); ablr += [dict(subject=sid, block=b, **m) for b, m in a.items()]
    print(f"[done] {sid}  fusion macro-F1={r['Proposed fusion|macro_f1']:.4f}  "
          f"raw={r['Raw EEG|macro_f1']:.4f}  ({time.time()-t0:.0f}s)")

D = pd.DataFrame(rows); A = pd.DataFrame(ablr)
D.to_csv(f"{OUT}/FINAL_v2_SUBJECTWISE.csv", index=False)
A.to_csv(f"{OUT}/FINAL_v2_ABLATION_SUBJECTWISE.csv", index=False)

# ----------------------------------------------------------------------------
# 8. STATISTICS
# ----------------------------------------------------------------------------
def boot(v, n=10000, seed=GLOBAL_SEED):
    rng = np.random.default_rng(seed); v = np.asarray(v, float)
    b = rng.choice(v, (n, len(v)), True).mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def boot_d(a, b, n=10000, seed=GLOBAL_SEED):
    rng = np.random.default_rng(seed); d = np.asarray(a, float) - np.asarray(b, float)
    s = rng.choice(d, (n, len(d)), True).mean(1)
    return float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

def rb(a, b):
    d = np.asarray(a, float) - np.asarray(b, float); d = d[np.abs(d) > 1e-12]
    if d.size == 0: return 0.0
    r = pd.Series(np.abs(d)).rank().values
    return float((r[d > 0].sum() - r[d < 0].sum()) / (d.size * (d.size + 1) / 2))

def holm(ps):
    idx = np.argsort(ps); out = np.empty(len(ps)); run = 0
    for i, j in enumerate(idx):
        run = max(run, (len(ps) - i) * ps[j]); out[j] = min(1.0, run)
    return out

METHODS = ["Proposed fusion", "MDRG branch", "Graph comparator", "Raw EEG"]
summ = []
for m in METHODS:
    r = {"method": m}
    for k in ["accuracy", "macro_f1", "macro_precision", "macro_sensitivity", "macro_specificity"]:
        v = D[f"{m}|{k}"].values
        r[f"{k}_mean"], r[f"{k}_std"] = float(v.mean()), float(v.std(ddof=1))
    summ.append(r)
if os.path.exists(REVE_CSV):
    rv = pd.read_csv(REVE_CSV)
    for m in rv["method"].unique():
        g = rv[rv["method"] == m]; r = {"method": m}
        for k in ["accuracy", "macro_f1", "macro_precision", "macro_sensitivity", "macro_specificity"]:
            col = k if k in g else k.replace("macro_", "macro_")
            if col in g: r[f"{k}_mean"], r[f"{k}_std"] = float(g[col].mean()), float(g[col].std(ddof=1))
        summ.append(r)
S = pd.DataFrame(summ).sort_values("macro_f1_mean", ascending=False)
S.to_csv(f"{OUT}/FINAL_v2_BASELINE_COMPARISON.csv", index=False)

fus = D["Proposed fusion|macro_f1"].values
comps, ps = [], []
for other in ["Raw EEG", "MDRG branch", "Graph comparator"]:
    o = D[f"{other}|macro_f1"].values
    d, lo, hi = boot_d(fus, o)
    p = float(wilcoxon(fus, o).pvalue) if np.any(np.abs(fus - o) > 1e-12) else 1.0
    comps.append(dict(comparison=f"Proposed fusion vs {other}", metric="macro_f1",
                      mean_difference=d, CI95_low=lo, CI95_high=hi, Wilcoxon_p=p,
                      rank_biserial=rb(fus, o),
                      subjects_better=int((fus > o).sum()), subjects_worse=int((fus < o).sum())))
    ps.append(p)
ad = holm(np.array(ps))
for c, a in zip(comps, ad): c["Holm_adjusted_p"] = float(a)
P = pd.DataFrame(comps); P.to_csv(f"{OUT}/FINAL_v2_PAIRED_STATISTICS.csv", index=False)

ab_s, ab_p, base = [], [], None
for b in BLOCKS:
    v = A[A["block"] == b]["macro_f1"].values
    m, lo, hi = boot(v)
    ab_s.append(dict(block=b, dimensions=int(A[A["block"] == b]["dimensions"].iloc[0]),
                     macro_f1_mean=m, ci_low=lo, ci_high=hi,
                     accuracy_mean=float(A[A["block"] == b]["accuracy"].mean())))
    if b == "Full_MDRG": base = v
pv = []
for b in BLOCKS:
    if b == "Full_MDRG": continue
    v = A[A["block"] == b]["macro_f1"].values
    d, lo, hi = boot_d(base, v)
    p = float(wilcoxon(base, v).pvalue) if np.any(np.abs(base - v) > 1e-12) else 1.0
    ab_p.append(dict(comparison=f"Full_MDRG vs {b}", mean_difference=d, CI95_low=lo,
                     CI95_high=hi, Wilcoxon_p=p, rank_biserial=rb(base, v))); pv.append(p)
for c, a in zip(ab_p, holm(np.array(pv))): c["Holm_adjusted_p"] = float(a)
AS = pd.DataFrame(ab_s).sort_values("macro_f1_mean", ascending=False)
AP = pd.DataFrame(ab_p)
AS.to_csv(f"{OUT}/FINAL_v2_ABLATION_SUMMARY.csv", index=False)
AP.to_csv(f"{OUT}/FINAL_v2_ABLATION_STATISTICS.csv", index=False)

cm = {}
for m in ["Raw EEG", "Proposed fusion"]:
    yy, pp = [], []
    for f in sorted(glob.glob(f"{PART}/*.npz")):
        z = np.load(f); yy.append(z["y_test"]); pp.append(z[f"prob_{m.replace(' ','_')}"].argmax(1))
    cm[m] = confusion_matrix(np.concatenate(yy), np.concatenate(pp),
                             labels=range(N_CLASSES)).tolist()
json.dump(cm, open(f"{OUT}/FINAL_v2_CONFUSION.json", "w"), indent=2)

json.dump({"mdrg_source": "stage2_MDRG_corrected (64-channel, 18 regions, D=1656)",
           "protocol": "train fits; validation selects; test scored once; refit on train+valid",
           "seeds": DEEP_SEEDS, "global_seed": GLOBAL_SEED,
           "note": "All MDRG-dependent results regenerated on the corrected features."},
          open(f"{OUT}/FINAL_v2_MANIFEST.json", "w"), indent=2)

# ----------------------------------------------------------------------------
# 9. PRINTOUT
# ----------------------------------------------------------------------------
print("\n" + "=" * 78 + "\nRESULTS\n" + "=" * 78)
print("\n--- OFFICIAL TEST, corrected 64-channel MDRG ---")
print(S.to_string(index=False))
print("\n--- PAIRED STATISTICS ---")
print(P.to_string(index=False))
print("\n--- ABLATION (official test) ---")
print(AS.to_string(index=False))
print("\n--- ABLATION STATISTICS ---")
print(AP.to_string(index=False))
print("\n--- SELECTED CONFIGURATIONS ---")
print(D[["subject","raw_arch","raw_seed","raw_epoch","mdrg_block","mdrg_clf","mdrg_C",
         "fusion_w","graph_gamma","graph_k"]].to_string(index=False))
print("\n--- SUBJECT-WISE macro-F1 (official test) ---")
print(D[["subject"] + [f"{m}|macro_f1" for m in METHODS]].to_string(index=False))
print("\n--- CONFUSION MATRICES ---")
print("Raw EEG:        ", cm["Raw EEG"])
print("Proposed fusion:", cm["Proposed fusion"])
print("\n" + "=" * 78)
print("Files written to:", OUT)
print("=" * 78)
