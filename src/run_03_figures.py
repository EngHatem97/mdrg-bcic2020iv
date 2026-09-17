"""
Publication figures for the revised manuscript.

Part of: Graph-Riemannian Multi-View Fusion for Subject-Specific EEG Imagined
Speech Decoding: A Held-Out Evaluation on the BCIC2020-IV Official Test Set.

This file is the original experimental script, unmodified except that the
hard-coded Google Drive paths have been replaced by a lookup in
configs/config.json (see src/paths.py). Run it from the repository root:

    python src/run_03_figures.py

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

import os, re, glob, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from scipy.stats import wilcoxon

warnings.filterwarnings('ignore')

# ----------------------------- CONFIG ---------------------------------
# ROOT now comes from configs/config.json via src/paths.py

OUT = os.path.join(ROOT, 'INASS_PAPER_FIGURES_FINAL')
os.makedirs(OUT, exist_ok=True)

CHANCE = 0.20
CLASS_NAMES = ['hello', 'help me', 'stop', 'thank you', 'yes']

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8.5,
    'axes.titlesize': 9.2,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 600,
})

# ----------------------------- HELPERS --------------------------------
def save_fig(fig, stem):
    paths = {}
    for ext in ['png', 'pdf', 'svg']:
        p = os.path.join(OUT, f'{stem}.{ext}')
        fig.savefig(p, dpi=600 if ext == 'png' else None, bbox_inches='tight')
        paths[ext] = p
    plt.close(fig)
    print('Saved:', paths['png'])
    return paths

def s_num(x):
    m = re.search(r'(\d{1,2})$', str(x))
    if not m:
        raise ValueError(f'Cannot parse subject from {x}')
    return int(m.group(1))

def s_lab(x):
    return f'S{s_num(x):02d}'

def boot_mean_ci(v, n=50000, seed=20265732):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    b = v[idx].mean(1)
    return float(v.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))

def boot_diff_ci(a, b, n=50000, seed=20265732):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bd = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(bd, .025)), float(np.quantile(bd, .975))

def first_existing(paths):
    return next((p for p in paths if p and os.path.exists(p)), None)

def rfind(name):
    x = glob.glob(os.path.join(ROOT, '**', name), recursive=True)
    return x[0] if x else None

# --------------------------- FIND SAVED FILES ---------------------------
PRED_DIR = first_existing([
    os.path.join(ROOT, 'stage3_dual_fusion_v3_final', 'predictions')
])
if PRED_DIR is None:
    ds = [p for p in glob.glob(os.path.join(ROOT, '**', 'predictions'), recursive=True) if os.path.isdir(p)]
    PRED_DIR = ds[0] if ds else None
if PRED_DIR is None:
    raise FileNotFoundError('Frozen Stage-3 predictions folder not found.')

TEST_LABELS_FILE = first_existing([
    os.path.join(ROOT, 'INASS_FINAL_TEST_REVISION', 'official_test_labels.npy'),
    rfind('official_test_labels.npy')
])
REVE_FILE = first_existing([
    os.path.join(ROOT, 'INASS_REVE_OFFICIAL_RAW_FINAL', 'REVE_SUBJECTWISE.csv'),
    rfind('REVE_SUBJECTWISE.csv')
])
FINAL_TEST_FILE = first_existing([
    os.path.join(ROOT, 'INASS_FINAL_TEST_REVISION', 'FINAL_TEST_SUBJECTWISE.csv'),
    rfind('FINAL_TEST_SUBJECTWISE.csv')
])
STAGE1_SUMMARY = first_existing([
    os.path.join(ROOT, 'prep_stage1_final_robust', 'GLOBAL_stage1final_summary.csv'),
    rfind('GLOBAL_stage1final_summary.csv')
])
ABLATION_DIR = os.path.join(ROOT, 'INASS_REVISION_FINAL', 'ablation')
if not os.path.isdir(ABLATION_DIR):
    ad = [p for p in glob.glob(os.path.join(ROOT, '**', 'ablation'), recursive=True) if os.path.isdir(p)]
    ABLATION_DIR = ad[0] if ad else None

if TEST_LABELS_FILE is None or REVE_FILE is None:
    raise FileNotFoundError('Required saved test-label or REVE file is missing.')

print('ROOT:', ROOT)
print('Predictions:', PRED_DIR)
print('Test labels:', TEST_LABELS_FILE)
print('REVE:', REVE_FILE)
print('Final-test results:', FINAL_TEST_FILE)
print('Ablation:', ABLATION_DIR)

# -------------------- RECONSTRUCT RAW/FUSION RESULTS --------------------
y_test_all = np.load(TEST_LABELS_FILE).astype(int)
pred_files = sorted(glob.glob(os.path.join(PRED_DIR, '*_predictions.npz')))
if len(pred_files) != 15:
    raise RuntimeError(f'Expected 15 prediction NPZ files, found {len(pred_files)}')

rows, pooled_y, pooled_raw, pooled_fus = [], [], [], []
for fp in pred_files:
    z = np.load(fp, allow_pickle=True)
    subject = str(np.asarray(z['subject']).item()) if 'subject' in z.files else os.path.basename(fp)
    sid = s_num(subject)
    yt = y_test_all[sid - 1]
    yv = z['valid_y'].astype(int)
    raw_t = np.argmax(z['test_logits_raw'], axis=1)
    fus_t = np.argmax(z['test_probs_fused'], axis=1)
    raw_v = np.argmax(z['valid_logits_raw'], axis=1)
    fus_v = np.argmax(z['valid_probs_fused'], axis=1)
    rows.append({
        'subject': f'S{sid:02d}', 'sid': sid,
        'raw_valid_f1': f1_score(yv, raw_v, average='macro', zero_division=0),
        'fusion_valid_f1': f1_score(yv, fus_v, average='macro', zero_division=0),
        'raw_test_f1': f1_score(yt, raw_t, average='macro', zero_division=0),
        'fusion_test_f1': f1_score(yt, fus_t, average='macro', zero_division=0),
        'raw_test_acc': accuracy_score(yt, raw_t),
        'fusion_test_acc': accuracy_score(yt, fus_t),
    })
    pooled_y.append(yt); pooled_raw.append(raw_t); pooled_fus.append(fus_t)

primary = pd.DataFrame(rows).sort_values('sid').reset_index(drop=True)
primary['delta'] = primary['fusion_test_f1'] - primary['raw_test_f1']
raw_t = primary.raw_test_f1.to_numpy()
fus_t = primary.fusion_test_f1.to_numpy()
raw_v = primary.raw_valid_f1.to_numpy()
fus_v = primary.fusion_valid_f1.to_numpy()
mean_d, ci_d_lo, ci_d_hi = boot_diff_ci(fus_t, raw_t)
p_primary = float(wilcoxon(fus_t, raw_t, alternative='two-sided', zero_method='wilcox').pvalue)
n_better = int((fus_t > raw_t).sum())

# ----------------------- BUILD BASELINE TABLE ---------------------------
frames = []
for method, col in [('Raw EEG','raw_test_f1'), ('Proposed fusion','fusion_test_f1')]:
    t = primary[['subject', col]].copy(); t.columns = ['subject','macro_f1']; t['method'] = method; frames.append(t)

reve = pd.read_csv(REVE_FILE)
reve['subject'] = reve['subject'].map(s_lab)
for raw_name, display in [('REVE_LinearProbe','REVE LP'), ('REVE_LoRA','REVE LoRA')]:
    t = reve[reve.method == raw_name][['subject','macro_f1']].copy(); t['method'] = display; frames.append(t)

if FINAL_TEST_FILE:
    ft = pd.read_csv(FINAL_TEST_FILE)
    g = ft[ft.method == 'MVGSF_Inspired_Graph'][['subject','macro_f1']].copy()
    if len(g):
        g['subject'] = g['subject'].map(s_lab); g['method'] = 'Graph comparator'; frames.append(g)

methods = pd.concat(frames, ignore_index=True)
ORDER = [m for m in ['Raw EEG','Graph comparator','REVE LP','REVE LoRA','Proposed fusion'] if m in set(methods.method)]

# -------------------------- QC SUMMARY ---------------------------------
qc_total, qc_removed, qc_subject = 4500, 3, 'S14'
if STAGE1_SUMMARY:
    try:
        q = pd.read_csv(STAGE1_SUMMARY)
        before = next((c for c in ['train_raw','train_before','n_train_raw','n_train_before'] if c in q.columns), None)
        after = next((c for c in ['train_after','train_after_qc','n_train_after'] if c in q.columns), None)
        if before and after:
            rem = q[before] - q[after]
            qc_total = int(q[before].sum()); qc_removed = int(rem.sum())
            if rem.max() > 0 and 'subject' in q.columns:
                qc_subject = s_lab(q.iloc[int(np.argmax(rem.values))]['subject'])
    except Exception:
        pass
qc_pct = 100 * qc_removed / qc_total

# -------------------------- ABLATION DATA -------------------------------
def parse_mean_std(v):
    s = str(v)
    if '±' in s:
        a,b = s.split('±',1); return float(a.strip()), float(b.strip())
    return float(s), np.nan

def load_ablation():
    if ABLATION_DIR is None:
        raise FileNotFoundError('Ablation directory not found.')
    csvs = glob.glob(os.path.join(ABLATION_DIR, '*.csv'))
    # Prefer subject-wise files
    for fp in csvs:
        try: d = pd.read_csv(fp)
        except Exception: continue
        sc = next((c for c in d.columns if 'subject' in c.lower()), None)
        bc = next((c for c in d.columns if c.lower() in ['block','ablation','component'] or 'block' in c.lower()), None)
        fc = next((c for c in d.columns if 'macro_f1' in c.lower()), None)
        if sc and bc and fc:
            out=[]
            for block,g in d.groupby(bc):
                v=pd.to_numeric(g[fc],errors='coerce').dropna().to_numpy()
                if len(v):
                    m,l,h=boot_mean_ci(v); out.append([str(block),m,l,h,len(v),fp])
            if out:
                return pd.DataFrame(out,columns=['block','mean','lo','hi','n','source'])
    # Otherwise summary
    fps = [p for p in csvs if 'summary' in os.path.basename(p).lower()]
    if not fps:
        raise FileNotFoundError('No ablation summary CSV found.')
    d = pd.read_csv(fps[0]); fp=fps[0]
    bc = next(c for c in d.columns if c.lower() in ['block','ablation','component'] or 'block' in c.lower())
    fc = next(c for c in d.columns if 'macro_f1' in c.lower())
    out=[]
    for _,r in d.iterrows():
        if pd.api.types.is_numeric_dtype(d[fc]):
            m=float(r[fc]); sd=np.nan
            sc=next((c for c in d.columns if 'macro_f1_std' in c.lower() or ('std' in c.lower() and 'f1' in c.lower())),None)
            if sc: sd=float(r[sc])
        else:
            m,sd=parse_mean_std(r[fc])
        if np.isfinite(sd):
            h=1.96*sd/math.sqrt(15); lo=m-h; hi=m+h
        else: lo=hi=m
        out.append([str(r[bc]),m,lo,hi,15,fp])
    return pd.DataFrame(out,columns=['block','mean','lo','hi','n','source'])

abl = load_ablation()
NAME = {
    'Full_MDRG':'Full MDRG','Riemannian_Graph':'Riemannian + Graph',
    'Spectral_Dynamic_Riemannian':'Spectral + Dynamic + Riemannian',
    'Riemannian':'Riemannian','Static_Spectral':'Static spectral',
    'Spectral_Dynamic':'Spectral + Dynamic','Dynamic':'Dynamic',
    'Spectral_Dynamic_Graph':'Spectral + Dynamic + Graph','Graph':'Graph'
}
DIMS = {'Full_MDRG':1656,'Riemannian_Graph':1440,'Spectral_Dynamic_Riemannian':1584,'Riemannian':1368,'Static_Spectral':144,'Spectral_Dynamic':216,'Dynamic':72,'Spectral_Dynamic_Graph':288,'Graph':72}
abl['label'] = abl.block.map(lambda x: NAME.get(x, str(x).replace('_',' + ')))
abl['dim'] = abl.block.map(DIMS)
abl = abl.sort_values('mean').reset_index(drop=True)

# ======================================================================
# FIGURE 1 — COMPACT EVALUATION PROTOCOL + QC
# ======================================================================
fig,ax=plt.subplots(figsize=(7.2,2.45)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
nodes=[(0.03,.25,'TRAIN','300 trials / subject','Model fitting','Training-derived preprocessing'),
       (0.375,.25,'VALIDATION','50 trials / subject','Selection and stopping','No final performance claim'),
       (0.72,.25,'OFFICIAL TEST','50 trials / subject','Final evaluation only','Predictions frozen before scoring')]
for x,w,title,l1,l2,foot in nodes:
    ax.add_patch(FancyBboxPatch((x,.43),w,.36,boxstyle='round,pad=.012,rounding_size=.014',fill=False,linewidth=1.0))
    cx=x+w/2
    ax.text(cx,.705,title,ha='center',va='center',fontweight='bold',fontsize=9)
    ax.text(cx,.61,l1,ha='center',va='center')
    ax.text(cx,.525,l2,ha='center',va='center')
    ax.text(cx,.355,foot,ha='center',va='center',fontsize=7.2)
for a,b in [(.285,.37),(.63,.715)]:
    ax.annotate('',xy=(b,.61),xytext=(a,.61),arrowprops=dict(arrowstyle='->',linewidth=1.1))
ax.plot([.03,.97],[.205,.205],linewidth=.6)
ax.text(.5,.105,f'QC audit: {qc_total-qc_removed:,}/{qc_total:,} eligible training trials retained; {qc_removed} removed ({qc_pct:.3f}%), all from {qc_subject}.',ha='center',va='center',fontsize=7.5)
F1=save_fig(fig,'Fig01_EvaluationProtocol')

# ======================================================================
# FIGURE 2 — SUBJECT-WISE FUSION EFFECT
# ======================================================================
fig,axs=plt.subplots(2,1,figsize=(7.2,5.5),sharex=True,gridspec_kw={'height_ratios':[1.55,1],'hspace':.12})
x=np.arange(1,16)
ax=axs[0]
for i in range(15): ax.plot([x[i],x[i]],[raw_t[i],fus_t[i]],linewidth=.8,alpha=.5)
ax.scatter(x,raw_t,s=28,marker='o',label='Raw EEG',zorder=3)
ax.scatter(x,fus_t,s=30,marker='s',label='Proposed fusion',zorder=3)
ax.axhline(CHANCE,linestyle='--',linewidth=.9,label='Chance = 0.20')
ax.set_ylabel('Official-test macro-F1'); ax.set_title('Subject-level performance')
ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.01))
ax.spines[['top','right']].set_visible(False)
ax=axs[1]; delta=primary.delta.to_numpy(); bars=ax.bar(x,delta,width=.68)
for b,d in zip(bars,delta):
    if d<0: b.set_hatch('///')
ax.axhline(0,linewidth=.9); ax.axhline(mean_d,linestyle='--',linewidth=.9)
ax.set_ylabel('Fusion − Raw macro-F1'); ax.set_xlabel('Subject'); ax.set_title('Within-subject fusion effect')
ax.set_xticks(x); ax.set_xticklabels(primary.subject)
ax.text(.015,.94,f'Mean Δ = {mean_d:+.3f} [95% bootstrap CI {ci_d_lo:+.3f}, {ci_d_hi:+.3f}]\n{n_better}/15 improved; Wilcoxon p = {p_primary:.3f}',transform=ax.transAxes,ha='left',va='top',fontsize=7.5)
ax.spines[['top','right']].set_visible(False)
F2=save_fig(fig,'Fig02_SubjectwiseFusion')

# ======================================================================
# FIGURE 3 — GENERALIZATION GAP + HEAD-TO-HEAD COMPARISON
# ======================================================================
fig,axs=plt.subplots(1,2,figsize=(7.2,3.55),gridspec_kw={'width_ratios':[.86,1.35],'wspace':.30})
ax=axs[0]
vals=[('Raw EEG',raw_v.mean(),raw_t.mean()),('Proposed fusion',fus_v.mean(),fus_t.mean())]
for name,v,t in vals:
    ax.plot([0,1],[v,t],marker='o',linewidth=1.25,label=name)
    ax.text(-.05,v+0.006,f'{v:.3f}',ha='right',fontsize=7)
    ax.text(1.05,t-0.006,f'{t:.3f}',ha='left',fontsize=7)
ax.axhline(CHANCE,linestyle='--',linewidth=.8)
ax.set_xlim(-.28,1.28); ax.set_xticks([0,1]); ax.set_xticklabels(['Validation','Official test'])
ax.set_ylabel('Macro-F1'); ax.set_title('Generalization gap'); ax.legend(frameon=False)
ax.spines[['top','right']].set_visible(False)

ax=axs[1]; rng=np.random.default_rng(20265732); baseline_stats={}
for i,m in enumerate(ORDER):
    v=methods.loc[methods.method==m,'macro_f1'].astype(float).to_numpy()
    j=rng.normal(0,.05,len(v)); ax.scatter(np.full(len(v),i)+j,v,s=14,alpha=.45,zorder=2)
    mean,lo,hi=boot_mean_ci(v); baseline_stats[m]={'mean':mean,'lo':lo,'hi':hi,'n':len(v)}
    ax.errorbar(i,mean,yerr=np.array([[mean-lo],[hi-mean]]),fmt='D',markersize=4.1,capsize=3,linewidth=1.05,zorder=4)
ax.axhline(CHANCE,linestyle='--',linewidth=.8)
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER,rotation=24,ha='right')
ax.set_ylabel('Official-test macro-F1'); ax.set_title('Head-to-head subject-wise comparison')
ax.spines[['top','right']].set_visible(False)
F3=save_fig(fig,'Fig03_GeneralizationAndBaselines')

# ======================================================================
# FIGURE 4 — MDRG ABLATION AS DOT-RANGE PLOT
# ======================================================================
fig,ax=plt.subplots(figsize=(7.2,4.45)); y=np.arange(len(abl)); m=abl['mean'].to_numpy(); lo=abl['lo'].to_numpy(); hi=abl['hi'].to_numpy()
ax.errorbar(m,y,xerr=np.vstack([m-lo,hi-m]),fmt='o',capsize=3,linewidth=1.0,markersize=4.2)
ax.set_yticks(y); ax.set_yticklabels(abl.label); ax.set_xlabel('Macro-F1'); ax.set_title('MDRG representation ablation'); ax.axvline(CHANCE,linestyle='--',linewidth=.8)
span=max(hi)-min(lo); tx=max(hi)+max(.008,.08*span)
for yi,r in abl.iterrows():
    if pd.notna(r['dim']): ax.text(tx,yi,f"{int(r['dim'])}D",va='center',fontsize=7)
ax.text(tx,len(abl)-.22,'Feature dimension',va='bottom',fontsize=7,fontweight='bold')
ax.spines[['top','right']].set_visible(False)
F4=save_fig(fig,'Fig04_MDRGAblation')

# ======================================================================
# FIGURE 5 — SHARED-SCALE CONFUSION MATRICES + RECALL CHANGE
# ======================================================================
y=np.concatenate(pooled_y); pr=np.concatenate(pooled_raw); pf=np.concatenate(pooled_fus)
cmr=confusion_matrix(y,pr,labels=np.arange(5)); cmf=confusion_matrix(y,pf,labels=np.arange(5))
prc=cmr/cmr.sum(1,keepdims=True)*100; pfc=cmf/cmf.sum(1,keepdims=True)*100; vmax=max(prc.max(),pfc.max())
fig,axs=plt.subplots(1,3,figsize=(7.2,2.9),gridspec_kw={'width_ratios':[1,1,.78],'wspace':.34})
ims=[]
for k,(ax,cm,pct,title) in enumerate([(axs[0],cmr,prc,'Raw EEG'),(axs[1],cmf,pfc,'Proposed fusion')]):
    im=ax.imshow(pct,vmin=0,vmax=vmax,aspect='equal'); ims.append(im); ax.set_title(title)
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(CLASS_NAMES,rotation=42,ha='right'); ax.set_yticklabels(CLASS_NAMES if k==0 else ['']*5)
    ax.set_xlabel('Predicted');
    if k==0: ax.set_ylabel('True')
    thr=vmax/2
    for i in range(5):
        for j in range(5):
            ax.text(j,i,f'{cm[i,j]}\n{pct[i,j]:.0f}%',ha='center',va='center',fontsize=6.2,color='white' if pct[i,j]>thr else 'black')
cbar=fig.colorbar(ims[0],ax=[axs[0],axs[1]],fraction=.035,pad=.03); cbar.set_label('Row percentage (%)',fontsize=7.5)
rec_delta=np.diag(pfc)-np.diag(prc); ax=axs[2]; bars=ax.barh(range(5),rec_delta)
for b,d in zip(bars,rec_delta):
    if d<0: b.set_hatch('///')
ax.axvline(0,linewidth=.8); ax.set_yticks(range(5)); ax.set_yticklabels(CLASS_NAMES); ax.set_xlabel('Recall change\n(percentage points)'); ax.set_title('Fusion effect by class'); ax.spines[['top','right']].set_visible(False)
F5=save_fig(fig,'Fig05_ConfusionComparison')

# ------------------------- SAVE AUDIT TABLES ---------------------------
primary.to_csv(os.path.join(OUT,'FIGURE_DATA_primary_subjectwise.csv'),index=False)
methods.to_csv(os.path.join(OUT,'FIGURE_DATA_baseline_subjectwise.csv'),index=False)
abl.to_csv(os.path.join(OUT,'FIGURE_DATA_ablation.csv'),index=False)

audit={
    'sources':{'predictions':PRED_DIR,'official_test_labels':TEST_LABELS_FILE,'reve_subjectwise':REVE_FILE,'final_test_subjectwise':FINAL_TEST_FILE,'stage1_summary':STAGE1_SUMMARY,'ablation_dir':ABLATION_DIR},
    'primary':{'raw_validation_macro_f1_mean':float(raw_v.mean()),'fusion_validation_macro_f1_mean':float(fus_v.mean()),'raw_test_macro_f1_mean':float(raw_t.mean()),'fusion_test_macro_f1_mean':float(fus_t.mean()),'mean_delta':mean_d,'delta_CI95':[ci_d_lo,ci_d_hi],'wilcoxon_p':p_primary,'subjects_improved':n_better},
    'qc':{'eligible_training_trials':qc_total,'removed':qc_removed,'removed_percent':qc_pct,'affected_subject':qc_subject},
    'baseline_stats':baseline_stats,
    'confusion':{'raw_counts':cmr.tolist(),'fusion_counts':cmf.tolist(),'recall_delta_pp':rec_delta.tolist()},
    'figures':{'Fig01':F1,'Fig02':F2,'Fig03':F3,'Fig04':F4,'Fig05':F5}
}
with open(os.path.join(OUT,'FIGURE_AUDIT.json'),'w') as f: json.dump(audit,f,indent=2)

print('\n'+'='*78)
print('PUBLICATION FIGURES COMPLETE')
print('='*78)
print(f'Validation macro-F1: Raw={raw_v.mean():.3f}, Fusion={fus_v.mean():.3f}')
print(f'Official-test macro-F1: Raw={raw_t.mean():.3f}, Fusion={fus_t.mean():.3f}')
print(f'Paired Δ={mean_d:+.3f}, 95% CI [{ci_d_lo:+.3f}, {ci_d_hi:+.3f}], p={p_primary:.4f}, improved={n_better}/15')
print('\nMain-text recommendation: Fig02, Fig03, Fig04, Fig05')
print('Use Fig01 in Methods only if your existing workflow figure does not already explain the split roles.')
print('Output folder:', OUT)