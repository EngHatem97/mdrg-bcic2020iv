# MDRG: Graph-Riemannian Multi-View Fusion for EEG Imagined Speech Decoding

Code, configuration and result files for:

> H. T. M. Duhair, M. bin Mat Ibrahim, J. A. J. Alsayaydeh and M. Farid,
> "Graph-Riemannian Multi-View Fusion for Subject-Specific EEG Imagined Speech
> Decoding: A Held-Out Evaluation on the BCIC2020-IV Official Test Set",
> *International Journal of Intelligent Engineering and Systems*, 2026.

## What this is

Six subject-specific decoders for the five-command BCIC2020-IV Track 3 imagined
speech benchmark (15 participants, 64 channels): a compact raw-EEG network, a
1656-dimensional graph-Riemannian multi-view descriptor (MDRG) read out by a
shallow classifier, their decision-level fusion, a multi-view graph comparator,
and a pretrained EEG encoder (REVE) used as a linear probe and with low-rank
adaptation. All six are configured on the official validation split and scored
once on the official test split.

## Evaluation protocol

| Split | Trials per participant | Role |
|---|---|---|
| Official train | 300 (297 for participant 14) | fits every candidate |
| Official validation | 50 | selects architecture, seed, stopping epoch, feature subset, classifier, regularisation constant, fusion weight and graph settings |
| Official test | 50 | scored once per system |

Each selected configuration is refit on train + validation before the test
split is scored (the encoder baseline follows its released recipe and is not
refit). In `src/run_final_v2.py` the test labels (`yte`) are passed only to
`metrics()` and to `np.savez_compressed()`, which stores test posteriors for
the statistics; no fitting, selection or fusion-weight call receives them.
You can check this with

```bash
grep -n "yte" src/run_final_v2.py
```

## Results on the official test split (15 participants, mean ± SD)

| System | Accuracy | Macro-F1 |
|---|---|---|
| Pretrained encoder, low-rank adaptation | 0.313 ± 0.066 | 0.284 ± 0.087 |
| Pretrained encoder, linear probe | 0.311 ± 0.054 | 0.279 ± 0.076 |
| Proposed fusion (raw + MDRG) | 0.291 ± 0.069 | 0.274 ± 0.072 |
| Raw EEG branch | 0.284 ± 0.047 | 0.268 ± 0.052 |
| MDRG branch | 0.277 ± 0.064 | 0.260 ± 0.061 |
| Multi-view graph comparator | 0.255 ± 0.071 | 0.243 ± 0.064 |
| Chance | 0.200 | 0.200 |

No paired difference survives Holm correction. On the validation split, where
the fusion weight is chosen, fusion beats the raw branch by 0.028 macro-F1
(p = 0.012); on the test split by 0.005 (p = 0.508).

## Getting the data

1. **Dataset.** BCIC2020-IV Track 3 is distributed by the organisers of the 2020
   International Brain-Computer Interface Competition (OSF project `pq7vb`,
   CC BY 4.0) and is not redistributed here. Set `paths.dataset_root` in
   `configs/config.json` to your copy.
2. **Test labels.** The official answer sheet is not redistributed either.
   `src/download_test_labels.py` fetches it from the organisers' OSF project,
   verifies its SHA-256 checksum, checks that every participant has exactly 10
   test trials per class, and writes `artifacts/official_test_labels.npy`
   (ignored by git).

## Reproducing the paper

```bash
pip install -r requirements.txt
# edit configs/config.json first

python src/download_test_labels.py   # once
python src/run_final_v2.py           # Tables 5 and 7 and the inputs of Table 6, Table 8 and Figs. 4 to 7 (GPU, about 30 min)
python src/run_reve_baseline.py      # encoder rows of Table 5 (GPU)
python src/stats_final_v2.py         # Tables 6 and 8 (and Table 7 SDs) from the CSVs alone, no data needed
python src/make_figures_v2.py --results results --out figures   # Figs. 1, 4, 5, 6, 7
```

`src/run_final_v2.py` starts from the stored Stage-1 tensors
(`prep_stage1_final_robust/`) and Stage-2 feature matrices
(`INASS_REVISION_FINAL/stage2_MDRG_corrected/npz/`). Stage 1 (signal
conditioning and quality control) and Stage 2 (the MDRG descriptor) are
specified in Sections 3.2 and 3.3 and Table 4 of the paper and in
`configs/config.json`.

`stats_final_v2.py` and `make_figures_v2.py` need only the files already in
`results/`, so every statistic and figure can be checked without the dataset.

## Repository layout

```
configs/config.json                 every setting in Table 4, and the data path
configs/reve_reproducibility.json   pinned encoder revisions and fine-tuning settings
src/paths.py                        path resolution from configs/config.json
src/download_test_labels.py         answer-sheet retrieval, checksum and alignment checks
src/run_final_v2.py                 selection, refit, fusion, graph comparator, ablation
src/run_reve_baseline.py            pretrained-encoder baseline
src/stats_final_v2.py               Tables 6 and 8 from the result files
src/make_figures_v2.py              Figs. 1 and 4 to 7 at print size
src/mdrg_blocks.py                  Table 2 region map, Table 3 column layout, ablation variants
src/make_artifacts.py               writes the two files below from mdrg_blocks.py
artifacts/channel_to_region.json    Table 2 (18 regions, all 64 channels)
artifacts/mdrg_block_layout.csv     Table 3 (column ranges of the 1656-dimensional vector)
artifacts/stage1_qc_audit.csv       per-participant quality-control audit
results/                            the values behind Tables 5 to 8 and Figs. 4 to 7
figures/                            Figs. 1 and 4 to 7 as printed
```

## Result files

| File | Contents |
|---|---|
| `FINAL_v2_SUBJECTWISE.csv` | per participant: selected configuration, validation and test scores of four systems |
| `REVE_SUBJECTWISE.csv`, `REVE_SUMMARY.csv` | per participant and mean test scores of the two encoder variants |
| `FINAL_v2_BASELINE_COMPARISON.csv` | Table 5 |
| `FINAL_v2_GENERALISATION_GAP.csv` | Table 8 |
| `FINAL_v2_PAIRED_STATISTICS_FAMILY5.csv` | Table 6 |
| `FINAL_v2_PAIRED_STATISTICS.csv` | the pipeline's own three-comparison summary; Table 6 uses the five-comparison family above |
| `FINAL_v2_ABLATION_SUBJECTWISE.csv`, `..._SUMMARY.csv`, `..._STATISTICS.csv` | Table 7 and Fig. 7 |
| `FINAL_v2_CONFUSION.json` | Fig. 6 (aggregate counts only) |

## The 1656 dimensions

For each of 4 bands and 2 windows: 18 regional log-bandpower + 171
tangent-space + 9 graph-spectral values = 198, giving 1584; then 4 x 18
late-minus-early contrasts = 72. The families occupy fixed, non-overlapping
column ranges (`artifacts/mdrg_block_layout.csv`), so any family is recovered
by column slicing. The 18 regions cover all 64 channels. No column is imputed,
clipped or removed: the stored matrices contain no non-finite values.

## Licence

MIT for the code. The dataset and the answer sheet remain under the terms set
by their providers and are not redistributed here.
