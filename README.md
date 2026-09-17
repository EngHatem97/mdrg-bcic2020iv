# MDRG: Graph–Riemannian Multi-View Fusion for EEG Imagined Speech Decoding

Code, configuration and result artefacts for:

> H. T. M. Duhair, M. bin Mat Ibrahim, J. A. J. Alsayaydeh, and M. Farid,
> "Graph–Riemannian Multi-View Fusion for Subject-Specific EEG Imagined Speech
> Decoding: A Held-Out Evaluation on the BCIC2020-IV Official Test Set",
> *International Journal of Intelligent Engineering and Systems*, 2026.

## What this is

A subject-specific decoder for the five-command BCIC2020-IV (Track 3) imagined
speech benchmark. A compact temporal-spatial CNN is fused at decision level with
a 1656-dimensional multi-view descriptor built from regional log-bandpower,
early-to-late dynamics, Riemannian tangent-space covariance and PLV graph
spectra. Everything is evaluated on the **official held-out test split**.

## Evaluation protocol

| Split | Trials per subject | Role |
|---|---|---|
| Official train | 300 | fits the models |
| Official validation | 50 | selects architecture, seed, stopping epoch, feature block, classifier, regularisation, fusion weight |
| Official test | 50 | scored **once**, after every parameter is fixed |

The selected configuration is refit on the 350 labelled development trials
(train + validation) using the validation-chosen architecture, seed and epoch,
and the test split is then scored a single time. The test predictions reported
in the paper were written to disk during the original experimental run, before
the answer key was retrieved, so no test trial influenced any modelling choice.

## Headline results (official test split, 15 subjects)

| System | Accuracy | Macro-F1 |
|---|---|---|
| Proposed fusion | 0.305 ± 0.064 | **0.297 ± 0.066** |
| Pretrained encoder, LoRA | **0.313 ± 0.066** | 0.284 ± 0.087 |
| Pretrained encoder, linear probe | 0.311 ± 0.054 | 0.279 ± 0.076 |
| MDRG branch alone (linear probe) | 0.277 ± 0.078 | 0.272 ± 0.079 |
| Consensus-graph comparator | 0.269 ± 0.071 | 0.264 ± 0.068 |
| Raw EEG branch | 0.269 ± 0.067 | 0.262 ± 0.067 |
| Chance | 0.200 | 0.200 |

No pairwise difference survives Holm correction across the 15 participants.
The same models scored macro-F1 0.487 and 0.399 on the **validation** split, so
the validation-to-test reduction is 39% and 34% and occurs for every
participant without exception.

## Getting the data

1. **Dataset.** BCIC2020-IV Track 3 is distributed by the organisers of the
   2020 International Brain-Computer Interface Competition and is not
   redistributed here. Download it, then set `paths.dataset_root` in
   `configs/config.json`. The folder must contain `Training Set`,
   `Validation Set` and `Testing Set`.

2. **Test labels.** The official answer sheet is likewise not redistributed.
   Run the retrieval script below; it fetches the sheet from the organisers'
   public OSF project (`pq7vb`, doi `10.17605/OSF.IO/PQ7VB`, CC BY 4.0),
   verifies it by SHA-256, and checks that every participant has exactly 10
   test trials per class before any label is used.

## Quick start

```bash
pip install -r requirements.txt

# edit configs/config.json first, then:
python src/download_test_labels.py        # fetch + verify the answer sheet
python src/make_artifacts.py              # regenerate Tables 2 and 3 as files
python src/run_01_pipeline.py             # preprocessing, MDRG features, training, fusion
python src/run_02_baselines_and_stats.py  # external baselines + paired statistics
python src/run_03_figures.py              # publication figures
```

`src/mdrg_blocks.py` runs standalone and prints the block layout, which is a
quick way to confirm the environment is set up:

```bash
python src/mdrg_blocks.py
```

## Repository layout

```
configs/config.json                 every hyperparameter (Table 4) and all paths
configs/reve_reproducibility.json   pinned encoder revisions and settings
src/paths.py                        path resolution, replaces the Colab Drive mounts
src/mdrg_blocks.py                  block index map (Table 3), region map (Table 2), ablation variants
src/download_test_labels.py         answer-sheet retrieval, checksum and alignment checks
src/make_artifacts.py               writes the machine-readable Tables 2 and 3
src/run_01_pipeline.py              Stage 1 to 3
src/run_02_baselines_and_stats.py   external baselines and final statistics
src/run_03_figures.py               figures
artifacts/channel_to_region.json    Table 2
artifacts/mdrg_block_layout.csv     Table 3
results/*.csv                       the values behind Tables 5 to 7 and Figures 4 to 7
```

## The 1656 dimensions

Per trial, for each of 4 bands × 2 windows: 18 regional log-bandpower + 171
tangent-space + 9 graph-spectral = 198, giving 1584; plus 4 × 18 late-minus-early
dynamic contrasts = 72. Total 1656. The blocks are contiguous and
non-overlapping, so any descriptor family is recoverable by column slicing
(`src/mdrg_blocks.py`), which is what makes the component ablation exact.

The 18 regions cover 60 of the 64 recorded channels. **FT9, FT10, TP7 and TP8
are unassigned**: each falls between two adjacent groupings and assigning it to
either would leave that region inconsistent with its contralateral counterpart.
Those four channels feed the raw EEG branch but not the MDRG branch.

## Notes on the code

`run_01`, `run_02` and `run_03` are the original experimental scripts,
unmodified except that the hard-coded Google Drive paths were replaced by a
lookup in `configs/config.json`. They are kept intact rather than refactored so
that the published numbers remain exactly reproducible.

No MDRG column is ever removed. Non-finite entries are repaired by
training-column median imputation followed by clipping to the training 0.1st and
99.9th percentiles; the dimensionality supplied to the classifier is 1656 for
every participant and every split.

## Citing

Please cite the paper above and the BCIC2020-IV dataset. `CITATION.cff` lets
GitHub generate a citation for the software itself.

## Licence

MIT for the code in this repository. The dataset and the answer sheet remain
under the terms set by their providers and are not redistributed here.
