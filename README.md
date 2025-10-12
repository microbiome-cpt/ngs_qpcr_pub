# ngs_qpcr_pub
Code and scripts from "Comparison of qPCR and 16S rRNA NGS for Quantitative Analysis of Human Gut Microbiota" study.

This repository contains a pipeline for classification from a tabular data with group-aware nested cross-validation, optional class rebalancing, and post-hoc explainability (SHAP).

## Table of Contents

-   [Environment](#environment)
-   [Repository Structure](#repository-structure)
-   [Input Data](#input-data)
-   [CLI arguments](#cli-arguments)
-   [How to run](#how-to-run)
-   [Outputs](#outputs)

## Environment

### Requirements

A requirements.txt is located in ngs_qpcr_pub/ML. Install it into a clean Python 3.10 environment:

``` bash
# using conda + pip
conda create -n micro-ml python=3.10 numpy=1.26.4 scipy=1.12.0 pandas=2.3.1 scikit-learn=1.4.2 -c conda-forge -y
conda activate micro-ml
pip install -r ngs_qpcr_pub/ML/requirements.txt
```

You can choose models and hyperparametrs in ml_utils/models.py.

## Repository Structure

```
ngs_qpcr_pub/
└─ ML/
   ├─ requirements.txt          # install this
   └─ ml_utils/
      ├─ models.py              # list of classifiers + hyperparam grids
      ├─ model_selection.py     # main entrypoint (nested CV, reports)
      ├─ transforms.py          # preprocessing / normalization
      ├─ importance_utils.py    # metrics, calibration, SHAP
      └─ io_utils.py            # CSV loading, target/ID handling
```

## Input Data

Input files are CSV. You specify:
- a target column (`--target-col`, for example "Disease"),
- a group/ID column (`--id-col`, e.g. patient or sample group),
- optional columns to exclude from features (`--exclude-cols`).

Helper utilities sanitize columns, encode Sex → Sex_enc, and automatically select numeric microbiome signatures while keeping common covariates (e.g., Sex_enc, Age, BMI).

Notes:

- Delimiter and encoding are auto-detected (,/;, UTF-8/CP1251 supported).
- Features = all non-excluded columns except target/ID; missing values are safely imputed.
- Multiclass is supporte and per-class scoring uses label names exactly as in the data.
- Grouped CV requires ≥2 samples per class per fold. Tiny classes will be auto-skipped in inner splits.


## CLI arguments

`model_selection.py` exposes the following arguments (main ones):

- `--data <paths...>`: one or more CSV files to process.
- `--target-col <str>`: target label column (e.g., Disease).
- `--id-col <str>`: group ID column (ensures group-aware splits).
- `--exclude-cols <str ...>`: columns to drop from features.
- `--norm {NONE,log10,CLR,DEICODE}`: normalization mode (default: NONE).
- `--outer-folds <int>` / `--inner-folds <int>`: CV folds (default 5/3).
- `--seed <int>`: random seed.
- `--scoring <expr>`: Scoring by chosen metric for model selection (see below).
- `--main_group <str>`: Optional group focus if not specified explicitly in the scoring argument.
- `--use-smote`: Optional enable SMOTE (default is off).
- `--corr-threshold <float>`: |corr(CLR)| threshold for redundancy filter (non-DEICODE).
- `--deicode-components <int>`: number of DEICODE RPCA components (DEICODE only).
- `--output-root <path>`: output directory (default msel_out).
- `--skip-models <names ...>`: exclude some classifiers by name.

### Available metrics & Scoring

- Binary/OVR: `roc_auc`, `pr_auc`, `f1`, `accuracy`, `balanced_accuracy`, `log_loss`

- Macro (class-balanced): `macro_pr_auc`, `macro_f1`

- Tuned: *_tuned (e.g., `f1_tuned`) — decision threshold is optimized on inner-CV, then used for evaluation.

#### Scoring

Combine metrics with +, apply weights with *.

- Per-class metric: metric[ClassName] (e.g., pr_auc[PF]). If class omitted, `--main_group` supplies the default.

- Macro metrics (macro_*) aggregate across classes and ignore `--main_group`.

``` bash
# Examples
--scoring "roc_auc+f1_tuned"
--scoring "pr_auc+f1_tuned" --main_group PF
--scoring "macro_pr_auc+macro_f1"
--scoring "pr_auc[HFpEF]*2 + pr_auc[HFrEF]*2 + macro_f1"
```

## How to run

**A) Minimal direct call (one file)**

```
# from the repository root
export PYTHONPATH="ngs_qpcr_pub/ML:${PYTHONPATH}"
python ngs_qpcr_pub/ML/model_selection.py \
  --data data.csv \
  --target-col Disease \
  --id-col Name \
  --exclude-cols Method \
  --scoring "pr_auc+f1_tuned" \
  --main_group PF \
  --outer-folds 5 --inner-folds 3 \
  --seed 42 \
  --output-root msel_out
```

**B) Multiple files + DEICODE example**

```
export PYTHONPATH="ngs_qpcr_pub/ML:${PYTHONPATH}"
python ngs_qpcr_pub/ML/model_selection.py \
  --data data.csv data2.csv data3.csv \
  --target-col Disease \
  --id-col Name \
  --exclude-cols Method \
  --norm DEICODE --deicode-components 50 \
  --scoring "macro_pr_auc+macro_f1"
  --outer-folds 5 --inner-folds 3 \
  --seed 42 \
  --output-root msel_out
```

You can also use a Bash wrapper.

## Outputs

For each input CSV, an output folder is created under `--output-root/<file_stem>`, containing:

- out-of-fold predictions and metrics, best params, summary JSON, and a Markdown report;
- feature importance / SHAP top-20 CSVs; optional DEICODE loadings;
- LOO accuracy summary.
