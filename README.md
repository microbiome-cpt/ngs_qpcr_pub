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
-   [R Scripts](#r-scripts)

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
├─ datasets/                    # studied datasets
├─ R/
|  ├─ spearman_boxplot.R        # correlation analysis and abundance comparison 
|  ├─ statistics_pipeline.R     # multivariate statistics workflow
|  └─ nmds_3d/                  # interactive 3D NDMS plots
└─ ML/
   ├─ ML_testing_subsets.txt    # 43 subsets of data
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

## CLI arguments

`model_selection.py` exposes the following arguments (main ones):

- `--data <paths...>`: one or more CSV files to process.
- `--target-col <str>`: target label column (e.g., Disease).
- `--id-col <str>`: group ID column (ensures group-aware splits).
- `--exclude-cols <str ...>`: columns to drop from features.
- `--norm {log10,CLR,DEICODE}`: normalization mode.
- `--outer-folds <int>` / `--inner-folds <int>`: CV folds (default 5/3).
- `--seed <int>`: random seed.
- `--no-smote`: disable SMOTE oversampling (enabled by default).
- `--corr-threshold <float>`: |corr(CLR)| threshold for redundancy filter (non-DEICODE).
- `--deicode-components <int>`: number of DEICODE RPCA components (DEICODE only).
- `--output-root <path>`: output directory (default msel_out).
- `--skip-models <names ...>`: exclude some classifiers by name.

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
  --norm CLR \
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

## R Scripts

Executable R code for multivariate statistics and correlation analysis is provided in the `R` subfolder.

**Key features**:  
- All required packages are automatically installed on first execution.  
- No command-line arguments are needed — the pipeline automatically processes predefined datasets located in the `datasets` folder.  
- To run: `Rscript {script_name.R}` from the project root. Output will be generated in the current working directory.  
- **Prerequisite**: the `datasets` folder must be present in the same working directory as the script.

### statistics_pipeline.R

This script executes a comprehensive, end-to-end statistical and exploratory analysis of microbiome data, integrating multivariate testing, ordination, correlation, and diagnostic visualization in a single automated workflow.

**Performed analyses**:  
- Multivariate group comparison: PERMANOVA, MANCOVA, multivariate normality assessment  
- Ordination: PCoA, dbRDA, NMDS  
- Spearman rank correlation  
- Diagnostic and visualization: boxplots, ordination biplots, taxonomic composition profiles, etc.

### spearman_boxplot.R

Independent script for conducting Spearman rank correlation analysis with FDR adjustment between specified taxa across microbiome datasets, producing comprehensive result tables and tailored boxplots of absolute abundances.

**Key features**:  
- Automatic taxon matching by nomenclature across input files; no predefined datasets required.  
- Generates detailed correlation matrices (ρ, p-values, q-values) and publication-quality boxplots with statistical annotations.  
- Highly adaptable: user-defined data paths and taxon list enable application to external datasets.