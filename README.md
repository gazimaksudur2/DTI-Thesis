# Re-evaluating Cross-Modal Feature Fusion in DTI: The Impact of Data Leakage and Cold-Start Splitting

## Abstract

This thesis project builds a reproducible evaluation pipeline for **drug-target interaction (DTI)** regression with a direct focus on **data leakage**. The core comparison is between a naive **random split** and strict **cold-start splits** (drug-cold/scaffold and target-cold/entity) to show how evaluation difficulty changes when the model must generalize to unseen chemistry or proteins.

This repository now implements **Phase 1, Phase 2, and Phase 3** from `guideline.md`: ingestion, standardization/splitting, and an XGBoost baseline experiment runner with report generation.

## Project status by phase

### Phase 1 (Initialization and ingestion) - Completed

- `README.md`, project structure, and environment instructions created.
- Ingestion pipeline implemented in `src/data_ingestion/fetch_datasets.py`.
- Datasets fetched via PyTDC (`DAVIS`, `KIBA`, `BindingDB_Kd`) and exported to `data/raw/*.parquet`.
- Logging includes dataset shape, columns, and missing-value summary.

### Phase 2 (Standardization and cold-start splits) - Completed

- `src/data_preprocessing/standardize.py`
  - Canonicalizes `drug_smiles` with RDKit (largest fragment strategy).
  - Suppresses RDKit C++ logs for cleaner terminal output.
  - Deduplicates by `(drug_smiles, target_sequence)` with median `affinity_label`.
  - Writes standardized outputs to `data/processed/standardized/*.parquet`.
- `src/data_preprocessing/splitters.py`
  - Implements `RandomSplitter`, `DrugColdScaffoldSplitter`, `TargetColdSplitter`.
  - Uses reproducible seed (`RANDOM_SEED = 42`).
- `src/data_preprocessing/generate_splits.py`
  - Exports train/val/test Parquets to:
    - `data/processed/splits/{dataset}/random/`
    - `data/processed/splits/{dataset}/drug_cold/`
    - `data/processed/splits/{dataset}/target_cold/`
  - Logs fold sizes and overlap checks (0 scaffold overlap for drug-cold, 0 target overlap for target-cold).

### Phase 3 (Classical ML baseline with XGBoost) - Completed

- `src/models/features.py`
  - Drug features: Morgan fingerprints (radius 2, 1024 bits).
  - Target features: AAC vector (20 amino acids + unknown bucket).
  - Feature fusion: concatenated drug+target vector.
- `src/models/train_xgboost.py`
  - Trains `xgboost.XGBRegressor` with validation-based early stopping.
  - Reports required metrics: **MSE**, **Pearson r**, **Concordance Index (CI)**.
- `src/models/run_experiments.py`
  - Runs experiments on Davis `random`, `drug_cold`, `target_cold` splits sequentially.
  - Writes markdown summary to `docs/experiment_results.md`.

## Directory structure

```text
.
├── README.md
├── guideline.md
├── requirements.txt
├── data/
│   ├── raw/                  # Raw dataset parquet exports from ingestion
│   ├── tdc_cache/            # PyTDC source cache (.tab/.csv), auto-managed
│   └── processed/
│       ├── standardized/     # Canonicalized + deduplicated parquet tables
│       └── splits/           # random / drug_cold / target_cold train/val/test
├── docs/
│   ├── README.md
│   ├── davis.md
│   ├── kiba.md
│   ├── bindingdb_kd.md
│   └── experiment_results.md
├── src/
│   ├── data_ingestion/
│   │   └── fetch_datasets.py
│   ├── data_preprocessing/
│   │   ├── standardize.py
│   │   ├── splitters.py
│   │   └── generate_splits.py
│   └── models/
│       ├── features.py
│       ├── train_xgboost.py
│       └── run_experiments.py
└── notebooks/
```

## Setup

Assume Windows + conda env `dti_research`.

```bash
conda activate dti_research
cd /path/to/DTIWork
conda install -c conda-forge pytdc -y
pip install -r requirements.txt
```

Notes:
- On Windows, installing `PyTDC` through conda-forge avoids common build issues from `pip`-only installation.
- Run one ingestion process at a time to avoid temporary file locking in `data/tdc_cache/`.

## Dataset overview

| Dataset | Role | Source name in TDC |
|--------|------|--------------------|
| Davis | Kinase-focused affinity benchmark | `DAVIS` |
| KIBA | Integrated kinase bioactivity benchmark | `KIBA` |
| BindingDB Kd | Larger affinity benchmark subset | `BindingDB_Kd` |

## End-to-end usage

### 1) Fetch raw datasets (Phase 1)

```bash
python src/data_ingestion/fetch_datasets.py
```

Optional CSV mirror:

```bash
python src/data_ingestion/fetch_datasets.py --csv
```

### 2) Standardize and split (Phase 2)

```bash
python src/data_preprocessing/standardize.py
python src/data_preprocessing/generate_splits.py
```

### 3) Run XGBoost baseline experiments (Phase 3)

```bash
python src/models/run_experiments.py --dataset davis
```

## Current Phase 3 result snapshot (Davis)

From `docs/experiment_results.md`:

| Split | MSE | Pearson r | CI |
|------|----:|----------:|---:|
| `random` | 7757754.000000 | 0.715890 | 0.866806 |
| `drug_cold` | 12079390.000000 | 0.355803 | 0.682012 |
| `target_cold` | 11368114.000000 | 0.565301 | 0.786867 |

This pattern is consistent with the thesis premise: performance drops under cold-start evaluation compared with random splitting, highlighting optimistic bias in leakage-prone settings.

## Documentation for reviewers

- Dataset schema and counts: `docs/README.md`, `docs/davis.md`, `docs/kiba.md`, `docs/bindingdb_kd.md`
- Experiment metrics table: `docs/experiment_results.md`
