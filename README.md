## Abstract

This thesis project builds a reproducible evaluation pipeline for **drug-target interaction (DTI)** regression with a direct focus on **data leakage**. The core comparison is between a naive **random split** and strict **cold-start splits** (drug-cold/scaffold and target-cold/entity) to show how evaluation difficulty changes when the model must generalize to unseen chemistry or proteins.

This repository now implements **Phase 1, Phase 2, Phase 3, and Phase 4** from `guideline.md`: ingestion, standardization/splitting, an XGBoost and Random Forest baseline runner with multi-model result tables and CSV export, matplotlib/seaborn bar charts, a SHAP summary plot, and Docker containerization for fully reproducible execution.

## Project status by phase

### Phase 1 (Initialization and ingestion)

- `README.md`, project structure, and environment instructions created.
- Ingestion pipeline implemented in `src/data_ingestion/fetch_datasets.py`.
- Datasets fetched via PyTDC (`DAVIS`, `KIBA`, `BindingDB_Kd`) and exported to `data/raw/*.parquet`.
- Logging includes dataset shape, columns, and missing-value summary.

### Phase 2 (Standardization and cold-start splits)

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

### Phase 3 (Classical ML baseline with XGBoost)

- `src/models/features.py`
  - Drug features: Morgan fingerprints (radius 2, 1024 bits).
  - Target features: AAC vector (20 amino acids + unknown bucket).
  - Feature fusion: concatenated drug+target vector.
- `src/models/train_xgboost.py`
  - Trains `xgboost.XGBRegressor` with validation-based early stopping.
  - Reports required metrics: **MSE**, **Pearson r**, **Concordance Index (CI)**.
  - Optionally persists the fitted model to disk for downstream SHAP analysis.
- `src/models/run_experiments.py`
  - Runs experiments on Davis `random`, `drug_cold`, `target_cold` splits sequentially.
  - Writes markdown summary to `docs/experiment_results.md`.
This thesis project builds a reproducible evaluation pipeline for **drug–target interaction (DTI)** models. Random train/test splits often let models see chemically or structurally similar drugs and proteins in both halves of the split, causing **optimistic bias** (**data leakage**). We prioritize **cold-start** protocols—drug-cold, target-cold, and drug+target-cold—that force generalization to unseen entities, and we compare them to naive random splitting. Phase 1 focuses on documentation and **raw data ingestion**; model training and split implementations come after ingestion is verified.

### Phase 4 (Multi-model validation, visualization, and MLOps)

- `src/models/train_rf.py`
  - Replicates the XGBoost feature + evaluation pipeline with `sklearn.ensemble.RandomForestRegressor` (CPU-only, no early stopping).
- `src/models/run_experiments.py` (extended)
  - Trains **both** XGBoost and Random Forest on all three splits via `--models xgboost,random_forest`.
  - Writes side-by-side markdown table (`docs/experiment_results.md`), long-form CSV (`docs/experiment_results.csv`), persisted XGBoost models (`models/{dataset}/xgboost_{split}.json`), and feature schema (`models/{dataset}/feature_meta.json`).
- `src/visualization/plot_metrics.py`
  - Reads `docs/experiment_results.csv` and generates grouped bar charts: `docs/figures/mse_comparison.png` and `docs/figures/ci_comparison.png`.
- `src/visualization/explain_model.py`
  - Loads a saved XGBoost model, runs `shap.TreeExplainer` on a deterministic test subset, saves `docs/figures/shap_summary.png`.
- `Dockerfile` / `docker-compose.yml`
  - CUDA 12.4 base image with Miniforge + `dti_research` env baked in; workspace mounted as volume; GPU reservation via `deploy.resources.reservations.devices`.

## Directory structure

```text
.
├── README.md
├── guideline.md
├── requirements.txt            # CPU dependencies (Windows + Linux)
├── requirements-gpu.txt        # Additional GPU wheel for WSL2/Linux
├── Dockerfile                  # CUDA 12.4 image with dti_research env
├── docker-compose.yml          # Volume-mounted service with GPU reservation
├── logs/                       # Timestamped run logs (gitignored)
├── models/                     # Runtime outputs: saved XGBoost .json + feature_meta.json (gitignored, auto-created by runner)
├── data/
│   ├── raw/                    # Raw dataset parquet exports from ingestion
│   ├── tdc_cache/              # PyTDC source cache (.tab/.csv), auto-managed
│   └── processed/
│       ├── standardized/       # Canonicalized + deduplicated parquet tables
│       └── splits/             # random / drug_cold / target_cold train/val/test
├── docs/
│   ├── README.md
│   ├── davis.md
│   ├── kiba.md
│   ├── bindingdb_kd.md
│   ├── experiment_results.md   # Human-readable multi-model results table
│   ├── experiment_results.csv  # Long-form CSV for visualization
│   └── figures/                # PNG charts and SHAP plots (gitignored)
│       ├── mse_comparison.png
│       ├── ci_comparison.png
│       └── shap_summary.png
│   ├── raw/          # Benchmark tables written by ingestion (ignored by git except .gitkeep)
│   ├── tdc_cache/    # PyTDC downloaded source files (.tab/.csv); auto-managed; gitignored
│   └── processed/    # Reserved for cleaned / feature-ready data (later phases)
├── src/
│   ├── data_ingestion/
│   │   └── fetch_datasets.py
│   ├── data_preprocessing/
│   │   ├── standardize.py
│   │   ├── splitters.py
│   │   └── generate_splits.py
│   ├── models/                 # Source code only (no data files live here)
│   │   ├── features.py         # Morgan fingerprint + AAC feature extraction
│   │   ├── train_xgboost.py    # XGBoost training + evaluation
│   │   ├── train_rf.py         # Random Forest training + evaluation
│   │   └── run_experiments.py  # Orchestrator: trains both models, writes results
│   ├── utils/
│   │   └── device.py           # CUDA detection + device resolution helpers
│   └── visualization/
│       ├── plot_metrics.py     # Bar charts (MSE, CI)
│       └── explain_model.py    # SHAP summary plot
└── notebooks/
│   └── models/       # Reserved for modeling code (later phases)
└── notebooks/        # Exploratory and reporting notebooks
```

## Setup

Assume **Git Bash on Windows** and a conda environment named **`dti_research`** (already created on your machine).

```bash
conda activate dti_research
cd /path/to/DTIWork
conda install -c conda-forge pytdc -y
pip install -r requirements.txt
```

Notes:
- On Windows, installing `PyTDC` through conda-forge avoids common build issues from `pip`-only installation.
- Run one ingestion process at a time to avoid temporary file locking in `data/tdc_cache/`.

## GPU acceleration (RTX 4060 Ti) + CPU fallback

This project supports `--device {auto,gpu,cpu}` for Phase 3 XGBoost runs:

- `auto` (default): use GPU if a usable CUDA device is detected; otherwise CPU.
- `gpu`: require GPU; fails fast if CUDA is not usable.
- `cpu`: force CPU.

### Quick check (Windows + WSL2)

On Windows (PowerShell or Git Bash):

```bash
nvidia-smi
```

Inside WSL2 Ubuntu:

```bash
nvidia-smi
```

### Windows-native (CPU)

CUDA-enabled XGBoost wheels are not reliably available as conda packages for Windows, so Windows-native is the CPU fallback path.

```bash
conda activate dti_research
pip install -r requirements.txt
python src/models/run_experiments.py --dataset davis --device cpu
```

### WSL2 Ubuntu (GPU)

Create/activate a `dti_research` env inside WSL2 Ubuntu, then:

```bash
conda activate dti_research
pip install -r requirements.txt -r requirements-gpu.txt
python src/models/run_experiments.py --dataset davis --device auto
```

Optional verification (prints `cuda` or `cpu`):

```bash
python -c "import sys; sys.path.insert(0, 'src'); from utils.device import resolve_device; print(resolve_device('auto'))"
```
On Windows, `pip install PyTDC` alone can fail while compiling optional dependencies such as **tiledbsoma**; conda-forge provides pre-built binaries.

PyTDC stores downloaded `.tab` / `.csv` sources under **`data/tdc_cache/`** (configured by `fetch_datasets.py`, ignored by git). Processed exports go to **`data/raw/`**.

## Dataset overview

| Dataset | Role in this repo | Notes |
|--------|-------------------|--------|
| **Davis** | Primary kinase inhibitor benchmark | \(K_d\)-focused screen; TDC exposes `DAVIS` with SMILES, sequence, and affinity. |
| **KIBA** | Integrated bioactivity matrix | Combines complementary assay types into a unified score (TDC: `KIBA`). |
| **BindingDB** | Large public affinity benchmark | In TDC, BindingDB is split by assay units. **Default ingestion:** `BindingDB_Kd` (regression, comparable affinity type to Davis-style \(K_d\)). Alternatives: `BindingDB_IC50`, `BindingDB_Ki` ([TDC DTI task](http://tdcommons.ai/multi_pred_tasks/dti/)). |

## Fetch raw data

From the repository root:

```bash
conda activate dti_research
python src/data_ingestion/fetch_datasets.py
```

- Writes **Parquet** files under `data/raw/` by default.
- Optional CSV mirror for each dataset:

  ```bash
  python src/data_ingestion/fetch_datasets.py --csv
  ```

### 2) Standardize and split (Phase 2)

```bash
python src/data_preprocessing/standardize.py
python src/data_preprocessing/generate_splits.py
```

### 3) Run classical baselines (Phase 3 & 4)

```bash
python src/models/run_experiments.py --dataset davis --device auto
```

This trains both XGBoost and Random Forest on the three splits and writes:

- `docs/experiment_results.md` and `docs/experiment_results.csv`
- `models/{dataset}/xgboost_{split}.json` and `models/{dataset}/feature_meta.json`
- `logs/phase4_{dataset}_*.log`

You can then generate the bar charts and SHAP summary plot:

```bash
python src/visualization/plot_metrics.py --dataset davis
python src/visualization/explain_model.py --dataset davis --split random
```

Figures are saved into `docs/figures/`.

### 4) Run via Docker (optional)

Build the image once (installs Miniforge, `dti_research` env, and all dependencies):

```bash
docker compose build
```

Run the Phase 4 experiment (default command trains both models on Davis):

```bash
docker compose run --rm dti
```

Override the command to run plotting or SHAP inside the container:

```bash
docker compose run --rm dti bash -lc "python src/visualization/plot_metrics.py --dataset davis"
docker compose run --rm dti bash -lc "python src/visualization/explain_model.py --dataset davis --split random"
```

All outputs (`docs/`, `models/`, `logs/`) are persisted on the host via the volume mount in `docker-compose.yml`.

## Documentation for reviewers

- Dataset schema and counts: `docs/README.md`, `docs/davis.md`, `docs/kiba.md`, `docs/bindingdb_kd.md`
- Multi-model results table: `docs/experiment_results.md`
- Long-form CSV for analysis: `docs/experiment_results.csv`
- Bar charts: `docs/figures/mse_comparison.png`, `docs/figures/ci_comparison.png`
- SHAP summary: `docs/figures/shap_summary.png`
- Run history: `logs/phase4_{dataset}_*.log`
Uses [PyTDC](https://github.com/mims-harvard/TDC) (`tdc.multi_pred.DTI`) to download benchmark-ready tables (no manual CSV hunting). Logs shapes, dtypes, missing-value counts, and output paths.

Run **one** ingestion at a time so Windows does not lock `data/tdc_cache/*.tab` while another process downloads or parses the same file.

## Phase boundary

Do **not** start model implementation or cold-start splitting until raw ingestion completes successfully and outputs are inspected.
