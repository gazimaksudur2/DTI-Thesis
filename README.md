# Re-evaluating Cross-Modal Feature Fusion in DTI: The Impact of Data Leakage and Cold-Start Splitting

## Abstract

This thesis project builds a reproducible evaluation pipeline for **drug–target interaction (DTI)** models. Random train/test splits often let models see chemically or structurally similar drugs and proteins in both halves of the split, causing **optimistic bias** (**data leakage**). We prioritize **cold-start** protocols—drug-cold, target-cold, and drug+target-cold—that force generalization to unseen entities, and we compare them to naive random splitting. Phase 1 focuses on documentation and **raw data ingestion**; model training and split implementations come after ingestion is verified.

## Directory structure

```text
.
├── README.md
├── guideline.md
├── requirements.txt
├── data/
│   ├── raw/          # Benchmark tables written by ingestion (ignored by git except .gitkeep)
│   ├── tdc_cache/    # PyTDC downloaded source files (.tab/.csv); auto-managed; gitignored
│   └── processed/    # Reserved for cleaned / feature-ready data (later phases)
├── src/
│   ├── data_ingestion/
│   │   └── fetch_datasets.py
│   └── models/       # Reserved for modeling code (later phases)
└── notebooks/        # Exploratory and reporting notebooks
```

## Setup

This repository uses Git LFS for large dataset files.

```bash
git lfs install
git clone https://github.com/gazimaksudur2/DTI-Thesis.git

Assume **Git Bash on Windows** and a conda environment named **`dti_research`** (already created on your machine).

```bash
conda activate dti_research
cd /path/to/DTIWork
conda install -c conda-forge pytdc -y
pip install -r requirements.txt
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

Uses [PyTDC](https://github.com/mims-harvard/TDC) (`tdc.multi_pred.DTI`) to download benchmark-ready tables (no manual CSV hunting). Logs shapes, dtypes, missing-value counts, and output paths.

Run **one** ingestion at a time so Windows does not lock `data/tdc_cache/*.tab` while another process downloads or parses the same file.

## Phase boundary

Do **not** start model implementation or cold-start splitting until raw ingestion completes successfully and outputs are inspected.
