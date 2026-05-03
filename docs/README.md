# Dataset documentation (Parquet exports)

This folder gives a **text-friendly** overview of tabular data under `data/raw/`, `data/processed/standardized/`, and `data/processed/splits/` so reviewers can skim **columns, dtypes, row counts, and split layout** without opening binary Parquet files.

## Per-dataset pages

| Benchmark | File stem | Doc |
|-----------|-----------|-----|
| Davis | `davis` | [davis.md](davis.md) |
| KIBA | `kiba` | [kiba.md](kiba.md) |
| BindingDB Kd subset | `bindingdb_kd` | [bindingdb_kd.md](bindingdb_kd.md) |

---

## Common schema (after ingestion)

All three raw Parquets are normalized to the **same column names** in `src/data_ingestion/fetch_datasets.py`:

| Column | Role |
|--------|------|
| `drug_smiles` | Drug structure (SMILES) |
| `target_sequence` | Full protein sequence string |
| `affinity_label` | Numeric regression target (TDC field `y`); interpretation depends on the benchmark (KIBA score vs. Kd-style affinity, etc.) |
| `Drug_ID` | Optional compound id when present in the TDC table |
| `Target_ID` | Optional target id (may be null in BindingDB export) |

Standardization (`src/data_preprocessing/standardize.py`): **largest-fragment canonical SMILES** via RDKit; drop invalid SMILES / empty targets; **deduplicate** `(drug_smiles, target_sequence)` with **`affinity_label` aggregated by median** and ids by **first** (where present).

---

## Split layout

Train / validation / test Parquets live at:

```text
data/processed/splits/{davis|kiba|bindingdb_kd}/{random|drug_cold|target_cold}/{train|val|test}.parquet
```

- **`random`:** row-level shuffle (approximately 80% / 10% / 10%); **overlap across drugs/targets/scaffolds is expected** (intentionally leaky baseline).
- **`drug_cold`:** scaffold-disjoint folds (whole Bemis-Murcko scaffolds stay in one fold).
- **`target_cold`:** target-disjoint folds (whole `target_sequence` stays in one fold).

---

## How to inspect Parquet directly

Examples (from repo root after `conda activate dti_research`):

```python
import pandas as pd
df = pd.read_parquet("data/processed/standardized/davis.parquet")
print(df.head(), df.info(), df.describe())
```

You can also use **duckdb**, **parquet-tools**, or a VS Code Parquet viewer extension.
