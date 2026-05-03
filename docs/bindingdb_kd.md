# BindingDB Kd subset (`bindingdb_kd`)

**Source:** [PyTDC](https://github.com/mims-harvard/TDC) `DTI` task, name **`BindingDB_Kd`**. Exported from this repo as `bindingdb_kd`.

**Purpose:** Regression benchmark from BindingDB emphasizing **dissociation constant** (Kd)-style annotations (tabular format via TDC; confirm unit conventions in BindingDB/TDC docs when interpreting `affinity_label` numerically).

---

## Locations and row counts

| Stage | Relative path | Rows | Notes |
|-------|---------------|-----:|-------|
| Raw | `data/raw/bindingdb_kd.parquet` | 52,274 | **`Target_ID` missing** on **4,333** rows in this snapshot |
| Standardized | `data/processed/standardized/bindingdb_kd.parquet` | **46,114** | Dedup **`(drug_smiles, target_sequence)`** with **median** `affinity_label` removes **6,160** raw rows (\(52274\to 46114\)); remaining **`Target_ID`** null count **3,888** |
| Splits root | `data/processed/splits/bindingdb_kd/` | — | Nine split Parquets |

**Entity diversity (standardized table):**

- Unique `drug_smiles`: **10,660**
- Unique `target_sequence`: **1,413**

---

## Columns (pandas dtypes from current files)

### Raw (`data/raw/bindingdb_kd.parquet`)

| Column | Dtype | Nulls | Description |
|--------|-------|------:|---------------|
| `Drug_ID` | `float64` | 0 | Numeric id in upstream export (`object`/`int`-like semantics) |
| `drug_smiles` | `object` | 0 | SMILES string |
| `Target_ID` | `object` | **4,333** | Protein/target id where provided |
| `target_sequence` | `object` | 0 | Full amino-acid sequence |
| `affinity_label` | `float64` | 0 | Regression target |
| Approx. in-memory size (pandas, `deep=True`) | ~50.0 MB | | |

Missing identifiers on `Target_ID` do **not** imply missing sequence: every raw row retains `target_sequence` in this export.

### Standardized (`data/processed/standardized/bindingdb_kd.parquet`)

| Column | Dtype | Nulls (current file) |
|--------|-------|----------------------:|
| `drug_smiles` | `object` | 0 |
| `target_sequence` | `object` | 0 |
| `affinity_label` | `float64` | 0 |
| `Drug_ID` | `float64` | 0 |
| `Target_ID` | `object` | **3,888** |

`Target_ID` nulls propagate when the **`first`** value in each dedupe group lacked an id after aggregation.

**`affinity_label` summary (standardized):** min `0.0`, max **10,000,000**, median ≈ `10000`; mean ≈ `46568` (**heavy tails** typical of Kd nM-style ranges mixed across assays—use robust statistics when reporting).

---

## Splits (`data/processed/splits/bindingdb_kd/`)

| Strategy | Train | Val | Test | Total rows |
|----------|------:|----:|-----:|-----------:|
| `random/` | 36,891 | 4,611 | 4,612 | 46,114 |
| `drug_cold/` | 38,374 | 2,438 | 5,302 | 46,114 |
| `target_cold/` | 37,163 | 4,184 | 4,767 | 46,114 |

---

## Caveats for reviewers

- Report whether analyses **require** stable `Target_ID` (e.g., external mapping); otherwise rely on **`target_sequence`** for identity.
- High dynamic range on `affinity_label` may warrant **log transform** inside modeling scripts (outside the scope of this schema doc).

---

## Reproducing exports

```bash
python src/data_ingestion/fetch_datasets.py
python src/data_preprocessing/standardize.py bindingdb_kd
python src/data_preprocessing/generate_splits.py --dataset bindingdb_kd
```
