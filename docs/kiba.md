# KIBA (`kiba`)

**Source:** [PyTDC](https://github.com/mims-harvard/TDC) `DTI` task, name **`KIBA`**. Exported from this repo as `kiba`.

**Purpose:** Large **integrated** kinase inhibitor bioactivity matrix (combined evidence from multiple assay types converted to a unified **KIBA** score in the benchmark).

---

## Locations and row counts

| Stage | Relative path | Rows | Notes |
|-------|---------------|-----:|-------|
| Raw | `data/raw/kiba.parquet` | 117,657 | Immediate post-ingestion |
| Standardized | `data/processed/standardized/kiba.parquet` | **117,606** | **`affinity_label`** dedup: 51 `(drug, target)` groups merged via **median** |
| Splits root | `data/processed/splits/kiba/` | — | Nine split Parquets |

**Entity diversity (standardized table):**

- Unique `drug_smiles`: **2,065**
- Unique `target_sequence`: **229**

---

## Columns (pandas dtypes from current files)

### Raw (`data/raw/kiba.parquet`)

| Column | Dtype | Nulls | Description |
|--------|-------|------:|---------------|
| `Drug_ID` | `object` | 0 | Compound identifier |
| `drug_smiles` | `object` | 0 | SMILES prior to canonicalization |
| `Target_ID` | `object` | 0 | Target identifier |
| `target_sequence` | `object` | 0 | Full amino-acid sequence |
| `affinity_label` | `float64` | 0 | **KIBA** integrated score (`y` from TDC) |
| Approx. in-memory size (pandas, `deep=True`) | ~121.4 MB | | |

### Standardized (`data/processed/standardized/kiba.parquet`)

Identifiers and sequences unchanged except invalid SMILES rows removed when applicable; **`drug_smiles`** canonicalized; duplicates on **`(drug_smiles, target_sequence)`** collapsed with **`affinity_label`** = **median**.

**`affinity_label` summary (standardized):** minimum `0.0`, maximum ≈ `17.20`; median ≈ `11.51`; mean ≈ `11.72`.

---

## Splits (`data/processed/splits/kiba/`)

| Strategy | Train | Val | Test | Total rows |
|----------|------:|----:|-----:|-----------:|
| `random/` | 94,084 | 11,761 | 11,761 | 117,606 |
| `drug_cold/` | 93,496 | 11,747 | 12,363 | 117,606 |
| `target_cold/` | 95,323 | 10,175 | 12,108 | 117,606 |

Drug-cold and target-cold splits enforce **no overlapping scaffolds** or **no overlapping target sequences**, respectively, between train / val / test.

---

## Reproducing exports

```bash
python src/data_ingestion/fetch_datasets.py
python src/data_preprocessing/standardize.py kiba
python src/data_preprocessing/generate_splits.py --dataset kiba
```
