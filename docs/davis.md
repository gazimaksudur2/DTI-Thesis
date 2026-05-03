# Davis kinase benchmark (`davis`)

**Source:** [PyTDC](https://github.com/mims-harvard/TDC) `DTI` task, name **`DAVIS`**. Exported from this repo as `davis`.

**Purpose:** Widely used **kinase inhibitor** measurement table with drug SMILES, target sequences, and a continuous **`affinity_label`** suited to **Kd-focused** benchmarking in the literature—exact units and any censoring follow the TDC `DAVIS` definition (see TDC for physical interpretation).

---

## Locations and row counts

| Stage | Relative path | Rows | Notes |
|-------|---------------|-----:|-------|
| Raw | `data/raw/davis.parquet` | **25,772** | Post-ingestion |
| Standardized | `data/processed/standardized/davis.parquet` | **25,772** | Same row count after dedup in this snapshot (`drug_smiles`, `target_sequence` pairs already unique upstream) |
| Splits root | `data/processed/splits/davis/` | — | Nine split Parquets |

**Entity diversity (standardized table):**

- Unique `drug_smiles`: **68**
- Unique `target_sequence`: **379**

---

## Columns (pandas dtypes from current files)

### Raw (`data/raw/davis.parquet`)

| Column | Dtype | Nulls | Description |
|--------|-------|------:|-------------|
| `Drug_ID` | `int64` | 0 | Compound identifier |
| `drug_smiles` | `object` | 0 | SMILES prior to canonicalization |
| `Target_ID` | `object` | 0 | Target identifier |
| `target_sequence` | `object` | 0 | Full amino-acid sequence |
| `affinity_label` | `float64` | 0 | Regression target |
| Approx. in-memory size (pandas, `deep=True`) | ~25.6 MB | | |

### Standardized (`data/processed/standardized/davis.parquet`)

Same five columns; column order may differ. **`drug_smiles`** is RDKit-canonicalized \(largest fragment\).

**`affinity_label` summary (standardized):** minimum ≈ `0.016`, maximum `10000.0`, median `10000.0`, mean ≈ `7558.1`. The large mass at the maximum matches **weak-binding / censored-style** coding typical of published Davis setups—keep this in mind before mapping values to literal Kd in nM.

---

## Splits (`data/processed/splits/davis/`)

| Strategy | Train | Val | Test | Total rows |
|----------|------:|----:|-----:|-----------:|
| `random/` | 20,617 | 2,577 | 2,578 | 25,772 |
| `drug_cold/` | 20,466 | 2,653 | 2,653 | 25,772 |
| `target_cold/` | 20,604 | 2,584 | 2,584 | 25,772 |

Drug-cold and target-cold splits enforce **no overlapping scaffolds** or **no overlapping target sequences**, respectively, between train / val / test.

---

## Reproducing exports

```bash
python src/data_ingestion/fetch_datasets.py
python src/data_preprocessing/standardize.py davis
python src/data_preprocessing/generate_splits.py --dataset davis
```
