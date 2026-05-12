# Phase 4 Experiment Results

Dataset: `davis`  
Timestamp: `2026-05-05 17:05:38 UTC`  
Seed: `42`  
Device: `cuda`

| Split | Model | MSE | Pearson r | CI |
|------|------|----:|----------:|---:|
| `random` | XGBoost | 7730278.500000 | 0.717201 | 0.868523 |
| `random` | Random Forest | 10066469.408728 | 0.607415 | 0.817390 |
| `drug_cold` | XGBoost | 12666642.000000 | 0.339042 | 0.675695 |
| `drug_cold` | Random Forest | 11594657.552453 | 0.496725 | 0.795102 |
| `target_cold` | XGBoost | 11461091.000000 | 0.560446 | 0.785974 |
| `target_cold` | Random Forest | 11403880.392310 | 0.564675 | 0.785488 |
