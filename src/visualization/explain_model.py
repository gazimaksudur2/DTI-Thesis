#!/usr/bin/env python3
"""
SHAP explanation for the trained XGBoost DTI model.

Loads a model saved by ``src/models/run_experiments.py``
(``models/{dataset}/xgboost_{split}.json``), reads the matching test split,
builds Morgan + AAC features for a deterministic random subset, runs
``shap.TreeExplainer``, and saves a summary plot to
``docs/figures/shap_summary.png``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_SRC_ROOT / "models", _SRC_ROOT):
    s = str(_path)
    if s not in sys.path:
        sys.path.insert(0, s)

from features import AAC_ALPHABET, build_matrix  # noqa: E402

DEFAULT_N_SAMPLES = 500


def repo_root() -> Path:
    return _REPO_ROOT


def build_feature_names(meta: dict) -> list[str]:
    """Construct human-readable feature names for the fused Morgan+AAC vector."""
    morgan_nbits = int(meta["morgan_nbits"])
    aac_alphabet = str(meta.get("aac_alphabet", AAC_ALPHABET))
    names = [f"morgan_b{i}" for i in range(morgan_nbits)]
    names.extend(f"aac_{aa}" for aa in aac_alphabet)
    names.append("aac_unknown")
    return names


def load_test_subset(
    test_parquet: Path,
    *,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    if not test_parquet.is_file():
        raise SystemExit(
            f"Missing test split: {test_parquet}. Run generate_splits.py first."
        )
    df = pd.read_parquet(test_parquet)
    if len(df) == 0:
        raise SystemExit(f"Test split is empty: {test_parquet}")
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SHAP summary plot for trained XGBoost model.")
    p.add_argument("--dataset", default="davis", help="Dataset stem under data/processed/splits (default: davis)")
    p.add_argument(
        "--split",
        default="random",
        choices=("random", "drug_cold", "target_cold"),
        help="Which split's saved model to explain (default: random)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"How many test rows to explain (default: {DEFAULT_N_SAMPLES})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the test subset (default: 42)",
    )
    p.add_argument(
        "--max-display",
        type=int,
        default=20,
        help="Number of top features to show on the SHAP summary plot (default: 20)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: docs/figures/shap_summary.png)",
    )
    p.add_argument(
        "--device",
        choices=("auto", "gpu", "cpu"),
        default="cpu",
        help="Device for SHAP/inference (default: cpu; SHAP TreeExplainer runs on CPU).",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    root = repo_root()
    dataset = str(args.dataset).strip().lower()

    model_path = root / "models" / dataset / f"xgboost_{args.split}.json"
    meta_path = root / "models" / dataset / "feature_meta.json"
    test_path = root / "data" / "processed" / "splits" / dataset / args.split / "test.parquet"
    out_path = Path(args.out) if args.out else (root / "docs" / "figures" / "shap_summary.png")

    if not model_path.is_file():
        raise SystemExit(
            f"Saved model not found: {model_path}. "
            "Run `python src/models/run_experiments.py` first to train and persist it."
        )
    if not meta_path.is_file():
        raise SystemExit(
            f"Feature metadata missing: {meta_path}. Re-run the experiment to regenerate."
        )

    logging.info("Loading model from %s", model_path)
    model = XGBRegressor()
    model.load_model(str(model_path))

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_names = build_feature_names(meta)
    if len(feature_names) != int(meta["total_dim"]):
        raise SystemExit(
            f"Feature-name count mismatch: built {len(feature_names)}, meta says {meta['total_dim']}"
        )

    logging.info(
        "Loading test subset (n=%s) from %s",
        args.n_samples,
        test_path,
    )
    test_df = load_test_subset(test_path, n_samples=args.n_samples, seed=args.seed)
    feats = build_matrix(test_df)
    x = feats.x.astype(np.float32, copy=False)

    logging.info("Computing SHAP values via TreeExplainer (n=%s, dim=%s)", x.shape[0], x.shape[1])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)

    plt.figure()
    shap.summary_plot(
        shap_values,
        x,
        feature_names=feature_names,
        max_display=args.max_display,
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(
        f"SHAP summary - XGBoost ({dataset}, {args.split} split, n={x.shape[0]})",
        y=1.02,
        fontsize=11,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", out_path.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
