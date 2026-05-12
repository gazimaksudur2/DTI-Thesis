#!/usr/bin/env python3
"""
Render publication-ready bar charts comparing models across splits.

Reads ``docs/experiment_results.csv`` (the long-form sidecar written by
``src/models/run_experiments.py``) and emits two PNGs:

- ``docs/figures/mse_comparison.png``  (lower is better)
- ``docs/figures/ci_comparison.png``   (higher is better)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SPLIT_ORDER: tuple[str, ...] = ("random", "drug_cold", "target_cold")
SPLIT_DISPLAY: dict[str, str] = {
    "random": "Random",
    "drug_cold": "Drug-cold",
    "target_cold": "Target-cold",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_results(csv_path: Path, dataset: str | None) -> pd.DataFrame:
    if not csv_path.is_file():
        raise SystemExit(
            f"Results CSV not found at {csv_path}. "
            "Run `python src/models/run_experiments.py` first."
        )
    df = pd.read_csv(csv_path)
    expected = {"split", "model_label", "mse", "ci"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"Results CSV missing columns: {sorted(missing)}")
    if dataset is not None and "dataset" in df.columns:
        df = df[df["dataset"].astype(str).str.lower() == dataset.lower()].copy()
        if df.empty:
            raise SystemExit(f"No rows for dataset='{dataset}' in {csv_path}.")
    df["split"] = df["split"].astype(str)
    df["split_display"] = df["split"].map(SPLIT_DISPLAY).fillna(df["split"])
    return df


def _grouped_bar(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    y_label: str,
    out_path: Path,
    y_lim: tuple[float, float] | None = None,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")

    present = [s for s in SPLIT_ORDER if s in df["split"].unique().tolist()]
    order = [SPLIT_DISPLAY.get(s, s) for s in present]
    hue_order = sorted(df["model_label"].unique().tolist())

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(
        data=df,
        x="split_display",
        y=metric,
        hue="model_label",
        order=order,
        hue_order=hue_order,
        ax=ax,
    )
    ax.set_xlabel("Evaluation split")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.legend(title="Model", loc="best", frameon=True)
    sns.despine(ax=ax)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", out_path.resolve())


def render_all(df: pd.DataFrame, out_dir: Path) -> None:
    _grouped_bar(
        df,
        metric="mse",
        title="MSE by split (lower is better)",
        y_label="Mean squared error",
        out_path=out_dir / "mse_comparison.png",
    )

    ci_min = max(0.0, float(df["ci"].min()) - 0.05)
    ci_max = min(1.0, float(df["ci"].max()) + 0.05)
    if ci_max - ci_min < 0.1:
        ci_max = min(1.0, ci_min + 0.1)
    _grouped_bar(
        df,
        metric="ci",
        title="Concordance Index by split (higher is better)",
        y_label="Concordance Index",
        out_path=out_dir / "ci_comparison.png",
        y_lim=(ci_min, ci_max),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Phase 4 metric bar charts.")
    p.add_argument(
        "--results-csv",
        default=None,
        help="Path to experiment results CSV (default: docs/experiment_results.csv)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for PNGs (default: docs/figures)",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset filter (matches the 'dataset' column).",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    root = repo_root()
    csv_path = Path(args.results_csv) if args.results_csv else (root / "docs" / "experiment_results.csv")
    out_dir = Path(args.out_dir) if args.out_dir else (root / "docs" / "figures")
    df = load_results(csv_path, args.dataset)
    render_all(df, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
