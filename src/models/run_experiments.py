#!/usr/bin/env python3
"""Run Phase 3 XGBoost baseline experiments across split strategies."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_MODELS = Path(__file__).resolve().parent
if str(_MODELS) not in sys.path:
    sys.path.insert(0, str(_MODELS))

from features import build_matrix
from train_xgboost import RANDOM_SEED, EvalMetrics, train_and_evaluate


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_split_frames(base: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_p = base / "train.parquet"
    val_p = base / "val.parquet"
    test_p = base / "test.parquet"
    for p in (train_p, val_p, test_p):
        if not p.is_file():
            raise FileNotFoundError(f"Missing split file: {p}. Run generate_splits.py first.")
    return pd.read_parquet(train_p), pd.read_parquet(val_p), pd.read_parquet(test_p)


def run_one_split(split_dir: Path, split_name: str) -> EvalMetrics:
    train_df, val_df, test_df = load_split_frames(split_dir)
    logging.info(
        "[%s] rows train=%s val=%s test=%s",
        split_name,
        len(train_df),
        len(val_df),
        len(test_df),
    )
    tr = build_matrix(train_df)
    va = build_matrix(val_df)
    te = build_matrix(test_df)
    metrics = train_and_evaluate(
        x_train=tr.x,
        y_train=tr.y,
        x_val=va.x,
        y_val=va.y,
        x_test=te.x,
        y_test=te.y,
    )
    logging.info(
        "[%s] MSE=%.6f Pearson_r=%.6f CI=%.6f best_iteration=%s",
        split_name,
        metrics.mse,
        metrics.pearson_r,
        metrics.ci,
        metrics.best_iteration,
    )
    return metrics


def write_results_md(
    *,
    out_path: Path,
    dataset: str,
    results: dict[str, EvalMetrics],
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Phase 3 Experiment Results",
        "",
        f"Dataset: `{dataset}`  ",
        f"Timestamp: `{timestamp}`  ",
        f"Seed: `{RANDOM_SEED}`",
        "",
        "| Split | MSE | Pearson r | CI |",
        "|------|----:|----------:|---:|",
    ]
    for split_name in ("random", "drug_cold", "target_cold"):
        m = results[split_name]
        lines.append(f"| `{split_name}` | {m.mse:.6f} | {m.pearson_r:.6f} | {m.ci:.6f} |")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run XGBoost baseline across dataset split strategies.")
    p.add_argument("--dataset", default="davis", help="Dataset stem under data/processed/splits (default: davis)")
    p.add_argument(
        "--repo-root",
        default=None,
        help="Optional repository root override",
    )
    p.add_argument(
        "--results-path",
        default=None,
        help="Optional markdown output path (default: docs/experiment_results.md)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    root = Path(args.repo_root) if args.repo_root else repo_root()
    dataset = str(args.dataset).strip().lower()
    splits_root = root / "data" / "processed" / "splits" / dataset
    if not splits_root.is_dir():
        raise SystemExit(f"Missing split directory: {splits_root}. Run generate_splits.py first.")

    results: dict[str, EvalMetrics] = {}
    for split_name in ("random", "drug_cold", "target_cold"):
        results[split_name] = run_one_split(splits_root / split_name, split_name)

    out_path = Path(args.results_path) if args.results_path else (root / "docs" / "experiment_results.md")
    write_results_md(out_path=out_path, dataset=dataset, results=results)
    logging.info("Wrote %s", out_path.resolve())


if __name__ == "__main__":
    main()
