#!/usr/bin/env python3
"""Run Phase 4 experiments across split strategies and multiple models.

Trains both XGBoost and Random Forest on each of the random / drug-cold /
target-cold splits and writes:

- ``docs/experiment_results.md``  — human-readable wide table.
- ``docs/experiment_results.csv`` — long-form rows (one per split x model)
  consumed by ``src/visualization/plot_metrics.py``.
- ``models/{dataset}/xgboost_{split}.json`` — persisted XGBoost models for
  SHAP explanation downstream.
- ``models/{dataset}/feature_meta.json`` — feature schema (Morgan + AAC
  layout) so the explainer can build feature names without re-deriving them.
- ``logs/phase4_{dataset}_...log`` — full INFO log of the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

_MODELS = Path(__file__).resolve().parent
_SRC_ROOT = _MODELS.parent
for _path in (_MODELS, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from features import AAC_ALPHABET, AAC_DIM, MORGAN_NBITS, build_matrix  # noqa: E402
from train_rf import train_and_evaluate as train_rf  # noqa: E402
from train_xgboost import RANDOM_SEED, EvalMetrics, train_and_evaluate as train_xgb  # noqa: E402
from utils.device import resolve_device  # noqa: E402


SPLIT_NAMES: tuple[str, ...] = ("random", "drug_cold", "target_cold")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    train_fn: Callable[..., EvalMetrics]
    persists: bool


_AVAILABLE_MODELS: dict[str, ModelSpec] = {
    "xgboost": ModelSpec(
        name="xgboost",
        label="XGBoost",
        train_fn=train_xgb,
        persists=True,
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        label="Random Forest",
        train_fn=train_rf,
        persists=False,
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def configure_logging(*, root: Path, dataset: str, device_requested: str, device_resolved: str) -> Path:
    """Set up dual console + file logging under ``<repo>/logs/``."""
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    log_path = log_dir / f"phase4_{dataset}_req-{device_requested}_use-{device_resolved}_{ts}.log"

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(levelname)s %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return log_path


def load_split_frames(base: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_p = base / "train.parquet"
    val_p = base / "val.parquet"
    test_p = base / "test.parquet"
    for p in (train_p, val_p, test_p):
        if not p.is_file():
            raise FileNotFoundError(f"Missing split file: {p}. Run generate_splits.py first.")
    return pd.read_parquet(train_p), pd.read_parquet(val_p), pd.read_parquet(test_p)


def parse_models_arg(raw: str) -> list[ModelSpec]:
    """Parse a comma-separated ``--models`` value into ordered specs."""
    seen: set[str] = set()
    specs: list[ModelSpec] = []
    for token in raw.split(","):
        key = token.strip().lower()
        if not key or key in seen:
            continue
        if key not in _AVAILABLE_MODELS:
            raise SystemExit(
                f"Unknown model '{key}'. Available: {sorted(_AVAILABLE_MODELS)}"
            )
        specs.append(_AVAILABLE_MODELS[key])
        seen.add(key)
    if not specs:
        raise SystemExit("--models produced an empty list")
    return specs


def write_feature_meta(out_path: Path) -> None:
    """Persist the Morgan+AAC feature layout next to saved XGBoost models."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "morgan_nbits": MORGAN_NBITS,
        "aac_alphabet": AAC_ALPHABET,
        "aac_dim": AAC_DIM,
        "morgan_offset": 0,
        "aac_offset": MORGAN_NBITS,
        "total_dim": MORGAN_NBITS + AAC_DIM,
    }
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_one_split(
    split_dir: Path,
    split_name: str,
    *,
    models: list[ModelSpec],
    device: str,
    models_out_dir: Path,
) -> dict[str, EvalMetrics]:
    """Train all requested models on a single split and return their metrics."""
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

    out: dict[str, EvalMetrics] = {}
    for spec in models:
        kwargs: dict[str, object] = {
            "x_train": tr.x,
            "y_train": tr.y,
            "x_val": va.x,
            "y_val": va.y,
            "x_test": te.x,
            "y_test": te.y,
            "device": device,
        }
        if spec.persists:
            kwargs["save_model_path"] = models_out_dir / f"{spec.name}_{split_name}.json"

        logging.info("[%s] Training model: %s", split_name, spec.label)
        metrics = spec.train_fn(**kwargs)
        out[spec.name] = metrics
        logging.info(
            "[%s][%s] MSE=%.6f Pearson_r=%.6f CI=%.6f best_iteration=%s",
            split_name,
            spec.label,
            metrics.mse,
            metrics.pearson_r,
            metrics.ci,
            metrics.best_iteration,
        )

    return out


def write_results_md(
    *,
    out_path: Path,
    dataset: str,
    results: dict[str, dict[str, EvalMetrics]],
    models: list[ModelSpec],
    device: str,
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Phase 4 Experiment Results",
        "",
        f"Dataset: `{dataset}`  ",
        f"Timestamp: `{timestamp}`  ",
        f"Seed: `{RANDOM_SEED}`  ",
        f"Device: `{device}`",
        "",
        "| Split | Model | MSE | Pearson r | CI |",
        "|------|------|----:|----------:|---:|",
    ]
    for split_name in SPLIT_NAMES:
        for spec in models:
            m = results[split_name][spec.name]
            lines.append(
                f"| `{split_name}` | {spec.label} | {m.mse:.6f} | "
                f"{m.pearson_r:.6f} | {m.ci:.6f} |"
            )
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_results_csv(
    *,
    out_path: Path,
    dataset: str,
    results: dict[str, dict[str, EvalMetrics]],
    models: list[ModelSpec],
    device: str,
) -> None:
    """Write one row per (split, model) for downstream visualization."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["dataset", "device", "split", "model", "model_label", "mse", "pearson_r", "ci", "best_iteration"]
        )
        for split_name in SPLIT_NAMES:
            for spec in models:
                m = results[split_name][spec.name]
                writer.writerow(
                    [
                        dataset,
                        device,
                        split_name,
                        spec.name,
                        spec.label,
                        f"{m.mse:.6f}",
                        f"{m.pearson_r:.6f}",
                        f"{m.ci:.6f}",
                        m.best_iteration,
                    ]
                )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Phase 4 baselines (XGBoost + Random Forest) across split strategies.",
    )
    p.add_argument("--dataset", default="davis", help="Dataset stem under data/processed/splits (default: davis)")
    p.add_argument("--repo-root", default=None, help="Optional repository root override")
    p.add_argument(
        "--results-path",
        default=None,
        help="Optional markdown output path (default: docs/experiment_results.md)",
    )
    p.add_argument(
        "--results-csv",
        default=None,
        help="Optional CSV output path (default: docs/experiment_results.csv)",
    )
    p.add_argument(
        "--models",
        default="xgboost,random_forest",
        help=(
            "Comma-separated list of models to train. "
            f"Available: {sorted(_AVAILABLE_MODELS)}. "
            "Default: xgboost,random_forest."
        ),
    )
    p.add_argument(
        "--device",
        choices=("auto", "gpu", "cpu"),
        default="auto",
        help=(
            "Compute device for XGBoost (default: auto). "
            "'auto' picks CUDA when available (e.g. RTX 4060 Ti) and falls back to CPU. "
            "'gpu' fails fast if no CUDA device is detected. 'cpu' forces CPU. "
            "Random Forest is CPU-only regardless."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root) if args.repo_root else repo_root()
    dataset = str(args.dataset).strip().lower()
    splits_root = root / "data" / "processed" / "splits" / dataset
    if not splits_root.is_dir():
        raise SystemExit(f"Missing split directory: {splits_root}. Run generate_splits.py first.")

    resolved_device = resolve_device(args.device)
    log_path = configure_logging(
        root=root,
        dataset=dataset,
        device_requested=args.device,
        device_resolved=resolved_device,
    )
    logging.info("Logging to %s", log_path.resolve())

    models = parse_models_arg(args.models)
    logging.info("Models: %s", ", ".join(spec.label for spec in models))

    models_out_dir = root / "models" / dataset
    models_out_dir.mkdir(parents=True, exist_ok=True)
    if any(spec.persists for spec in models):
        write_feature_meta(models_out_dir / "feature_meta.json")

    results: dict[str, dict[str, EvalMetrics]] = {}
    for split_name in SPLIT_NAMES:
        results[split_name] = run_one_split(
            splits_root / split_name,
            split_name,
            models=models,
            device=args.device,
            models_out_dir=models_out_dir,
        )

    md_path = Path(args.results_path) if args.results_path else (root / "docs" / "experiment_results.md")
    csv_path = Path(args.results_csv) if args.results_csv else (root / "docs" / "experiment_results.csv")
    write_results_md(
        out_path=md_path,
        dataset=dataset,
        results=results,
        models=models,
        device=resolved_device,
    )
    write_results_csv(
        out_path=csv_path,
        dataset=dataset,
        results=results,
        models=models,
        device=resolved_device,
    )
    logging.info("Wrote %s", md_path.resolve())
    logging.info("Wrote %s", csv_path.resolve())


if __name__ == "__main__":
    main()
