#!/usr/bin/env python3
"""
Load standardized Parquet benchmarks, produce random / scaffold / target-cold splits.
Writes ``train.parquet``, ``val.parquet``, ``test.parquet`` per strategy under
``data/processed/splits/{dataset}/{random|drug_cold|target_cold}/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_PREPROC = Path(__file__).resolve().parent
if str(_PREPROC) not in sys.path:
    sys.path.insert(0, str(_PREPROC))

from splitters import (
    DrugColdScaffoldSplitter,
    RandomSplitter,
    TargetColdSplitter,
    column_scaffold_set,
    column_target_set,
    intersect_sizes,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def standardized_dir() -> Path:
    return repo_root() / "data" / "processed" / "standardized"


def splits_base_dir() -> Path:
    return repo_root() / "data" / "processed" / "splits"


def log_fold_sizes(name: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    logging.info(
        "[%s] rows train=%s val=%s test=%s (total=%s)",
        name,
        len(train),
        len(val),
        len(test),
        len(train) + len(val) + len(test),
    )


def verify_random_note(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    ts_tr = column_target_set(train)
    ts_te = column_target_set(test)
    sc_tr = column_scaffold_set(train)
    sc_te = column_scaffold_set(test)
    logging.info(
        "Random split note: scaffold train∩test=%s targets train∩test=%s "
        "(overlap expected for this baseline)",
        intersect_sizes(sc_tr, sc_te),
        intersect_sizes(ts_tr, ts_te),
    )


def verify_drug_cold(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    st, sv, ste = column_scaffold_set(train), column_scaffold_set(val), column_scaffold_set(test)
    a, b, c = intersect_sizes(st, sv), intersect_sizes(st, ste), intersect_sizes(sv, ste)
    logging.info(
        "Verified drug-cold (scaffold): train∩val=%s train∩test=%s val∩test=%s",
        a,
        b,
        c,
    )
    if b != 0 or a != 0 or c != 0:
        logging.warning(
            "Expected zero scaffold overlap between folds; "
            "train∩val=%s train∩test=%s val∩test=%s",
            a,
            b,
            c,
        )


def verify_target_cold(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    tt, tv, te = column_target_set(train), column_target_set(val), column_target_set(test)
    a, b, c = intersect_sizes(tt, tv), intersect_sizes(tt, te), intersect_sizes(tv, te)
    logging.info(
        "Verified target-cold: train∩val=%s train∩test=%s val∩test=%s",
        a,
        b,
        c,
    )
    if b != 0 or a != 0 or c != 0:
        logging.warning(
            "Expected zero target overlap between folds; "
            "train∩val=%s train∩test=%s val∩test=%s",
            a,
            b,
            c,
        )


def write_split_dir(out_dir: Path, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    logging.info("Wrote %s", out_dir.resolve())


def run_for_dataset(stem: str, df: pd.DataFrame, base: Path) -> None:
    strategies = [
        ("random", RandomSplitter(), verify_random_note),
        ("drug_cold", DrugColdScaffoldSplitter(), verify_drug_cold),
        ("target_cold", TargetColdSplitter(), verify_target_cold),
    ]
    for subdir, splitter, verify in strategies:
        tr, va, te = splitter.split(df)
        key = f"{stem}/{subdir}"
        log_fold_sizes(key, tr, va, te)
        verify(tr, va, te)
        write_split_dir(base / stem / subdir, tr, va, te)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate train/val/test parquet splits from standardized data.")
    p.add_argument(
        "--dataset",
        nargs="*",
        default=None,
        help="Standardized parquet stem(s) without .parquet (default: all files in standardized dir)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    std = standardized_dir()
    base = splits_base_dir()

    if args.dataset:
        paths = [std / f"{s.removesuffix('.parquet')}.parquet" for s in args.dataset]
    else:
        paths = sorted(std.glob("*.parquet"))

    if not paths:
        raise SystemExit(f"No standardized parquet files under {std}. Run standardize.py first.")

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        stem = path.stem
        logging.info("Loading standardized %s", path)
        df = pd.read_parquet(path)
        run_for_dataset(stem, df, base)

    logging.info("Split generation complete (%s dataset(s)).", len(paths))


if __name__ == "__main__":
    main()
