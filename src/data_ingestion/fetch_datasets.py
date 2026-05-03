#!/usr/bin/env python3
"""
Download Davis, KIBA, and BindingDB (Kd) DTI benchmarks via PyTDC and save tabular exports.

Designed to run inside the conda environment ``dti_research`` after installing
packages from ``requirements.txt``. Run from the repository root::

    python src/data_ingestion/fetch_datasets.py
    python src/data_ingestion/fetch_datasets.py --csv
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.errors import ParserError

DEFAULT_DATASETS: tuple[tuple[str, str], ...] = (
    ("DAVIS", "davis"),
    ("KIBA", "kiba"),
    ("BindingDB_Kd", "bindingdb_kd"),
)


def repo_root() -> Path:
    """Return repository root (parent of ``src``)."""
    return Path(__file__).resolve().parents[2]


def raw_dir() -> Path:
    """Directory for raw parquet/CSV outputs."""
    return repo_root() / "data" / "raw"


def tdc_cache_dir() -> Path:
    """Resolved PyTDC ``path=`` directory (downloads live here; separate from ``data/raw``)."""
    d = repo_root() / "data" / "tdc_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _unlink_with_retry(path: Path, *, max_attempts: int = 15) -> bool:
    """
    Delete a single file; retry on transient Windows locks (e.g. another download).

    Returns
    -------
    bool
        True if the path no longer exists after the call.
    """
    if not path.is_file():
        return True
    for attempt in range(max_attempts):
        try:
            path.unlink()
            logging.warning("Removed cached TDC file (will re-download): %s", path)
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            gc.collect()
            time.sleep(min(4.0, 0.2 * (2**min(attempt, 4))))
    logging.error(
        "Could not delete %s after %s attempts — close other ingestion processes and retry.",
        path,
        max_attempts,
    )
    raise PermissionError(
        "File is locked (often caused by multiple fetch scripts). "
        "Close other Python/TDC downloads and retry.",
    )


def purge_tdc_cached_files(tdc_query_name: str, cache_dir: Path) -> None:
    """
    Delete local PyTDC artifacts for a dataset so the next open re-downloads.

    TDC normalizes names to lowercase (e.g. ``KIBA`` -> ``kiba``); files are
    ``{name}.{ext}`` per ``tdc.metadata.name2type``.
    """
    from tdc.metadata import name2type

    key = tdc_query_name.lower()
    if key.startswith("tdc."):
        key = key[4:]
    candidates: list[Path] = []
    ext = name2type.get(key)
    if ext:
        candidates.append(cache_dir / f"{key}.{ext}")
    candidates.extend(cache_dir.glob(f"{key}.*"))

    seen: set[Path] = set()
    for path in candidates:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _unlink_with_retry(path)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(message)s",
    )


def load_dti_dataframe(
    tdc_name: str,
    *,
    max_load_attempts: int = 3,
) -> pd.DataFrame:
    """
    Load the full DTI table for a TDC dataset name (e.g. ``DAVIS``).

    Uses a fixed cache under ``data/tdc_cache/``. If parsing fails (e.g.
    truncated download), cached files for that dataset are removed and loading
    is retried.

    Parameters
    ----------
    tdc_name:
        PyTDC ``DTI`` task name.

    Returns
    -------
    pd.DataFrame
        Raw frame as returned by ``get_data()``.
    """
    from tdc.multi_pred import DTI

    cache_path = tdc_cache_dir()
    cache = str(cache_path)

    for attempt in range(max_load_attempts):
        try:
            data = DTI(name=tdc_name, path=cache)
            return data.get_data()
        except ParserError as exc:
            if attempt >= max_load_attempts - 1:
                raise
            logging.warning(
                "[%s] Parse failed (attempt %s/%s); corrupt cache likely. "
                "Purging and retrying: %s",
                tdc_name,
                attempt + 1,
                max_load_attempts,
                exc,
            )
            gc.collect()
            time.sleep(0.5)
            purge_tdc_cached_files(tdc_name, cache_path)

    raise RuntimeError("internal error: load retry loop exited without return")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map TDC DTI columns to a stable schema:
    ``drug_smiles``, ``target_sequence``, ``affinity_label``.

    Any additional columns (e.g. identifiers) are preserved under their
    original names after lower-specific renames for the three core fields.
    """
    # Build case-insensitive lookup for the three canonical TDC names.
    col_by_lower = {c.lower().strip(): c for c in df.columns}
    renames: dict[str, str] = {}
    if "drug" in col_by_lower:
        renames[col_by_lower["drug"]] = "drug_smiles"
    if "target" in col_by_lower:
        renames[col_by_lower["target"]] = "target_sequence"
    if "y" in col_by_lower:
        renames[col_by_lower["y"]] = "affinity_label"

    if len(renames) < 3:
        logging.warning(
            "Expected Drug/Target/Y-style columns; got %s. Renaming may be incomplete.",
            list(df.columns),
        )

    out = df.rename(columns=renames, copy=True)
    return out


def summarize_missing(logger: logging.Logger, df: pd.DataFrame, label: str) -> None:
    """Log per-column counts of missing values."""
    counts = df.isna().sum()
    bad = counts[counts > 0]
    if bad.empty:
        logger.info("[%s] No missing values.", label)
    else:
        logger.info("[%s] Missing values per column:\n%s", label, bad.to_string())


def save_raw(
    df: pd.DataFrame,
    stem: str,
    out_dir: Path,
    *,
    write_csv: bool,
) -> list[Path]:
    """
    Write ``df`` to Parquet under ``out_dir``, optionally also CSV.

    Returns
    -------
    list of Path
        Paths that were written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    parquet_path = out_dir / f"{stem}.parquet"
    df.to_parquet(parquet_path, index=False)
    written.append(parquet_path)
    if write_csv:
        csv_path = out_dir / f"{stem}.csv"
        df.to_csv(csv_path, index=False)
        written.append(csv_path)
    return written


def process_one_dataset(
    tdc_name: str,
    file_stem: str,
    out_dir: Path,
    *,
    write_csv: bool,
    logger: logging.Logger,
) -> None:
    """
    Fetch one dataset via TDC, normalize columns, summarize quality, save files.

    Raises
    ------
    Exception
        On download or I/O failures (caught by the caller for per-dataset isolation).
    """
    logger.info("Starting fetch: %s", tdc_name)
    df = load_dti_dataframe(tdc_name)
    df = normalize_columns(df)

    logger.info(
        "[%s] Shape rows=%s cols=%s | columns=%s",
        tdc_name,
        len(df),
        len(df.columns),
        list(df.columns),
    )
    summarize_missing(logger, df, tdc_name)

    paths = save_raw(df, file_stem, out_dir, write_csv=write_csv)
    for p in paths:
        logger.info("[%s] Wrote %s", tdc_name, p.resolve())


def fetch_all(
    specs: Iterable[tuple[str, str]],
    out_dir: Path,
    *,
    write_csv: bool,
) -> int:
    """
    Fetch every ``(tdc_name, file_stem)`` pair; continue on failures.

    Returns
    -------
    int
        0 if all succeeded, 1 otherwise.
    """
    logger = logging.getLogger(__name__)
    failures: list[str] = []

    for tdc_name, file_stem in specs:
        try:
            process_one_dataset(
                tdc_name,
                file_stem,
                out_dir,
                write_csv=write_csv,
                logger=logger,
            )
        except Exception as exc:
            failures.append(tdc_name)
            logger.exception("[%s] Failed: %s", tdc_name, exc)

    if failures:
        logger.error(
            "Completed with failures for: %s",
            ", ".join(failures),
        )
        return 1
    logger.info("All datasets fetched successfully.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download DTI benchmarks (Davis, KIBA, BindingDB Kd) via PyTDC.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write CSV alongside Parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory (default: <repo>/data/raw)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    out = args.out_dir if args.out_dir is not None else raw_dir()

    logger = logging.getLogger(__name__)
    logger.debug("Repo root %s", repo_root())
    logger.debug("Raw output dir %s", out.resolve())

    code = fetch_all(DEFAULT_DATASETS, out, write_csv=args.csv)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
