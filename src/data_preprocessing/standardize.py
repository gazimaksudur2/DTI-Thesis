#!/usr/bin/env python3
"""
Canonicalize SMILES with RDKit (largest fragment), dedupe (median affinity), write Parquet.

Reads ``data/raw/*.parquet`` into ``data/processed/standardized/*.parquet``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def raw_parquet_dir() -> Path:
    return repo_root() / "data" / "raw"


def standardized_dir() -> Path:
    return repo_root() / "data" / "processed" / "standardized"


def suppress_rdkit_logging() -> None:
    RDLogger.DisableLog("rdApp.*")


def largest_fragment_canonical_smiles(smiles: str | float | None) -> str | None:
    """
    Parse SMILES, keep largest heavy-fragment by atom count, return canonical SMILES.
    Returns None if parsing fails or result is unusable.
    """
    if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return None
    parent = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(parent)
    except Chem.MolSanitizeException:
        return None
    canon = Chem.MolToSmiles(parent, canonical=True)
    return canon if canon else None


def standardize_frame(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Return cleaned dataframe with canonical ``drug_smiles`` and aggregated duplicates."""
    if "drug_smiles" not in df.columns or "target_sequence" not in df.columns:
        raise ValueError(f"{label}: expected columns drug_smiles, target_sequence")

    suppress_rdkit_logging()

    base = df.copy()
    base["target_sequence"] = base["target_sequence"].astype(str)
    missing_tgt = base["target_sequence"].isin(("", "nan", "None"))
    base = base.loc[~missing_tgt].reset_index(drop=True)

    logging.info("[%s] Applying RDKit canonicalization (largest fragment)...", label)
    base["drug_smiles"] = base["drug_smiles"].apply(largest_fragment_canonical_smiles)
    before_drop = len(base)
    base = base.loc[base["drug_smiles"].notna()].reset_index(drop=True)
    logging.info("[%s] Dropped %s rows with invalid / empty SMILES.", label, before_drop - len(base))

    agg_map: dict[str, str] = {"affinity_label": "median"}
    for col in ("Drug_ID", "Target_ID"):
        if col in base.columns:
            agg_map[col] = "first"

    logging.info("[%s] Deduplicating by (drug_smiles, target_sequence); median affinity.", label)
    grouped = (
        base.groupby(["drug_smiles", "target_sequence"], as_index=False, sort=False)
        .agg(agg_map)
    )
    logging.info(
        "[%s] Final shape rows=%s cols=%s (from %s raw rows).",
        label,
        len(grouped),
        len(grouped.columns),
        len(df),
    )
    return grouped


def process_parquet(path: Path, out_dir: Path) -> Path:
    stem = path.stem
    logging.info("Loading %s", path)
    df = pd.read_parquet(path)
    clean = standardize_frame(df, label=stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.parquet"
    clean.to_parquet(out_path, index=False)
    logging.info("Wrote %s", out_path.resolve())
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standardize raw DTI parquet files with RDKit.")
    p.add_argument(
        "files",
        nargs="*",
        help="Specific raw parquet stems or paths (default: all data/raw/*.parquet)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    raw_dir = raw_parquet_dir()
    out_dir = standardized_dir()

    if args.files:
        targets: list[Path] = []
        for f in args.files:
            p = Path(f)
            if not p.suffix:
                p = raw_dir / f"{p.name}.parquet"
            elif not p.is_absolute():
                p = raw_dir / p.name if p.parent == Path(".") else p
            targets.append(p)
    else:
        targets = sorted(raw_dir.glob("*.parquet"))

    if not targets:
        raise SystemExit(f"No parquet inputs found under {raw_dir}")

    for path in targets:
        if not path.is_file():
            raise FileNotFoundError(path)
        process_parquet(path, out_dir)

    logging.info("Standardization complete (%s file(s)).", len(targets))


if __name__ == "__main__":
    main()
