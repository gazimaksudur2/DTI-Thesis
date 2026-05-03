"""
Train / validation / test split strategies for DTI benchmarking.

Splits use an 80% / 10% / 10% partition (by row for random split, by scaffold or
target entity for cold splits). ``RANDOM_SEED`` fixes all shuffles for reproducibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


def suppress_rdkit_logging() -> None:
    RDLogger.DisableLog("rdApp.*")


def scaffold_smiles_from_drug(canonical_smiles: str) -> str:
    """
    Bemis-Murcko scaffold canonical SMILES; fall back to the input SMILES when empty.
    """
    suppress_rdkit_logging()
    mol = Chem.MolFromSmiles(str(canonical_smiles))
    if mol is None:
        return str(canonical_smiles)
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if core is None:
        return str(canonical_smiles)
    smi = Chem.MolToSmiles(core, canonical=True)
    return smi if smi else str(canonical_smiles)


def column_scaffold_set(df: pd.DataFrame, smiles_col: str = "drug_smiles") -> Set[str]:
    """Scaffold set for molecules in ``df``."""
    suppress_rdkit_logging()
    scaffolds: Set[str] = set()
    for smi in df[smiles_col].astype(str).unique():
        scaffolds.add(scaffold_smiles_from_drug(smi))
    return scaffolds


def column_target_set(df: pd.DataFrame, target_col: str = "target_sequence") -> Set[str]:
    """Unique targets (exact sequence strings)."""
    return set(df[target_col].astype(str).unique())


def _partition_unique_entities(entities: list[str]) -> tuple[set[str], set[str], set[str]]:
    """
    Assign each distinct entity to exactly one of train / val / test (approximately 80/10/10).

    Uses two stratified sklearn splits on index arrays for reproducibility.
    """
    n = len(entities)
    if n == 0:
        return set(), set(), set()
    if n == 1:
        return {entities[0]}, set(), set()
    ent_arr = np.array(entities)
    idx = np.arange(n)
    train_idx, temp_idx = train_test_split(
        idx,
        train_size=0.8,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    if len(temp_idx) <= 1:
        val_idx = np.array([], dtype=int)
        test_idx = temp_idx if len(temp_idx) else np.array([], dtype=int)
    else:
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=0.5,
            random_state=RANDOM_SEED + 1,
            shuffle=True,
        )
    train_s = set(ent_arr[train_idx].tolist())
    val_s = set(ent_arr[val_idx].tolist())
    test_s = set(ent_arr[test_idx].tolist())
    return train_s, val_s, test_s


class BaseSplitter(ABC):
    """Produce ``train, val, test`` tables from a standardized interaction frame."""

    @abstractmethod
    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ...


class RandomSplitter(BaseSplitter):
    """Naive shuffle split at the row level (leakage-prone baseline)."""

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        if n == 0:
            empty = df.iloc[0:0].copy()
            return empty, empty, empty.copy()
        idx = np.arange(n)
        train_idx, temp_idx = train_test_split(
            idx,
            train_size=0.8,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
        if len(temp_idx) <= 1:
            val_idx = np.array([], dtype=int)
            test_idx = temp_idx if len(temp_idx) else np.array([], dtype=int)
        else:
            val_idx, test_idx = train_test_split(
                temp_idx,
                train_size=0.5,
                random_state=RANDOM_SEED + 1,
                shuffle=True,
            )
        return (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
        )


class DrugColdScaffoldSplitter(BaseSplitter):
    """
    Murcko-scaffold disjoint split: rows sharing a scaffold appear in exactly one fold.
    """

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if len(df) == 0:
            empty = df.iloc[0:0].copy()
            return empty, empty, empty.copy()
        work = df.copy()
        suppress_rdkit_logging()
        work["_scaffold"] = work["drug_smiles"].map(scaffold_smiles_from_drug)
        uniq = sorted(work["_scaffold"].unique().tolist())
        train_s, val_s, test_s = _partition_unique_entities(uniq)
        tr = work[work["_scaffold"].isin(train_s)].drop(columns=["_scaffold"]).reset_index(
            drop=True
        )
        va = work[work["_scaffold"].isin(val_s)].drop(columns=["_scaffold"]).reset_index(
            drop=True
        )
        te = work[work["_scaffold"].isin(test_s)].drop(columns=["_scaffold"]).reset_index(
            drop=True
        )
        return tr, va, te


class TargetColdSplitter(BaseSplitter):
    """Target-sequence disjoint split: each protein occurs in exactly one fold."""

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if len(df) == 0:
            empty = df.iloc[0:0].copy()
            return empty, empty, empty.copy()
        if "target_sequence" not in df.columns:
            raise ValueError("TargetColdSplitter requires column target_sequence")
        uniq = sorted(df["target_sequence"].astype(str).unique().tolist())
        train_t, val_t, test_t = _partition_unique_entities(uniq)
        tr = df[df["target_sequence"].astype(str).isin(train_t)].reset_index(drop=True)
        va = df[df["target_sequence"].astype(str).isin(val_t)].reset_index(drop=True)
        te = df[df["target_sequence"].astype(str).isin(test_t)].reset_index(drop=True)
        return tr, va, te


def intersect_sizes(a: Set[str], b: Set[str]) -> int:
    return len(a & b)
