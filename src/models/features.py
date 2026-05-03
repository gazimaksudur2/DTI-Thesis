#!/usr/bin/env python3
"""Feature extraction for classical DTI baselines."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

MORGAN_RADIUS = 2
MORGAN_NBITS = 1024
AAC_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AAC_UNKNOWN = "X"
AAC_DIM = len(AAC_ALPHABET) + 1


def suppress_rdkit_logging() -> None:
    RDLogger.DisableLog("rdApp.*")


def smiles_to_morgan_bits(smiles: str) -> np.ndarray:
    """Return Morgan bit vector as float32 array; invalid SMILES => zeros."""
    out = np.zeros(MORGAN_NBITS, dtype=np.float32)
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return out
    bv = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=MORGAN_RADIUS,
        nBits=MORGAN_NBITS,
    )
    ConvertToNumpyArray(bv, out)
    return out


def sequence_to_aac(seq: str) -> np.ndarray:
    """AAC vector over 20 amino acids + unknown bucket."""
    vec = np.zeros(AAC_DIM, dtype=np.float32)
    text = str(seq).strip().upper()
    if not text:
        return vec
    idx_map = {aa: i for i, aa in enumerate(AAC_ALPHABET)}
    unknown_idx = AAC_DIM - 1
    for ch in text:
        vec[idx_map.get(ch, unknown_idx)] += 1.0
    vec /= float(len(text))
    return vec


@dataclass(frozen=True)
class FeatureFrame:
    x: np.ndarray
    y: np.ndarray
    invalid_smiles: int


def build_matrix(df: pd.DataFrame) -> FeatureFrame:
    """Build fused feature matrix and target vector from split dataframe."""
    needed = {"drug_smiles", "target_sequence", "affinity_label"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    suppress_rdkit_logging()
    n = len(df)
    drug = np.zeros((n, MORGAN_NBITS), dtype=np.float32)
    prot = np.zeros((n, AAC_DIM), dtype=np.float32)
    invalid = 0

    for i, smi in enumerate(df["drug_smiles"].astype(str).tolist()):
        fp = smiles_to_morgan_bits(smi)
        if not fp.any():
            invalid += 1
        drug[i] = fp

    for i, seq in enumerate(df["target_sequence"].astype(str).tolist()):
        prot[i] = sequence_to_aac(seq)

    x = np.concatenate([drug, prot], axis=1)
    y = df["affinity_label"].to_numpy(dtype=np.float32, copy=True)
    if invalid:
        logging.warning("Feature build: %s/%s SMILES invalid; using zero fingerprints.", invalid, n)
    return FeatureFrame(x=x, y=y, invalid_smiles=invalid)
