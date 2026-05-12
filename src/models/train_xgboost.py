#!/usr/bin/env python3
"""Train and evaluate an XGBoost regressor for DTI."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.device import resolve_device, xgb_device_params  # noqa: E402

RANDOM_SEED = 42


@dataclass(frozen=True)
class EvalMetrics:
    mse: float
    pearson_r: float
    ci: float
    best_iteration: int


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray, *, chunk_size: int = 4096) -> float:
    """
    Concordance index for regression ranking quality.

    CI = fraction of comparable pairs (y_i != y_j) whose predicted order matches true order.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    n = len(yt)
    if n <= 1:
        return 0.0

    if n <= 8192:
        dy = np.subtract.outer(yt, yt)
        dp = np.subtract.outer(yp, yp)
        mask = dy > 0
        denom = float(mask.sum())
        if denom == 0:
            return 0.0
        num = float((dp[mask] > 0).sum() + 0.5 * (dp[mask] == 0).sum())
        return num / denom

    # Memory-safe fallback
    num = 0.0
    denom = 0.0
    for i0 in range(0, n, chunk_size):
        i1 = min(n, i0 + chunk_size)
        dy = yt[i0:i1, None] - yt[None, :]
        dp = yp[i0:i1, None] - yp[None, :]
        mask = dy > 0
        d = float(mask.sum())
        if d == 0:
            continue
        denom += d
        num += float((dp[mask] > 0).sum() + 0.5 * (dp[mask] == 0).sum())
    return (num / denom) if denom else 0.0


def train_and_evaluate(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_params: dict[str, Any] | None = None,
    device: str = "auto",
    save_model_path: Path | None = None,
) -> EvalMetrics:
    """
    Train XGBRegressor with validation early stopping and evaluate on test split.

    Parameters
    ----------
    device:
        One of ``"auto"``, ``"gpu"``, or ``"cpu"``. ``"auto"`` uses CUDA when a
        usable NVIDIA device is detected (e.g. RTX 4060 Ti) and falls back to
        CPU otherwise. ``"gpu"`` raises if no CUDA device is available.
    save_model_path:
        If provided, the fitted model is serialized to this path (JSON format
        recommended, e.g. ``models/davis/xgboost_random.json``) so downstream
        consumers like the SHAP explainer can reload it.
    """
    resolved_device = resolve_device(device)
    params: dict[str, Any] = {
        "objective": "reg:squarederror",
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "eval_metric": "rmse",
        "early_stopping_rounds": 50,
    }
    params.update(xgb_device_params(resolved_device))
    if model_params:
        params.update(model_params)

    model = XGBRegressor(**params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )

    if save_model_path is not None:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(save_model_path))

    preds = model.predict(x_test)
    mse = float(mean_squared_error(y_test, preds))
    if len(y_test) < 2 or np.std(y_test) == 0 or np.std(preds) == 0:
        pr = 0.0
    else:
        pr = float(pearsonr(y_test, preds).statistic)
    ci = float(concordance_index(y_test, preds))
    best_iteration = int(getattr(model, "best_iteration", -1))
    return EvalMetrics(mse=mse, pearson_r=pr, ci=ci, best_iteration=best_iteration)
