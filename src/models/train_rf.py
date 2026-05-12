#!/usr/bin/env python3
"""
Train and evaluate a Random Forest regressor for DTI.

Mirrors the public surface of :mod:`train_xgboost` so that the experiment
runner can treat both models uniformly. ``scikit-learn``'s
``RandomForestRegressor`` is **CPU-only**: the ``device`` argument is accepted
for API compatibility and a single warning is emitted if a non-CPU device is
requested. We intentionally do not pull in cuML to keep the ``dti_research``
conda environment lightweight on Windows.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from train_xgboost import EvalMetrics, concordance_index

RANDOM_SEED = 42
_logger = logging.getLogger(__name__)
_logged_cpu_only = False


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
) -> EvalMetrics:
    """
    Train ``RandomForestRegressor`` and evaluate on the held-out test split.

    Notes
    -----
    - ``x_val`` / ``y_val`` are accepted for signature parity with
      :func:`train_xgboost.train_and_evaluate` but sklearn's RF has no native
      early-stopping hook, so the validation split is unused. This is by
      design: cold-start evaluation comes from the test split anyway.
    - ``device`` is accepted for parity. Random Forest in scikit-learn runs on
      CPU regardless; if the caller asked for ``"gpu"`` we log a one-time
      warning and continue on CPU instead of raising.
    """
    global _logged_cpu_only
    if device.strip().lower() in ("gpu", "cuda") and not _logged_cpu_only:
        _logger.warning(
            "Random Forest requested device=%s, but scikit-learn RF is CPU-only. "
            "Continuing on CPU.",
            device,
        )
        _logged_cpu_only = True

    del x_val, y_val

    params: dict[str, Any] = {
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    if model_params:
        params.update(model_params)

    model = RandomForestRegressor(**params)
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    mse = float(mean_squared_error(y_test, preds))
    if len(y_test) < 2 or np.std(y_test) == 0 or np.std(preds) == 0:
        pr = 0.0
    else:
        pr = float(pearsonr(y_test, preds).statistic)
    ci = float(concordance_index(y_test, preds))
    return EvalMetrics(mse=mse, pearson_r=pr, ci=ci, best_iteration=-1)
