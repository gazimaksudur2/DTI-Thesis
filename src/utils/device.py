"""
Device selection helpers for XGBoost training.

Centralizes the logic for choosing between an NVIDIA CUDA device (e.g. an
RTX 4060 Ti) and CPU execution, so every entrypoint in the repository
exposes the same ``--device {auto,gpu,cpu}`` semantics:

- ``auto``: use CUDA if a usable device is detected, otherwise fall back to CPU.
- ``gpu``:  use CUDA; raise a clear ``RuntimeError`` if no CUDA device is usable
            (we never silently fall back when the user explicitly asked for GPU).
- ``cpu``:  force CPU regardless of GPU presence.

Detection probes XGBoost itself with a tiny fit on ``device="cuda"`` so that
what we report matches what XGBoost will actually do at runtime. The result
is cached per-process so the probe only runs once.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

_VALID_REQUESTS = ("auto", "gpu", "cpu")
_logger = logging.getLogger(__name__)
_logged_resolution = False


@lru_cache(maxsize=1)
def detect_cuda() -> bool:
    """
    Return True if XGBoost can train on a CUDA device on this machine.

    The check first inspects ``xgboost.build_info()`` to see whether the
    installed XGBoost wheel was compiled with CUDA support. If yes, a 4-row
    smoke fit on ``device="cuda"`` confirms a usable runtime device.
    """
    try:
        import xgboost as xgb
    except Exception as exc:
        _logger.debug("CUDA detection: xgboost import failed (%s)", exc)
        return False

    build_info_fn = getattr(xgb, "build_info", None)
    if callable(build_info_fn):
        try:
            info = build_info_fn()
            use_cuda = bool(info.get("USE_CUDA", False)) if isinstance(info, dict) else False
            if not use_cuda:
                _logger.debug("CUDA detection: xgboost wheel was built without CUDA support.")
                return False
        except Exception as exc:
            _logger.debug("CUDA detection: build_info() failed (%s); will probe directly.", exc)

    try:
        x = np.zeros((4, 2), dtype=np.float32)
        y = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        probe = xgb.XGBRegressor(
            device="cuda",
            tree_method="hist",
            n_estimators=1,
            verbosity=0,
        )
        probe.fit(x, y)
        return True
    except Exception as exc:
        _logger.debug("CUDA detection: probe fit failed (%s); falling back to CPU.", exc)
        return False


def resolve_device(requested: str) -> str:
    """
    Translate a user-facing request into the XGBoost device string.

    Parameters
    ----------
    requested:
        One of ``"auto"``, ``"gpu"``, or ``"cpu"`` (case-insensitive).

    Returns
    -------
    str
        ``"cuda"`` when XGBoost should run on GPU, ``"cpu"`` otherwise.

    Raises
    ------
    ValueError
        If ``requested`` is not one of the accepted values.
    RuntimeError
        If the user explicitly asked for ``"gpu"`` but no CUDA device is usable.
    """
    global _logged_resolution

    if not isinstance(requested, str):
        raise ValueError(f"device must be a string, got {type(requested).__name__}")
    key = requested.strip().lower()
    if key not in _VALID_REQUESTS:
        raise ValueError(
            f"device must be one of {_VALID_REQUESTS}, got {requested!r}"
        )

    if key == "cpu":
        resolved = "cpu"
    elif key == "gpu":
        if not detect_cuda():
            raise RuntimeError(
                "device='gpu' requested but no CUDA-capable XGBoost runtime was "
                "detected. On Windows-native, install via WSL2 Ubuntu or run with "
                "--device cpu. Verify with `nvidia-smi` and a CUDA-enabled XGBoost build."
            )
        resolved = "cuda"
    else:
        resolved = "cuda" if detect_cuda() else "cpu"

    if not _logged_resolution:
        _logger.info("Resolved compute device: requested=%s -> %s", key, resolved)
        _logged_resolution = True
    return resolved


def xgb_device_params(device: str) -> dict[str, str]:
    """
    Return XGBoost ``>=2.0`` parameters for the chosen device.

    Both branches use ``tree_method="hist"`` so that switching between CPU and
    GPU does not change the algorithm, only where it executes. The legacy
    ``tree_method="gpu_hist"`` form is deprecated in XGBoost 2.x.
    """
    key = device.strip().lower()
    if key in ("cuda", "gpu"):
        return {"device": "cuda", "tree_method": "hist"}
    if key == "cpu":
        return {"device": "cpu", "tree_method": "hist"}
    raise ValueError(f"Unknown XGBoost device: {device!r}")
