"""Recency context strategies for foundation-model fits.

``hard_pool``: each ensemble estimator draws ``K`` rows without replacement from
the ``M`` most-recent training rows by cutoff time. The public hyperparameter is
``recency_window`` (= ``M``); bag size defaults to ``K = M // 4`` and
``n_estimators = 8``.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

DEFAULT_N_ESTIMATORS = 8


def _to_int64_ns(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t)
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[ns]").view("int64")
    if t.dtype == object or pd.api.types.is_datetime64_any_dtype(t):
        return pd.to_datetime(t).astype("int64").to_numpy()
    return t.astype("int64")


def hard_pool_subsample_indices(
    t: np.ndarray, *, K: int, M: int, n_estimators: int, seed: int
) -> list[np.ndarray] | None:
    """Most-recent-``M`` pool; each estimator draws ``K`` rows (``replace=False``).

    Returns ``None`` when ``N <= M`` so the caller can fall back to TabPFN's
    uniform ``SUBSAMPLE_SAMPLES=K`` path.
    """
    n = len(t)
    if n <= M:
        return None
    order = np.argsort(-_to_int64_ns(t), kind="stable")
    pool = order[:M]
    rng = np.random.default_rng(seed)
    return [rng.choice(pool, size=K, replace=False) for _ in range(n_estimators)]


def hard_pool_sizes(recency_window: int) -> tuple[int, int]:
    """Return ``(K, M)`` from the public ``recency_window`` (= ``M``)."""
    m = int(recency_window)
    if m < 4:
        raise ValueError(f"recency_window must be >= 4, got {m}")
    return m // 4, m


def prepare_hard_pool_fit(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    context_time: np.ndarray,
    *,
    recency_window: int = 800_000,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Shrink ``X,y`` to the hard-pool union and build TabPFN constructor overrides.

    Overrides always include ``n_estimators`` and
    ``inference_config["SUBSAMPLE_SAMPLES"]`` (either ``K`` or per-estimator
    index lists remapped into the union).
    """
    if context_time is None:
        raise ValueError("hard_pool requires per-row context_time (cutoff timestamps)")
    t = np.asarray(context_time)
    if len(t) != len(X):
        raise ValueError(f"context_time length {len(t)} != X length {len(X)}")

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    K, M = hard_pool_sizes(recency_window)
    idx = hard_pool_subsample_indices(t, K=K, M=M, n_estimators=n_estimators, seed=seed)
    if idx is None:
        subsample: Any = K
    else:
        union = np.unique(np.concatenate(idx))
        X = X.iloc[union].reset_index(drop=True)
        y = y.iloc[union].reset_index(drop=True)
        subsample = [np.searchsorted(union, e) for e in idx]

    overrides = {
        "n_estimators": n_estimators,
        "inference_config": {"SUBSAMPLE_SAMPLES": subsample},
    }
    return X, y, overrides
