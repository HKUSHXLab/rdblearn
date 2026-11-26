"""Auto hurdle regression wrapping TabPFN stages with optional hard_pool contexts."""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, clone


class _StageEstimator(BaseEstimator):
    """Sklearn-compatible wrapper that builds a fresh TabPFN (+ hard_pool) per stage."""

    def __init__(
        self,
        make: Callable[..., Any],
        *,
        task_type: str,
        context_time: np.ndarray | None = None,
        context_strategy: str | None = None,
        recency_window: int = 800_000,
        seed: int = 0,
        model_params: Optional[dict] = None,
    ) -> None:
        self.make = make
        self.task_type = task_type
        self.context_time = context_time
        self.context_strategy = context_strategy
        self.recency_window = recency_window
        self.seed = seed
        self.model_params = model_params or {}

    def fit(self, X: pd.DataFrame, y) -> "_StageEstimator":
        X = X.reset_index(drop=True)
        y_arr = np.asarray(y)
        overrides = dict(self.model_params)
        X_fit, y_fit = X, y_arr
        if self.context_strategy == "hard_pool":
            from .context import prepare_hard_pool_fit

            if self.context_time is None:
                raise ValueError("hard_pool hurdle stage requires context_time")
            # AutoHurdle may pass a subset of rows; recover timestamps via index if present
            if hasattr(X, "index") and len(X) != len(self.context_time):
                # Prefer positional subset when X was reset; use stored row ids if available
                raise ValueError(
                    "hurdle stage X length does not match context_time; "
                    "pass row-aligned timestamps via index alignment"
                )
            cutoff = self.context_time
            X_fit, y_s, pool_overrides = prepare_hard_pool_fit(
                X_fit,
                y_fit,
                cutoff,
                recency_window=self.recency_window,
                seed=self.seed,
            )
            y_fit = y_s.to_numpy()
            ic = dict(overrides.get("inference_config") or {})
            ic.update(pool_overrides.get("inference_config") or {})
            overrides = {**overrides, **pool_overrides, "inference_config": ic}

        self.estimator_ = self.make(**overrides)
        self.estimator_.fit(X_fit, y_fit)
        if self.task_type == "classification":
            self.classes_ = getattr(self.estimator_, "classes_", np.unique(y_fit))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator_.predict_proba(X))

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> Any:
        return self.estimator_.predict(X, **kwargs)


def fit_auto_hurdle(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    make_classifier: Callable[..., Any],
    make_regressor: Callable[..., Any],
    context_time: np.ndarray | None,
    context_strategy: str | None = "hard_pool",
    recency_window: int = 800_000,
    seed: int = 0,
    zero_threshold: float = 0.05,
    model_params: Optional[dict] = None,
) -> Any:
    """Fit ``tabpfn_extensions.hurdle.AutoHurdleRegressor`` on ``X, y``."""
    try:
        from tabpfn_extensions.hurdle import AutoHurdleRegressor
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "hurdle='auto' requires tabpfn-extensions (AutoHurdleRegressor)"
        ) from e

    params = dict(model_params or {})
    # Row-align timestamps for stage subsetting: store on a RangeIndex frame
    X = X.reset_index(drop=True)
    y_arr = np.asarray(y)
    if context_time is not None and len(context_time) != len(X):
        raise ValueError("context_time length must match X")

    # AutoHurdle filters rows; pass full context_time and a custom stage that
    # reindexes by the positional index AutoHurdle preserves after reset_index.
    class _AlignedStage(_StageEstimator):
        def fit(self, X_stage: pd.DataFrame, y_stage):  # type: ignore[override]
            X_stage = X_stage.copy()
            # When AutoHurdle subsets, index may be a subset of the original RangeIndex
            if self.context_time is not None and len(X_stage) != len(self.context_time):
                idx = X_stage.index.to_numpy()
                cutoff = np.asarray(self.context_time)[idx]
            else:
                cutoff = self.context_time
            # Temporarily swap context_time for this subset
            original = self.context_time
            self.context_time = cutoff
            X_stage = X_stage.reset_index(drop=True)
            try:
                return super().fit(X_stage, y_stage)
            finally:
                self.context_time = original

    clf = _AlignedStage(
        make_classifier,
        task_type="classification",
        context_time=context_time,
        context_strategy=context_strategy,
        recency_window=recency_window,
        seed=seed,
        model_params=params,
    )
    reg = _AlignedStage(
        make_regressor,
        task_type="regression",
        context_time=context_time,
        context_strategy=context_strategy,
        recency_window=recency_window,
        seed=seed,
        model_params=params,
    )
    estimator = AutoHurdleRegressor(
        classifier=clf,
        regressor=reg,
        hurdle="auto",
        zero_threshold=zero_threshold,
    )
    logger.info("Fitting auto hurdle regressor (zero_threshold={})", zero_threshold)
    estimator.fit(X, y_arr)
    return estimator
