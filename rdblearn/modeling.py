"""Fit and predict a single-table model on an already-featurized frame.

Shared by ``RDBLearnClassifier`` / ``RDBLearnRegressor`` and ``StandardPipeline``, so the
pipeline calls foundation models directly and still applies the class-cap fallback and the
regression point-prediction choice.
"""
import inspect
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from .base10_hierarchical import Base10Decomposer, reconstruct_class_proba_from_digit_probs


def base_estimator_max_classes(estimator) -> Optional[int]:
    """Classes one head of ``estimator`` can take; ``None`` means unlimited.

    Above this, base-10-hierarchical heads are used. Estimators may declare
    ``rdblearn_max_classes``; TabPFN reports the cap of its checkpoint (10 for v2.5,
    160 for v3.5) and TabICL handles any number of classes when ``support_many_classes``
    is on. Anything else is assumed to take at most 10.
    """
    if hasattr(estimator, "rdblearn_max_classes"):
        return estimator.rdblearn_max_classes
    if type(estimator).__module__.startswith("tabicl") and getattr(estimator, "support_many_classes", False):
        return None
    if hasattr(estimator, "get_inference_config"):
        try:
            return int(estimator.get_inference_config().MAX_NUMBER_OF_CLASSES)
        except Exception as e:
            logger.debug(f"Could not read the base estimator's class cap: {e}")
    return 10


def predict_batches(predict_func, X: pd.DataFrame, batch_size: Optional[int], **kwargs):
    """Call ``predict_func`` on ``X``, concatenating batches when ``batch_size`` is set."""
    if not (batch_size and len(X) > batch_size):
        return predict_func(X, **kwargs)
    results = [predict_func(X.iloc[i:i + batch_size], **kwargs) for i in range(0, len(X), batch_size)]
    if isinstance(results[0], dict):
        aggregated = {}
        for key in results[0].keys():
            key_results = [r[key] for r in results]
            if isinstance(key_results[0], np.ndarray):
                aggregated[key] = np.concatenate(key_results)
            elif isinstance(key_results[0], (pd.Series, pd.DataFrame)):
                aggregated[key] = pd.concat(key_results, axis=0)
            else:
                print(
                    f"Warning: Unexpected type of key_results: {type(key_results[0])} "
                    f"when aggregating results for key {key}, skipping this key"
                )
        return aggregated
    if isinstance(results[0], np.ndarray):
        return np.concatenate(results)
    if isinstance(results[0], (pd.Series, pd.DataFrame)):
        return pd.concat(results, axis=0)
    return np.concatenate(results)


def regression_predict_kwargs(estimator, output_type: Optional[str], kwargs: dict) -> dict:
    """Pass ``output_type`` when the regressor's ``predict`` accepts it."""
    kwargs = dict(kwargs)
    if (
        output_type is not None
        and "output_type" not in kwargs
        and "output_type" in inspect.signature(estimator.predict).parameters
    ):
        kwargs["output_type"] = output_type
    return kwargs


def fit_classification(state: Any, estimator, X: pd.DataFrame, y, **kwargs):
    """Fit ``estimator`` on ``X, y``, using base-10-hierarchical heads above its class cap.

    Fitted attributes (``classes_``, ``base10_hierarchical_``, ``digit_estimators_``, ...)
    are written onto ``state``. ``state.base_estimator`` is the single head when the
    label count is within the cap.
    """
    y_series = y if isinstance(y, pd.Series) else pd.Series(y, name="target")
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series)
    n_classes = int(len(le.classes_))

    state.base_estimator = estimator
    state.label_encoder_ = le
    state.classes_ = np.asarray(le.classes_)
    state.n_classes_ = n_classes
    state.base10_hierarchical_ = False
    state.base10_decomposer_ = None
    state.digit_estimators_ = None

    max_classes = base_estimator_max_classes(estimator)
    if max_classes is None or n_classes <= max_classes:
        logger.info(f"Fitting base estimator (C={n_classes}, single head) ...")
        estimator.fit(X, y_series, **kwargs)
        return state

    state.base10_hierarchical_ = True
    decomposer = Base10Decomposer(n_classes)
    state.base10_decomposer_ = decomposer
    train_digits = decomposer.digits_for_array(y_enc)
    digit_estimators = []
    for i in range(decomposer.D):
        est = clone(estimator)
        y_digit = pd.Series(train_digits[i], index=X.index, name=y_series.name)
        est.fit(X, y_digit, **kwargs)
        digit_estimators.append(est)
    state.digit_estimators_ = digit_estimators
    logger.info(f"Fitted base-10-hierarchical classifier (C={n_classes}, D={decomposer.D} digit heads).")
    return state


def predict_proba_batches(estimator, X: pd.DataFrame, batch_size: Optional[int], **kwargs) -> np.ndarray:
    fn = getattr(estimator, "predict_proba")
    if batch_size and len(X) > batch_size:
        parts = [np.asarray(fn(X.iloc[i:i + batch_size], **kwargs)) for i in range(0, len(X), batch_size)]
        return np.concatenate(parts, axis=0)
    return np.asarray(fn(X, **kwargs))


def predict_proba_hierarchical(state: Any, X: pd.DataFrame, batch_size: Optional[int], **kwargs) -> np.ndarray:
    check_is_fitted(state, attributes=["digit_estimators_", "base10_decomposer_"])
    logger.info("Making predictions (base-10-hierarchical) ...")
    digit_probs = [
        predict_proba_batches(est, X, batch_size, **kwargs) for est in state.digit_estimators_
    ]
    return reconstruct_class_proba_from_digit_probs(state.base10_decomposer_, digit_probs, state.n_classes_)


class FittedModel:
    """A foundation model fit on one featurized frame, plus the classification fallback."""

    def __init__(self, estimator, task_type: str, config):
        self.estimator = estimator
        self.task_type = task_type
        self.config = config
        self.base_estimator = estimator
        self.classes_ = None
        self.base10_hierarchical_ = False

    def fit(self, X: pd.DataFrame, y, **kwargs):
        if self.task_type == "classification":
            fit_classification(self, self.estimator, X, y, **kwargs)
        else:
            logger.info("Fitting base estimator ...")
            self.estimator.fit(X, y, **kwargs)
            self.base_estimator = self.estimator
        return self

    def predict(self, X: pd.DataFrame, **kwargs):
        if self.task_type == "classification" and self.base10_hierarchical_:
            proba = self.predict_proba(X, **kwargs)
            return self.classes_.take(np.argmax(proba, axis=1).astype(np.intp, copy=False))
        if self.task_type == "regression":
            kwargs = regression_predict_kwargs(self.estimator, self.config.regression_output_type, kwargs)
        logger.info("Making predictions ...")
        return predict_batches(
            getattr(self.estimator, "predict"), X, self.config.predict_batch_size, **kwargs
        )

    def predict_proba(self, X: pd.DataFrame, **kwargs):
        if self.base10_hierarchical_:
            return predict_proba_hierarchical(self, X, self.config.predict_batch_size, **kwargs)
        logger.info("Making predictions ...")
        return predict_batches(
            self.estimator.predict_proba, X, self.config.predict_batch_size, **kwargs
        )
