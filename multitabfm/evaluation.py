from typing import Optional, List, Union
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, r2_score


def _to_proba_df(proba: Union[pd.DataFrame, np.ndarray, pd.Series]) -> pd.DataFrame:
    """Coerce probabilities to a DataFrame with class labels as columns.

    - If 1D array/series is provided, assume binary probabilities of the positive class and
      construct a two-column DataFrame with classes [0, 1].
    - If 2D array is provided, use integer class labels [0..n_classes-1].
    """
    if isinstance(proba, pd.DataFrame):
        return proba
    if isinstance(proba, pd.Series):
        p1 = proba.to_numpy()
        return pd.DataFrame({0: 1 - p1, 1: p1})
    arr = np.asarray(proba)
    if arr.ndim == 1:
        p1 = arr
        return pd.DataFrame({0: 1 - p1, 1: p1})
    if arr.ndim == 2:
        n_classes = arr.shape[1]
        return pd.DataFrame(arr, columns=list(range(n_classes)))
    raise ValueError("Unsupported probability input shape")


def _coerce_labels_dtype(labels: pd.Series, proba_df: pd.DataFrame) -> pd.Series:
    """Attempt to coerce label dtype to match proba_df columns when possible."""
    if proba_df.empty:
        return labels
    cols = list(proba_df.columns)
    target_type = type(cols[0])
    try:
        if target_type is int:
            return labels.astype(int)
        if target_type is float:
            return labels.astype(float)
        if target_type is str:
            return labels.astype(str)
    except Exception:
        pass
    return labels


def _binary_positive_col(labels: pd.Series, proba_df: pd.DataFrame):
    """Pick a positive-class column name for binary ROC-AUC.

    Preference order: 1, '1', True, the second column.
    """
    candidates = [1, "1", True]
    for c in candidates:
        if c in proba_df.columns:
            return c
    # fallback to the second column
    return proba_df.columns[1] if len(proba_df.columns) >= 2 else proba_df.columns[0]


def compute_metrics(labels: pd.Series, preds_or_proba: Union[pd.DataFrame, np.ndarray, pd.Series], metrics: Optional[List[str]] = None) -> dict:
    """Compute requested metrics for classification or regression.

    Inputs:
        labels: pandas Series of true labels
        preds_or_proba: predictions (1D) for regression or probabilities (2D) for classification
        metrics: list of metric names

    Supported metrics:
        - Classification: 'accuracy', 'auroc', 'logloss'
        - Regression: 'rmse', 'mse', 'mae', 'r2'

    Returns a dict of metric -> value (float) or None if unavailable.
    """
    metrics = metrics or ["accuracy"]

    # Heuristic: if DataFrame with 2+ columns -> classification probabilities
    # If 1D array/series -> treat as regression predictions unless classification metrics requested only.
    is_df = isinstance(preds_or_proba, pd.DataFrame)
    arr = np.asarray(preds_or_proba) if not is_df else None
    is_1d = (not is_df) and (arr.ndim == 1)

    cls_metrics = {"accuracy", "auroc", "logloss"}
    reg_metrics = {"rmse", "mse", "mae", "r2"}

    results: dict = {}

    if is_df:
        # Classification path
        proba_df = _to_proba_df(preds_or_proba)
        y_true = _coerce_labels_dtype(labels, proba_df)

        if "accuracy" in metrics:
            try:
                y_pred = proba_df.idxmax(axis=1)
                try:
                    y_pred = y_pred.astype(y_true.dtype)
                except Exception:
                    pass
                results["accuracy"] = float((y_pred.to_numpy() == y_true.to_numpy()).mean())
            except Exception:
                results["accuracy"] = None

        if "auroc" in metrics:
            try:
                classes = list(proba_df.columns)
                n_classes = len(classes)
                if n_classes == 2:
                    pos_col = _binary_positive_col(y_true, proba_df)
                    y_score = proba_df[pos_col].to_numpy()
                    results["auroc"] = float(roc_auc_score(y_true.to_numpy(), y_score))
                else:
                    y_score = proba_df.to_numpy()
                    results["auroc"] = float(
                        roc_auc_score(y_true.to_numpy(), y_score, multi_class="ovr", average="macro", labels=classes)
                    )
            except Exception:
                results["auroc"] = None

        if "logloss" in metrics:
            try:
                classes = list(proba_df.columns)
                results["logloss"] = float(log_loss(y_true.to_numpy(), proba_df.to_numpy(), labels=classes))
            except Exception:
                results["logloss"] = None

    elif is_1d:
        # Regression path
        y_true = labels.to_numpy()
        y_pred = np.asarray(preds_or_proba)

        if "mse" in metrics:
            try:
                results["mse"] = float(mean_squared_error(y_true, y_pred))
            except Exception:
                results["mse"] = None
        if "rmse" in metrics:
            try:
                results["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            except Exception:
                results["rmse"] = None
        if "mae" in metrics:
            try:
                results["mae"] = float(mean_absolute_error(y_true, y_pred))
            except Exception:
                results["mae"] = None
        if "r2" in metrics:
            try:
                results["r2"] = float(r2_score(y_true, y_pred))
            except Exception:
                results["r2"] = None

    # Fill any requested metrics not computed with None
    for m in metrics:
        if m not in results:
            # Only populate None if the metric belongs to a supported set
            if m in cls_metrics.union(reg_metrics):
                results[m] = None

    return results
