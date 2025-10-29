from typing import Optional, List, Union
import pandas as pd
import numpy as np

def _to_1d_proba(proba: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    if isinstance(proba, pd.DataFrame):
        # if single proba column named 'proba' use it, else take first column
        if 'proba' in proba.columns:
            return proba['proba'].to_numpy()
        return proba.iloc[:, 0].to_numpy()
    arr = np.asarray(proba)
    if arr.ndim == 2 and arr.shape[1] > 1:
        # assume last column is positive class? take column 1 if binary
        if arr.shape[1] == 2:
            return arr[:, 1]
        return arr[:, 0]
    return arr.ravel()

def compute_metrics(labels: pd.Series, proba: Union[pd.DataFrame, np.ndarray], metrics: Optional[List[str]] = None) -> dict:
    """Compute requested metrics. Default is ['accuracy'] if metrics is None."""
    if metrics is None:
        metrics = ['accuracy']
    proba_1d = _to_1d_proba(proba)
    results = {}
    for m in metrics:
        if m == 'accuracy':
            preds = (proba_1d >= 0.5).astype(int)
            results['accuracy'] = float((preds == labels.to_numpy()).mean())
        elif m == 'roc_auc':
            try:
                from sklearn.metrics import roc_auc_score

                results['roc_auc'] = float(roc_auc_score(labels.to_numpy(), proba_1d))
            except Exception:
                results['roc_auc'] = None
        elif m == 'logloss':
            try:
                from sklearn.metrics import log_loss

                results['logloss'] = float(log_loss(labels.to_numpy(), proba_1d))
            except Exception:
                results['logloss'] = None
        else:
            results[m] = None
    return results
