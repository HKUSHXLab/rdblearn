from typing import Optional, List, Union
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_metrics(labels: pd.Series, proba: Union[pd.DataFrame, np.ndarray], metrics: Optional[List[str]] = None) -> dict:
    """Compute requested metrics. Default is ['accuracy'] if metrics is None."""
    if metrics is None:
        metrics = ['accuracy']
    results = {}
    for m in metrics:
        if m == 'roc_auc':
            results['roc_auc'] = float(roc_auc_score(labels.to_numpy(), proba))
        else:
            results[m] = None
    return results
