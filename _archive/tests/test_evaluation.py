import numpy as np
import pandas as pd

from multitabfm.evaluation import compute_metrics


def test_binary_metrics():
    y = pd.Series([0, 1, 1, 0])
    proba = pd.DataFrame(
        {
            0: [0.8, 0.3, 0.2, 0.6],
            1: [0.2, 0.7, 0.8, 0.4],
        }
    )

    metrics = compute_metrics(y, proba, metrics=["accuracy", "roc_auc", "logloss"])

    assert isinstance(metrics, dict)
    assert 0.99 <= metrics["accuracy"] <= 1.0
    assert 0.9 <= (metrics["roc_auc"] or 0.0) <= 1.0
    assert metrics["logloss"] is not None
    assert np.isfinite(metrics["logloss"]).item() is True


def test_multiclass_metrics():
    y = pd.Series([0, 1, 2, 1])
    proba = pd.DataFrame(
        {
            0: [0.9, 0.1, 0.1, 0.1],
            1: [0.05, 0.8, 0.1, 0.8],
            2: [0.05, 0.1, 0.8, 0.1],
        }
    )

    metrics = compute_metrics(y, proba, metrics=["accuracy", "roc_auc", "logloss"])

    assert isinstance(metrics, dict)
    assert 0.99 <= metrics["accuracy"] <= 1.0
    # multiclass roc_auc (macro OVR) should be well above chance
    assert 0.8 <= (metrics["roc_auc"] or 0.0) <= 1.0
    assert metrics["logloss"] is not None
    assert np.isfinite(metrics["logloss"]).item() is True
