import numpy as np
import pandas as pd
import pytest

from multitabfm.evaluation import compute_metrics
from multitabfm.model import AGAdapter


class _RegPredictor:
    # Mimic AutoGluon predictor for regression
    problem_type = "regression"

    def predict(self, X):
        # simple linear-ish predictions
        return pd.Series(np.linspace(0.0, 1.0, len(X)))


def test_evaluation_regression_metrics():
    y_true = pd.Series([0.0, 0.5, 1.0, 1.5])
    y_pred = pd.Series([0.1, 0.4, 0.9, 1.4])

    metrics = compute_metrics(y_true, y_pred, metrics=["mse", "rmse", "mae", "r2"])

    assert set(metrics.keys()) == {"mse", "rmse", "mae", "r2"}
    assert metrics["mse"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
    # r2 can be negative for bad models; just ensure it is a float
    assert isinstance(metrics["r2"], float)


def test_agadapter_predict_regression_and_no_proba():
    adapter = AGAdapter()
    adapter.predictor = _RegPredictor()
    adapter.problem_type = "regression"

    X = pd.DataFrame({"a": [1, 2, 3]})
    preds = adapter.predict(X)
    assert isinstance(preds, pd.Series)
    assert len(preds) == 3

    with pytest.raises(RuntimeError):
        _ = adapter.predict_proba(X)
