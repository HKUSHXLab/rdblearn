import numpy as np
import pandas as pd

from multitabfm.model import AGAdapter


class _DFPredictor:
    def predict_proba(self, X):
        # returns a well-formed DataFrame with class labels as columns
        return pd.DataFrame({0: [0.7, 0.2], 1: [0.3, 0.8]})


class _SeriesPredictor:
    def predict_proba(self, X):
        # positive-class probabilities as a Series
        return pd.Series([0.3, 0.8], name="proba")


class _ArrayPredictor:
    class_labels = [0, 1]

    def predict_proba(self, X):
        # numpy array 2D
        return np.array([[0.7, 0.3], [0.2, 0.8]])


def test_agadapter_returns_dataframe_from_dataframe_input():
    adapter = AGAdapter()
    adapter.predictor = _DFPredictor()
    X = pd.DataFrame({"a": [1, 2]})
    proba = adapter.predict_proba(X)
    assert isinstance(proba, pd.DataFrame)
    assert list(proba.columns) == [0, 1]
    assert proba.shape == (2, 2)


def test_agadapter_converts_series_to_two_column_dataframe():
    adapter = AGAdapter()
    adapter.predictor = _SeriesPredictor()
    X = pd.DataFrame({"a": [1, 2]})
    proba = adapter.predict_proba(X)
    assert isinstance(proba, pd.DataFrame)
    assert list(proba.columns) == [0, 1]
    # row sums ~ 1
    assert np.allclose(proba.sum(axis=1).to_numpy(), np.ones(2))


def test_agadapter_converts_array_to_dataframe_with_labels():
    adapter = AGAdapter()
    adapter.predictor = _ArrayPredictor()
    X = pd.DataFrame({"a": [1, 2]})
    proba = adapter.predict_proba(X)
    assert isinstance(proba, pd.DataFrame)
    assert list(proba.columns) == [0, 1]
    assert proba.shape == (2, 2)
