import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from rdblearn.estimator import RDBLearnClassifier, RDBLearnRegressor
from rdblearn.config import RDBLearnConfig, TemporalDiffConfig
from rdblearn.constants import RDBLEARN_DEFAULT_CONFIG
from fastdfs import RDB, DFSConfig
from fastdfs.api import create_rdb

from loguru import logger
logger.enable("rdblearn")

class MockTabPFNClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, **kwargs):
        self.X_fit = X
        self.y_fit = y
        return self

    def predict(self, X, **kwargs):
        return np.random.randint(0, 2, size=len(X))

    def predict_proba(self, X, **kwargs):
        return np.random.rand(len(X), 2)

class MockTabPFNRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y, **kwargs):
        self.X_fit = X
        self.y_fit = y
        return self

    def predict(self, X, **kwargs):
        return np.random.rand(len(X))

class TestRDBLearnEstimator(unittest.TestCase):
    def setUp(self):
        # Create synthetic RDB
        # Users table
        self.users_df = pd.DataFrame({
            'user_id': range(100),
            'age': np.random.randint(18, 80, 100),
            'city': ['New York', 'London', 'Paris', 'Tokyo'] * 25
        })
        
        # Transactions table
        self.tx_df = pd.DataFrame({
            'tx_id': range(500),
            'user_id': np.random.randint(0, 100, 500),
            'amount': np.random.rand(500) * 100,
            'timestamp': pd.date_range('2023-01-01', periods=500, freq='H')
        })
        
        self.rdb = create_rdb(
            name="test_db",
            tables={
                "users": self.users_df,
                "transactions": self.tx_df
            },
            primary_keys={
                "users": "user_id",
                "transactions": "tx_id"
            },
            foreign_keys=[
                ("transactions", "user_id", "users", "user_id")
            ],
            time_columns={
                "transactions": "timestamp"
            }
        )
        
        # Create target dataframe (X)
        # We want to predict something for users at a specific time
        self.X_train = pd.DataFrame({
            'user_id': range(100),
            'cutoff_time': pd.date_range('2023-01-10', periods=100),
            'cat_col': ['a', 'b'] * 50 # Extra column in X
        })
        self.X_train['user_id'] = self.X_train['user_id'].astype(str)

        self.y_train_cls = pd.Series(np.random.randint(0, 2, 100), name='target')
        self.y_train_reg = pd.Series(np.random.rand(100), name='target')
        
        self.X_test = pd.DataFrame({
            'user_id': range(20),
            'cutoff_time': pd.date_range('2023-02-01', periods=20),
            'cat_col': ['a', 'c'] * 10
        })
        self.X_test['user_id'] = self.X_test['user_id'].astype(str)

        self.key_mappings = {'user_id': 'users.user_id'}
        self.cutoff_time_column = 'cutoff_time'

    def test_classifier_fit_predict(self):
        base_model = MockTabPFNClassifier()
        clf = RDBLearnClassifier(base_estimator=base_model)
        
        clf.fit(
            self.X_train, 
            self.y_train_cls, 
            rdb=self.rdb, 
            key_mappings=self.key_mappings, 
            cutoff_time_column=self.cutoff_time_column
        )
        
        # Check if base model was fitted
        self.assertTrue(hasattr(base_model, 'X_fit'))
        self.assertEqual(len(base_model.X_fit), 100)
        # Check if features were generated (more columns than original X)
        # Original X has 3 columns (user_id, cutoff_time, cat_col)
        # DFS should add features from transactions
        self.assertGreater(base_model.X_fit.shape[1], 3)
        
        # Predict
        preds = clf.predict(self.X_test)
        self.assertEqual(len(preds), 20)
        
        # Predict proba
        proba = clf.predict_proba(self.X_test)
        self.assertEqual(len(proba), 20)
        self.assertEqual(proba.shape[1], 2)

    def test_regressor_fit_predict(self):
        base_model = MockTabPFNRegressor()
        reg = RDBLearnRegressor(base_estimator=base_model)
        
        reg.fit(
            self.X_train, 
            self.y_train_reg, 
            rdb=self.rdb, 
            key_mappings=self.key_mappings, 
            cutoff_time_column=self.cutoff_time_column
        )
        
        # Check if base model was fitted
        self.assertTrue(hasattr(base_model, 'X_fit'))
        self.assertGreater(base_model.X_fit.shape[1], 3)
        
        # Predict
        preds = reg.predict(self.X_test)
        self.assertEqual(len(preds), 20)

    def test_downsampling(self):
        base_model = MockTabPFNClassifier()
        config = RDBLearnConfig(max_train_samples=50)
        clf = RDBLearnClassifier(base_estimator=base_model, config=config)
        
        clf.fit(
            self.X_train, 
            self.y_train_cls, 
            rdb=self.rdb, 
            key_mappings=self.key_mappings, 
            cutoff_time_column=self.cutoff_time_column
        )
        
        # Check if base model was fitted with downsampled data
        self.assertEqual(len(base_model.X_fit), 50)
        self.assertEqual(len(base_model.y_fit), 50)

    def test_config_passing(self):
        base_model = MockTabPFNClassifier()
        config_dict = {
            "dfs": {"max_depth": 1},
            "max_train_samples": 80
        }
        clf = RDBLearnClassifier(base_estimator=base_model, config=config_dict)
        
        self.assertIsInstance(clf.config, RDBLearnConfig)
        self.assertEqual(clf.config.max_train_samples, 80)
        self.assertIsNotNone(clf.config.dfs)

    def test_config_override(self):
        # Test 1: Default config
        base_model = MockTabPFNClassifier()
        clf = RDBLearnClassifier(base_estimator=base_model)
        self.assertEqual(clf.config.max_train_samples, RDBLEARN_DEFAULT_CONFIG["max_train_samples"])
        self.assertEqual(clf.config.dfs.max_depth, RDBLEARN_DEFAULT_CONFIG["dfs"]["max_depth"])

        # Test 2: Partial override
        override_config = {"max_train_samples": 500}
        clf = RDBLearnClassifier(base_estimator=base_model, config=override_config)
        
        # Check overridden value
        self.assertEqual(clf.config.max_train_samples, 500)
        
        # Check preserved default value (nested)
        self.assertEqual(clf.config.dfs.max_depth, RDBLEARN_DEFAULT_CONFIG["dfs"]["max_depth"])

    def test_classifier_with_temporal_diff(self):
        """Test classifier with temporal diff feature enabled."""
        base_model = MockTabPFNClassifier()
        temporal_config = TemporalDiffConfig(enabled=True, exclude_columns=[])
        config = RDBLearnConfig(
            temporal_diff=temporal_config,
            dfs=DFSConfig(
                agg_primitives=["max", "min", "mean", "std", "count"],
                max_depth=2
            )
        )
        clf = RDBLearnClassifier(base_estimator=base_model, config=config)

        clf.fit(
            self.X_train,
            self.y_train_cls,
            rdb=self.rdb,
            key_mappings=self.key_mappings,
            cutoff_time_column=self.cutoff_time_column
        )

        self.assertIsNotNone(clf.preprocessor_.temporal_transformer)
        self.assertTrue(clf.preprocessor_.temporal_transformer.is_fitted_)

        # Verify that expected diff columns are present after fitting
        x_fit_cols = clf.base_estimator.X_fit.columns
        self.assertTrue(any(c.endswith('_diff') for c in x_fit_cols))
        self.assertFalse(any('timestamp_epochtime' in c and 'diff' not in c for c in x_fit_cols))

        preds = clf.predict(self.X_test)
        self.assertEqual(len(preds), 20)


if __name__ == '__main__':
    unittest.main()
