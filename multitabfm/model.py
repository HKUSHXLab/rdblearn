from typing import Optional
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

class ModelPredictor:
    """Base class for model training and prediction."""
    
    def fit(self, X: pd.DataFrame, y: pd.Series, label_column: str, *, config: Optional[dict] = None):
        """Fit the model on training data.
        
        Args:
            X: Feature dataframe
            y: Target series
            label_column: Name of the label column
            config: Optional model configuration
        """
        raise NotImplementedError()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of prediction probabilities
        """
        raise NotImplementedError()


class AGAdapter(ModelPredictor):
    """AutoGluon TabularPredictor adapter based on ag_dfs_experiment.py."""

    def __init__(self):
        self.predictor = None
        
    def fit(self, X: pd.DataFrame, label_column: str, config: Optional[dict] = None):
        # Default config based on ag_dfs_experiment.py
        default_config = {
            "hyperparameters": {
                "TABPFNV2": {
                    "random_state": 42,
                    "n_jobs": -1,
                    "n_estimators": 8,
                    "inference_config": {
                        "SUBSAMPLE_SAMPLES": 10000,
                    },
                    "ignore_pretraining_limits": True,
                    "ag.max_rows": 20000,
                    "ag.max_features": 600
                }
            },
            "ag_args_fit": {"ag.max_memory_usage_ratio": 1.2},
            "num_bag_folds": 0,
            "num_bag_sets": None,
            "num_stack_levels": 0,
        }
        
        # Update with provided config
        if config:
            default_config.update(config)
            
        # Setup feature generator
        feature_generator = AutoMLPipelineFeatureGenerator(
            enable_datetime_features=True,
            enable_raw_text_features=False,
            enable_text_special_features=False,
            enable_text_ngram_features=False,
        )
        
        
        # Create and train predictor
        self.predictor = TabularPredictor(label=label_column).fit(
            train_data= X,
            feature_generator=feature_generator,
            **default_config
        )
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.predictor is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        proba = self.predictor.predict_proba(X)
        
        # Return positive class probabilities
        if proba.shape[1] == 2:
            return proba.iloc[:, 1].values
        return proba.iloc[:, 0].values
