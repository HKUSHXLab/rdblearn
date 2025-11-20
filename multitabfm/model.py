from typing import Optional
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier, TabPFNRegressor


class CustomTabPFN:
    """Custom TabPFN model wrapper for classification and regression."""
    
    def __init__(self, model_path: str = None, task_type: str = "regression", **kwargs):
        self.model_path = model_path or "/root/autodl-tmp/tabpfn_2_5"
        self.task_type = task_type
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit the TabPFN model."""
        # Choose the appropriate TabPFN model based on task type
        if self.task_type == "regression":
            regressor_model_path = self.model_path + "/tabpfn-v2.5-regressor-v2.5_default.ckpt"
            self.model = TabPFNRegressor(model_path=regressor_model_path)
        else:
            classifier_model_path = self.model_path + "/tabpfn-v2.5-classifier-v2.5_default.ckpt"
            self.model = TabPFNClassifier(model_path=classifier_model_path)
        
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Predict class probabilities. Always returns DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
            
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)
        else:
            # For regression, return predictions as probabilities
            preds = self.model.predict(X)
            proba = np.column_stack([1 - preds, preds])  # Dummy probabilities
        
        # Ensure DataFrame output
        if isinstance(proba, pd.DataFrame):
            return proba
        else:
            # Convert ndarray to DataFrame
            arr = np.asarray(proba)
            if arr.ndim == 1:
                # Binary case
                return pd.DataFrame({0: 1 - arr, 1: arr})
            else:
                # Multi-class case
                n_classes = arr.shape[1]
                return pd.DataFrame(arr, columns=list(range(n_classes)))

    def predict(self, X: pd.DataFrame, **kwargs) -> pd.Series:
        """Predict target values. Always returns Series."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
            
        preds = self.model.predict(X, output_type="mean" if self.task_type == "regression" else None)
        
        # Ensure Series output
        if isinstance(preds, pd.Series):
            return preds
        else:
            return pd.Series(np.asarray(preds))
