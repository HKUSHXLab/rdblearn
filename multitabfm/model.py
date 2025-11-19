from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

class ModelPredictor:
    """Base class for model training and prediction."""
    
    def fit(self, X: pd.DataFrame, label_column: str, config: Optional[dict] = None):
        """Fit the model on training data.
        
        Contract:
            - X must include the label column specified by `label_column`.
            - Implementations should split X into features/labels internally.

        Args:
            X: Feature dataframe including the label column
            label_column: Name of the label column in X
            config: Optional model configuration
        """
        raise NotImplementedError()

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities.
        
        Args:
            X: Feature dataframe
            
        Returns:
            DataFrame of prediction probabilities with class labels as columns.
        """
        raise NotImplementedError()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict target values (for regression) or class labels.

        Default implementation raises. Implement in adapters that support regression.

        Args:
            X: Feature dataframe

        Returns:
            Series of predictions.
        """
        raise NotImplementedError()


class AGAdapter(ModelPredictor):
    """AutoGluon TabularPredictor adapter based on ag_dfs_experiment.py."""

    def __init__(self, model_config: Optional[dict] = None):
        self.predictor = None
        self.problem_type: Optional[str] = None  # 'binary', 'multiclass', 'regression', or None
        self.model_config = {
            "hyperparameters": {
                "TABPFNV2": {
                    "random_state": 42,
                    "n_jobs": -1,
                    "n_estimators": 8,
                    "inference_config": {
                        "SUBSAMPLE_SAMPLES": 2048,
                    },
                    "ignore_pretraining_limits": True,
                    # Align with test_autogluon to avoid unexpected safety cutoffs
                    "ag.max_rows": None,
                    "ag.max_features": None,
                    "ag.max_memory_usage_ratio": None,
                }
            },
            "num_bag_folds": 0,
            "num_bag_sets": None,
            "num_stack_levels": 0,
        }
        self.predictor_kwargs: dict = {}
        if model_config:
            # Deep merge dictionaries
            def deep_merge(d1, d2):
                for k, v in d2.items():
                    if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                        d1[k] = deep_merge(d1[k], v)
                    else:
                        d1[k] = v
                return d1
            predictor_kwargs = model_config.get("predictor_kwargs", {}) or {}
            self.predictor_kwargs.update(predictor_kwargs)

            # Allow simple aliases for convenience
            for key in ("path", "predictor_path", "model_output_path", "output_path"):
                if key in model_config and model_config[key]:
                    self.predictor_kwargs.setdefault("path", model_config[key])

            filtered_config = {
                k: v for k, v in model_config.items()
                if k not in {"predictor_kwargs", "path", "predictor_path", "model_output_path", "output_path"}
            }
            deep_merge(self.model_config, filtered_config)

    def fit(self, X: pd.DataFrame, label_column: str, task_type: Optional[str] = None, eval_metric: Optional[str] = None) -> None:
        # Setup feature generator
        feature_generator = AutoMLPipelineFeatureGenerator(
            enable_datetime_features=True,
            enable_raw_text_features=False,
            enable_text_special_features=False,
            enable_text_ngram_features=False,
        )

        # Create and train predictor
        predictor_init_kwargs = {
            "label": label_column,
            "problem_type": task_type,
            "eval_metric": eval_metric,
        }
        predictor_init_kwargs.update(self.predictor_kwargs)

        model_path = predictor_init_kwargs.get("path")
        if model_path:
            Path(model_path).mkdir(parents=True, exist_ok=True)  # Ensure AutoGluon can write here

        self.predictor = TabularPredictor(**predictor_init_kwargs).fit(
            train_data=X,
            feature_generator=feature_generator,
            **self.model_config,
        )
    
    def predict_proba(self, X: pd.DataFrame, batch_size: int = 5000) -> pd.DataFrame:
        if self.predictor is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        # Process in batches to avoid CUDA memory issues
        proba_batches = []
        num_rows = len(X)
        
        for start in range(0, num_rows, batch_size):
            end = min(start + batch_size, num_rows)
            batch = X.iloc[start:end]
            proba_batch = self.predictor.predict_proba(batch)
            proba_batches.append(proba_batch)
        
        # Concatenate all batches
        if len(proba_batches) == 1:
            proba = proba_batches[0]
        else:
            proba = pd.concat(proba_batches, axis=0).reset_index(drop=True)

        # Normalize to DataFrame output with class labels as columns
        if isinstance(proba, pd.Series):
            # Assume binary positive-class probabilities; expand to two columns [0,1]
            p1 = proba.to_numpy()
            p0 = 1 - p1
            return pd.DataFrame({0: p0, 1: p1})
        if isinstance(proba, pd.DataFrame):
            return proba

        # Fallback: construct DataFrame from ndarray
        arr = np.asarray(proba)
        if arr.ndim == 1:
            # Assume binary probabilities of positive class; create [0,1]
            p1 = arr
            p0 = 1 - p1
            return pd.DataFrame({0: p0, 1: p1})
        n_classes = arr.shape[1]
        class_labels = None
        try:
            class_labels = list(self.predictor.class_labels)
        except Exception:
            class_labels = list(range(n_classes))
        return pd.DataFrame(arr, columns=class_labels)

    def predict(self, X: pd.DataFrame, batch_size: int = 5000) -> pd.Series:
        if self.predictor is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        # Process in batches to avoid CUDA memory issues
        pred_batches = []
        num_rows = len(X)
        
        for start in range(0, num_rows, batch_size):
            end = min(start + batch_size, num_rows)
            batch = X.iloc[start:end]
            pred_batch = self.predictor.predict(batch)
            pred_batches.append(pred_batch)
        
        # Concatenate all batches
        if len(pred_batches) == 1:
            preds = pred_batches[0]
        else:
            preds = pd.concat(pred_batches, axis=0).reset_index(drop=True)
        
        # Ensure pandas Series output
        if isinstance(preds, pd.Series):
            return preds
        return pd.Series(np.asarray(preds))
