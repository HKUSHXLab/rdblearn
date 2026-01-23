#!/usr/bin/env python3
"""
WandB Sweep agent script for RDBLearn experiments.

This script is called by wandb agent for each experiment configuration.
It uses the RDBLearn framework with fit/predict pattern instead of train_and_predict.

Usage:
    # Create sweep and run agent
    wandb sweep config/sweep_config.yaml
    CUDA_VISIBLE_DEVICES=0 wandb agent <entity>/<project>/<sweep_id>
"""

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

# Add parent directory to path for rdblearn imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, r2_score

from rdblearn.datasets import RDBDataset
from rdblearn.estimator import RDBLearnClassifier, RDBLearnRegressor


# ============================================================================
# Custom Model Wrappers (moved from rdblearn.models)
# ============================================================================

# LimiX root path - adjust as needed
LIMIX_ROOT = Path("/root/yl_project/LimiX")


class CustomTabPFN:
    """Custom TabPFN model wrapper for classification and regression.

    Uses TabPFN's built-in subsample API for efficient ensemble diversity.
    """

    def __init__(
        self,
        model_path: str = None,
        task_type: str = "regression",
        n_estimators: int = 8,
        device: str = "cuda",
        n_preprocessing_jobs: int = -1,
        eval_metric: str = None,
        **kwargs
    ):
        if model_path is None:
            raise ValueError("CustomTabPFN requires a model_path pointing to the TabPFN checkpoint.")
        self.model_path = model_path
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.device = device
        self.n_preprocessing_jobs = n_preprocessing_jobs
        self.eval_metric = eval_metric
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit the TabPFN model."""
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        # Choose the appropriate TabPFN model based on task type
        if self.task_type == "regression":
            self.model = TabPFNRegressor(
                model_path=self.model_path,
                n_estimators=self.n_estimators,
                ignore_pretraining_limits=True,  # Allow datasets larger than 10K samples
                device=self.device,
                n_preprocessing_jobs=self.n_preprocessing_jobs,
            )
        else:
            self.model = TabPFNClassifier(
                model_path=self.model_path,
                n_estimators=self.n_estimators,
                ignore_pretraining_limits=True,  # Allow datasets larger than 10K samples
                device=self.device,
                n_preprocessing_jobs=self.n_preprocessing_jobs
            )

        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict class probabilities. Always returns numpy array."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)
        else:
            # For regression, return predictions as probabilities
            preds = self.model.predict(X)
            proba = np.column_stack([1 - preds, preds])  # Dummy probabilities

        # Ensure numpy array output
        if isinstance(proba, pd.DataFrame):
            return proba.to_numpy()
        else:
            arr = np.asarray(proba)
            if arr.ndim == 1:
                # Binary case - convert to 2D
                return np.column_stack([1 - arr, arr])
            return arr

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict target values. Always returns numpy array."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        output_type = None
        if self.task_type == "regression":
            output_type = "main"

        preds = self.model.predict(X, output_type=output_type)

        # Ensure numpy array output
        if isinstance(preds, dict):
            return preds
        elif isinstance(preds, (pd.Series, pd.DataFrame)):
            return preds.to_numpy().ravel()
        else:
            return np.asarray(preds).ravel()


class CustomLimiX:
    """Wrapper around LimiXPredictor to expose a sklearn-like API."""

    def __init__(
        self,
        task_type: str = "classification",
        model_path: Union[str, Path] = None,
        limix_root: Union[str, Path] = LIMIX_ROOT,
        inference_config: Optional[Union[str, Path]] = None,
        normalize_target: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        eval_metric: str = None,  # Accept but don't use - LimiX doesn't support eval_metric
        **predictor_kwargs,
    ) -> None:
        task_type = task_type.lower()
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'.")

        if model_path is None:
            raise ValueError("CustomLimiX requires a model_path pointing to a LimiX checkpoint.")

        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device is required for CustomLimiX but was not found.")
            device = torch.device("cuda")
        else:
            device = torch.device(device)
            if device.type != "cuda":
                raise ValueError("CustomLimiX currently supports CUDA devices only.")

        self.task_type = task_type
        self.normalize_target = normalize_target and task_type == "regression"
        self.device = device
        self.limix_root = Path(limix_root)
        self.model_path = self._resolve_model_path(model_path)

        if inference_config is None:
            config_name = "config/cls_default_noretrieval.json" if task_type == "classification" else "config/reg_default_noretrieval.json"
            inference_config = self.limix_root / config_name
        self.inference_config = str(inference_config)

        self.predictor_kwargs = predictor_kwargs
        self.model = None
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_processed_: Optional[np.ndarray] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None
        self._is_fitted = False

    def _resolve_model_path(self, model_path: Union[str, Path]) -> str:
        resolved = Path(model_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Provided model_path does not exist: {resolved}")
        return str(resolved)

    def _load_limix_predictor(self):
        """Lazily load LimiXPredictor to avoid import issues."""
        if self.limix_root.exists() and str(self.limix_root) not in sys.path:
            sys.path.append(str(self.limix_root))

        try:
            from inference.predictor import LimiXPredictor
            return LimiXPredictor
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"LimiX repository is required for CustomLimiX. Expected at {self.limix_root}."
            ) from exc

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **_) -> "CustomLimiX":
        """Store training data so LimiX can perform retrieval-based inference later."""

        X_arr, _ = self._prepare_features(X)
        y_arr = self._prepare_targets(y)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")

        if self.task_type == "regression" and self.normalize_target:
            self.y_mean_ = float(y_arr.mean())
            self.y_std_ = float(y_arr.std() or 1.0)
            if np.isclose(self.y_std_, 0.0):
                self.normalize_target = False
                y_processed = y_arr.astype(np.float32)
            else:
                y_processed = ((y_arr - self.y_mean_) / self.y_std_).astype(np.float32)
        else:
            y_processed = y_arr

        self.X_train_ = np.asarray(X_arr)
        self.y_train_processed_ = np.asarray(y_processed)

        if self.model is None:
            LimiXPredictor = self._load_limix_predictor()
            self.model = LimiXPredictor(
                device=self.device,
                model_path=self.model_path,
                inference_config=self.inference_config,
                **self.predictor_kwargs,
            )

        self._is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray], **_) -> np.ndarray:
        """Predict target values. Always returns numpy array."""
        self._ensure_fitted()
        if self.task_type == "classification":
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)

        X_arr, index = self._prepare_features(X)
        preds = self._run_inference(X_arr)
        preds = preds.reshape(-1)
        if self.task_type == "regression" and self.normalize_target:
            preds = preds * self.y_std_ + self.y_mean_
        return preds

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray], **_) -> np.ndarray:
        """Predict class probabilities. Always returns numpy array."""
        self._ensure_fitted()
        if self.task_type != "classification":
            raise NotImplementedError("predict_proba is only defined for classification tasks in CustomLimiX.")

        X_arr, index = self._prepare_features(X)
        proba = self._run_inference(X_arr)
        return np.asarray(proba)

    def _run_inference(self, X_new: np.ndarray) -> np.ndarray:
        task = "Classification" if self.task_type == "classification" else "Regression"
        result = self.model.predict(self.X_train_, self.y_train_processed_, X_new, task_type=task)
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()
        return np.asarray(result)

    def _prepare_features(self, X: Union[pd.DataFrame, np.ndarray]):
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(), X.index
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr, pd.RangeIndex(arr.shape[0])

    def _prepare_targets(self, y: Union[pd.Series, np.ndarray]):
        if isinstance(y, pd.Series):
            return y.to_numpy().ravel()
        arr = np.asarray(y)
        if arr.ndim != 1:
            return arr.reshape(-1)
        return arr

    def _ensure_fitted(self) -> None:
        if not self._is_fitted or self.X_train_ is None or self.y_train_processed_ is None or self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")


# ============================================================================
# Configuration and Helper Functions
# ============================================================================

# Model paths configuration
MODEL_PATHS = {
    "tabpfn_v2": {
        "classification": "/root/.cache/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
        "regression": "/root/.cache/tabpfn/tabpfn-v2-regressor.ckpt",
    },
    "tabpfn_v2.5": {
        "classification": "/root/autodl-tmp/tabpfn_2_5/tabpfn-v2.5-classifier-v2.5_default.ckpt",
        "regression": "/root/autodl-tmp/tabpfn_2_5/tabpfn-v2.5-regressor-v2.5_default.ckpt",
    },
    "limix": "/root/autodl-tmp/limix/cache/LimiX-16M.ckpt",
}


# Default DFS configuration
DEFAULT_DFS_CONFIG = {
    "max_depth": 2,
}


def get_base_estimator(model_pair_name: str, task_type_str: str):
    """
    Create base estimator based on model pair and task type.
    
    Args:
        model_pair_name: One of "tabpfn_v2", "tabpfn_v2.5", "limix"
        task_type_str: Either "classification" or "regression"
        
    Returns:
        A sklearn-compatible estimator instance
    """
    if model_pair_name in ["tabpfn_v2", "tabpfn_v2.5"]:
        model_path = MODEL_PATHS[model_pair_name][task_type_str]
        return CustomTabPFN(
            model_path=model_path,
            task_type=task_type_str,
            n_estimators=8,
            device="cuda",
        )
    
    elif model_pair_name == "limix":
        # Use CustomLimiX (same checkpoint for both tasks)
        return CustomLimiX(
            model_path=MODEL_PATHS["limix"],
            task_type=task_type_str,
            device="cuda",
        )
    
    else:
        raise ValueError(
            f"Unknown model_pair_name: {model_pair_name}. "
            f"Supported: 'tabpfn_v2', 'tabpfn_v2.5', 'limix'"
        )


def compute_metrics(y_true, y_pred, task_type: str, metric_name: str = None):
    """
    Compute evaluation metrics.
    
    Args:
        y_true: Ground truth labels/values
        y_pred: Predicted labels/values (or probabilities for classification)
        task_type: "classification" or "regression"
        metric_name: Optional specific metric to compute
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    if task_type == "classification":
        # For binary classification with predict_proba output
        if hasattr(y_pred, 'ndim') and y_pred.ndim == 2:
            if y_pred.shape[1] == 2:
                y_scores = y_pred[:, 1]
            else:
                y_scores = y_pred[:, 1] if y_pred.shape[1] > 1 else y_pred.ravel()
        elif isinstance(y_pred, pd.DataFrame):
            if y_pred.shape[1] == 2:
                y_scores = y_pred.iloc[:, 1].values
            else:
                y_scores = y_pred.iloc[:, 1].values if y_pred.shape[1] > 1 else y_pred.values.ravel()
        else:
            y_scores = np.asarray(y_pred).ravel()
        
        try:
            auc = roc_auc_score(y_true, y_scores)
            metrics["roc_auc"] = auc
            metrics["test_score"] = auc
        except Exception as e:
            print(f"Warning: Could not compute ROC AUC: {e}")
            # Fall back to accuracy
            y_pred_labels = (y_scores > 0.5).astype(int)
            acc = accuracy_score(y_true, y_pred_labels)
            metrics["accuracy"] = acc
            metrics["test_score"] = acc
    else:
        # Regression metrics
        if isinstance(y_pred, dict):
            assert "mean" in y_pred and "median" in y_pred, "y_pred must contain 'mean' and 'median' keys"
            print("Prediction is a dictionary including 'mean' and 'median' keys")
            y_pred_mean = y_pred["mean"]
            y_pred_median = y_pred["median"]

            y_pred_mean_values = np.asarray(y_pred_mean).ravel()
            y_pred_median_values = np.asarray(y_pred_median).ravel()
            y_true_values = np.asarray(y_true).ravel()

            # Calculate MSE and RMSE for mean prediction
            mse = mean_squared_error(y_true_values, y_pred_mean_values)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_values, y_pred_mean_values)

            mae = np.mean(np.abs(y_true_values - y_pred_median_values))

            metrics["r2"] = r2
            metrics["mse"] = mse
            metrics["rmse"] = rmse
            metrics["mae"] = mae
            # Use MAE for test_score
            metrics["test_score"] = mae

        else:
            y_pred_values = np.asarray(y_pred).ravel()
            y_true_values = np.asarray(y_true).ravel()
            
            mse = mean_squared_error(y_true_values, y_pred_values)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_values, y_pred_values)

            mae = np.mean(np.abs(y_true_values - y_pred_values))

            metrics["r2"] = r2
            metrics["mse"] = mse
            metrics["rmse"] = rmse
            metrics["mae"] = mae
            # Use MAE for test_score
            metrics["test_score"] = mae
    
    return metrics


def load_dataset(adapter_type: str, dataset_name: str, task_name: str):
    """
    Load dataset using remote download.

    Args:
        adapter_type: "4dbinfer" or "relbench"
        dataset_name: Name of the dataset
        task_name: Name of the task to load

    Returns:
        RDBDataset instance
    """
    print(f"Loading dataset via remote adapter: {adapter_type}, dataset: {dataset_name}")
    if adapter_type == "4dbinfer":
        dataset = RDBDataset.from_4dbinfer(dataset_name)
    elif adapter_type == "relbench":
        dataset = RDBDataset.from_relbench(dataset_name)
    else:
        raise ValueError(
            f"Unknown adapter type: {adapter_type}. "
            "Supported: '4dbinfer', 'relbench'"
        )

    return dataset


def main(engine_path: str):
    """Main entry point for sweep agent."""
    # Initialize wandb - config comes from sweep controller
    run = wandb.init()
    config = wandb.config
    
    # Parse sweep parameters
    model_pair_name = config.model_config_pair
    data_pair_str = config.data_pair
    use_dfs = config.get("use_dfs", True)
    max_train_samples = config.get("max_train_samples", 10000)
    dfs_max_depth = config.get("dfs_max_depth", 2)
    
    # Split data pair: "adapter_type::dataset_name::task_name"
    # For relbench: "relbench::rel-f1::driver-dnf"
    # For 4dbinfer: "4dbinfer::stackexchange::task_name"
    parts = data_pair_str.split("::")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid data_pair format: {data_pair_str}. "
            "Expected format: 'adapter_type::dataset_name::task_name' "
            "(e.g., 'relbench::rel-f1::driver-dnf' or '4dbinfer::stackexchange::votes')"
        )
    adapter_type, dataset_name, task_name = parts
    
    # Validate model pair
    valid_models = ["tabpfn_v2", "tabpfn_v2.5", "limix"]
    if model_pair_name not in valid_models:
        raise ValueError(
            f"Unknown model_config_pair: {model_pair_name}. "
            f"Available: {valid_models}"
        )
    
    # Load Dataset using RDBDataset (local path preferred if available)
    dataset = load_dataset(adapter_type, dataset_name, task_name)
    
    rdb = dataset.rdb
    
    # Get task from dataset
    if task_name not in dataset.tasks:
        available_tasks = list(dataset.tasks.keys())
        raise ValueError(
            f"Task '{task_name}' not found in dataset. "
            f"Available tasks: {available_tasks}"
        )
    task = dataset.tasks[task_name]
    
    # Determine task type
    task_type_str = task.metadata.task_type
    if task_type_str is None:
        task_type_str = "classification"
        print(f"Warning: Task type not specified, defaulting to {task_type_str}")
    elif task_type_str not in ("classification", "regression"):
        # Handle enum values from relbench
        task_type_lower = str(task_type_str).lower()
        if "classification" in task_type_lower:
            task_type_str = "classification"
        elif "regression" in task_type_lower:
            task_type_str = "regression"
        else:
            task_type_str = "classification"
            print(f"Warning: Unknown task type '{task_type_str}', defaulting to classification")
    
    print(f"Task type: {task_type_str}")
    
    # Create base estimator
    print(f"Creating base estimator: {model_pair_name} ({task_type_str})")
    base_estimator = get_base_estimator(model_pair_name, task_type_str)
    
    # Create RDBLearn estimator
    if task_type_str == "classification":
        EstimatorClass = RDBLearnClassifier
    else:
        EstimatorClass = RDBLearnRegressor
    
    # Build config for RDBLearn estimator
    rdblearn_config = {
        "max_train_samples": max_train_samples,
    }
    
    if use_dfs:
        rdblearn_config["dfs"] = {"max_depth": dfs_max_depth, "engine_path": engine_path}
    
    clf = EstimatorClass(
        base_estimator=base_estimator,
        config=rdblearn_config,
    )
    
    # Prepare training data
    target_col = task.metadata.target_col
    X_train = task.train_df.drop(columns=[target_col])
    y_train = task.train_df[target_col]

    X_dev = task.val_df.drop(columns=[target_col])
    y_dev = task.val_df[target_col]
    
    # Prepare test data
    X_test = task.test_df.drop(columns=[target_col])
    y_test = task.test_df[target_col]
    
    # Update wandb config with additional metadata
    wandb.config.update({
        "adapter_type": adapter_type,
        "dataset_name": dataset_name,
        "task_name": task_name,
        "task_type": task_type_str,
        "model_name": model_pair_name,
        "train_samples": len(X_train),
        "dev_samples": len(X_dev),
        "test_samples": len(X_test),
        "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }, allow_val_change=True)
    
    # Run the experiment
    print(f"\n{'='*60}")
    print(f"Running experiment: {dataset_name}/{task_name}")
    print(f"Model: {model_pair_name} ({task_type_str})")
    print(f"DFS: {'enabled (depth=' + str(dfs_max_depth) + ')' if use_dfs else 'disabled'}")
    print(f"Max train samples: {max_train_samples}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set')}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"{'='*60}\n")
    
    try:
        # Fit with timing
        start_fit = time.time()
        clf.fit(
            X=X_train,
            y=y_train,
            rdb=rdb,
            key_mappings=task.metadata.key_mappings,
            cutoff_time_column=task.metadata.time_col,
        )
        fit_seconds = time.time() - start_fit
        
        # Predict with timing
        start_dev_predict = time.time()
        if task_type_str == "classification":
            dev_preds = clf.predict_proba(X=X_dev)
        else:
            dev_preds = clf.predict(X=X_dev)
        dev_predict_seconds = time.time() - start_dev_predict
        start_test_predict = time.time()

        if task_type_str == "classification":
            test_preds = clf.predict_proba(X=X_test)
        else:
            test_preds = clf.predict(X=X_test)
        test_predict_seconds = time.time() - start_test_predict
        
        total_seconds = fit_seconds + dev_predict_seconds + test_predict_seconds
        
        # Compute metrics
        dev_metrics = compute_metrics(
            y_dev, dev_preds, task_type_str, 
            task.metadata.evaluation_metric
        )
        test_metrics = compute_metrics(
            y_test, test_preds, task_type_str, 
            task.metadata.evaluation_metric
        )

        # Log timing metrics
        wandb.log({
            "fit_seconds": fit_seconds,
            "dev_predict_seconds": dev_predict_seconds,
            "test_predict_seconds": test_predict_seconds,
            "total_seconds": total_seconds,
            "num_dev_predictions": len(dev_preds) if dev_preds is not None else 0,
            "num_test_predictions": len(test_preds) if test_preds is not None else 0,
        })
        
        # Log evaluation metrics
        for metric_name, metric_value in dev_metrics.items():
            wandb.log({f"dev_metric/{metric_name}": metric_value})
        
        for metric_name, metric_value in test_metrics.items():
            wandb.log({f"test_metric/{metric_name}": metric_value})

        # Set primary metric in summary for sweep comparison
        primary_metric = test_metrics.get("test_score")
        if primary_metric is not None:
            wandb.summary["metric/test_score"] = primary_metric
        
        wandb.summary["status"] = "success"
        
        print(f"\n✓ Experiment completed successfully!")
        print(f"  Fit time: {fit_seconds:.2f}s")
        print(f"  Dev predict time: {dev_predict_seconds:.2f}s")
        print(f"  Test predict time: {test_predict_seconds:.2f}s")
        print(f"  Total time: {total_seconds:.2f}s")
        print(f"  Dev metrics: {dev_metrics}")
        print(f"  Test metrics: {test_metrics}")
        
        return {"status": "success", "dev_metrics": dev_metrics, "test_metrics": test_metrics}
        
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        wandb.summary["status"] = "failed"
        wandb.summary["error"] = error_msg
        wandb.log({"error": error_msg})
        
        print(f"\n✗ Experiment failed!")
        print(f"  Error: {error_msg}")
        print(f"  Traceback:\n{error_traceback}")
        
        # Re-raise to signal failure to wandb
        raise


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(dir="/root/autodl-tmp") as temp_file:
        main(temp_file.name)