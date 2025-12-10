from typing import Optional, Tuple, Union
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tabpfn import TabPFNClassifier, TabPFNRegressor
from autogluon.tabular import TabularDataset, TabularPredictor


LIMIX_ROOT = Path("/root/yl_project/LimiX")

if LIMIX_ROOT.exists() and str(LIMIX_ROOT) not in sys.path:
    sys.path.append(str(LIMIX_ROOT))

try:
    from inference.predictor import LimiXPredictor  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - surface clear error for missing repo
    raise ImportError(
        "LimiX repository is required for CustomLimiX. Expected at /root/yl_project/LimiX."
    ) from exc


class CustomTabPFN:
    """Custom TabPFN model wrapper for classification and regression.
    
    Uses TabPFN's built-in subsample API for efficient ensemble diversity.
    """
    
    def __init__(
        self,
        model_path: str = None,
        task_type: str = "regression",
        max_samples: int = None,
        n_estimators: int = 8,
        device: str = "cuda",
        n_preprocessing_jobs: int = -1,
        eval_metric: str = None,
        **kwargs
    ):
        if model_path is None:
             raise ValueError("CustomTabPFN requires a model_path pointing to the TabPFN directory.")
        self.model_path = model_path
        self.task_type = task_type
        self.max_samples = max_samples  # Maximum samples per estimator (handled by TabPFN internally)
        self.n_estimators = n_estimators
        self.device = device
        self.n_preprocessing_jobs = n_preprocessing_jobs
        self.eval_metric = eval_metric
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit the TabPFN model using its built-in subsample mechanism.
        
        Each of the n_estimators will independently subsample max_samples from the data,
        providing ensemble diversity without manual preprocessing.
        """
        # Configure inference_config for TabPFN's internal subsampling
        inference_config = {}
        if self.max_samples is not None:
            # Let TabPFN handle subsampling internally for each estimator
            inference_config['SUBSAMPLE_SAMPLES'] = self.max_samples
            print(f"TabPFN will subsample {self.max_samples} samples per estimator from {len(X)} total samples")
        
        # Choose the appropriate TabPFN model based on task type
        if self.task_type == "regression":
            self.model = TabPFNRegressor(
                model_path=self.model_path,
                n_estimators=self.n_estimators,
                ignore_pretraining_limits=True,  # Allow datasets larger than 10K samples
                inference_config=inference_config if inference_config else None,
                device=self.device,
                n_preprocessing_jobs=self.n_preprocessing_jobs
            )
        else:
            self.model = TabPFNClassifier(
                model_path=self.model_path,
                n_estimators=self.n_estimators,
                ignore_pretraining_limits=True,  # Allow datasets larger than 10K samples
                inference_config=inference_config if inference_config else None,
                device=self.device,
                n_preprocessing_jobs=self.n_preprocessing_jobs
            )
        
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
            
        output_type = None
        if self.task_type == "regression":
            output_type = "median"
            # if self.eval_metric == "mae":
            #     output_type = "median"
            # else:
            #     output_type = "mean"

        preds = self.model.predict(X, output_type=output_type)
        
        # Ensure Series output
        if isinstance(preds, pd.Series):
            return preds
        else:
            return pd.Series(np.asarray(preds))

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
        max_samples: Optional[int] = None,
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
        self.max_samples = max_samples

        if inference_config is None:
            # config_name = "config/cls_default_16M_retrieval.json" if task_type == "classification" else "config/reg_default_16M_retrieval.json"
            config_name = "config/cls_default_noretrieval.json" if task_type == "classification" else "config/reg_default_noretrieval.json"
            inference_config = self.limix_root / config_name
        self.inference_config = str(inference_config)

        self.predictor_kwargs = predictor_kwargs
        self.model: Optional[LimiXPredictor] = None
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

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **_) -> None:
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
            self.model = LimiXPredictor(
                device=self.device,
                model_path=self.model_path,
                inference_config=self.inference_config,
                **self.predictor_kwargs,
            )

        self._is_fitted = True

    def predict(self, X: Union[pd.DataFrame, np.ndarray], **_) -> pd.Series:
        self._ensure_fitted()
        if self.task_type == "classification":
            proba = self.predict_proba(X)
            return proba.idxmax(axis=1)

        X_arr, index = self._prepare_features(X)
        preds = self._run_inference(X_arr)
        preds = preds.reshape(-1)
        if self.task_type == "regression" and self.normalize_target:
            preds = preds * self.y_std_ + self.y_mean_
        return pd.Series(preds, index=index)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray], **_) -> pd.DataFrame:
        self._ensure_fitted()
        if self.task_type != "classification":
            raise NotImplementedError("predict_proba is only defined for classification tasks in CustomLimiX.")

        X_arr, index = self._prepare_features(X)
        proba = self._run_inference(X_arr)
        classes = getattr(self.model, "classes", None)
        if classes is not None:
            columns = list(classes)
        else:
            columns = list(range(proba.shape[1]))
        return pd.DataFrame(proba, columns=columns, index=index)

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

class AutoGluon:
    """Wrapper around AutoGluon TabularPredictor with full model suite."""
    
    # Mapping of common metric names to AutoGluon's expected names
    METRIC_MAPPING = {
        'auroc': 'roc_auc',
        'auc': 'roc_auc',
        'roc': 'roc_auc',
        'mae': 'mean_absolute_error',
        'mse': 'mean_squared_error',
        'rmse': 'root_mean_squared_error',
        'r2': 'r2',
    }
    
    def __init__(
        self,
        task_type: str = "classification",
        eval_metric: str = None,
        time_limit: int = None,
        use_ensembling: bool = True,
        use_feature_generator: bool = True,
        presets: str = None,
        hyperparameters: dict = None,
        num_gpus: int = 0,
        verbosity: int = 2,
        **kwargs
    ):
        self.task_type = task_type.lower()
        if self.task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'.")
        
        self.problem_type = "binary" if self.task_type == "classification" else "regression"
        
        # Map metric names to AutoGluon's expected format
        if eval_metric and eval_metric.lower() in self.METRIC_MAPPING:
            self.eval_metric = self.METRIC_MAPPING[eval_metric.lower()]
            print(f"Mapped eval_metric '{eval_metric}' to AutoGluon's '{self.eval_metric}'")
        else:
            self.eval_metric = eval_metric
        self.time_limit = time_limit
        self.use_ensembling = use_ensembling
        self.use_feature_generator = use_feature_generator
        self.presets = presets
        self.hyperparameters = hyperparameters
        self.num_gpus = num_gpus
        self.verbosity = verbosity
        self.predictor_kwargs = kwargs
        self.predictor: Optional[TabularPredictor] = None
        self._is_fitted = False
    
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        **kwargs
    ):
        """Fit the AutoGluon model using full model suite."""
        # Prepare training data
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        train_data = X.copy()
        train_data['target'] = y.values
        
        # Configure ensembling
        extra_kwargs = {}
        if not self.use_ensembling:
            extra_kwargs['auto_stack'] = False
            extra_kwargs['num_bag_folds'] = 0
            extra_kwargs['num_bag_sets'] = 1
            extra_kwargs['num_stack_levels'] = 0
        else:
            # Only set these if presets is NOT provided, as presets usually handle this
            if not self.presets:
                extra_kwargs['auto_stack'] = True
                extra_kwargs['use_bag_holdout'] = True
        
        if not self.use_feature_generator:
            extra_kwargs['feature_generator'] = None
        
        if self.presets:
            extra_kwargs['presets'] = self.presets

        if self.hyperparameters:
            extra_kwargs['hyperparameters'] = self.hyperparameters
        
        # Initialize predictor
        self.predictor = TabularPredictor(
            label='target',
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            verbosity=self.verbosity,
            **self.predictor_kwargs
        )
        
        # Fit the model
        self.predictor.fit(
            train_data,
            time_limit=self.time_limit,
            num_gpus=self.num_gpus,
            **extra_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ['hyperparameters', 'presets']}
        )
        
        self._is_fitted = True
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> pd.Series:
        """Predict target values."""
        self._ensure_fitted()
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        preds = self.predictor.predict(X, **kwargs)
        
        if not isinstance(preds, pd.Series):
            preds = pd.Series(preds)
        
        return preds
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> pd.DataFrame:
        """Predict class probabilities for classification tasks."""
        self._ensure_fitted()
        
        if self.task_type != "classification":
            raise NotImplementedError("predict_proba is only available for classification tasks.")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        proba = self.predictor.predict_proba(X, **kwargs)
        
        if not isinstance(proba, pd.DataFrame):
            proba = pd.DataFrame(proba)
        
        return proba
    
    def _ensure_fitted(self):
        """Check if the model has been fitted."""
        if not self._is_fitted or self.predictor is None:
            raise RuntimeError("Model not trained. Call fit() first.")