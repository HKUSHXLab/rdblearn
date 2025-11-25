from typing import Optional, Union
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNClassifier, TabPFNRegressor


LIMIX_ROOT = Path("/root/yl_project/LimiX")
DEFAULT_TABPFN_ROOT = Path("/root/autodl-tmp/tabpfn_2_5")
DEFAULT_REGRESSOR_CHECKPOINT = "tabpfn-v2.5-regressor-v2.5_default.ckpt"
DEFAULT_CLASSIFIER_CHECKPOINT = "tabpfn-v2.5-classifier-v2.5_default.ckpt"

if LIMIX_ROOT.exists() and str(LIMIX_ROOT) not in sys.path:
    sys.path.append(str(LIMIX_ROOT))

try:
    from inference.predictor import LimiXPredictor  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - surface clear error for missing repo
    raise ImportError(
        "LimiX repository is required for CustomLimiX. Expected at /root/yl_project/LimiX."
    ) from exc


class CustomTabPFN:
    """Custom TabPFN model wrapper for classification and regression."""
    
    def __init__(self, model_path: str = None, task_type: str = "regression", max_samples: int = None):
        self.model_path = model_path or "/root/autodl-tmp/tabpfn_2_5"
        self.task_type = task_type
        self.max_samples = max_samples  # Maximum samples to use for training
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit the TabPFN model with optional random sampling."""
        # Apply random sampling if dataset exceeds max_samples
        if self.max_samples is not None and len(X) > self.max_samples:
            print(f"Sampling {self.max_samples} from {len(X)} samples for TabPFN training...")
            indices = np.random.choice(len(X), size=self.max_samples, replace=False)
            X = X.iloc[indices].reset_index(drop=True)
            y = y.iloc[indices].reset_index(drop=True)
        
        # Choose the appropriate TabPFN model based on task type
        if self.task_type == "regression":
            regressor_model_path = self.model_path
            self.model = TabPFNRegressor(model_path=regressor_model_path)
        else:
            classifier_model_path = self.model_path
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
            
        preds = self.model.predict(X, output_type="median" if self.task_type == "regression" else None)
        
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
