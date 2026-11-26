from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

class LimiXWrapperClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible wrapper for the LimiX classifier.
    
    This wrapper handles the in-context learning nature of LimiX by storing 
    the training data during `fit` and passing it to `predict` along with the query data.
    """
    
    def __init__(self, predictor):
        """
        Initialize the wrapper.
        
        Args:
            predictor: An initialized LimiXPredictor instance.
                       The instance must implement a `predict` method with the signature:
                       `predict(x_train, y_train, x_test, task_type='Classification') -> np.ndarray`
        """
        self.predictor = predictor

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Store the training data for in-context inference.
        """
        # LimiX expects numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class labels for X.
        """
        check_is_fitted(self, ['X_train_', 'y_train_'])
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for X.
        """
        check_is_fitted(self, ['X_train_', 'y_train_'])
        X = np.array(X)
        
        # Call the underlying predictor
        result = self.predictor.predict(
            self.X_train_, 
            self.y_train_, 
            X,
            task_type="Classification"
        )
            
        return np.asarray(result)


class LimiXWrapperRegressor(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible wrapper for the LimiX regressor.

    LimiX's regression head predicts one value per row (averaged over its ensemble),
    so unlike TabPFN, TabICL and Kumo-Tabular there is no median to choose.
    """
    
    def __init__(self, predictor):
        """
        Initialize the wrapper.
        
        Args:
            predictor: An initialized LimiXPredictor instance.
        """
        self.predictor = predictor

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Store the training data.
        """
        X = np.array(X)
        y = np.array(y)
        
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict target values for X.
        """
        check_is_fitted(self, ['X_train_', 'y_train_'])
        X = np.array(X)
        
        result = self.predictor.predict(
            self.X_train_, 
            self.y_train_, 
            X,
            task_type="Regression"
        )
        
        return np.asarray(result)


# ---------------------------------------------------------------------------
# TabFM (google/tabfm-1.0.0-pytorch) integration
# ---------------------------------------------------------------------------

def _materialize_tabfm_bin(checkpoint_path: str, model_type: str) -> None:
    """Materialize ``pytorch_model.bin`` from ``model.safetensors`` if missing.

    The released ``tabfm==1.0.0`` wheel's loader reads
    ``{checkpoint_path}/{model_type}/pytorch_model.bin`` via ``torch.load``, but the
    Hugging Face repo ``google/tabfm-1.0.0-pytorch`` only ships ``model.safetensors``.
    This converts once (lossless) so the stock loader works.
    """
    import os

    sub = os.path.join(checkpoint_path, model_type)
    bin_path = os.path.join(sub, "pytorch_model.bin")
    st_path = os.path.join(sub, "model.safetensors")
    if os.path.exists(bin_path) or not os.path.exists(st_path):
        return
    import torch
    from safetensors.torch import load_file

    torch.save(load_file(st_path), bin_path)


def _enable_tabfm_chunking(backbone, n_features: int, chunk_size) -> None:
    """Turn on TabFM's activation chunking (off by default in tabfm==1.0.0).

    Without it the model materializes the whole ``(context + test) x n_features``
    activation in one forward pass, which OOMs feature-rich DFS outputs (hundreds of
    columns) on 32 GB cards. ``chunk_size="auto"`` keeps a constant cells-per-chunk
    budget so wide feature sets automatically get smaller row chunks.
    """
    chunk = chunk_size
    if str(chunk) == "auto":
        # ~150k cells/chunk stays within 32 GB in fp32 (measured on RTX 5090).
        chunk = max(128, min(2048, 150_000 // max(1, int(n_features))))
    for mod in backbone.modules():
        if hasattr(mod, "row_chunk_size"):
            mod.row_chunk_size = int(chunk)
        if hasattr(mod, "col_chunk_size"):
            mod.col_chunk_size = 8
        if hasattr(mod, "ffn_chunk_size"):
            mod.ffn_chunk_size = max(int(chunk), 2048)


class _TabFMBf16IO:
    """Forward-boundary dtype adapter for a bf16-cast TabFM backbone.

    tabfm 1.0.0's estimators build fp32 input tensors while ``use_amp`` is documented
    as "informational only" (no autocast), so simply casting the weights to bf16 fails
    at the first matmul. This adapter casts floating-point inputs to bf16 on entry and
    the logits back to fp32 on exit, halving the activation memory that row-chunking
    cannot reduce (the full in-context rows x features embedding). bf16 is TabFM's
    upstream (JAX) design dtype; parity vs fp32 measured at ~1e-3 AUROC.
    """

    def __new__(cls, inner):
        import torch

        class _Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.inner = m

            def forward(self, *args, **kwargs):
                def cast(v):
                    if torch.is_tensor(v) and v.is_floating_point():
                        return v.to(torch.bfloat16)
                    return v

                out = self.inner(
                    *[cast(a) for a in args], **{k: cast(v) for k, v in kwargs.items()}
                )
                return out.float()

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(super().__getattr__("inner"), name)

        return _Wrapper(inner)


class _TabFMWrapperBase(BaseEstimator):
    """Shared TabFM loading/config logic. Use the Classifier/Regressor subclasses."""

    _task_type = "classification"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        n_estimators: int = 8,
        max_num_features: Optional[int] = None,
        chunk_size: Union[int, str, None] = "auto",
        dtype: str = "float32",
    ):
        # Stored verbatim so sklearn.clone() works (required by RDBLearn multiclass heads).
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.n_estimators = n_estimators
        self.max_num_features = max_num_features
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.model_ = None

    def _build(self, n_features: int):
        try:
            import tabfm
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "tabfm is required for the TabFM wrappers. Install with "
                "`pip install tabfm --no-deps` (plus jaxtyping<0.3, typeguard<3, "
                "absl-py, safetensors) to avoid disturbing pinned torch builds."
            ) from e

        if self.checkpoint_path is not None:
            _materialize_tabfm_bin(self.checkpoint_path, self._task_type)
        backbone = tabfm.tabfm_v1_0_0_pytorch.load(
            model_type=self._task_type,
            checkpoint_path=self.checkpoint_path,  # None -> download from Hugging Face
            device=self.device,
        )
        use_bf16 = str(self.dtype).lower() in ("bfloat16", "bf16")
        if use_bf16:
            import torch

            backbone = backbone.to(torch.bfloat16)
        if self.chunk_size:
            _enable_tabfm_chunking(backbone, n_features, self.chunk_size)
        if use_bf16:
            backbone = _TabFMBf16IO(backbone)

        est_cls = (
            tabfm.TabFMClassifier if self._task_type == "classification" else tabfm.TabFMRegressor
        )
        return est_cls(
            model=backbone,
            n_estimators=self.n_estimators,
            max_num_features=self.max_num_features,
            random_state=42,
        )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Prepare the in-context estimator on the training data (no weight updates)."""
        n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])
        self.model_ = self._build(n_features)
        self.model_.fit(X, y)
        if self._task_type == "classification":
            self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        preds = self.model_.predict(X)
        if isinstance(preds, (pd.Series, pd.DataFrame)):
            return preds.to_numpy().ravel()
        return np.asarray(preds).ravel()


class TabFMWrapperClassifier(_TabFMWrapperBase, ClassifierMixin):
    """Scikit-learn compatible wrapper for Google's TabFM 1.0.0 classifier.

    Loads the PyTorch backbone from ``checkpoint_path`` (a local directory holding the
    ``classification/`` and ``regression/`` subfolders of ``google/tabfm-1.0.0-pytorch``;
    ``None`` downloads from Hugging Face) and wraps ``tabfm.TabFMClassifier``.

    Practical notes baked in (all optional):
      * auto-converts ``model.safetensors`` -> ``pytorch_model.bin`` (wheel expects .bin);
      * ``chunk_size="auto"`` enables activation chunking so wide DFS feature sets fit
        on 32 GB GPUs without dropping features;
      * ``dtype="bfloat16"`` halves activation memory for very wide/deep feature sets
        (upstream design dtype; adds an fp32<->bf16 IO adapter around the backbone).

    Constraints inherited from TabFM: max 10 classes (architectural), single GPU.
    """

    _task_type = "classification"

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities, shape (n_samples, n_classes)."""
        check_is_fitted(self, ["model_"])
        proba = self.model_.predict_proba(X)
        if isinstance(proba, pd.DataFrame):
            proba = proba.to_numpy()
        arr = np.asarray(proba)
        if arr.ndim == 1:
            return np.column_stack([1 - arr, arr])
        return arr

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels."""
        check_is_fitted(self, ["model_"])
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class TabFMWrapperRegressor(_TabFMWrapperBase, RegressorMixin):
    """Scikit-learn compatible wrapper for Google's TabFM 1.0.0 regressor.

    See :class:`TabFMWrapperClassifier` for loading behaviour and memory options.
    Note: TabFM's regression head emits a single scalar per row (out_dim=1); unlike
    TabPFN there is no predictive distribution, so no separate median/quantiles.
    """

    _task_type = "regression"


# ---------------------------------------------------------------------------
# TabPFN (tabpfn>=9.0.0) and TabICL (tabicl>=2.0.0) model selection
# ---------------------------------------------------------------------------

TABPFN_VERSIONS = ("v2", "v2.5", "v2.6", "v3", "v3.5", "v3.5-fast")


def _make_tabpfn(task: str, version: str, overrides: Dict[str, Any]):
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    from .constants import TABPFN_DEFAULT_CONFIG

    if version not in TABPFN_VERSIONS:
        raise ValueError(f"Unknown TabPFN version {version!r}; expected one of {TABPFN_VERSIONS}")
    est_cls = TabPFNClassifier if task == "classification" else TabPFNRegressor
    return est_cls.create_default_for_version(
        ModelVersion(version), **{**TABPFN_DEFAULT_CONFIG, **overrides}
    )


def make_tabpfn_classifier(version: str = "v3.5", **overrides):
    """Build a ``TabPFNClassifier`` for ``version`` with RDBLearn's defaults.

    ``version`` is one of :data:`TABPFN_VERSIONS`: ``"v3.5"`` is TabPFN-3.5 and
    ``"v3.5-fast"`` the smaller TabPFN-3.5-Fast. Defaults come from
    ``TABPFN_DEFAULT_CONFIG`` (including ``fit_mode="fit_with_cache"``);
    ``overrides`` take precedence.
    """
    return _make_tabpfn("classification", version, overrides)


def make_tabpfn_regressor(version: str = "v3.5", **overrides):
    """Build a ``TabPFNRegressor``; see :func:`make_tabpfn_classifier`."""
    return _make_tabpfn("regression", version, overrides)


def make_tabicl_classifier(**overrides):
    """Build a TabICL v2 ``TabICLClassifier`` with ``TABICL_DEFAULT_CONFIG`` (KV cache on)."""
    from tabicl import TabICLClassifier

    from .constants import TABICL_DEFAULT_CONFIG

    return TabICLClassifier(**{**TABICL_DEFAULT_CONFIG, **overrides})


def make_tabicl_regressor(**overrides):
    """Build a TabICL v2 ``TabICLRegressor``; see :func:`make_tabicl_classifier`."""
    from tabicl import TabICLRegressor

    from .constants import TABICL_DEFAULT_CONFIG

    return TabICLRegressor(**{**TABICL_DEFAULT_CONFIG, **overrides})


# ---------------------------------------------------------------------------
# NVIDIA Kumo-Tabular (nvidia/Kumo-Tabular, structured-data-models>=0.2.1)
# ---------------------------------------------------------------------------

KUMO_TABULAR_SIZES = ("small", "medium", "large")


def _kumo_feature_stypes(X: pd.DataFrame) -> Dict[str, str]:
    stypes = {}
    for col in X.columns:
        dtype = X[col].dtype
        if isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype):
            stypes[col] = "categorical"
        else:
            stypes[col] = "numerical"
    return stypes


class _KumoTabularWrapperBase(BaseEstimator):
    """Shared Kumo-Tabular logic. Use the Classifier/Regressor subclasses."""

    _task_type = "classification"

    def __init__(
        self,
        size: str = "medium",
        device: str = "cuda",
        n_estimators: int = 8,
        estimator_batch_size: Optional[int] = 1,
        random_state: Optional[int] = 0,
    ):
        self.size = size
        self.device = device
        self.n_estimators = n_estimators
        self.estimator_batch_size = estimator_batch_size
        self.random_state = random_state

    def _as_frame(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.rename(columns=str)
        X = np.asarray(X)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    def _table(self, X: pd.DataFrame):
        import sdm

        return sdm.TableTensor.from_pandas(X, stypes=self.feature_stypes_, device=self.device)

    def _fit_context(self, X: Union[pd.DataFrame, np.ndarray], y_frame: pd.DataFrame, y_stype: str):
        import sdm
        import torch

        if self.size not in KUMO_TABULAR_SIZES:
            raise ValueError(f"Unknown Kumo-Tabular size {self.size!r}; expected one of {KUMO_TABULAR_SIZES}")

        X = self._as_frame(X)
        self.feature_stypes_ = _kumo_feature_stypes(X)
        # Loaded per instance: the fitted context cache lives on the model object.
        self.model_ = sdm.models.KumoTabular(task=self._task_type, size=self.size, device=self.device)
        generator = None
        if self.random_state is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(self.random_state))
        # Encodes the context rows once and caches their keys/values; predict()
        # then only runs the query rows against the cache.
        self.model_.fit(
            x=self._table(X),
            y=sdm.TableTensor.from_pandas(y_frame, stypes={"target": y_stype}, device=self.device),
            num_estimators=self.n_estimators,
            estimator_batch_size=self.estimator_batch_size,
            generator=generator,
        )
        return self

    def _predict_table(self, X: Union[pd.DataFrame, np.ndarray]):
        check_is_fitted(self, ["model_"])
        return self.model_.predict(self._table(self._as_frame(X)))


class KumoTabularWrapperClassifier(_KumoTabularWrapperBase, ClassifierMixin):
    """Scikit-learn compatible wrapper for NVIDIA Kumo-Tabular classification.

    ``size`` selects ``"small"`` (~28M params), ``"medium"`` (~62M, default) or
    ``"large"`` (~215M). Weights download from ``nvidia/Kumo-Tabular`` on first use.
    ``fit`` caches the encoded context, so repeated ``predict_proba`` calls only
    process the query rows. Any number of classes is handled natively (ECOC), so
    RDBLearn does not apply its base-10-hierarchical decomposition.
    """

    _task_type = "classification"
    rdblearn_max_classes = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        codes = le.fit_transform(np.asarray(y))
        self.classes_ = le.classes_
        return self._fit_context(X, pd.DataFrame({"target": codes.astype(np.int64)}), "categorical")

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        out = self._predict_table(X)
        values = out.numerical.float().cpu().numpy()
        # Output columns are named by class code; order them to match classes_.
        codes = [int(c) for c in out.columns["numerical"]]
        proba = np.zeros((values.shape[0], len(self.classes_)), dtype=np.float64)
        proba[:, codes] = values
        return proba

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class KumoTabularWrapperRegressor(_KumoTabularWrapperBase, RegressorMixin):
    """Scikit-learn compatible wrapper for NVIDIA Kumo-Tabular regression.

    The model predicts 999 quantiles (``q001`` .. ``q999``). ``output_type="median"``
    (default) returns ``q500``; ``"mean"`` returns the average over all quantiles.
    See :class:`KumoTabularWrapperClassifier` for ``size`` and caching.
    """

    _task_type = "regression"

    def __init__(
        self,
        size: str = "medium",
        device: str = "cuda",
        n_estimators: int = 8,
        estimator_batch_size: Optional[int] = 1,
        random_state: Optional[int] = 0,
        output_type: str = "median",
    ):
        super().__init__(
            size=size,
            device=device,
            n_estimators=n_estimators,
            estimator_batch_size=estimator_batch_size,
            random_state=random_state,
        )
        self.output_type = output_type

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        if self.output_type not in ("median", "mean"):
            raise ValueError(f"output_type must be 'median' or 'mean', got {self.output_type!r}")
        y_frame = pd.DataFrame({"target": np.asarray(y, dtype=np.float64)})
        return self._fit_context(X, y_frame, "numerical")

    def predict_quantiles(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """All 999 predicted quantiles, shape (n_samples, 999), levels 0.001 .. 0.999."""
        return self._predict_table(X).numerical.double().cpu().numpy()

    def predict(self, X: Union[pd.DataFrame, np.ndarray], output_type: Optional[str] = None) -> np.ndarray:
        """Point prediction: ``output_type`` (default ``self.output_type``) is ``"median"`` or ``"mean"``."""
        output_type = output_type or self.output_type
        if output_type not in ("median", "mean"):
            raise ValueError(f"output_type must be 'median' or 'mean', got {output_type!r}")
        quantiles = self.predict_quantiles(X)
        if output_type == "median":
            return quantiles[:, 499]
        return quantiles.mean(axis=1)


# Named foundation models for ``StandardPipeline``: name -> (classifier factory, regressor factory).
TFM_FACTORIES = {
    "tabpfn-v3.5": (
        lambda **kw: make_tabpfn_classifier("v3.5", **kw),
        lambda **kw: make_tabpfn_regressor("v3.5", **kw),
    ),
    "tabpfn-v3.5-fast": (
        lambda **kw: make_tabpfn_classifier("v3.5-fast", **kw),
        lambda **kw: make_tabpfn_regressor("v3.5-fast", **kw),
    ),
    "tabicl-v2": (make_tabicl_classifier, make_tabicl_regressor),
    **{
        f"kumo-tabular-{size}": (
            lambda size=size, **kw: KumoTabularWrapperClassifier(size=size, **kw),
            lambda size=size, **kw: KumoTabularWrapperRegressor(size=size, **kw),
        )
        for size in KUMO_TABULAR_SIZES
    },
}
