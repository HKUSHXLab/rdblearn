from typing import Optional, Dict, Union

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from fastdfs import RDB

from .config import RDBLearnConfig
from .constants import RDBLEARN_DEFAULT_CONFIG
from .featurizer import RDBFeaturizer
from .modeling import (
    base_estimator_max_classes,
    fit_classification,
    predict_batches,
    predict_proba_hierarchical,
    regression_predict_kwargs,
)


class RDBLearnEstimator(BaseEstimator):
    def __init__(
        self,
        base_estimator,
        config: Optional[Union[RDBLearnConfig, dict]] = None
    ):
        self.base_estimator = base_estimator

        if isinstance(config, RDBLearnConfig):
            self.config = config
        else:
            config_dict = RDBLEARN_DEFAULT_CONFIG.copy()
            if isinstance(config, dict):
                config_dict.update(config)
            self.config = RDBLearnConfig(**config_dict)

        self.featurizer_ = None
        self.downstream_feature_columns_ = None

    def __getattr__(self, name):
        """Fitted RDB state lives on the featurizer; expose it as before."""
        if name.startswith("__"):
            raise AttributeError(name)
        featurizer = self.__dict__.get("featurizer_")
        if featurizer is not None and hasattr(featurizer, name):
            return getattr(featurizer, name)
        if name in (
            "rdb_", "preprocessor_", "key_mappings_", "cutoff_time_column_",
            "history_df_", "target_history_fks_", "train_cutoff_time_column_",
            "rdb_category_encoders_", "task_category_encoders_", "dfs_session_",
        ):
            return None
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _compute_dfs(self, X: pd.DataFrame, rdb: Optional[RDB] = None) -> pd.DataFrame:
        """DFS features for ``X``. Overridable; the default uses the featurizer's session."""
        return self.featurizer_.compute(X, rdb)

    def _close_dfs_session(self) -> None:
        if self.featurizer_ is not None:
            self.featurizer_.close()

    def _dfs_for_prediction(self, X: pd.DataFrame, rdb: Optional[RDB]) -> pd.DataFrame:
        return self.featurizer_.features_for_prediction(X, rdb, self._compute_dfs)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        rdb: RDB,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str] = None,
        **kwargs
    ):
        task_type = "regression" if isinstance(self, RegressorMixin) else "classification"
        self.featurizer_ = RDBFeaturizer(self.config, task_type)
        X, y = self.featurizer_.fit_rdb(X, y, rdb, key_mappings, cutoff_time_column)
        logger.info("Computing DFS features...")
        X_dfs = self._compute_dfs(X)
        logger.debug(f"DFS features: {X_dfs.columns.tolist()}")
        return self._fit_from_dfs(X_dfs, y, **kwargs)

    def _fit_from_dfs(self, X_dfs: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit the preprocessor and base estimator on DFS features of the training rows."""
        X_transformed = self.featurizer_.fit_preprocessor(X_dfs)
        self.downstream_feature_columns_ = list(X_transformed.columns)
        return self._fit_model(X_transformed, y, **kwargs)

    def _fit_model(self, X_transformed: pd.DataFrame, y: pd.Series, **kwargs):
        """Train the base estimator on preprocessed features (regression default)."""
        logger.info("Fitting base estimator ...")
        self.base_estimator.fit(X_transformed, y, **kwargs)
        return self

    def _predict_common(self, X: pd.DataFrame, rdb: Optional[RDB], method: str, **kwargs):
        return self._predict_from_dfs(self._dfs_for_prediction(X, rdb), method, **kwargs)

    def _predict_from_dfs(self, X_dfs: pd.DataFrame, method: str, **kwargs):
        X_transformed = self.featurizer_.transform(X_dfs)
        logger.info("Making predictions ...")
        predict_func = getattr(self.base_estimator, method)
        return predict_batches(predict_func, X_transformed, self.config.predict_batch_size, **kwargs)


class RDBLearnClassifier(RDBLearnEstimator, ClassifierMixin):
    def _base_estimator_max_classes(self) -> Optional[int]:
        """Classes one base-estimator head can take; ``None`` means unlimited."""
        return base_estimator_max_classes(self.base_estimator)

    def _fit_model(self, X_transformed: pd.DataFrame, y: pd.Series, **kwargs):
        fit_classification(self, self.base_estimator, X_transformed, y, **kwargs)
        return self

    def predict(self, X: pd.DataFrame, rdb: Optional[RDB] = None, **kwargs):
        if getattr(self, "base10_hierarchical_", False):
            proba = self.predict_proba(X, rdb, **kwargs)
            idx = np.argmax(proba, axis=1)
            return self.classes_.take(idx.astype(np.intp, copy=False))
        return self._predict_common(X, rdb, method="predict", **kwargs)

    def predict_proba(self, X: pd.DataFrame, rdb: Optional[RDB] = None, **kwargs):
        return self._predict_proba_from_dfs(self._dfs_for_prediction(X, rdb), **kwargs)

    def _predict_proba_from_dfs(self, X_dfs: pd.DataFrame, **kwargs):
        if getattr(self, "base10_hierarchical_", False):
            X_transformed = self.featurizer_.transform(X_dfs)
            return predict_proba_hierarchical(
                self, X_transformed, self.config.predict_batch_size, **kwargs
            )
        return self._predict_from_dfs(X_dfs, method="predict_proba", **kwargs)


class RDBLearnRegressor(RDBLearnEstimator, RegressorMixin):
    def predict(self, X: pd.DataFrame, rdb: Optional[RDB] = None, **kwargs):
        return self._predict_common(X, rdb, method="predict", **kwargs)

    def _predict_from_dfs(self, X_dfs: pd.DataFrame, method: str, **kwargs):
        if method == "predict":
            kwargs = regression_predict_kwargs(self.base_estimator, self.config.regression_output_type, kwargs)
        return super()._predict_from_dfs(X_dfs, method, **kwargs)
