"""Standard RDBLearn pipeline.

1. Run DFS once at the deepest hop (``max(hops)``) on train and validation, with
   target-history augmentation and temporal-difference features on. The fastdfs
   session belongs to that featurizer, not to a model.
2. Sweep every (hop, foundation model) pair on validation. A hop-``h`` candidate uses
   exactly the columns a DFS with ``max_depth=h`` would produce, taken from the deep DFS.
   Models are fit on the preprocessed frame directly.
3. Refit the best pair on train + validation: a new featurizer (its own session, because
   the target history now includes validation) and one model.
"""
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from fastdfs import RDB, DFSConfig
from fastdfs.dfs import dfs_feature_column_name, get_dfs_engine
from loguru import logger
from sklearn import metrics as skm

from .config import RDBLearnConfig, TemporalDiffConfig
from .constants import RDBLEARN_DEFAULT_CONFIG
from .featurizer import RDBFeaturizer
from .modeling import FittedModel
from .preprocessing import TabularPreprocessor
from .utils import TFM_FACTORIES

# name -> (score function taking (y_true, prediction, classes), greater is better)
# Regression metrics score the point prediction chosen by ``regression_output_type``
# (median by default). The median minimises absolute error (``mae``); the mean minimises
# squared error, so prefer ``regression_output_type="mean"`` when selecting on ``rmse``
# or ``r2``.
METRICS: Dict[str, Any] = {
    "roc_auc": (lambda y, p, classes: _roc_auc(y, p, classes), True),
    "accuracy": (lambda y, p, classes: skm.accuracy_score(y, classes[np.argmax(p, axis=1)]), True),
    "log_loss": (lambda y, p, classes: skm.log_loss(y, p, labels=classes), False),
    "mae": (lambda y, p, classes: skm.mean_absolute_error(y, p), False),
    "rmse": (lambda y, p, classes: float(np.sqrt(skm.mean_squared_error(y, p))), False),
    "r2": (lambda y, p, classes: skm.r2_score(y, p), True),
}


def _roc_auc(y, proba, classes) -> float:
    if len(classes) == 2:
        return skm.roc_auc_score(np.asarray(y) == classes[1], proba[:, 1])
    return skm.roc_auc_score(y, proba, multi_class="ovr", labels=classes)


def planned_dfs_columns(
    rdb: RDB,
    X: pd.DataFrame,
    key_mappings: Dict[str, str],
    cutoff_time_column: Optional[str],
    config: DFSConfig,
) -> List[str]:
    """Names of the features DFS plans for ``config`` (no feature values are computed).

    fastdfs may still drop planned features while computing, so intersect with the
    computed columns.
    """
    engine = get_dfs_engine(config.engine, config)
    _, target = engine._prepare_target_for_engine(X, key_mappings, cutoff_time_column)
    features = engine.prepare_features(
        rdb.canonicalize_key_types(), target, key_mappings, cutoff_time_column, config
    )
    return [dfs_feature_column_name(f) for f in features]


class StandardPipeline:
    """Hop x foundation-model sweep on validation, then refit of the best pair on train + val.

    Args:
        task_type: ``"classification"`` or ``"regression"``.
        hops: DFS depths to sweep; DFS itself runs once at ``max(hops)``.
        models: Names from ``rdblearn.utils.TFM_FACTORIES`` (e.g. ``"tabpfn-v3.5"``,
            ``"kumo-tabular-medium"``, ``"tabicl-v2"``), or a dict of name ->
            zero-argument estimator factory.
        metric: Key of :data:`METRICS`; defaults to ``roc_auc`` / ``mae``.
        config: Base ``RDBLearnConfig``. Target augmentation and temporal differences
            are switched on and ``dfs.max_depth`` is set by the pipeline.
        model_params: Keyword overrides passed to every named model factory (e.g. ``device``).
        regression_output_type: Point prediction of regressors, ``"median"`` or ``"mean"``.
            ``None`` keeps ``config.regression_output_type`` (``"median"`` by default).

    A candidate that runs out of memory is recorded with a NaN score and an ``error``
    in ``results_``; if the best candidate runs out of memory during the refit, the
    next best is refit instead.

    After ``fit``, ``featurizer_`` holds the refit relational database and its fastdfs
    session, and ``model_`` is the refit foundation model. ``predict`` runs new rows
    through that featurizer and then the model.
    """

    def __init__(
        self,
        task_type: str,
        hops: Sequence[int] = (2, 3, 4),
        models: Union[Sequence[str], Dict[str, Callable[[], Any]]] = ("tabpfn-v3.5", "kumo-tabular-medium"),
        metric: Optional[str] = None,
        config: Optional[RDBLearnConfig] = None,
        model_params: Optional[Dict[str, Any]] = None,
        regression_output_type: Optional[str] = None,
    ):
        if task_type not in ("classification", "regression"):
            raise ValueError(f"task_type must be 'classification' or 'regression', got {task_type!r}")
        if regression_output_type not in (None, "median", "mean"):
            raise ValueError(
                f"regression_output_type must be 'median', 'mean' or None, got {regression_output_type!r}"
            )
        self.regression_output_type = regression_output_type
        self.task_type = task_type
        self.hops = sorted(set(int(h) for h in hops))
        self.models = models
        self.metric = metric or ("roc_auc" if task_type == "classification" else "mae")
        if self.metric not in METRICS:
            raise ValueError(f"Unknown metric {self.metric!r}; expected one of {sorted(METRICS)}")
        self.config = config
        self.model_params = model_params or {}

    # ------------------------------------------------------------------ helpers

    def _model_factories(self) -> Dict[str, Callable[[], Any]]:
        if isinstance(self.models, dict):
            return dict(self.models)
        factories = {}
        for name in self.models:
            if name not in TFM_FACTORIES:
                raise ValueError(f"Unknown model {name!r}; expected one of {sorted(TFM_FACTORIES)}")
            make = TFM_FACTORIES[name][0 if self.task_type == "classification" else 1]
            factories[name] = lambda make=make: make(**self.model_params)
        return factories

    def _config_for_depth(self, depth: int) -> RDBLearnConfig:
        base = self.config or RDBLearnConfig(**RDBLEARN_DEFAULT_CONFIG)
        dfs = (base.dfs or DFSConfig()).model_copy(update={"max_depth": depth})
        temporal = (base.temporal_diff or TemporalDiffConfig()).model_copy(update={"enabled": True})
        update = {"dfs": dfs, "enable_target_augmentation": True, "temporal_diff": temporal}
        if self.regression_output_type is not None:
            update["regression_output_type"] = self.regression_output_type
        return base.model_copy(update=update)

    def _score(self, model, y, predictions) -> float:
        fn, _ = METRICS[self.metric]
        return float(fn(np.asarray(y), predictions, getattr(model, "classes_", None)))

    def _transformed(self, X: pd.DataFrame) -> pd.DataFrame:
        X_dfs = self.featurizer_.features_for_prediction(X, None, self.featurizer_.compute)
        return self.featurizer_.transform(X_dfs)

    @staticmethod
    def _free_gpu_memory():
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    @staticmethod
    def _is_out_of_memory(error: BaseException) -> bool:
        if isinstance(error, MemoryError):
            return True
        try:
            import torch

            if isinstance(error, torch.cuda.OutOfMemoryError):
                return True
        except ImportError:  # pragma: no cover
            pass
        return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()

    def _fit_model(self, make, config, X, y) -> FittedModel:
        model = FittedModel(make(), self.task_type, config)
        model.fit(X, y)
        return model

    # ------------------------------------------------------------------ API

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str,
        rdb: RDB,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str] = None,
    ):
        """Run the sweep and the refit. One call is the supported use.

        The refit session stays open on ``featurizer_`` for later ``predict`` and
        ``score``. A second ``fit`` on the same object opens a new sweep session
        before that session is closed. With the default temporary DuckDB file the
        two sessions do not meet. With a shared ``dfs.engine_path`` the second open
        fails (``Table already exists``). Close ``featurizer_`` at the start of
        ``fit`` before treating a repeated ``fit`` as supported.
        """
        factories = self._model_factories()
        max_hop = max(self.hops)
        self.target_col_ = target_col

        # 1. One DFS at the deepest hop, shared by every candidate.
        t0 = time.perf_counter()
        featurizer = RDBFeaturizer(self._config_for_depth(max_hop), self.task_type)
        X_train, y_train = featurizer.fit_rdb(
            train_df.drop(columns=[target_col]), train_df[target_col],
            rdb, key_mappings, cutoff_time_column,
        )
        D_train = featurizer.compute(X_train)
        D_val = featurizer.features_for_prediction(
            val_df.drop(columns=[target_col]), None, featurizer.compute
        )
        y_val = val_df[target_col]
        featurizer.close()
        self.dfs_seconds_ = time.perf_counter() - t0

        base_columns = [c for c in D_train.columns if c in X_train.columns]
        computed = set(D_train.columns) - set(base_columns)
        self.hop_columns_ = {}
        for hop in self.hops:
            planned = set(planned_dfs_columns(
                featurizer.rdb_, X_train, key_mappings, cutoff_time_column,
                self._config_for_depth(hop).dfs,
            ))
            self.hop_columns_[hop] = base_columns + [c for c in D_train.columns if c in planned & computed]
        logger.info(
            f"DFS depth {max_hop}: {len(computed)} features in {self.dfs_seconds_:.1f}s; "
            + ", ".join(f"hop {h}: {len(c) - len(base_columns)}" for h, c in self.hop_columns_.items())
        )

        # 2. Sweep (hop x model) on validation. The preprocessor depends on the hop's
        #    columns, not on the model, so it is fit once per hop.
        rows = []
        for hop in self.hops:
            cols = self.hop_columns_[hop]
            config = self._config_for_depth(hop)
            preprocessor = TabularPreprocessor(
                ag_config=config.ag_config,
                temporal_diff_config=config.temporal_diff,
                cutoff_time=cutoff_time_column,
            )
            Xt_train = preprocessor.fit(D_train[cols]).transform(D_train[cols])
            Xt_val = preprocessor.transform(D_val[cols])
            for name, make in factories.items():
                t0 = time.perf_counter()
                score, error = float("nan"), None
                try:
                    model = self._fit_model(make, config, Xt_train, y_train)
                    predictions = model.predict_proba(Xt_val) if self.task_type == "classification" else model.predict(Xt_val)
                    score = self._score(model, y_val, np.asarray(predictions))
                    logger.info(f"hop={hop} model={name}: val {self.metric}={score:.4f}")
                    del model
                except Exception as e:
                    if not self._is_out_of_memory(e):
                        raise
                    error = f"{type(e).__name__}: {e}"
                    logger.warning(f"hop={hop} model={name}: out of memory, skipped ({error})")
                finally:
                    self._free_gpu_memory()
                rows.append({
                    "hop": hop, "model": name, "n_features": len(cols) - len(base_columns),
                    f"val_{self.metric}": score, "seconds": time.perf_counter() - t0, "error": error,
                })
        self.results_ = pd.DataFrame(rows)

        # 3. Refit the best (hop, model) on train + validation. The history table now
        #    includes validation, so this is a new featurizer and a new session.
        _, greater_is_better = METRICS[self.metric]
        score_col = f"val_{self.metric}"
        ranked = self.results_.dropna(subset=[score_col]).sort_values(
            score_col, ascending=not greater_is_better, kind="stable"
        )
        if ranked.empty:
            raise RuntimeError(f"Every candidate ran out of memory:\n{self.results_}")

        train_val = pd.concat([train_df, val_df], ignore_index=True)
        self.featurizer_ = None
        self.model_ = None
        for _, row in ranked.iterrows():
            best = {"hop": int(row["hop"]), "model": row["model"], score_col: float(row[score_col])}
            logger.info(f"Best: {best}; refitting on train + validation.")
            t0 = time.perf_counter()
            config = self._config_for_depth(best["hop"])
            featurizer = RDBFeaturizer(config, self.task_type)
            try:
                X, y = featurizer.fit_rdb(
                    train_val.drop(columns=[target_col]), train_val[target_col],
                    rdb, key_mappings, cutoff_time_column,
                )
                Xt = featurizer.fit_preprocessor(featurizer.compute(X))
                model = self._fit_model(factories[best["model"]], config, Xt, y)
            except Exception as e:
                if not self._is_out_of_memory(e):
                    featurizer.close()
                    raise
                logger.warning(
                    f"Refit of {best} ran out of memory ({type(e).__name__}: {e}); trying the next best."
                )
                featurizer.close()
                self._free_gpu_memory()
                continue
            self.best_ = best
            self.featurizer_ = featurizer
            self.model_ = model
            self.refit_seconds_ = time.perf_counter() - t0
            break
        if self.model_ is None:
            raise RuntimeError("Every candidate ran out of memory during the refit on train + validation.")
        return self

    def predict(self, X: pd.DataFrame):
        return self.model_.predict(self._transformed(X))

    def predict_proba(self, X: pd.DataFrame):
        return self.model_.predict_proba(self._transformed(X))

    def score(self, df: pd.DataFrame) -> float:
        """``metric`` of the refitted model on ``df`` (which must contain the target column)."""
        X = self._transformed(df.drop(columns=[self.target_col_]))
        predictions = self.model_.predict_proba(X) if self.task_type == "classification" else self.model_.predict(X)
        return self._score(self.model_, df[self.target_col_], np.asarray(predictions))
