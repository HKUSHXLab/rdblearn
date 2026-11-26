"""Standard RDBLearn pipeline.

1. Run DFS once at the deepest hop (``max(hops)``) on train and validation, with
   target-history augmentation and temporal-difference features on. The fastdfs
   session belongs to that featurizer, not to a model.
2. Optionally append hub-completion and multi-period SQL features.
3. Sweep every (hop, foundation model) pair on validation. A hop-``h`` candidate uses
   exactly the columns a DFS with ``max_depth=h`` would produce, taken from the deep DFS,
   plus any hub/period/text columns. Models are fit on the preprocessed frame directly
   (optional hard_pool context / auto hurdle).
4. Refit the best pair on train + validation.
"""
from __future__ import annotations

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
from .context import prepare_hard_pool_fit
from .featurizer import RDBFeaturizer
from .modeling import FittedModel
from .preprocessing import TabularPreprocessor
from .text import AnchorTextEncoder, merge_transform_text
from .utils import TFM_FACTORIES

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
    """Names of the features DFS plans for ``config`` (no feature values are computed)."""
    engine = get_dfs_engine(config.engine, config)
    _, target = engine._prepare_target_for_engine(X, key_mappings, cutoff_time_column)
    features = engine.prepare_features(
        rdb.canonicalize_key_types(), target, key_mappings, cutoff_time_column, config
    )
    return [dfs_feature_column_name(f) for f in features]


class StandardPipeline:
    """Hop x foundation-model sweep on validation, then refit of the best pair on train + val.

    Extra feature / model knobs (RelBench v1 recipe):

    - ``hub`` / ``period``: post-DFS peer completion and multi-period SQL features.
    - ``text_encoder``: ``None`` or ``"tfidf"`` (TabPFN ``TRANSFORM_TEXT``, no custom TF-IDF).
    - ``context_strategy``: ``None``/``"random"`` or ``"hard_pool"`` with ``recency_window``.
    - ``hurdle``: ``None`` or ``"auto"`` (regression only).
    """

    def __init__(
        self,
        task_type: str,
        hops: Sequence[int] = (2, 3, 4),
        models: Union[Sequence[str], Dict[str, Callable[..., Any]]] = ("tabpfn-v3.5", "kumo-tabular-medium"),
        metric: Optional[str] = None,
        config: Optional[RDBLearnConfig] = None,
        model_params: Optional[Dict[str, Any]] = None,
        regression_output_type: Optional[str] = None,
        *,
        hub: bool = False,
        period: bool = False,
        task_id: Optional[str] = None,
        max_peer_features: int = 256,
        text_encoder: Optional[str] = None,
        context_strategy: Optional[str] = None,
        recency_window: int = 800_000,
        hurdle: Optional[str] = None,
        hurdle_zero_threshold: float = 0.05,
        seed: int = 0,
    ):
        if task_type not in ("classification", "regression"):
            raise ValueError(f"task_type must be 'classification' or 'regression', got {task_type!r}")
        if regression_output_type not in (None, "median", "mean"):
            raise ValueError(
                f"regression_output_type must be 'median', 'mean' or None, got {regression_output_type!r}"
            )
        if text_encoder not in (None, "tfidf"):
            raise ValueError(f"text_encoder must be None or 'tfidf', got {text_encoder!r}")
        if context_strategy not in (None, "random", "hard_pool"):
            raise ValueError(
                f"context_strategy must be None, 'random', or 'hard_pool', got {context_strategy!r}"
            )
        if hurdle not in (None, "auto"):
            raise ValueError(f"hurdle must be None or 'auto', got {hurdle!r}")
        self.regression_output_type = regression_output_type
        self.task_type = task_type
        self.hops = sorted(set(int(h) for h in hops))
        self.models = models
        self.metric = metric or ("roc_auc" if task_type == "classification" else "mae")
        if self.metric not in METRICS:
            raise ValueError(f"Unknown metric {self.metric!r}; expected one of {sorted(METRICS)}")
        self.config = config
        self.model_params = model_params or {}
        self.hub = bool(hub)
        self.period = bool(period)
        self.task_id = task_id
        self.max_peer_features = int(max_peer_features)
        self.text_encoder = text_encoder
        self.context_strategy = context_strategy
        self.recency_window = int(recency_window)
        self.hurdle = hurdle
        self.hurdle_zero_threshold = float(hurdle_zero_threshold)
        self.seed = int(seed)

    # ------------------------------------------------------------------ helpers

    def _effective_model_params(self) -> Dict[str, Any]:
        params = dict(self.model_params)
        if self.text_encoder == "tfidf":
            params = merge_transform_text(params)
        return params

    def _model_factories(self) -> Dict[str, Callable[..., Any]]:
        params = self._effective_model_params()
        if isinstance(self.models, dict):
            return dict(self.models)
        factories: Dict[str, Callable[..., Any]] = {}
        for name in self.models:
            if name not in TFM_FACTORIES:
                raise ValueError(f"Unknown model {name!r}; expected one of {sorted(TFM_FACTORIES)}")
            make = TFM_FACTORIES[name][0 if self.task_type == "classification" else 1]

            def _factory(make=make, params=params, **extra):
                return make(**{**params, **extra})

            factories[name] = _factory
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

    def _append_hub(self, feats: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        if not self.hub:
            return feats
        from .hub_completion import append_hub_completion

        out = append_hub_completion(
            feats,
            X,
            self.cutoff_time_column_,
            rdb=self.rdb_,
            key_mappings=self.key_mappings_,
            max_peer_features=self.max_peer_features,
        )
        logger.info("hub: cols {} -> {}", feats.shape[1], out.shape[1])
        return out

    def _append_period(self, feats: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        if not self.period:
            return feats
        if not self.task_id:
            logger.warning("period=True but task_id is missing; skipping period features")
            return feats
        from .sql_extra_features import append_sql_features_for_task

        try:
            out = append_sql_features_for_task(
                feats, X, self.task_id, cutoff_time_column=self.cutoff_time_column_
            )
        except Exception as e:
            logger.warning("period: SKIP {}: {}", self.task_id, e)
            return feats
        logger.info("period: cols {} -> {}", feats.shape[1], out.shape[1])
        return out

    def _append_extras(self, feats: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        return self._append_period(self._append_hub(feats, X), X)

    def _attach_text(self, feats: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        if self.text_encoder != "tfidf" or self.text_encoder_ is None:
            return feats
        return self.text_encoder_.transform(self.rdb_, feats, X)

    def _transformed(self, X: pd.DataFrame) -> pd.DataFrame:
        X_dfs = self.featurizer_.features_for_prediction(X, None, self.featurizer_.compute)
        X_dfs = self._append_extras(X_dfs, X)
        # Preprocessor was fit without __raw_text columns; keep that split here.
        wanted = [c for c in self.refit_feature_columns_ if not str(c).endswith("__raw_text")]
        cols = [c for c in wanted if c in X_dfs.columns]
        missing = [c for c in wanted if c not in X_dfs.columns]
        frame = X_dfs[cols].copy()
        for c in missing:
            frame[c] = np.nan
        frame = frame[wanted]
        Xt = self.featurizer_.transform(frame)
        with_text = self._attach_text(frame, X)
        text_cols = [c for c in with_text.columns if c.endswith("__raw_text")]
        if text_cols:
            Xt = pd.concat(
                [Xt.reset_index(drop=True), with_text[text_cols].reset_index(drop=True).astype("string")],
                axis=1,
            )
        return Xt

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

    def _fit_model(
        self,
        make: Callable[..., Any],
        config: RDBLearnConfig,
        X: pd.DataFrame,
        y,
        *,
        context_time: Optional[np.ndarray] = None,
        model_name: Optional[str] = None,
    ) -> FittedModel:
        use_hurdle = (
            self.hurdle == "auto"
            and self.task_type == "regression"
            and not isinstance(self.models, dict)
        )
        if use_hurdle:
            from .hurdle import fit_auto_hurdle
            from .utils import TFM_FACTORIES

            # Named models only: need both clf and reg factories from the same family
            name = model_name or "tabpfn-v3.5"
            if name not in TFM_FACTORIES:
                raise ValueError(f"hurdle='auto' needs a known TFM name, got {name!r}")
            params = self._effective_model_params()
            make_clf = TFM_FACTORIES[name][0]
            make_reg = TFM_FACTORIES[name][1]
            estimator = fit_auto_hurdle(
                X,
                y,
                make_classifier=make_clf,
                make_regressor=make_reg,
                context_time=context_time,
                context_strategy=self.context_strategy,
                recency_window=self.recency_window,
                seed=self.seed,
                zero_threshold=self.hurdle_zero_threshold,
                model_params=params,
            )
            model = FittedModel(estimator, self.task_type, config)
            model.base_estimator = estimator
            return model

        overrides: Dict[str, Any] = {}
        X_fit, y_fit = X, y
        if self.context_strategy == "hard_pool":
            X_fit, y_fit, overrides = prepare_hard_pool_fit(
                X,
                y,
                context_time,
                recency_window=self.recency_window,
                seed=self.seed,
            )
        try:
            estimator = make(**overrides) if overrides else make()
        except TypeError:
            # Zero-arg custom factories (tests)
            if overrides:
                logger.warning(
                    "hard_pool overrides ignored for custom factory that accepts no kwargs"
                )
            estimator = make()
        model = FittedModel(estimator, self.task_type, config)
        model.fit(X_fit, y_fit)
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
        *,
        task_id: Optional[str] = None,
    ):
        """Run the sweep and the refit. One call is the supported use."""
        if task_id is not None:
            self.task_id = task_id
        factories = self._model_factories()
        max_hop = max(self.hops)
        self.target_col_ = target_col
        self.rdb_ = rdb
        self.key_mappings_ = key_mappings
        self.cutoff_time_column_ = cutoff_time_column

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
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]

        base_columns = [c for c in D_train.columns if c in X_train.columns]
        computed = set(D_train.columns) - set(base_columns)
        prepared_rdb = featurizer.rdb_
        hop_dfs_columns: Dict[int, List[str]] = {}
        for hop in self.hops:
            planned = set(planned_dfs_columns(
                prepared_rdb, X_train, key_mappings, cutoff_time_column,
                self._config_for_depth(hop).dfs,
            ))
            hop_dfs_columns[hop] = base_columns + [
                c for c in D_train.columns if c in planned & computed
            ]

        D_train = self._append_extras(D_train, X_train)
        D_val = self._append_extras(D_val, X_val)

        self.text_encoder_ = None
        if self.text_encoder == "tfidf":
            # Look up text on the original RDB — the featurizer's prepared copy may
            # drop free-text columns that DFS does not use.
            self.text_encoder_ = AnchorTextEncoder().fit(rdb, X_train, key_mappings)
            D_train = self._attach_text(D_train, X_train)
            D_val = self._attach_text(D_val, X_val)

        extra_columns = [c for c in D_train.columns if c not in set(hop_dfs_columns[max_hop])]
        self.extra_columns_ = extra_columns
        self.hop_columns_ = {
            hop: hop_dfs_columns[hop] + [c for c in extra_columns if c in D_train.columns]
            for hop in self.hops
        }

        featurizer.close()
        self.dfs_seconds_ = time.perf_counter() - t0

        logger.info(
            f"DFS depth {max_hop}: {len(computed)} DFS features + {len(extra_columns)} extras "
            f"in {self.dfs_seconds_:.1f}s; "
            + ", ".join(
                f"hop {h}: {len(hop_dfs_columns[h]) - len(base_columns)} dfs "
                f"+ {len(extra_columns)} extras"
                for h in self.hops
            )
        )

        train_times = (
            pd.to_datetime(X_train[cutoff_time_column]).to_numpy()
            if cutoff_time_column and cutoff_time_column in X_train.columns
            else None
        )

        # 2. Sweep (hop x model) on validation.
        rows = []
        for hop in self.hops:
            cols = [c for c in self.hop_columns_[hop] if c in D_train.columns]
            config = self._config_for_depth(hop)
            preprocessor = TabularPreprocessor(
                ag_config=config.ag_config,
                temporal_diff_config=config.temporal_diff,
                cutoff_time=cutoff_time_column,
            )
            # Preserve string text columns for TabPFN TRANSFORM_TEXT: temporarily
            # exclude __raw_text from AG by transforming only non-text, then concat.
            text_cols = [c for c in cols if c.endswith("__raw_text")]
            model_cols = [c for c in cols if c not in text_cols]
            Xt_train = preprocessor.fit(D_train[model_cols]).transform(D_train[model_cols])
            Xt_val = preprocessor.transform(D_val[model_cols])
            if text_cols:
                Xt_train = pd.concat(
                    [Xt_train.reset_index(drop=True), D_train[text_cols].reset_index(drop=True).astype("string")],
                    axis=1,
                )
                Xt_val = pd.concat(
                    [Xt_val.reset_index(drop=True), D_val[text_cols].reset_index(drop=True).astype("string")],
                    axis=1,
                )
            for name, make in factories.items():
                t0 = time.perf_counter()
                score, error = float("nan"), None
                try:
                    model = self._fit_model(
                        make, config, Xt_train, y_train,
                        context_time=train_times, model_name=name if not isinstance(self.models, dict) else None,
                    )
                    predictions = (
                        model.predict_proba(Xt_val)
                        if self.task_type == "classification"
                        else model.predict(Xt_val)
                    )
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

        # 3. Refit the best (hop, model) on train + validation.
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
                D = self._append_extras(featurizer.compute(X), X)
                if self.text_encoder == "tfidf":
                    self.text_encoder_ = AnchorTextEncoder().fit(rdb, X, key_mappings)
                    D = self._attach_text(D, X)
                cols = [c for c in self.hop_columns_[best["hop"]] if c in D.columns]
                # Ensure all hop extras present
                for c in self.hop_columns_[best["hop"]]:
                    if c not in cols and c in D.columns:
                        cols.append(c)
                text_cols = [c for c in cols if c.endswith("__raw_text")]
                model_cols = [c for c in cols if c not in text_cols]
                Xt = featurizer.fit_preprocessor(D[model_cols])
                if text_cols:
                    Xt = pd.concat(
                        [Xt.reset_index(drop=True), D[text_cols].reset_index(drop=True).astype("string")],
                        axis=1,
                    )
                self.refit_feature_columns_ = list(cols)
                refit_times = (
                    pd.to_datetime(X[cutoff_time_column]).to_numpy()
                    if cutoff_time_column and cutoff_time_column in X.columns
                    else None
                )
                model = self._fit_model(
                    factories[best["model"]],
                    config,
                    Xt,
                    y,
                    context_time=refit_times,
                    model_name=best["model"] if not isinstance(self.models, dict) else None,
                )
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
