"""Single-config RelBench v1 recipe (one DFS hop, fixed TabPFN stack).

Used by the RelArena model wrapper and by standalone callers that do not want
``StandardPipeline``'s internal hop×model sweep. RelArena owns hop selection
via a ``fixed_grid`` over ``max_depth`` ∈ {2, 3, 4}.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastdfs.dataset.rdb import RDB
from fastdfs.dfs import DFSConfig
from loguru import logger

from .config import RDBLearnConfig, TemporalDiffConfig
from .constants import RDBLEARN_DEFAULT_CONFIG
from .context import prepare_hard_pool_fit
from .featurizer import RDBFeaturizer
from .modeling import FittedModel
from .text import AnchorTextEncoder, merge_transform_text
from .utils import TFM_FACTORIES

DEFAULT_MODEL = "tabpfn-v3.5"
DEFAULT_RECENCY_WINDOW = 800_000
DEFAULT_HOPS = (2, 3, 4)


def database_to_rdb(db, *, name: str = "relbench-v1") -> RDB:
    """Build an untransformed fastdfs ``RDB`` from a RelBench ``Database``."""
    from fastdfs import create_rdb
    from fastdfs.utils.type_utils import safe_convert_to_string

    table_dict = db.table_dict
    tables: Dict[str, pd.DataFrame] = {}
    primary_keys: Dict[str, str] = {}
    time_columns: Dict[str, str] = {}
    foreign_keys: list[tuple[str, str, str, str]] = []

    key_cols: Dict[str, set[str]] = {tname: set() for tname in table_dict}
    for tname, table in table_dict.items():
        if table.pkey_col:
            key_cols[tname].add(table.pkey_col)
        for child_col in table.fkey_col_to_pkey_table:
            key_cols[tname].add(child_col)

    for tname, table in table_dict.items():
        frame = table.df.copy()
        for col in key_cols[tname]:
            if col in frame.columns:
                frame[col] = safe_convert_to_string(frame[col])
        tables[tname] = frame
        if table.pkey_col:
            primary_keys[tname] = table.pkey_col
        if table.time_col:
            time_columns[tname] = table.time_col
        for child_col, parent_table in table.fkey_col_to_pkey_table.items():
            parent_pk = table_dict[parent_table].pkey_col
            foreign_keys.append((tname, child_col, parent_table, parent_pk))

    return create_rdb(
        tables,
        name=name,
        primary_keys=primary_keys,
        foreign_keys=foreign_keys,
        time_columns=time_columns,
    )


def key_mappings_for_task(task, db) -> Dict[str, str]:
    entity = db.table_dict[task.entity_table]
    if not entity.pkey_col:
        raise ValueError(f"Entity table {task.entity_table!r} has no primary key.")
    return {task.entity_col: f"{task.entity_table}.{entity.pkey_col}"}


def label_frame(table_df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    from fastdfs.utils.type_utils import safe_convert_to_string

    frame = table_df.copy()
    if entity_col in frame.columns:
        frame[entity_col] = safe_convert_to_string(frame[entity_col])
    return frame


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class RelBenchV1Estimator:
    """Fit / predict one RelBench v1 config: DFS @ ``max_depth`` + hub/period/TPR knobs."""

    def __init__(
        self,
        task_type: str,
        *,
        max_depth: int = 2,
        model: str = DEFAULT_MODEL,
        hub: bool = True,
        period: bool = True,
        task_id: Optional[str] = None,
        max_peer_features: int = 256,
        text_encoder: Optional[str] = "tfidf",
        context_strategy: Optional[str] = "hard_pool",
        recency_window: int = DEFAULT_RECENCY_WINDOW,
        hurdle: Optional[str] = "auto",
        hurdle_zero_threshold: float = 0.05,
        seed: int = 0,
        model_params: Optional[Dict[str, Any]] = None,
        config: Optional[RDBLearnConfig] = None,
    ) -> None:
        if task_type not in ("classification", "regression"):
            raise ValueError(f"task_type must be classification or regression, got {task_type!r}")
        if text_encoder not in (None, "tfidf"):
            raise ValueError(f"text_encoder must be None or 'tfidf', got {text_encoder!r}")
        if context_strategy not in (None, "random", "hard_pool"):
            raise ValueError(
                f"context_strategy must be None, 'random', or 'hard_pool', got {context_strategy!r}"
            )
        if hurdle not in (None, "auto"):
            raise ValueError(f"hurdle must be None or 'auto', got {hurdle!r}")
        if model not in TFM_FACTORIES:
            raise ValueError(f"Unknown model {model!r}; expected one of {sorted(TFM_FACTORIES)}")

        self.task_type = task_type
        self.max_depth = int(max_depth)
        self.model_name = model
        self.hub = bool(hub)
        self.period = bool(period)
        self.task_id = task_id
        self.max_peer_features = int(max_peer_features)
        self.text_encoder = text_encoder
        self.context_strategy = context_strategy
        self.recency_window = int(recency_window)
        self.hurdle = hurdle if task_type == "regression" else None
        self.hurdle_zero_threshold = float(hurdle_zero_threshold)
        self.seed = int(seed)
        self.model_params = dict(model_params or {})
        self.config = config

    def _rdb_config(self) -> RDBLearnConfig:
        base = self.config or RDBLearnConfig(**RDBLEARN_DEFAULT_CONFIG)
        dfs = (base.dfs or DFSConfig()).model_copy(update={"max_depth": self.max_depth})
        temporal = (base.temporal_diff or TemporalDiffConfig()).model_copy(update={"enabled": True})
        return base.model_copy(
            update={
                "dfs": dfs,
                "enable_target_augmentation": True,
                "temporal_diff": temporal,
            }
        )

    def _effective_model_params(self) -> Dict[str, Any]:
        params = dict(self.model_params)
        if self.text_encoder == "tfidf":
            params = merge_transform_text(params)
        return params

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

    def _fit_tfm(self, X: pd.DataFrame, y, *, context_time: Optional[np.ndarray]) -> FittedModel:
        config = self._rdb_config()
        params = self._effective_model_params()
        if self.hurdle == "auto":
            from .hurdle import fit_auto_hurdle

            make_clf, make_reg = TFM_FACTORIES[self.model_name]
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

        make = TFM_FACTORIES[self.model_name][0 if self.task_type == "classification" else 1]
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
        estimator = make(**{**params, **overrides}) if overrides else make(**params)
        model = FittedModel(estimator, self.task_type, config)
        model.fit(X_fit, y_fit)
        return model

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        rdb: RDB,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str] = None,
        *,
        task_id: Optional[str] = None,
    ) -> "RelBenchV1Estimator":
        if task_id is not None:
            self.task_id = task_id
        self.target_col_ = target_col
        self.rdb_ = rdb
        self.key_mappings_ = key_mappings
        self.cutoff_time_column_ = cutoff_time_column

        config = self._rdb_config()
        featurizer = RDBFeaturizer(config, self.task_type)
        X_train, y_train = featurizer.fit_rdb(
            train_df.drop(columns=[target_col]),
            train_df[target_col],
            rdb,
            key_mappings,
            cutoff_time_column,
        )
        D_train = featurizer.compute(X_train)
        D_train = self._append_extras(D_train, X_train)

        self.text_encoder_ = None
        if self.text_encoder == "tfidf":
            self.text_encoder_ = AnchorTextEncoder().fit(rdb, X_train, key_mappings)
            D_train = self._attach_text(D_train, X_train)

        text_cols = [c for c in D_train.columns if str(c).endswith("__raw_text")]
        model_cols = [c for c in D_train.columns if c not in text_cols]
        Xt = featurizer.fit_preprocessor(D_train[model_cols])
        if text_cols:
            # Drop any prior text placeholders, then re-attach once.
            drop = [c for c in Xt.columns if str(c).endswith("__raw_text")]
            if drop:
                Xt = Xt.drop(columns=drop)
            Xt = pd.concat(
                [Xt.reset_index(drop=True), D_train[text_cols].reset_index(drop=True).astype("string")],
                axis=1,
            )

        self.feature_columns_ = list(Xt.columns)
        train_times = (
            pd.to_datetime(X_train[cutoff_time_column]).to_numpy()
            if cutoff_time_column and cutoff_time_column in X_train.columns
            else None
        )
        self.model_ = self._fit_tfm(Xt, y_train, context_time=train_times)
        self.featurizer_ = featurizer
        return self

    def _transformed(self, X: pd.DataFrame) -> pd.DataFrame:
        X_dfs = self.featurizer_.features_for_prediction(X, None, self.featurizer_.compute)
        X_dfs = self._append_extras(X_dfs, X)
        wanted = [c for c in self.feature_columns_ if not str(c).endswith("__raw_text")]
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

    @property
    def classes_(self):
        return getattr(self.model_, "classes_", None)

    def predict(self, X: pd.DataFrame):
        return self.model_.predict(self._transformed(X))

    def predict_proba(self, X: pd.DataFrame):
        return self.model_.predict_proba(self._transformed(X))
