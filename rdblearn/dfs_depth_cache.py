"""Process-level max-depth DFS cache (built-in RelArena rdblearn style).

The RelArena harness builds one model instance per ``max_depth``. Shallower
hops reuse the deepest DFS matrix and featurizer session computed once per
``(db, train_frame)`` scope. A new RelBench ``Database`` object (e.g. outer
split) clears the memos — matching the inner vs outer protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from fastdfs.dataset.rdb import RDB
from fastdfs.dfs import DFSConfig
from loguru import logger

from .config import RDBLearnConfig, TemporalDiffConfig
from .constants import RDBLEARN_DEFAULT_CONFIG
from .featurizer import RDBFeaturizer
from .pipeline import planned_dfs_columns
from .text import AnchorTextEncoder

#: Deepest hop in the RelBench v1 hop grid; shallower trials slice this matrix.
DFS_CACHE_MAX_DEPTH = 4


@dataclass
class TrainDFSBundle:
    """Cached deep DFS (+ hub/period/text) for one train table under one db."""

    featurizer: RDBFeaturizer
    rdb: RDB
    key_mappings: Dict[str, str]
    cutoff_time_column: Optional[str]
    X_train: pd.DataFrame
    y_train: pd.Series
    D_train: pd.DataFrame
    hop_columns: Dict[int, List[str]]
    text_encoder: Optional[AnchorTextEncoder]
    pred_matrices: Dict[int, pd.DataFrame] = field(default_factory=dict)

    def features_for_prediction(self, X: pd.DataFrame, *, append_extras, attach_text) -> pd.DataFrame:
        """Full max-depth features for ``X``, memoized by frame identity."""
        key = id(X)
        if key not in self.pred_matrices:
            X_dfs = self.featurizer.features_for_prediction(
                X, None, self.featurizer.compute
            )
            X_dfs = append_extras(X_dfs, X)
            X_dfs = attach_text(X_dfs, X)
            self.pred_matrices[key] = X_dfs
        return self.pred_matrices[key]


class DFSDepthCache:
    """Module-level cache shared across RelArena model instances in one process."""

    def __init__(self) -> None:
        self._db: Any = None
        self._bundles: Dict[tuple, TrainDFSBundle] = {}
        self.matrix_computations = 0

    def clear(self) -> None:
        for bundle in self._bundles.values():
            try:
                bundle.featurizer.close()
            except Exception:  # pragma: no cover
                pass
        self._db = None
        self._bundles = {}
        self.matrix_computations = 0

    def _scope(self, db: Any) -> None:
        if self._db is db:
            return
        for bundle in self._bundles.values():
            try:
                bundle.featurizer.close()
            except Exception:  # pragma: no cover
                pass
        self._db = db
        self._bundles = {}

    def train_bundle(
        self,
        *,
        db: Any,
        train_df: pd.DataFrame,
        target_col: str,
        rdb: RDB,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        task_type: str,
        hub: bool,
        period: bool,
        task_id: Optional[str],
        max_peer_features: int,
        text_encoder: Optional[str],
        append_extras,
        cache_max_depth: int = DFS_CACHE_MAX_DEPTH,
        config: Optional[RDBLearnConfig] = None,
        train_key: Any = None,
    ) -> TrainDFSBundle:
        """Return the deep DFS bundle for ``train_df``, computing it on a miss.

        ``train_key`` should be a stable identity for the RelArena train table
        (e.g. ``id(train_table)``). Do not key on ``id(train_df)`` alone — callers
        often copy the frame per trial, which would defeat the cache.
        """
        self._scope(db)
        # Prefer an explicit stable key; fall back to row-count + column fingerprint.
        if train_key is None:
            train_key = (
                len(train_df),
                tuple(train_df.columns),
                int(pd.util.hash_pandas_object(train_df, index=True).sum()),
            )
        key = (
            train_key,
            target_col,
            cache_max_depth,
            hub,
            period,
            task_id,
            text_encoder,
            tuple(sorted(key_mappings.items())),
            cutoff_time_column,
        )
        if key in self._bundles:
            logger.info(
                "DFS cache HIT (max_depth={}, train_rows={}, key={})",
                cache_max_depth,
                len(train_df),
                train_key if not isinstance(train_key, tuple) else "fingerprint",
            )
            return self._bundles[key]

        logger.info(
            "DFS cache MISS — computing depth {} once (train_rows={})",
            cache_max_depth,
            len(train_df),
        )
        base = config or RDBLearnConfig(**RDBLEARN_DEFAULT_CONFIG)
        dfs = (base.dfs or DFSConfig()).model_copy(update={"max_depth": cache_max_depth})
        temporal = (base.temporal_diff or TemporalDiffConfig()).model_copy(
            update={"enabled": True}
        )
        rdb_config = base.model_copy(
            update={
                "dfs": dfs,
                "enable_target_augmentation": True,
                "temporal_diff": temporal,
            }
        )
        featurizer = RDBFeaturizer(rdb_config, task_type)
        X_train, y_train = featurizer.fit_rdb(
            train_df.drop(columns=[target_col]),
            train_df[target_col],
            rdb,
            key_mappings,
            cutoff_time_column,
        )
        D_train = featurizer.compute(X_train)
        # Extras on the full deep matrix; hop slices keep them (StandardPipeline).
        D_train = append_extras(D_train, X_train)

        text_enc = None
        if text_encoder == "tfidf":
            text_enc = AnchorTextEncoder().fit(rdb, X_train, key_mappings)
            D_train = text_enc.transform(rdb, D_train, X_train)

        base_columns = [c for c in D_train.columns if c in X_train.columns]
        computed = set(D_train.columns) - set(base_columns)
        prepared_rdb = featurizer.rdb_
        hop_dfs_columns: Dict[int, List[str]] = {}
        for hop in range(1, cache_max_depth + 1):
            hop_cfg = rdb_config.dfs.model_copy(update={"max_depth": hop})
            planned = set(
                planned_dfs_columns(
                    prepared_rdb, X_train, key_mappings, cutoff_time_column, hop_cfg
                )
            )
            hop_dfs_columns[hop] = base_columns + [
                c for c in D_train.columns if c in planned & computed
            ]

        max_hop_dfs = set(hop_dfs_columns[cache_max_depth])
        extra_columns = [c for c in D_train.columns if c not in max_hop_dfs]
        hop_columns = {
            hop: hop_dfs_columns[hop]
            + [c for c in extra_columns if c in D_train.columns]
            for hop in hop_dfs_columns
        }

        self.matrix_computations += 1
        bundle = TrainDFSBundle(
            featurizer=featurizer,
            rdb=rdb,
            key_mappings=key_mappings,
            cutoff_time_column=cutoff_time_column,
            X_train=X_train,
            y_train=y_train,
            D_train=D_train,
            hop_columns=hop_columns,
            text_encoder=text_enc,
        )
        self._bundles[key] = bundle
        return bundle


#: Shared by every RelArena trial in this process.
DEPTH_CACHE = DFSDepthCache()
