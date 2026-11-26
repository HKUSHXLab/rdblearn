"""RelArena model plugin: RelBench v1 recipe with hop-only search space.

Install ``rdblearn`` (this package) where RelArena can import it, then
``discover_models()`` loads the ``relarena.models`` entry point. Search space
is ``max_depth`` ∈ {2, 3, 4}; TabPFN v3.5 + hub/period/hard_pool/tfidf/hurdle
are fixed recipe knobs, not tuned.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from relbench.base import Database, EntityTask, Table, TaskType

from relarena_core.model import RelArenaModel
from relarena_core.predict_contract import predict_to_contract
from relarena_core.registry import register_model, registry
from relarena_core.search_space import SearchSpace

from .relbench_v1 import (
    DEFAULT_HOPS,
    DEFAULT_MODEL,
    DEFAULT_RECENCY_WINDOW,
    RelBenchV1Estimator,
    database_to_rdb,
    default_device,
    key_mappings_for_task,
    label_frame,
)

_NAME = "rdblearn-relbench-v1-2026-11-26"

#: Hop-only grid. Default depth 4 first so ``n_trials`` truncation still reports it.
RELBENCH_V1_SPACE = SearchSpace(
    default_overrides={"max_depth": 4},
    fixed_grid=[{"max_depth": d} for d in DEFAULT_HOPS],
)


def _task_type_name(task: EntityTask) -> str:
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return "classification"
    if task.task_type == TaskType.REGRESSION:
        return "regression"
    raise ValueError(f"Unsupported task type: {task.task_type}")


def _resolve_task_id(identity) -> Optional[str]:
    if identity is None:
        return None
    if not identity.dataset or not identity.task:
        return None
    return f"relbench::{identity.dataset}::{identity.task}"


@register_model(search_space=RELBENCH_V1_SPACE)
class RDBLearnRelBenchV1Model(RelArenaModel):
    """DFS @ tuned hop + fixed TabPFN RelBench v1 stack (hub/period/hard_pool/…)."""

    name = _NAME
    supported_task_types = frozenset(
        {TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION}
    )
    #: Match StandardPipeline / TPR: select hop on val, refit train∪val for test.
    refit_on_full_data = True

    def fit(
        self,
        task: EntityTask,
        db: Database,
        train_table: Table,
        val_table: Table | None,
        *,
        seed: int,
        time_limit: float | None = None,
    ) -> None:
        del val_table, time_limit  # selection is harness-owned; no early stopping.
        depth = int(self.config.get("max_depth", 4))
        task_type = _task_type_name(task)
        task_id = _resolve_task_id(self.run_identity)
        train_df = label_frame(train_table.df, task.entity_col)
        rdb = database_to_rdb(db, name="rdblearn-relbench-v1")
        self.estimator_ = RelBenchV1Estimator(
            task_type,
            max_depth=depth,
            model=DEFAULT_MODEL,
            hub=True,
            period=True,
            task_id=task_id,
            text_encoder="tfidf",
            context_strategy="hard_pool",
            recency_window=DEFAULT_RECENCY_WINDOW,
            hurdle="auto",
            seed=seed,
            model_params={"device": default_device()},
        )
        self.estimator_.fit(
            train_df,
            task.target_col,
            rdb,
            key_mappings_for_task(task, db),
            task.time_col,
            task_id=task_id,
            db=db,  # scopes the process-level max-depth DFS cache
            train_key=id(train_table),  # stable across hop trials (not id of copied df)
        )
        self._depth = depth
        self._task_type = task.task_type

    def predict(self, task: EntityTask, db: Database, table: Table) -> np.ndarray:
        del db  # featurizer holds the fit-time RDB / session
        frame = label_frame(table.df, task.entity_col)
        if task.target_col in frame.columns:
            frame = frame.drop(columns=[task.target_col])
        return predict_to_contract(self.estimator_, frame, task.task_type)


registry.register_alias("rdblearn-relbench-v1-latest", RDBLearnRelBenchV1Model.name)

__all__ = [
    "RELBENCH_V1_SPACE",
    "RDBLearnRelBenchV1Model",
]
