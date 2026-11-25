"""Relational featurizer: prepared RDB, one fastdfs session, tabular preprocessor.

A session belongs to one prepared database (including its target-history table), not to a
model. Several models can fit on ``compute`` output from the same featurizer.
"""
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import fastdfs
from fastdfs import RDB, DFSConfig
from fastdfs.dfs import DFSSession
from fastdfs.utils.type_utils import safe_convert_to_string
from fastdfs.transform import (
    RDBTransformWrapper, RDBTransformPipeline, HandleDummyTable,
    FeaturizeDatetime, FillMissingPrimaryKey,
    FilterColumn, CanonicalizeTypes, EncodeCategoryColumns,
)
from fastdfs.transform.encode_categorical import encode_series_with_label_encoder
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from .config import RDBLearnConfig
from .constants import TARGET_HISTORY_TABLE_NAME
from .preprocessing import TabularPreprocessor


class RDBFeaturizer:
    def __init__(self, config: RDBLearnConfig, task_type: str):
        if task_type not in ("classification", "regression"):
            raise ValueError(f"task_type must be 'classification' or 'regression', got {task_type!r}")
        self.config = config
        self.task_type = task_type
        self.rdb_ = None
        self.preprocessor_ = None
        self.key_mappings_ = None
        self.cutoff_time_column_ = None
        self.history_df_ = None
        self.target_history_fks_ = None
        self.train_cutoff_time_column_ = None
        self.rdb_category_encoders_ = None
        self.task_category_encoders_ = None
        self.dfs_session_ = None

    def __getstate__(self):
        # The session holds a DuckDB connection, which cannot be pickled; the next
        # ``compute`` opens a new one on the prepared RDB.
        state = self.__dict__.copy()
        state["dfs_session_"] = None
        return state

    def close(self) -> None:
        if self.dfs_session_ is not None:
            self.dfs_session_.close()
            self.dfs_session_ = None

    def _dfs_config(self) -> DFSConfig:
        return self.config.dfs or DFSConfig()

    def fit_rdb(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        rdb: RDB,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
    ):
        """Prepare the RDB and return the training rows ready for :meth:`compute`.

        Target history is stored from the full training set before downsampling.
        """
        X = X.copy()
        self._ensure_keys_are_strings(X, key_mappings)
        self.key_mappings_ = key_mappings
        self.cutoff_time_column_ = cutoff_time_column
        self.close()

        if self.config.enable_target_augmentation:
            if cutoff_time_column is None:
                logger.debug(
                    "enable_target_augmentation is True but cutoff_time_column is None. "
                    "Skipping augmentation to prevent leakage."
                )
            else:
                logger.info("Storing target history for augmentation.")
                self.history_df_ = X.copy()
                target_col = y.name or "_RDBL_target"
                self.history_df_[target_col] = y.copy()
                self.train_cutoff_time_column_ = cutoff_time_column
                self.target_history_fks_ = []
                for x_col, rdb_ref in key_mappings.items():
                    if "." in rdb_ref:
                        rdb_table, rdb_col = rdb_ref.split(".", 1)
                        self.target_history_fks_.append((x_col, rdb_table, rdb_col))

        self.rdb_ = self._prepare_rdb(rdb)

        if len(X) > self.config.max_train_samples:
            data = X
            target_col = y.name or "_RDBL_target"
            data[target_col] = y
            downsampled = self._downsample(
                data, target_col, self.task_type,
                self.config.max_train_samples, self.config.stratified_sampling,
            )
            X = downsampled.drop(columns=[target_col])
            y = downsampled[target_col]

        if self.config.encode_categorical_as_float:
            self._fit_task_categorical_encoders(X, key_mappings, cutoff_time_column)
            X = self._apply_task_categorical_encoders(X)
        return X, y

    def compute(self, X: pd.DataFrame, rdb: Optional[RDB] = None) -> pd.DataFrame:
        """DFS features for ``X``.

        Calls against the fitted RDB (``rdb is None``) share one :class:`DFSSession`.
        A different ``rdb`` is computed one-shot and does not touch that session.
        """
        config = self._dfs_config()
        if rdb is not None or config.engine != "dfs2sql":
            return fastdfs.compute_dfs_features(
                rdb if rdb is not None else self.rdb_,
                X,
                key_mappings=self.key_mappings_,
                cutoff_time_column=self.cutoff_time_column_,
                config=config,
            )
        if self.dfs_session_ is None:
            self.dfs_session_ = DFSSession(
                self.rdb_, self.key_mappings_, self.cutoff_time_column_, config
            )
        return self.dfs_session_.compute(X)

    def features_for_prediction(
        self,
        X: pd.DataFrame,
        rdb: Optional[RDB],
        compute: Callable[..., pd.DataFrame],
    ) -> pd.DataFrame:
        """DFS features for prediction rows.

        ``compute`` is passed in so a caller can replace the DFS implementation
        (the session by default) without the featurizer knowing about it.
        """
        X = X.copy()
        if self.key_mappings_:
            self._ensure_keys_are_strings(X, self.key_mappings_)
        selected_rdb = None if rdb is None else self._prepare_rdb(rdb)
        if self.config.encode_categorical_as_float:
            X = self._apply_task_categorical_encoders(X)
        logger.info("Computing DFS features...")
        return compute(X, selected_rdb)

    def fit_preprocessor(self, X_dfs: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing augmented features ...")
        self.preprocessor_ = TabularPreprocessor(
            ag_config=self.config.ag_config,
            temporal_diff_config=self.config.temporal_diff,
            cutoff_time=self.cutoff_time_column_,
        )
        return self.preprocessor_.fit(X_dfs).transform(X_dfs)

    def transform(self, X_dfs: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing augmented features ...")
        return self.preprocessor_.transform(X_dfs)

    def _prepare_rdb(self, rdb: RDB) -> RDB:
        if (
            self.config.enable_target_augmentation
            and self.history_df_ is not None
            and self.target_history_fks_ is not None
            and self.train_cutoff_time_column_ is not None
        ):
            logger.info(f"Augmenting RDB with {TARGET_HISTORY_TABLE_NAME} table.")
            rdb = rdb.add_table(
                dataframe=self.history_df_,
                name=TARGET_HISTORY_TABLE_NAME,
                time_column=self.train_cutoff_time_column_,
                foreign_keys=self.target_history_fks_,
            )
            rdb = rdb.canonicalize_key_types()
            rdb.validate_key_consistency()

        logger.info("Preparing RDB with transformation pipeline.")
        steps = [
            HandleDummyTable(),
            FillMissingPrimaryKey(),
            RDBTransformWrapper(FeaturizeDatetime(features=["epochtime"])),
            RDBTransformWrapper(FilterColumn(drop_dtypes=["text"])),
        ]
        encode_transform = None
        if self.config.encode_categorical_as_float:
            encode_transform = EncodeCategoryColumns(encoders=self.rdb_category_encoders_)
            steps.append(RDBTransformWrapper(encode_transform))
        steps.append(RDBTransformWrapper(CanonicalizeTypes()))
        rdb = RDBTransformPipeline(steps)(rdb)
        if encode_transform is not None and self.rdb_category_encoders_ is None:
            self.rdb_category_encoders_ = encode_transform.encoders
        return rdb

    @staticmethod
    def _ensure_keys_are_strings(X: pd.DataFrame, key_mappings: Dict[str, str]) -> None:
        for col in key_mappings.keys():
            if col in X.columns:
                X[col] = safe_convert_to_string(X[col])

    def _task_categorical_columns(
        self,
        X: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
    ) -> List[str]:
        exclude = set(key_mappings.keys())
        if cutoff_time_column is not None:
            exclude.add(cutoff_time_column)
        cols: List[str] = []
        for col in X.columns:
            if col in exclude:
                continue
            dtype = X[col].dtype
            if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                cols.append(col)
        return cols

    def _fit_task_categorical_encoders(
        self,
        X: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
    ) -> None:
        cols = self._task_categorical_columns(X, key_mappings, cutoff_time_column)
        self.task_category_encoders_ = {}
        for col in cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.task_category_encoders_[col] = le
            logger.debug(f"Task categorical encoder fitted for {col} ({len(le.classes_)} classes)")

    def _apply_task_categorical_encoders(self, X: pd.DataFrame) -> pd.DataFrame:
        encoders = self.task_category_encoders_
        if not encoders:
            return X
        out = X.copy()
        for col, le in encoders.items():
            if col not in out.columns:
                continue
            out[col] = encode_series_with_label_encoder(out[col], le)
        return out

    @staticmethod
    def _downsample(
        data: pd.DataFrame,
        target_column: str,
        task_type: str,
        max_samples: int,
        stratified_sampling: bool = False,
    ) -> pd.DataFrame:
        if len(data) <= max_samples:
            return data

        logger.info(f"Downsampling training set from {len(data)} to {max_samples} samples.")
        X = data.drop(columns=[target_column])
        y = data[target_column].values

        if task_type == "regression":
            idx = np.random.choice(len(X), max_samples, replace=False)
            return data.iloc[idx].reset_index(drop=True)

        if not stratified_sampling:
            unique_labels = np.unique(y)
            selected_indices = []
            for label in unique_labels:
                class_indices = np.where(y == label)[0]
                if len(class_indices) > 0:
                    selected_indices.append(np.random.choice(class_indices, 1)[0])
            remaining_samples = max_samples - len(selected_indices)
            if remaining_samples > 0:
                mask = np.ones(len(X), dtype=bool)
                mask[selected_indices] = False
                eligible_indices = np.where(mask)[0]
                if len(eligible_indices) > 0:
                    additional = np.random.choice(
                        eligible_indices, min(remaining_samples, len(eligible_indices)), replace=False
                    )
                    selected_indices.extend(additional)
            np.random.shuffle(selected_indices)
            return data.iloc[np.array(selected_indices)].reset_index(drop=True)

        unique_labels = np.unique(y)
        n_classes = len(unique_labels)
        samples_per_class = max(1, max_samples // n_classes)
        balanced_indices = []
        remaining_indices = []
        for label in unique_labels:
            class_indices = np.where(y == label)[0]
            if len(class_indices) == 0:
                continue
            if len(class_indices) <= samples_per_class:
                balanced_indices.extend(class_indices)
            else:
                sampled = np.random.choice(class_indices, samples_per_class, replace=False)
                balanced_indices.extend(sampled)
                mask = np.ones(len(class_indices), dtype=bool)
                mask[np.isin(class_indices, sampled)] = False
                remaining_indices.extend(class_indices[mask])
        samples_needed = max_samples - len(balanced_indices)
        if samples_needed > 0 and len(remaining_indices) > 0:
            balanced_indices.extend(np.random.choice(
                remaining_indices, min(samples_needed, len(remaining_indices)), replace=False
            ))
        np.random.shuffle(balanced_indices)
        return data.iloc[np.array(balanced_indices[:max_samples])].reset_index(drop=True)
