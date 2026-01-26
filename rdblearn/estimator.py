from typing import Optional, Dict, Union, List
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import fastdfs
from fastdfs import RDB, DFSConfig
from fastdfs.transform import (
    RDBTransformWrapper, RDBTransformPipeline, HandleDummyTable, 
    FeaturizeDatetime, FillMissingPrimaryKey, 
    FilterColumn, CanonicalizeTypes
)

from .config import RDBLearnConfig
from .preprocessing import TabularPreprocessor
from .constants import RDBLEARN_DEFAULT_CONFIG, TARGET_HISTORY_TABLE_NAME

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
            # Start with defaults
            config_dict = RDBLEARN_DEFAULT_CONFIG.copy()
            # Update with user provided dict if any
            if isinstance(config, dict):
                config_dict.update(config)
            
            self.config = RDBLearnConfig(**config_dict)
            
        self.rdb_ = None
        self.preprocessor_ = None
        self.key_mappings_ = None
        self.cutoff_time_column_ = None
        
        self.history_df_ = None
        self.target_history_fks_ = None
        self.train_cutoff_time_column_ = None

    def _ensure_keys_are_strings(self, X: pd.DataFrame, key_mappings: Dict[str, str]) -> None:
        """Modifies X in place."""
        for col in key_mappings.keys():
            if col in X.columns:
                X[col] = X[col].astype(str)

    def _downsample(
        self,
        data: pd.DataFrame,
        target_column: str,
        task_type: str,
        max_samples: int,
        stratified_sampling: bool = False
    ) -> pd.DataFrame:
        """Downsample data to max_samples."""
        if len(data) <= max_samples:
            return data

        logger.info(f"Downsampling training set from {len(data)} to {max_samples} samples.")
        
        X = data.drop(columns=[target_column])
        y = data[target_column].values

        if task_type == "regression":
            idx = np.random.choice(len(X), max_samples, replace=False)
            return data.iloc[idx].reset_index(drop=True)

        # Classification
        if not stratified_sampling:
            unique_labels = np.unique(y)
            selected_indices = []
            for label in unique_labels:
                class_indices = np.where(y == label)[0]
                if len(class_indices) > 0:
                    selected_idx = np.random.choice(class_indices, 1)[0]
                    selected_indices.append(selected_idx)

            remaining_samples = max_samples - len(selected_indices)
            if remaining_samples > 0:
                mask = np.ones(len(X), dtype=bool)
                mask[selected_indices] = False
                eligible_indices = np.where(mask)[0]

                if len(eligible_indices) > 0:
                    additional_indices = np.random.choice(
                        eligible_indices,
                        min(remaining_samples, len(eligible_indices)),
                        replace=False
                    )
                    selected_indices.extend(additional_indices)

            np.random.shuffle(selected_indices)
            idx = np.array(selected_indices)
            return data.iloc[idx].reset_index(drop=True)

        else:
            unique_labels, label_counts = np.unique(y, return_counts=True)
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
                    sampled_indices = np.random.choice(class_indices, samples_per_class, replace=False)
                    balanced_indices.extend(sampled_indices)
                    mask = np.ones(len(class_indices), dtype=bool)
                    mask[np.isin(class_indices, sampled_indices)] = False
                    remaining_indices.extend(class_indices[mask])

            samples_needed = max_samples - len(balanced_indices)
            if samples_needed > 0 and len(remaining_indices) > 0:
                additional_samples = np.random.choice(
                    remaining_indices,
                    min(samples_needed, len(remaining_indices)),
                    replace=False
                )
                balanced_indices.extend(additional_samples)

            np.random.shuffle(balanced_indices)
            balanced_indices = balanced_indices[:max_samples]
            idx = np.array(balanced_indices)
            return data.iloc[idx].reset_index(drop=True)

    def _prepare_rdb(self, rdb: RDB) -> RDB:
        # Augment with target history if enabled and available
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
                foreign_keys=self.target_history_fks_
            )
            rdb = rdb.canonicalize_key_types()
            rdb.validate_key_consistency()

        logger.info("Preparing RDB with transformation pipeline.")
        pipeline = RDBTransformPipeline([
            HandleDummyTable(),
            FillMissingPrimaryKey(),
            RDBTransformWrapper(FeaturizeDatetime(features=["epochtime"])),
            RDBTransformWrapper(FilterColumn(drop_dtypes=["text"])),
            RDBTransformWrapper(CanonicalizeTypes()),
        ])
        return pipeline(rdb)

    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        rdb: RDB,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str] = None,
        **kwargs
    ):
        # 0. Copy and ensure keys are string
        X = X.copy()
        self._ensure_keys_are_strings(X, key_mappings)
        self.key_mappings_ = key_mappings
        self.cutoff_time_column_ = cutoff_time_column
        
        # 1. Setup Target History Augmentation (Using FULL X, y)
        if self.config.enable_target_augmentation:
            if cutoff_time_column is None:
                logger.debug("enable_target_augmentation is True but cutoff_time_column is None. Skipping augmentation to prevent leakage.")
            else:
                logger.info("Storing target history for augmentation.")
                
                # Create history dataframe (using current X which is the full train set)
                self.history_df_ = X.copy()
                target_col = y.name or "_RDBL_target"
                self.history_df_[target_col] = y.copy()
                
                self.train_cutoff_time_column_ = cutoff_time_column
                
                # Construct foreign keys for the history table
                self.target_history_fks_ = []
                for x_col, rdb_ref in key_mappings.items():
                    if "." in rdb_ref:
                        rdb_table, rdb_col = rdb_ref.split(".", 1)
                        # (this_table, this_col, other_table, other_col)
                        self.target_history_fks_.append((x_col, rdb_table, rdb_col))

        # 2. RDB Transformation (Augments RDB using stored history)
        self.rdb_ = self._prepare_rdb(rdb)

        # 3. Downsampling (Modifies X and y for training)
        if len(X) > self.config.max_train_samples:
            data = X
            target_col = y.name or "_RDBL_target"
            data[target_col] = y
            
            task_type = "regression" if isinstance(self, RegressorMixin) else "classification"
            
            downsampled_data = self._downsample(
                data, target_col, task_type, 
                self.config.max_train_samples, 
                self.config.stratified_sampling
            )
            X = downsampled_data.drop(columns=[target_col])
            y = downsampled_data[target_col]

        # 4. Feature Augmentation
        logger.info("Computing DFS features...")
        dfs_config = self.config.dfs or DFSConfig()
        
        X_dfs = fastdfs.compute_dfs_features(
            self.rdb_,
            X,
            key_mappings=key_mappings,
            cutoff_time_column=cutoff_time_column,
            config=dfs_config
        )
        logger.debug(f"DFS features: {X_dfs.columns.tolist()}")

        # 5. Preprocessing
        logger.info("Preprocessing augmented features ...")
        self.preprocessor_ = TabularPreprocessor(
            ag_config=self.config.ag_config,
            temporal_diff_config=self.config.temporal_diff,
            cutoff_time=cutoff_time_column
        )
        X_transformed = self.preprocessor_.fit(X_dfs).transform(X_dfs)
        
        # 6. Model Training
        logger.info("Fitting base estimator ...")
        self.base_estimator.fit(X_transformed, y, **kwargs)
        
        return self

    def _predict_common(self, X: pd.DataFrame, rdb: Optional[RDB], method: str, **kwargs):
        # 0. Copy and ensure keys are string
        X = X.copy()
        if self.key_mappings_:
            self._ensure_keys_are_strings(X, self.key_mappings_)

        # 2. RDB Selection
        if rdb is None:
            selected_rdb = self.rdb_
        else:
            # Augment new RDB with stored training history!
            selected_rdb = self._prepare_rdb(rdb)
            
        # 3. Feature Augmentation
        logger.info("Computing DFS features...")
        
        dfs_config = self.config.dfs or DFSConfig()
        
        X_dfs = fastdfs.compute_dfs_features(
            selected_rdb, 
            X, 
            key_mappings=self.key_mappings_, 
            cutoff_time_column=self.cutoff_time_column_, 
            config=dfs_config
        )


        # 4. Preprocessing
        logger.info("Preprocessing augmented features ...")
        X_transformed = self.preprocessor_.transform(X_dfs)
        
        # 5. Prediction
        logger.info("Making predictions ...")
        predict_func = getattr(self.base_estimator, method)
        
        if self.config.predict_batch_size and len(X_transformed) > self.config.predict_batch_size:
            results = []
            for i in range(0, len(X_transformed), self.config.predict_batch_size):
                batch = X_transformed.iloc[i:i+self.config.predict_batch_size]
                results.append(predict_func(batch, **kwargs))
             
            if isinstance(results[0], dict):
                # Aggregate dictionary results
                aggregated = {}
                for key in results[0].keys():
                    key_results = [r[key] for r in results]
                    if isinstance(key_results[0], np.ndarray):
                        aggregated[key] = np.concatenate(key_results)
                    elif isinstance(key_results[0], (pd.Series, pd.DataFrame)):
                        aggregated[key] = pd.concat(key_results, axis=0)
                    else:
                        print(f"Warning: Unexpected type of key_results: {type(key_results[0])} when aggregating results for key {key}, skipping this key")
                return aggregated
            elif isinstance(results[0], np.ndarray):
                return np.concatenate(results)
            elif isinstance(results[0], (pd.Series, pd.DataFrame)):
                return pd.concat(results, axis=0)
            else:
                return np.concatenate(results)
        else:
            return predict_func(X_transformed, **kwargs)

class RDBLearnClassifier(RDBLearnEstimator, ClassifierMixin):
    def predict(self, X: pd.DataFrame, rdb: Optional[RDB] = None, **kwargs):
        return self._predict_common(X, rdb, method="predict", **kwargs)

    def predict_proba(self, X: pd.DataFrame, rdb: Optional[RDB] = None, **kwargs):
        return self._predict_common(X, rdb, method="predict_proba", **kwargs)

class RDBLearnRegressor(RDBLearnEstimator, RegressorMixin):
    def predict(self, X: pd.DataFrame, rdb: Optional[RDB] = None, **kwargs):
        return self._predict_common(X, rdb, method="predict", **kwargs)
