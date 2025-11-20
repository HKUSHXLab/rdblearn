from typing import Optional, List, Tuple, Union, Set
import pandas as pd
import numpy as np
import warnings

from .feature_engineer import generate_features,prepare_feature_inputs,merge_original_and_dfs,ag_transform
from .model import AGAdapter, CustomModelAdapter, CustomTabPFN
from .evaluation import compute_metrics
from .utils import load_dataset


class MultiTabFM:
    """Simplified multi-table feature modeling framework."""
    
    def __init__(self, dfs_config: Optional[dict] = None, model_config: Optional[dict] = None, batch_size: int = 5000, custom_model_class: Optional[type] = None):
        self.model_config = model_config or {}
        self.dfs_config = dfs_config or {}
        self.batch_size = batch_size
        self.custom_model_class = custom_model_class
        
        # Choose adapter based on whether custom model is provided
        if custom_model_class is not None:
            self.model_adapter = CustomModelAdapter(model_config, custom_model_class)
        else:
            self.model_adapter = AGAdapter(model_config)

    def fit(self, train_features: pd.DataFrame, label_column: str, task_type:Optional[str]=None, eval_metric:Optional[str] = None) -> None:
        """Fit the model on feature-augmented training data."""
        return self.model_adapter.fit(train_features, label_column, task_type, eval_metric)

    def predict_proba(self, test_features: pd.DataFrame) -> pd.DataFrame:
        proba = self.model_adapter.predict_proba(test_features, batch_size=self.batch_size)
        return proba

    def predict(self, test_features: pd.DataFrame) -> pd.Series:
        return self.model_adapter.predict(test_features, batch_size=self.batch_size)

    def evaluate(self, labels: pd.Series, preds_or_proba: Union[pd.DataFrame, np.ndarray, pd.Series], metrics: Optional[List[str]] = None) -> dict:
        """Evaluate predictions against true labels for classification or regression."""
        return compute_metrics(labels, preds_or_proba, metrics)

    def train_and_predict(self,
                         rdb_data_path: str,
                         task_data_path: str,
                         enable_dfs: bool = True,
                         eval_metrics: Optional[List[str]] = None) -> Tuple[Union[pd.DataFrame, np.ndarray, pd.Series], Optional[dict]]:
        """Main API: End-to-end training and prediction from data paths.
        
        Args:
            rdb_data_path: Path to RDB data directory (e.g., "data/rel-event")
            task_data_path: Path to task data directory (e.g., "data/rel-event/user-ignore")
            enable_dfs: Whether to enable deep feature synthesis to augment features
            eval_metrics: Optional evaluation metrics
            
        Returns:
            Tuple of (predictions, metrics)
        """
        
        # Load dataset
        train_data, test_data, metadata, rdb = load_dataset(rdb_data_path, task_data_path)
        
        # Parse metadata
        key_mappings = {k: v for d in metadata['key_mappings'] for k, v in d.items()}
        time_column = metadata['time_column']
        target_column = metadata['target_column']
        # if metadata has task_type, use it; else task_type is None
        task_type = metadata.get('task_type', None)

        # Prepare target dataframes
        X_train, Y_train = train_data.drop(columns=[target_column]), train_data[target_column]
        X_test, Y_test = test_data.drop(columns=[target_column]), test_data[target_column]

        (
            base_train,
            base_test,
            sanitized_train,
            sanitized_test,
            dfs_input_cols,
        ) = prepare_feature_inputs(X_train, X_test, key_mappings, time_column)

        if enable_dfs:
            train_dfs = generate_features(
                sanitized_train,
                rdb,
                key_mappings,
                time_column,
                self.dfs_config,
            )
            test_dfs = generate_features(
                sanitized_test,
                rdb,
                key_mappings,
                time_column,
                self.dfs_config,
            )

            train_features = merge_original_and_dfs(base_train, train_dfs, dfs_input_cols)
            test_features = merge_original_and_dfs(base_test, test_dfs, dfs_input_cols)
        else:
            train_features = base_train
            test_features = base_test
        
        X_train_transformed, y_train_transformed, X_test_transformed = ag_transform(
            train_features, Y_train, test_features, task_type=task_type
        )
        
        train_data = pd.concat([X_train_transformed, y_train_transformed], axis=1)
        # 2. Train model
        self.fit(train_data, label_column=target_column, task_type=task_type, eval_metric=eval_metrics[0] if eval_metrics else None)

        # 3. Predict
        if task_type == "regression":
            preds_or_proba = self.predict(X_test_transformed)
        else:
            preds_or_proba = self.predict_proba(X_test_transformed)

        # 4. Evaluate if possible
        metrics = None
        if eval_metrics and Y_test is not None:
            labels = Y_test
            metrics = self.evaluate(labels, preds_or_proba, metrics=eval_metrics)

        return preds_or_proba, metrics