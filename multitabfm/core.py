from typing import Optional, List, Tuple, Union
import pandas as pd
import numpy as np

from .feature_engineer import generate_features
from .model import AGAdapter
from .evaluation import compute_metrics
from .utils import load_dataset

class MultiTabFM:
    """Simplified multi-table feature modeling framework."""
    
    def __init__(self, dfs_config: Optional[dict] = None,model_config: Optional[dict] = None):
        self.model_config = model_config or {}
        self.dfs_config = dfs_config or {}
        self.model_adapter = AGAdapter(model_config)

    def fit(self, train_features: pd.DataFrame, label_column: str, task_type: str) -> None:
        """Fit the model on feature-augmented training data."""
        return self.model_adapter.fit(train_features, label_column, task_type)

    def predict_proba(self, test_features: pd.DataFrame) -> pd.DataFrame:
        proba = self.model_adapter.predict_proba(test_features)
        return proba

    def predict(self, test_features: pd.DataFrame) -> pd.Series:
        return self.model_adapter.predict(test_features)

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

        if enable_dfs:
        # augment features using DFS
            train_features = generate_features(X_train, rdb, key_mappings, time_column, self.dfs_config)
            test_features = generate_features(X_test, rdb, key_mappings, time_column, self.dfs_config)
        else:
            train_features = X_train
            test_features = X_test
        train_data = pd.concat([train_features, Y_train], axis=1)

        # 2. Train model
        self.fit(train_data, label_column=target_column, task_type=task_type)

        # 3. Predict
        if task_type == "regression":
            preds_or_proba = self.predict(test_features)
        else:
            preds_or_proba = self.predict_proba(test_features)

        # 4. Evaluate if possible
        metrics = None
        if eval_metrics and Y_test is not None:
            labels = Y_test
            metrics = self.evaluate(labels, preds_or_proba, metrics=eval_metrics)

        return preds_or_proba, metrics