from typing import Optional, List, Tuple, Union
import pandas as pd
import numpy as np

from .feature_engineer import generate_features
from .model import AGAdapter
from .evaluation import compute_metrics
from .utils import load_dataset, prepare_target_dataframes

class MultiTabFM:
    """Simplified multi-table feature modeling framework."""
    
    def __init__(self, dfs_config: Optional[dict] = None, model_config: Optional[dict] = None):
        self.dfs_config = dfs_config or {}
        self.model_config = model_config or {}
        self.model_adapter = AGAdapter()

    def fit(self, train_features: pd.DataFrame, label_column: str, model_config: Optional[dict] = None) -> None:
        """Fit the model on feature-augmented training data."""
        return self.model_adapter.fit(train_features, label_column, model_config)

    def predict_proba(self, test_features: pd.DataFrame) -> np.ndarray:
        proba = self.model_adapter.predict_proba(test_features)
        return proba

    def evaluate(self, labels: pd.Series, proba: np.ndarray, metrics: Optional[List[str]] = None) -> dict:
        """Evaluate predictions against true labels."""
        return compute_metrics(labels, proba, metrics)

    def train_and_predict(self,
                         rdb_data_path: str,
                         task_data_path: str,
                         *,
                         dfs_config: Optional[dict] = None,
                         model_config: Optional[dict] = None,
                         eval_metrics: Optional[List[str]] = None) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[dict]]:
        """Main API: End-to-end training and prediction from data paths.
        
        Args:
            rdb_data_path: Path to RDB data directory (e.g., "data/rel-event")
            task_data_path: Path to task data directory (e.g., "data/rel-event/user-ignore")
            dfs_config: Optional DFS configuration
            model_config: Optional model configuration
            eval_metrics: Optional evaluation metrics
            
        Returns:
            Tuple of (predictions, metrics)
        """
        
        # Load dataset
        train_data, test_data, metadata, rdb = load_dataset(rdb_data_path, task_data_path)
        
        # Prepare target dataframes
        train_df, test_df = prepare_target_dataframes(train_data, test_data, metadata)
        
        # Parse metadata
        key_mappings = {k: v for d in metadata['key_mappings'] for k, v in d.items()}
        time_column = metadata['time_column']
        target_column = metadata['target_column']
        
        # 1. Generate features
        train_features = generate_features(train_df, rdb, key_mappings, time_column, dfs_config)
        test_features = generate_features(test_df, rdb, key_mappings, time_column, dfs_config)
        
        # 2. Train model
        self.fit(train_features, label_column=target_column, model_config=model_config)
        
        # 3. Predict
        proba = self.predict_proba(test_features)
        
        # 4. Evaluate if possible
        metrics = None
        if eval_metrics and target_column in test_features.columns:
            labels = test_features[target_column]
            metrics = self.evaluate(labels, proba, metrics=eval_metrics)
            
        return proba, metrics

