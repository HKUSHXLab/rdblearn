from typing import Optional, List, Tuple, Union
import pandas as pd
import numpy as np

from .feature_engineer import generate_features, prepare_feature_inputs, merge_original_and_dfs, ag_transform
from .model import CustomTabPFN
from .evaluation import compute_metrics
from .utils import load_dataset


class MultiTabFM:
    
    def __init__(self, dfs_config: Optional[dict] = None, model_config: Optional[dict] = None, batch_size: int = 5000):
        self.model_config = (model_config or {}).copy()
        self.dfs_config = dfs_config or {}
        
        # Extract batch_size from model_config if present, otherwise use argument
        if 'batch_size' in self.model_config:
            self.batch_size = self.model_config.pop('batch_size')
        else:
            self.batch_size = batch_size
            
        self.custom_model_class = self.model_config.pop('custom_model_class', None)
        self.model = None

    def _batch_predict_proba(self, X: pd.DataFrame, batch_size: int) -> pd.DataFrame:
        """Batch-wise probability prediction. Returns DataFrame."""
        proba_batches = []
        num_rows = len(X)
        
        for start in range(0, num_rows, batch_size):
            end = min(start + batch_size, num_rows)
            batch = X.iloc[start:end]
            proba_batch = self.model.predict_proba(batch)
            proba_batches.append(proba_batch)
        
        # Concatenate all batches
        if len(proba_batches) == 1:
            return proba_batches[0]
        else:
            return pd.concat(proba_batches, axis=0).reset_index(drop=True)

    def _batch_predict(self, X: pd.DataFrame, batch_size: int) -> pd.Series:
        """Batch-wise value prediction. Returns Series."""
        pred_batches = []
        num_rows = len(X)
        
        for start in range(0, num_rows, batch_size):
            end = min(start + batch_size, num_rows)
            batch = X.iloc[start:end]
            pred_batch = self.model.predict(batch)
            pred_batches.append(pred_batch)
        
        # Concatenate all batches
        if len(pred_batches) == 1:
            return pred_batches[0]
        else:
            return pd.concat(pred_batches, axis=0).reset_index(drop=True)

    def fit(self, train_features: pd.DataFrame, label_column: str, task_type: Optional[str] = None, eval_metric: Optional[str] = None) -> None:
        """Fit the model on feature-augmented training data."""
        # Separate features and labels
        features = train_features.drop(columns=[label_column])
        labels = train_features[label_column]
        
        # Add task_type to model config
        model_config = self.model_config.copy()
        model_config['task_type'] = task_type
        model_config['eval_metric'] = eval_metric
        
        # Use custom model class if provided, otherwise default to CustomTabPFN
        model_class = self.custom_model_class or CustomTabPFN
        
        # Initialize and fit the model
        self.model = model_class(**model_config)
        self.model.fit(X=features, y=labels)

    def predict_proba(self, test_features: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities using batch processing."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._batch_predict_proba(test_features, self.batch_size)

    def predict(self, test_features: pd.DataFrame) -> pd.Series:
        """Predict target values using batch processing."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._batch_predict(test_features, self.batch_size)

    def evaluate(self, labels: pd.Series, preds_or_proba: Union[pd.DataFrame, np.ndarray, pd.Series], metrics: Optional[List[str]] = None) -> dict:
        """Evaluate predictions against true labels for classification or regression."""
        return compute_metrics(labels, preds_or_proba, metrics)

    def train_and_predict(self,
                         rdb_data_path: str,
                         task_data_path: str) -> Tuple[Union[pd.DataFrame, np.ndarray, pd.Series], Optional[dict]]:
        """Main API: End-to-end training and prediction from data paths.
        
        Args:
            rdb_data_path: Path to RDB data directory (e.g., "data/rel-event")
            task_data_path: Path to task data directory (e.g., "data/rel-event/user-ignore")
            
        Returns:
            Tuple of (predictions, metrics)
        """
        
        # Load dataset
        train_data, test_data, metadata, rdb = load_dataset(rdb_data_path, task_data_path)
        
        # Sampling target_table if max_samples is set
        max_samples = self.model_config.get('max_samples')
        if max_samples is not None and len(train_data) > max_samples:
            print(f"Sampling {max_samples} from {len(train_data)} samples for training before DFS...")
            train_data = train_data.sample(n=max_samples, random_state=42).reset_index(drop=True)

        # Parse metadata
        key_mappings = {k: v for d in metadata['key_mappings'] for k, v in d.items()}
        time_column = metadata['time_column']
        target_column = metadata['target_column']
        # Prioritize task_type from metadata, fallback to model_config
        task_type = metadata.get('task_type')
        metric = metadata.get('evaluation_metric')
        eval_metrics = [metric] if metric else None

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

        if self.dfs_config:
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
            train_features.reset_index(drop=True), 
            Y_train.reset_index(drop=True), 
            test_features.reset_index(drop=True)
        )
        # save X_train_transformed
        X_train_transformed.to_parquet('/root/yl_project/multitabfm/debug/X_train_transformed.pqt', index=False)
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