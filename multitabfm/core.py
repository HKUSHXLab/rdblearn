from typing import Optional, List, Tuple, Union, Set
import pandas as pd
import numpy as np
import warnings

from .feature_engineer import generate_features
from .model import AGAdapter, CustomModelAdapter, CustomTabPFN
from .evaluation import compute_metrics
from .utils import load_dataset


def _is_array_like(value: object) -> bool:
    """Return True for list/tuple/np.ndarray values."""
    return isinstance(value, (list, tuple, np.ndarray))


def _filter_array_columns(
    df: pd.DataFrame,
    protected_cols: Set[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns containing array-like objects, except for protected ones."""
    if df.empty:
        return df.copy(), []

    array_cols: List[str] = []
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        sample = df[col].dropna().head(100)
        if sample.empty:
            continue
        if sample.apply(_is_array_like).any():
            if col in protected_cols:
                raise ValueError(
                    f"Column '{col}' contains array-like values but is required for DFS joins. "
                    "Please sanitize this column before enabling DFS."
                )
            array_cols.append(col)

    sanitized = df.drop(columns=array_cols, errors="ignore").copy()
    return sanitized, array_cols


def _coerce_array_columns_to_strings(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert array-like entries into delimited strings so models see hashable values."""
    if not columns:
        return df

    def _to_string(value: object) -> object:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return value
        if _is_array_like(value):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            return ",".join(map(str, value))
        return value

    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = result[col].apply(_to_string)
    return result


def _merge_original_and_dfs(
    original: pd.DataFrame,
    dfs_features: pd.DataFrame,
    dfs_input_cols: List[str]
) -> pd.DataFrame:
    """Append DFS-produced features back onto the original frame."""
    extra_features = dfs_features.drop(columns=dfs_input_cols, errors="ignore")
    combined = pd.concat([
        original.reset_index(drop=True),
        extra_features.reset_index(drop=True)
    ], axis=1)
    return combined


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

        if enable_dfs:
            protected_cols = set(key_mappings.keys())
            if time_column:
                protected_cols.add(time_column)

            sanitized_train, dropped_train = _filter_array_columns(X_train, protected_cols)
            sanitized_test, dropped_test = _filter_array_columns(X_test, protected_cols)
            dropped_columns = sorted(set(dropped_train) | set(dropped_test))
            if dropped_columns:
                warnings.warn(
                    "Excluded array-valued columns from DFS input: "
                    + ", ".join(dropped_columns)
                )
                X_train = _coerce_array_columns_to_strings(X_train, dropped_columns)
                X_test = _coerce_array_columns_to_strings(X_test, dropped_columns)

            dfs_input_cols = list(sanitized_train.columns)

            train_dfs = generate_features(sanitized_train, rdb, key_mappings, time_column, self.dfs_config)
            test_dfs = generate_features(sanitized_test, rdb, key_mappings, time_column, self.dfs_config)

            train_features = _merge_original_and_dfs(X_train, train_dfs, dfs_input_cols)
            test_features = _merge_original_and_dfs(X_test, test_dfs, dfs_input_cols)
        else:
            train_features = X_train
            test_features = X_test
        train_data = pd.concat([train_features, Y_train], axis=1)

        # 2. Train model
        self.fit(train_data, label_column=target_column, task_type=task_type, eval_metric=eval_metrics[0] if eval_metrics else None)

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