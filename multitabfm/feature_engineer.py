from typing import Optional, Union
import pandas as pd
import fastdfs
from fastdfs.transform import RDBTransformWrapper, RDBTransformPipeline, HandleDummyTable, FeaturizeDatetime
from fastdfs.api import compute_dfs_features
from fastdfs.dfs import DFSConfig
from typing import Optional, List, Tuple, Union, Set
import pandas as pd
import numpy as np
import warnings
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type

    
def generate_features(
    target_df: pd.DataFrame,
    rdb,
    key_mappings: dict,
    time_column: str,
    dfs_config: Optional[Union[dict, DFSConfig]] = None,
) -> pd.DataFrame:
    """Generate DFS features using fastdfs.api.compute_dfs_features.
    
    Args:
        target_df: Target dataframe with ID and time columns
        rdb: Loaded RDB object from fastdfs
        key_mappings: Dict mapping local columns to RDB table.column (e.g., {'user_id': 'users.user_id'})
        time_column: Name of the time/cutoff column
        dfs_config: Optional DFS configuration overrides
    
    Returns:
        Feature-augmented dataframe
    """
    if rdb is None:
        raise ValueError("rdb must be provided")

    effective_config = DFSConfig()
    if dfs_config:
        for key, value in dfs_config.items():
            if hasattr(effective_config, key):
                setattr(effective_config, key, value)

    pipeline = fastdfs.DFSPipeline(
                # transform_pipeline=None,  # No transforms for this test
                transform_pipeline=RDBTransformPipeline([
                    HandleDummyTable(),
                    RDBTransformWrapper(FeaturizeDatetime(features=["year", "month", "day", "hour", "dayofweek"])), #, cyclic=cyclic)),
                ]),
                dfs_config=effective_config
            )
    
    features = pipeline.compute_features(
        rdb=rdb,
        target_dataframe=target_df,
        key_mappings=key_mappings,
        cutoff_time_column=time_column
    )
    features.loc[:, ~features.columns.duplicated()]
    features = features.replace({pd.NA: None})
    return features


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

def merge_original_and_dfs(
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


def prepare_feature_inputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    key_mappings: dict,
    time_column: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Shared preprocessing for DFS-enabled and vanilla flows."""
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

    coerced_train = _coerce_array_columns_to_strings(X_train, dropped_columns)
    coerced_test = _coerce_array_columns_to_strings(X_test, dropped_columns)

    dfs_input_cols = list(sanitized_train.columns)
    return coerced_train, coerced_test, sanitized_train, sanitized_test, dfs_input_cols


def merge_original_and_dfs(
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


def ag_transform(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    feature_generator_config: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], AutoMLPipelineFeatureGenerator, LabelCleaner, str]:
    """
    Transform features and labels using AutoGluon's feature generator and label cleaner.
    
    Args:
        X_train: Training features dataframe
        y_train: Training labels series
        X_test: Optional test features dataframe
        feature_generator_config: Optional config for AutoMLPipelineFeatureGenerator
        
    Returns:
        Tuple of (X_train_transformed, y_train_transformed, X_test_transformed, 
                 feature_generator, label_cleaner, problem_type)
    """
    # Infer problem type if not provided
    problem_type = infer_problem_type(y_train)
    
    # Setup label cleaner
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y_train)
    y_train_transformed = label_cleaner.transform(y_train)
    
    # Setup feature generator with default config
    default_config = {
        "enable_datetime_features": True,
        "enable_raw_text_features": False,
        "enable_text_special_features": False,
        "enable_text_ngram_features": False,
    }
    if feature_generator_config:
        default_config.update(feature_generator_config)
    
    feature_generator = AutoMLPipelineFeatureGenerator(**default_config)
    
    # Transform training features
    X_train_transformed = feature_generator.fit_transform(X_train)
    
    # Transform test features if provided
    X_test_transformed = None
    if X_test is not None:
        X_test_transformed = feature_generator.transform(X_test)
    
    return X_train_transformed, y_train_transformed, X_test_transformed
