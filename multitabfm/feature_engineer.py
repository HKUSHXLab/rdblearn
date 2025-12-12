from typing import Optional, List, Tuple, Union, Set, Dict
import pandas as pd
import numpy as np
import warnings
import fastdfs

from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type

    
def generate_features(
    target_df: pd.DataFrame,
    rdb,
    key_mappings: dict,
    time_column: str,
    pipeline: fastdfs.DFSPipeline,
) -> pd.DataFrame:
    """Generate DFS features using fastdfs.api.compute_dfs_features.
    
    Args:
        target_df: Target dataframe with ID and time columns
        rdb: Loaded RDB object from fastdfs
        key_mappings: Dict mapping local columns to RDB table.column (e.g., {'user_id': 'users.user_id'})
        time_column: Name of the time/cutoff column
    
    Returns:
        Feature-augmented dataframe
    """
    if rdb is None:
        raise ValueError("rdb must be provided")

    
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

def prepare_for_dfs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    key_mappings: dict,
    time_column: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove array-valued columns that would break DFS joins.
    
    Args:
        X_train: Training features dataframe
        X_test: Test features dataframe
        key_mappings: Dict mapping local columns to RDB table.column
        time_column: Name of the time/cutoff column
        
    Returns:
        Tuple of (sanitized_train, sanitized_test) ready for DFS
    """
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
    
    return sanitized_train, sanitized_test


def add_dfs_features(
    original: pd.DataFrame,
    dfs_features: pd.DataFrame
) -> pd.DataFrame:
    """Add DFS features to original dataframe, avoiding duplicates.
    
    Args:
        original: Original dataframe with all features
        dfs_features: DFS-generated features dataframe
        
    Returns:
        Combined dataframe with original features + new DFS features
    """
    # Get only new columns from DFS output (avoid duplicates)
    new_cols = [col for col in dfs_features.columns if col not in original.columns]
    return pd.concat([
        original.reset_index(drop=True),
        dfs_features[new_cols].reset_index(drop=True)
    ], axis=1)


def ag_label_transform(
    y_train: pd.Series,
    y_test: Optional[pd.Series] = None,
    normalize_regression: bool = True,
    skew_threshold: float = 0.99,
    impute_strategy: str = "median"
) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Transform labels using AutoGluon's LabelCleaner for non-regression tasks.
    For regression problems, optionally normalize the labels using the normalize_numeric_columns logic.
    
    Args:
        y_train: Training labels series
        y_test: Optional test labels series
        normalize_regression: Whether to normalize regression labels (default: True for 4DBInfer, set False for RelBench)
        skew_threshold: Threshold for detecting skewed distributions (default: 0.99)
        impute_strategy: Strategy for imputing missing values (default: "median")
        
    Returns:
        Tuple of (y_train_transformed, y_test_transformed)
    """
    # Infer problem type
    problem_type = infer_problem_type(y_train)
    
    if problem_type == 'regression' and normalize_regression:
        # For regression with normalization enabled, skip LabelCleaner and apply normalization
        y_train_data = y_train.to_numpy().reshape(-1, 1)
        skew_score = pd.Series(y_train_data.flatten()).skew()
        
        # Use QuantileTransformer for highly skewed data, StandardScaler otherwise
        if np.abs(skew_score) > skew_threshold:
            print(f"Label: Skew detected (skew={skew_score:.4f}). Using QuantileTransformer.")
            scaler = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=impute_strategy)),
                ('scaler', QuantileTransformer(output_distribution='normal'))
            ])
        else:
            print(f"Label: Normal distribution (skew={skew_score:.4f}). Using StandardScaler.")
            scaler = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=impute_strategy)),
                ('scaler', StandardScaler())
            ])
        
        # Fit and transform training labels
        y_train_normalized = scaler.fit_transform(y_train_data).flatten()
        y_train_transformed = pd.Series(y_train_normalized, index=y_train.index)
        
        # Transform test labels with the same scaler
        y_test_transformed = None
        if y_test is not None:
            y_test_data = y_test.to_numpy().reshape(-1, 1)
            y_test_normalized = scaler.transform(y_test_data).flatten()
            y_test_transformed = pd.Series(y_test_normalized, index=y_test.index)
    else:
        # For non-regression tasks or regression without normalization, use LabelCleaner
        label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y_train)
        
        # Transform training labels
        y_train_transformed = label_cleaner.transform(y_train)
        
        # Transform test labels if provided
        y_test_transformed = None
        if y_test is not None:
            y_test_transformed = label_cleaner.transform(y_test)
    
    return y_train_transformed, y_test_transformed


def ag_transform(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    feature_generator_config: Optional[dict] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Transform features using categorical conversion and AutoGluon's feature generator.
    
    Args:
        X_train: Training features dataframe
        X_test: Optional test features dataframe
        feature_generator_config: Optional config for AutoMLPipelineFeatureGenerator
        
    Returns:
        Tuple of (X_train_transformed, X_test_transformed)
    """
    # Step 1: Convert categorical columns to numeric to avoid ag dropping high cardinality categoricals
    X_train_numeric, X_test_numeric = convert_categoricals_to_numeric(X_train, X_test)
    
    # Step 2: Setup feature generator with default config
    default_config = {
        "enable_datetime_features": True,
        "enable_raw_text_features": False,
        "enable_text_special_features": False,
        "enable_text_ngram_features": False,
    }
    if feature_generator_config:
        default_config.update(feature_generator_config)
    
    feature_generator = AutoMLPipelineFeatureGenerator(**default_config)
    
    # Step 3: Transform training features
    X_train_transformed = feature_generator.fit_transform(X_train_numeric)
    
    # Step 4: Transform test features if provided
    X_test_transformed = None
    if X_test_numeric is not None:
        X_test_transformed = feature_generator.transform(X_test_numeric)
    
    return X_train_transformed, X_test_transformed


def convert_categoricals_to_numeric(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Convert all categorical (object/category dtype) columns to numeric using label encoding.
    
    Args:
        train_df: Training dataframe
        test_df: Optional test dataframe
        
    Returns:
        Tuple of (train_transformed, test_transformed)
    """
    train_transformed = train_df.copy()
    test_transformed = test_df.copy() if test_df is not None else None
    
    # Identify categorical columns
    cat_columns = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in cat_columns:
        le = LabelEncoder()
        
        # Fit on training data
        train_transformed[col] = le.fit_transform(train_df[col].astype(str))
        
        # Transform test data, handling unseen categories
        if test_transformed is not None:
            # Get unique values in test that aren't in train
            test_vals = test_df[col].astype(str)
            unseen_mask = ~test_vals.isin(le.classes_)
            
            # Add unseen categories to the encoder
            if unseen_mask.any():
                le.classes_ = np.append(le.classes_, test_vals[unseen_mask].unique())
            
            test_transformed[col] = le.transform(test_vals)
    
    return train_transformed, test_transformed


