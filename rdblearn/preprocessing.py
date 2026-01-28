from typing import Optional, Dict, Any, List, Set, Tuple
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from loguru import logger
from .config import TemporalDiffConfig


class TemporalDiffTransformer:
    def __init__(self, config: TemporalDiffConfig):
        self.config = config
        self.timestamp_columns_: List[str] = []
        self.is_fitted_ = False

    def _sanitize_column_name(self, col: str) -> str:
        sanitized = col.replace('(', '_').replace(')', '').replace('.', '_')
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        sanitized = sanitized.strip('_')
        return sanitized

    def fit(self, X_dfs: pd.DataFrame) -> 'TemporalDiffTransformer':
        self.timestamp_columns_ = [
            col for col in X_dfs.columns
            if '_epochtime' in col and col not in self.config.exclude_columns
        ]
        if self.timestamp_columns_:
            logger.info(f"TemporalDiffTransformer: Found {len(self.timestamp_columns_)} timestamp columns for transformation.")
        else:
            logger.info("TemporalDiffTransformer: No timestamp columns detected.")
        self.is_fitted_ = True
        return self

    def transform(self, X_dfs: pd.DataFrame, cutoff_time: pd.Series) -> pd.DataFrame:
        if not self.is_fitted_ or not self.timestamp_columns_:
            return X_dfs

        result = X_dfs.copy()
        cutoff_nano = (cutoff_time.astype('datetime64[ns]') - np.array(0).astype('datetime64[ns]')).astype('int64')

        for col in self.timestamp_columns_:
            if col not in result.columns:
                continue

            # Timestamp columns already converted to int64 nanoseconds by fastdfs
            timestamp_nano = result[col].values

            # Calculate time difference and convert to float (following RDBColumnDType.float_t practice)
            time_diff = (cutoff_nano - timestamp_nano).astype('float64')

            sanitized_name = self._sanitize_column_name(col)
            feature_name = f"{sanitized_name}_diff"

            result[feature_name] = time_diff
            result = result.drop(columns=[col])

        logger.info(f"TemporalDiffTransformer: Generated {len(self.timestamp_columns_)} temporal difference features.")
        return result

class TabularPreprocessor:
    def __init__(self, ag_config: Optional[Dict[str, Any]] = None, temporal_diff_config: Optional[TemporalDiffConfig] = None, cutoff_time: Optional[str] = None):
        self.ag_config = ag_config or {}
        self.temporal_diff_config = temporal_diff_config
        self.cutoff_time = cutoff_time
        self.label_encoders = {}
        self.feature_generator = None
        self.temporal_transformer = None
        if self.temporal_diff_config and self.temporal_diff_config.enabled:
            self.temporal_transformer = TemporalDiffTransformer(self.temporal_diff_config)

    def fit(self, X: pd.DataFrame):
        """Fit the preprocessor on training data."""
        X_processed = X.copy()

        # 1. Fit temporal features if enabled
        if self.temporal_transformer:
            self.temporal_transformer.fit(X_processed)
        
        # 2. Fit LabelEncoders
        cat_columns = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            self.label_encoders[col] = le
            
        # 3. Fit AutoGluon Feature Generator
        default_ag_config = {
            "enable_datetime_features": True,
            "enable_raw_text_features": False,
            "enable_text_special_features": False,
            "enable_text_ngram_features": False,
        }
        if self.ag_config:
            default_ag_config.update(self.ag_config)
            
        self.feature_generator = AutoMLPipelineFeatureGenerator(**default_ag_config)
        self.feature_generator.fit(X=X_processed)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        if self.feature_generator is None:
            raise RuntimeError("Preprocessor not fitted.")

        X_processed = X.copy()

        # 1. Apply temporal difference transformation
        if self.temporal_transformer and self.cutoff_time is not None:
            cutoff_time = X_processed[self.cutoff_time]
            X_processed = self.temporal_transformer.transform(X_processed, cutoff_time)
        
        # 2. Apply LabelEncoders
        for col, le in self.label_encoders.items():
            if col in X_processed.columns:
                vals = X_processed[col].astype(str)
                unseen_mask = ~vals.isin(le.classes_)
                if unseen_mask.any():
                    le.classes_ = np.append(le.classes_, vals[unseen_mask].unique())
                
                X_processed[col] = le.transform(vals)
        
        # 3. Apply AutoGluon Feature Generator
        return self.feature_generator.transform(X_processed)
