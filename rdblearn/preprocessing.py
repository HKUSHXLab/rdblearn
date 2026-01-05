from typing import Optional, Dict, Any, List, Set, Tuple
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

class TabularPreprocessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feature_generator = None
        self.label_encoders = {} # For categorical columns in X

    def fit(self, X: pd.DataFrame):
        """Fit the preprocessor on training data."""
        # 1. Identify categorical columns and fit LabelEncoders
        X_numeric = X.copy()
        cat_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in cat_columns:
            le = LabelEncoder()
            # Convert to string to handle mixed types
            X_numeric[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            
        # 2. Fit AutoGluon Feature Generator
        default_config = {
            "enable_datetime_features": True,
            "enable_raw_text_features": False,
            "enable_text_special_features": False,
            "enable_text_ngram_features": False,
        }
        if self.config:
            default_config.update(self.config)
            
        self.feature_generator = AutoMLPipelineFeatureGenerator(**default_config)
        self.feature_generator.fit(X_numeric)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        if self.feature_generator is None:
            raise RuntimeError("Preprocessor not fitted.")
            
        X_numeric = X.copy()
        
        # 1. Apply LabelEncoders
        for col, le in self.label_encoders.items():
            if col in X_numeric.columns:
                vals = X_numeric[col].astype(str)
                
                # Handle unseen categories by extending the encoder's classes
                # This mimics the behavior of convert_categoricals_to_numeric in multitabfm
                unseen_mask = ~vals.isin(le.classes_)
                if unseen_mask.any():
                    le.classes_ = np.append(le.classes_, vals[unseen_mask].unique())
                
                X_numeric[col] = le.transform(vals)
        
        # 2. Apply AutoGluon Feature Generator
        return self.feature_generator.transform(X_numeric)

