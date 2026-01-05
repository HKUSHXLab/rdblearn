from pydantic import BaseModel, Field
from typing import Optional, Union, Dict, Any
from fastdfs import DFSConfig

class RDBLearnConfig(BaseModel):
    """
    Configuration for RDBLearnEstimator.
    """
    # DFS Configuration (passed to fastdfs)
    # If None, defaults to: {"max_depth": 3, "agg_primitives": ["max", "min", "mean", "count", "mode", "std"], "engine": "dfs2sql"}
    dfs: Optional[DFSConfig] = None
    
    # Preprocessing Configuration (passed to AutoGluon feature generator)
    # If None, defaults to: {"enable_datetime_features": True, "enable_raw_text_features": False, ...}
    ag_config: Optional[Dict[str, Any]] = None
    
    # Sampling Configuration
    max_train_samples: int = 10000
    stratified_sampling: bool = False  # Ignored for RDBLearnRegressor
    
    # Prediction Configuration
    predict_batch_size: int = 5000

    class Config:
        arbitrary_types_allowed = True
