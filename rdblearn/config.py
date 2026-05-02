from pydantic import BaseModel, Field
from typing import Optional, Union, Dict, Any, List
from dataclasses import dataclass, field
import os
from fastdfs import DFSConfig


class TemporalDiffConfig(BaseModel):
    """Configuration for temporal difference feature generation."""
    enabled: bool = True
    # Columns to explicitly exclude from transformation
    exclude_columns: List[str] = Field(default_factory=list)


class RDBLearnConfig(BaseModel):
    """
    Configuration for RDBLearnEstimator.
    """
    # DFS Configuration (passed to fastdfs)
    # If None, defaults to: {"max_depth": 2, "agg_primitives": ["max", "min", "mean", "count", "mode", "std"], "engine": "dfs2sql"}
    dfs: Optional[DFSConfig] = None
    
    # Preprocessing Configuration (passed to AutoGluon feature generator)
    # If None, defaults to: {"enable_datetime_features": True, "enable_raw_text_features": False, ...}
    ag_config: Optional[Dict[str, Any]] = None
    
    # Sampling Configuration
    max_train_samples: int = 10000
    stratified_sampling: bool = False  # Ignored for RDBLearnRegressor
    
    # Target History Augmentation
    enable_target_augmentation: bool = True
    
    # Prediction Configuration
    predict_batch_size: int = 5000

    # Temporal Difference Configuration (post-DFS transformation)
    temporal_diff: Optional[TemporalDiffConfig] = TemporalDiffConfig()

    # DFS Disk Cache Configuration
    dfs_cache_enabled: bool = False
    dfs_cache_dir: str = Field(
        default_factory=lambda: os.environ.get("RDBLEARN_DFS_CACHE_DIR", "/tmp/rdblearn_dfs_cache")
    )
    dfs_cache_rebuild: bool = False
    # requested: compute cache at requested depth
    # fixed_max: compute at dfs_cache_max_depth (if set), then slice down to requested depth
    dfs_cache_max_depth_mode: str = "requested"
    dfs_cache_max_depth: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True
