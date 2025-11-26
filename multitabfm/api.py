"""
Public API for the multitabfm package.

This module exposes a convenient, single-call function for training and
predicting from raw data paths, mirroring the style of fastdfs.api.
"""
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .core import MultiTabFM

__all__ = [
    "train_and_predict",
]


def train_and_predict(
    rdb_data_path: str,
    task_data_path: str,
    dfs_config: Optional[dict] = None,
    model_config: Optional[dict] = None,
) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[dict]]:
    """End-to-end convenience function to train and predict.

    Args:
        rdb_data_path: Path to the RDB data directory (e.g., "data/rel-event")
        task_data_path: Path to the task data directory (e.g., "data/rel-event/user-ignore")
        dfs_config: Optional DFS configuration dict. If provided, DFS is enabled.
        model_config: Optional model configuration dict

    Returns:
        Tuple of (predictions, metrics). If eval_metrics is provided and labels are
        available in the test set, metrics will be a dict; otherwise None.
    """
    engine = MultiTabFM(dfs_config=dfs_config, model_config=model_config)
    preds, metrics = engine.train_and_predict(
        rdb_data_path=rdb_data_path,
        task_data_path=task_data_path,
    )
    return preds, metrics