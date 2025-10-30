"""Utility functions for loading data and running experiments."""

import pandas as pd
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from fastdfs.api import load_rdb


def load_dataset(rdb_data_path: str, task_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Any]:
    """Load complete dataset from paths.
    
    Args:
        rdb_data_path: Path to RDB data directory (e.g., "data/rel-event")
        task_data_path: Path to task data directory (e.g., "data/rel-event/user-ignore")
        
    Returns:
        Tuple of (train_df, test_df, metadata, rdb)
    """
    # Load RDB
    rdb = load_rdb(rdb_data_path)
    
    # Load task data
    task_path = Path(task_data_path)
    train_df = pd.read_parquet(task_path / "train.pqt")
    test_df = pd.read_parquet(task_path / "test.pqt")
    
    # Load metadata
    with open(task_path / "metadata.yaml", "r") as f:
        metadata = yaml.safe_load(f)
    
    return train_df, test_df, metadata, rdb


def prepare_target_dataframes(train_data: pd.DataFrame, test_data: pd.DataFrame, metadata: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare target dataframes with ID + time + label columns for DFS.
    
    Args:
        train_data: Raw training data
        test_data: Raw test data  
        metadata: Dataset metadata containing key_mappings, time_column, target_column
        
    Returns:
        Tuple of (train_df, test_df) ready for DFS
    """
    # Extract required columns from metadata
    target_column = metadata["target_column"]
    key_mappings = {k: v for d in metadata.get("key_mappings", []) for k, v in d.items()}
    id_columns = list(key_mappings.keys())
    time_column = metadata["time_column"]
    
    # Prepare target dataframes (ID + time columns)
    train_df = train_data[id_columns + [time_column]].copy()
    test_df = test_data[id_columns + [time_column]].copy()
    
    # Add labels
    train_df[target_column] = train_data[target_column]
    test_df[target_column] = test_data[target_column]
    
    return train_df, test_df
