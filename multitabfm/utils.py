"""Utility functions for loading data and running experiments."""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import yaml
from fastdfs import load_rdb

def _read_npz_df(file_path: Path) -> pd.DataFrame:
    """Read a .npz file and convert to DataFrame."""
    data = np.load(file_path)
    df = pd.DataFrame({key: data[key] for key in data.files})
    return df


def load_dataset(rdb_data_path: str, task_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Any]:
    """Load complete dataset from paths.
    
    Args:
        rdb_data_path: Path to RDB data directory (e.g., "data/rel-event")
        task_data_path: Path to task data directory (e.g., "data/rel-event/user-ignore")
        
    Returns:
        Tuple of (train_df, test_df, metadata, rdb)
    """
    # Load RDB (lazy, dynamic import to avoid static dependency issues)
    rdb = load_rdb(rdb_data_path)
    task_path = Path(task_data_path)
    
    # Find files ending with train.pqt/test.pqt or train.npz/test.npz
    train_pqt_files = list(task_path.glob("*train.pqt"))
    test_pqt_files = list(task_path.glob("*test.pqt"))
    train_npz_files = list(task_path.glob("*train.npz"))
    test_npz_files = list(task_path.glob("*test.npz"))

    if train_pqt_files and test_pqt_files:
        # Use the first match
        train_df = pd.read_parquet(train_pqt_files[0])
        test_df = pd.read_parquet(test_pqt_files[0])
    elif train_npz_files and test_npz_files:
        # Use the first match
        train_df = _read_npz_df(train_npz_files[0])
        test_df = _read_npz_df(test_npz_files[0])
    else:
        raise FileNotFoundError(
            f"Could not find train/test as .pqt or .npz in {task_path}. "
            "Expected files ending with 'train.pqt' & 'test.pqt' or 'train.npz' & 'test.npz'."
        )
    
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