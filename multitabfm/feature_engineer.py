from typing import Optional, Union
import pandas as pd
from fastdfs.api import compute_dfs_features
from fastdfs.dfs import DFSConfig

    
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
    
    return compute_dfs_features(
        rdb=rdb,
        target_dataframe=target_df,
        key_mappings=key_mappings,
        cutoff_time_column=time_column,
        config_overrides=dfs_config or {}
    )

