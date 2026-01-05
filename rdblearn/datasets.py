from typing import List, Optional, Dict
import pandas as pd
from pydantic import BaseModel
from fastdfs import RDB

class TaskMetadata(BaseModel):
    key_mappings: Dict[str, str]
    target_col: str
    time_col: Optional[str] = None
    task_type: Optional[str] = None
    evaluation_metric: Optional[str] = None

class Task(BaseModel):
    name: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    val_df: Optional[pd.DataFrame] = None
    metadata: TaskMetadata

class RDBDataset:
    def __init__(self, rdb: RDB, tasks: List[Task]):
        self.rdb = rdb
        self.tasks = {t.name: t for t in tasks}

    @classmethod
    def from_relbench(cls, dataset_name: str):
        # Uses fastdfs.adapter.relbench to load RDB
        # Loads tasks using relbench library
        raise NotImplementedError("RelBench adapter not implemented yet.")

    @classmethod
    def from_4dbinfer(cls, dataset_name: str):
        # Uses fastdfs.adapter.dbinfer
        raise NotImplementedError("4DBInfer adapter not implemented yet.")
