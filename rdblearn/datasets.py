from typing import List, Optional, Dict
import os
import pickle
import yaml
import pandas as pd
from pydantic import BaseModel
from fastdfs import RDB, load_rdb

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

    class Config:
        arbitrary_types_allowed = True

class RDBDataset:
    def __init__(self, rdb: RDB, tasks: List[Task]):
        self.rdb = rdb
        self.tasks = {t.name: t for t in tasks}

    def save(self, path: str):
        """Save the dataset to disk."""
        os.makedirs(path, exist_ok=True)
        
        # Save RDB
        rdb_path = os.path.join(path, "rdb")
        self.rdb.save(rdb_path)
        
        # Save Tasks
        tasks_dir = os.path.join(path, "tasks")
        os.makedirs(tasks_dir, exist_ok=True)
        
        for task_name, task in self.tasks.items():
            task_path = os.path.join(tasks_dir, task_name)
            os.makedirs(task_path, exist_ok=True)
            
            # Save DataFrames
            task.train_df.to_parquet(os.path.join(task_path, "train.parquet"))
            task.test_df.to_parquet(os.path.join(task_path, "test.parquet"))
            if task.val_df is not None:
                task.val_df.to_parquet(os.path.join(task_path, "val.parquet"))
            
            # Save Metadata
            task_info = {
                "name": task.name,
                "metadata": task.metadata.model_dump()
            }
            
            with open(os.path.join(task_path, "metadata.yaml"), "w") as f:
                yaml.dump(task_info, f)

    @classmethod
    def load(cls, path: str):
        """Load the dataset from disk."""
        # Load RDB
        rdb_path = os.path.join(path, "rdb")
        rdb = load_rdb(rdb_path)
        
        # Load Tasks
        tasks = []
        tasks_dir = os.path.join(path, "tasks")
        
        if os.path.exists(tasks_dir):
            for task_name in os.listdir(tasks_dir):
                task_path = os.path.join(tasks_dir, task_name)
                if not os.path.isdir(task_path):
                    continue
                
                # Load Metadata
                metadata_path = os.path.join(task_path, "metadata.yaml")
                if not os.path.exists(metadata_path):
                    continue
                    
                with open(metadata_path, "r") as f:
                    task_info = yaml.safe_load(f)
                
                metadata = TaskMetadata(**task_info["metadata"])
                name = task_info["name"]
                
                # Load DataFrames
                train_df = pd.read_parquet(os.path.join(task_path, "train.parquet"))
                test_df = pd.read_parquet(os.path.join(task_path, "test.parquet"))
                
                val_path = os.path.join(task_path, "val.parquet")
                val_df = pd.read_parquet(val_path) if os.path.exists(val_path) else None
                
                tasks.append(Task(
                    name=name,
                    train_df=train_df,
                    test_df=test_df,
                    val_df=val_df,
                    metadata=metadata
                ))
        
        return cls(rdb=rdb, tasks=tasks)

    @classmethod
    def from_relbench(cls, dataset_name: str):
        try:
            import relbench.tasks
            from fastdfs.adapter import RelBenchAdapter
        except ImportError as e:
            raise ImportError("relbench and fastdfs must be installed to use from_relbench") from e

        # Load RDB
        adapter = RelBenchAdapter(dataset_name)
        rdb = adapter.load()

        # Load Tasks
        tasks = []
        task_names = relbench.tasks.get_task_names(dataset_name)
        
        for task_name in task_names:
            # Load task (download=True ensures it's available/generated)
            rb_task = relbench.tasks.get_task(dataset_name, task_name, download=True)
            
            # Get tables (convert to pandas)
            train_df = rb_task.get_table("train", mask_input_cols=False).df
            test_df = rb_task.get_table("test", mask_input_cols=False).df
            val_df = rb_task.get_table("val", mask_input_cols=False).df
            
            # Metadata
            entity_table = rb_task.entity_table
            entity_col = rb_task.entity_col
            
            # Find PK of entity table in RDB
            pk = rdb.get_table_metadata(entity_table).primary_key
            
            key_mappings = {entity_col: f"{entity_table}.{pk}"}
            
            # Extract metric name
            metric_name = None
            if rb_task.metrics:
                metric_name = rb_task.metrics[0].__name__

            metadata = TaskMetadata(
                key_mappings=key_mappings,
                target_col=rb_task.target_col,
                time_col=rb_task.time_col,
                task_type=rb_task.task_type.value if hasattr(rb_task.task_type, "value") else rb_task.task_type,
                evaluation_metric=metric_name
            )
            
            tasks.append(Task(
                name=task_name,
                train_df=train_df,
                test_df=test_df,
                val_df=val_df,
                metadata=metadata
            ))
            
        return cls(rdb=rdb, tasks=tasks)

    @classmethod
    def from_4dbinfer(cls, dataset_name: str):
        try:
            from fastdfs.adapter import DBInferAdapter
        except ImportError as e:
            raise ImportError("fastdfs must be installed to use from_4dbinfer") from e

        # Load RDB
        adapter = DBInferAdapter(dataset_name)
        rdb = adapter.load()
        
        # The adapter stores the underlying DBBRDBDataset in self.dataset
        dbb_dataset = adapter.dataset
        
        tasks = []
        for dbb_task in dbb_dataset._tasks:
            task_meta = dbb_task.metadata
            
            # Convert numpy dicts to DataFrames
            train_df = pd.DataFrame(dbb_task.train_set)
            test_df = pd.DataFrame(dbb_task.test_set)
            val_df = pd.DataFrame(dbb_task.validation_set)
            
            # Metadata
            # Find key mappings: look for foreign keys in the task table
            key_mappings = {}
            for col_schema in task_meta.columns:
                if col_schema.dtype == "foreign_key":
                    # link_to is in format "table.column"
                    link_to = getattr(col_schema, "link_to", None)
                    if link_to:
                        key_mappings[col_schema.name] = link_to
            
            metadata = TaskMetadata(
                key_mappings=key_mappings,
                target_col=task_meta.target_column,
                time_col=task_meta.time_column,
                task_type=task_meta.task_type.value if hasattr(task_meta.task_type, "value") else task_meta.task_type,
                evaluation_metric=task_meta.evaluation_metric.value if hasattr(task_meta.evaluation_metric, "value") else task_meta.evaluation_metric
            )
            
            tasks.append(Task(
                name=task_meta.name,
                train_df=train_df,
                test_df=test_df,
                val_df=val_df,
                metadata=metadata
            ))
            
        return cls(rdb=rdb, tasks=tasks)
