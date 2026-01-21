from typing import List, Optional, Dict
import os
import pickle
import yaml
import pandas as pd
from pydantic import BaseModel
from fastdfs import RDB, load_rdb
from fastdfs.dataset.meta import RDBColumnDType

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
            
            # Skip link prediction tasks as they are not supported yet
            task_type_val = rb_task.task_type.value if hasattr(rb_task.task_type, "value") else rb_task.task_type
            if task_type_val == "link_prediction":
                continue

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
            # Find key mappings from both foreign keys and primary key + target_table
            key_mappings = {}
            
            # 1. Add mappings for foreign keys in the task table
            for col_schema in task_meta.columns:
                if col_schema.dtype == "foreign_key":
                    # link_to is in format "table.column"
                    link_to = getattr(col_schema, "link_to", None)
                    if link_to:
                        key_mappings[col_schema.name] = link_to
            
            # 2. Also add primary key + target_table mapping if it exists
            # This handles cases like stackexchange::upvote where task.Id -> Posts.Id
            target_table = getattr(task_meta, "target_table", None)
            if target_table:
                # Find primary key column in task data
                task_pk_col = None
                for col_schema in task_meta.columns:
                    if col_schema.dtype == "primary_key":
                        task_pk_col = col_schema.name
                        break
                
                # Only add if not already mapped via FK
                if task_pk_col and task_pk_col not in key_mappings:
                    # Find primary key of target table in RDB
                    try:
                        table_meta = rdb.get_table_metadata(target_table)
                        table_pk = table_meta.primary_key
                        if table_pk:
                            key_mappings[task_pk_col] = f"{target_table}.{table_pk}"
                    except ValueError:
                        # Target table not found in RDB, skip
                        pass
            
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
        
        # Apply correction for stackexchange upvote task:
        # Increase timestamp column's value by one second
        # This is needed because the timestamp in the original task table match the ones in RDB, so FastDFS cannot join the features.
        if dataset_name == "stackexchange":
            # The data types of these Id columns are float because it's a mix of NaN and int values.
            # Convert them to string
            def convert_series(series: pd.Series) -> pd.Series:
                series.where(series.isna(), series.astype('Int64').astype(str), inplace=True)

            # Convert all primary keys and foreign keys in all tables using convert_series
            for table_name in rdb.table_names:
                table = rdb.tables[table_name]
                metadata = rdb.get_table_metadata(table_name)
                
                for col_schema in metadata.columns:
                    col_name = col_schema.name
                    if col_schema.dtype in [RDBColumnDType.primary_key, RDBColumnDType.foreign_key]:
                        if col_name in table.columns:
                            convert_series(table[col_name])
            
            for task in tasks:
                convert_series(task.train_df['Id'])
                convert_series(task.val_df['Id'])
                convert_series(task.test_df['Id'])
                if task.name == "upvote":
                    time_col = task.metadata.time_col
                    if time_col is not None:
                        # Add one second to the timestamp column in train, test, and val dataframes
                        if time_col in task.train_df.columns:
                            task.train_df[time_col] = task.train_df[time_col] + pd.Timedelta(seconds=1)
                        if time_col in task.test_df.columns:
                            task.test_df[time_col] = task.test_df[time_col] + pd.Timedelta(seconds=1)
                        if task.val_df is not None and time_col in task.val_df.columns:
                            task.val_df[time_col] = task.val_df[time_col] + pd.Timedelta(seconds=1)
            
        return cls(rdb=rdb, tasks=tasks)

