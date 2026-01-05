import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import sys
from loguru import logger
logger.enable("rdblearn")

# Ensure modules are loaded so we can patch them
try:
    import relbench.tasks
except ImportError:
    from unittest.mock import MagicMock
    mock_relbench = MagicMock()
    sys.modules["relbench"] = mock_relbench
    sys.modules["relbench.tasks"] = mock_relbench.tasks

import fastdfs.adapter

from rdblearn.datasets import RDBDataset, Task, TaskMetadata

class TestRDBDataset(unittest.TestCase):
    @patch('rdblearn.datasets.load_rdb')
    def test_save_load(self, mock_load_rdb):
        import tempfile
        import shutil
        import os
        
        # Create a dummy dataset
        mock_rdb = MagicMock()
        metadata = TaskMetadata(
            key_mappings={"id": "table.id"},
            target_col="target"
        )
        task = Task(
            name="task1",
            train_df=pd.DataFrame({'a': [1]}),
            test_df=pd.DataFrame({'a': [2]}),
            metadata=metadata
        )
        dataset = RDBDataset(rdb=mock_rdb, tasks=[task])
        
        # Create temp dir
        temp_dir = tempfile.mkdtemp()
        try:
            # Save
            dataset.save(temp_dir)
            
            # Verify rdb.save called
            mock_rdb.save.assert_called_with(os.path.join(temp_dir, "rdb"))
            
            # Verify tasks directory structure exists
            task_dir = os.path.join(temp_dir, "tasks", "task1")
            self.assertTrue(os.path.exists(os.path.join(task_dir, "metadata.yaml")))
            self.assertTrue(os.path.exists(os.path.join(task_dir, "train.parquet")))
            
            # Mock load_rdb return
            mock_load_rdb.return_value = mock_rdb
            
            # Load
            loaded_dataset = RDBDataset.load(temp_dir)
            
            # Verify
            self.assertEqual(loaded_dataset.rdb, mock_rdb)
            self.assertIn("task1", loaded_dataset.tasks)
            self.assertEqual(loaded_dataset.tasks["task1"].name, "task1")
            
        finally:
            shutil.rmtree(temp_dir)

    @patch('relbench.tasks')
    @patch('fastdfs.adapter.RelBenchAdapter')
    def test_from_relbench(self, mock_adapter_cls, mock_relbench_tasks):
        # Mock RDB
        mock_rdb = MagicMock()
        mock_rdb.get_table_metadata.return_value.primary_key = "id"
        
        # Mock Adapter
        mock_adapter = mock_adapter_cls.return_value
        mock_adapter.load.return_value = mock_rdb
        
        # Mock Tasks
        mock_relbench_tasks.get_task_names.return_value = ["task1"]
        
        mock_task = MagicMock()
        mock_task.get_table.return_value.df = pd.DataFrame({'col': [1, 2]})
        mock_task.entity_table = "users"
        mock_task.entity_col = "user_id"
        mock_task.target_col = "target"
        mock_task.time_col = "timestamp"
        mock_task.task_type.value = "binary_classification"
        mock_task.metrics = [MagicMock(__name__="auc")]
        
        mock_relbench_tasks.get_task.return_value = mock_task
        
        # Call method
        dataset = RDBDataset.from_relbench("dummy_dataset")
        
        # Assertions
        self.assertEqual(len(dataset.tasks), 1)
        self.assertIn("task1", dataset.tasks)
        task = dataset.tasks["task1"]
        
        self.assertEqual(task.metadata.key_mappings, {"user_id": "users.id"})
        self.assertEqual(task.metadata.target_col, "target")
        self.assertEqual(task.metadata.evaluation_metric, "auc")
        
        mock_adapter_cls.assert_called_with("dummy_dataset")
        mock_relbench_tasks.get_task_names.assert_called_with("dummy_dataset")

    @patch('fastdfs.adapter.DBInferAdapter')
    def test_from_4dbinfer(self, mock_adapter_cls):
        # Mock RDB
        mock_rdb = MagicMock()
        
        # Mock Adapter
        mock_adapter = mock_adapter_cls.return_value
        mock_adapter.load.return_value = mock_rdb
        
        # Mock DBBRDBDataset
        mock_dbb_dataset = MagicMock()
        mock_adapter.dataset = mock_dbb_dataset
        
        # Mock Task
        mock_dbb_task = MagicMock()
        mock_dbb_task.train_set = {'col': [1]}
        mock_dbb_task.test_set = {'col': [2]}
        mock_dbb_task.validation_set = {'col': [3]}
        
        mock_task_meta = MagicMock()
        mock_task_meta.name = "task1"
        mock_task_meta.target_column = "target"
        mock_task_meta.time_column = "timestamp"
        # Test both Enum-like and string-like task_type/metric
        mock_task_meta.task_type = "classification"
        mock_task_meta.evaluation_metric = MagicMock(value="auroc")
        
        # Mock columns for key_mappings
        mock_col = MagicMock()
        mock_col.name = "user_id"
        mock_col.dtype = "foreign_key"
        mock_col.link_to = "users.id"
        mock_task_meta.columns = [mock_col]
        
        mock_dbb_task.metadata = mock_task_meta
        mock_dbb_dataset._tasks = [mock_dbb_task]
        
        # Call method
        dataset = RDBDataset.from_4dbinfer("dummy_4db")
        
        # Assertions
        self.assertEqual(len(dataset.tasks), 1)
        self.assertIn("task1", dataset.tasks)
        task = dataset.tasks["task1"]
        
        self.assertEqual(task.metadata.key_mappings, {"user_id": "users.id"})
        self.assertEqual(task.metadata.target_col, "target")
        self.assertEqual(task.metadata.task_type, "classification")
        self.assertEqual(task.metadata.evaluation_metric, "auroc")
        
        mock_adapter_cls.assert_called_with("dummy_4db")
        mock_adapter.load.assert_called_once()

if __name__ == '__main__':
    unittest.main()
