import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import sys
from loguru import logger
logger.enable("rdblearn")

# Ensure modules are loaded so we can patch them
try:
    import relbench.tasks
    import fastdfs.adapter
except ImportError:
    pass

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

if __name__ == '__main__':
    unittest.main()
