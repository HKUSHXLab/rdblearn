"""
WandB API client for fetching experiment results.
"""

import wandb
import pandas as pd
from typing import List, Dict, Any, Optional

from constants import (
    WANDB_ENTITY, WANDB_PROJECT,
    RELBENCH_DATASET_MAP, RELBENCH_TASK_MAP,
    DB4INFER_DATASET_MAP, DB4INFER_TASK_MAP,
    MODEL_MAP
)


class WandBClient:
    """Client for fetching and processing WandB experiment results."""

    def __init__(
        self,
        entity: str = WANDB_ENTITY,
        project: str = WANDB_PROJECT
    ):
        self.entity = entity
        self.project = project
        self.api = wandb.Api()

    def fetch_sweep_runs(self, sweep_id: str) -> pd.DataFrame:
        """
        Fetch all finished runs from a WandB sweep.

        Args:
            sweep_id: The sweep ID to fetch runs from

        Returns:
            DataFrame with detailed table columns
        """
        # Fetch runs from sweep
        sweep_path = f"{self.entity}/{self.project}/{sweep_id}"
        try:
            sweep = self.api.sweep(sweep_path)
            runs = sweep.runs
        except Exception as e:
            print(f"Error fetching sweep {sweep_path}: {e}")
            return pd.DataFrame()

        results = []
        for run in runs:
            # Only process finished runs
            if run.state != "finished":
                continue

            processed = self._process_run(run)
            if processed:
                results.append(processed)

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def _process_run(self, run) -> Optional[Dict[str, Any]]:
        """
        Process a single WandB run into the detailed table format.

        Returns dict with columns:
            Adapter, Dataset, Task, Metric, Direction, DFS_Depth,
            model_name, dev_metric, test_metric
        """
        config = run.config
        summary = run.summary

        # Extract required fields
        adapter_type = config.get('adapter_type')
        dataset_name = config.get('dataset_name')
        task_name = config.get('task_name')
        model_name = config.get('model_name')
        task_type = config.get('task_type')
        dfs_depth = config.get('dfs_max_depth', 2)

        # Skip incomplete runs
        if not all([adapter_type, dataset_name, task_name, model_name]):
            return None

        # Get metric values based on task type
        if task_type == 'classification':
            dev_metric = summary.get('dev_metric/roc_auc')
            test_metric = summary.get('test_metric/roc_auc') or summary.get('metric/roc_auc')
            metric_name = 'AUC'
            higher_better = True
        else:  # regression
            dev_metric = summary.get('dev_metric/mae')
            test_metric = summary.get('test_metric/mae') or summary.get('metric/mae')
            metric_name = 'MAE'
            higher_better = False

        # Skip if metrics are missing
        if dev_metric is None or test_metric is None:
            return None

        # Standardize names using mappings
        if adapter_type == 'relbench':
            dataset_display = RELBENCH_DATASET_MAP.get(dataset_name, dataset_name)
            task_display = RELBENCH_TASK_MAP.get(task_name, task_name)
        elif adapter_type == '4dbinfer':
            dataset_display = DB4INFER_DATASET_MAP.get(dataset_name, dataset_name)
            task_display = DB4INFER_TASK_MAP.get(task_name, task_name)
        else:
            dataset_display = dataset_name
            task_display = task_name

        model_display = MODEL_MAP.get(model_name, model_name)

        return {
            'Adapter': adapter_type,
            'Dataset': dataset_display,
            'Task': task_display,
            'Metric': metric_name,
            'Direction': 'up' if higher_better else 'down',
            'DFS_Depth': int(dfs_depth),
            'model_name': model_display,
            'dev_metric': float(dev_metric),
            'test_metric': float(test_metric),
        }


def fetch_sweep_as_detailed_table(sweep_id: str) -> pd.DataFrame:
    """
    Convenience function to fetch a sweep and return as detailed table.

    Args:
        sweep_id: WandB sweep ID

    Returns:
        DataFrame with detailed table format
    """
    client = WandBClient()
    return client.fetch_sweep_runs(sweep_id)
