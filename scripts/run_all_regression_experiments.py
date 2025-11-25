#!/usr/bin/env python3
"""
Script to automatically run MultiTabFM experiments on TabPFN datasets.

This script:
1. Scans all datasets under /root/autodl-tmp/tabpfn_data
2. Identifies tasks by reading metadata.yaml (currently regression by default)
3. Supports filtering datasets by "small" or "large" buckets
4. Runs experiments using the CustomLimiX model via train_and_predict
5. Saves detailed + summary reports for the completed experiments
"""

import argparse
import yaml
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import traceback

# Add the multitabfm package to the path
import sys
sys.path.append('/root/yl_project/multitabfm')

from multitabfm.api import train_and_predict
from multitabfm.model import CustomLimiX, CustomTabPFN


# SMALL_DATASET_NAMES = {"rel-amazon-post-dfs-3", "rel-avito-post-dfs-3", "rel-stack-post-dfs-3","rel-hm-post-dfs-3","rel-trial-post-dfs-3", "rel-event-post-dfs-3", "rel-f1-post-dfs-3"}

# SMALL_DATASET_NAMES = {"rel-trial-post-dfs-3", "rel-event-post-dfs-3", "rel-f1-post-dfs-3"}
SMALL_DATASET_NAMES = {"rel-trial"}
def _find_tasks_by_type(base_path: str, target_task_type: str) -> List[Tuple[str, str]]:
    """Shared helper to discover tasks that match the requested task type."""
    target_task_type = target_task_type.lower()
    tasks = []
    base_path = Path(base_path)

    print(f"Scanning for {target_task_type} tasks in {base_path}...")

    for dataset_dir in base_path.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        print(f"  Checking dataset: {dataset_name}")

        for task_dir in dataset_dir.iterdir():
            if not task_dir.is_dir():
                continue

            metadata_file = task_dir / "metadata.yaml"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, "r") as f:
                    metadata = yaml.safe_load(f)

                task_type = metadata.get("task_type", "").lower()
                if task_type in ("binary", "multiclass"):
                    task_type = "classification"
                task_name = metadata.get("task_name", task_dir.name)

                if task_type == target_task_type:
                    tasks.append((str(dataset_dir), str(task_dir)))
                    print(f"    ✓ Found {target_task_type} task: {task_name}")
                else:
                    print(f"    - Skipping {task_name} (task_type: {task_type or 'unknown'})")

            except Exception as e:
                print(f"    ! Error reading {metadata_file}: {e}")
                continue

    print(f"\nFound {len(tasks)} {target_task_type} tasks total.")
    return tasks


def find_regression_tasks(base_path: str = "/root/autodl-tmp/tabpfn_data") -> List[Tuple[str, str]]:
    """Discover all regression tasks underneath the TabPFN data root."""
    return _find_tasks_by_type(base_path, "regression")


def find_classification_tasks(base_path: str = "/root/autodl-tmp/tabpfn_data") -> List[Tuple[str, str]]:
    """Discover all classification tasks underneath the TabPFN data root."""
    return _find_tasks_by_type(base_path, "classification")


def classify_dataset_size(dataset_name: str) -> str:
    """Return "small" if dataset is in the curated set, otherwise "large"."""
    return "small" if dataset_name.lower() in SMALL_DATASET_NAMES else "large"


def dataset_size_matches(dataset_name: str, desired_size: str) -> bool:
    """Check whether the dataset matches the requested size filter (or if filter is 'all')."""
    desired_size = desired_size.lower()
    if desired_size == "all":
        return True
    return classify_dataset_size(dataset_name) == desired_size


def build_experiment_config(task_type: str,
                            model_path: str = "/root/autodl-tmp/limix/cache/LimiX-16M.ckpt") -> Dict:
    """Return an experiment configuration inspired by examples/pipeline_example.py."""
    task_type = task_type.lower()
    eval_metrics = ['mae'] if task_type == 'regression' else ['auroc']

    return {
        'enable_dfs': False,
        'dfs_config': {
            "max_depth": 3,
            "agg_primitives": ["max", "min", "mean", "count", "mode", "std"],
            "engine": "dfs2sql",
        },
        'model_config': {
            # "model_path": "/root/autodl-tmp/tabpfn_2_5/",
            # "model_path":"/root/.cache/tabpfn/tabpfn-v2-regressor.ckpt",
            # "model_path":"/root/.cache/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
            "model_path": "/root/autodl-tmp/limix/cache/LimiX-16M.ckpt",
            "task_type": task_type,
            # "max_samples": 10000
        },
        'eval_metrics': eval_metrics,
        'batch_size': 2000,
        'custom_model_class': CustomLimiX,
    }


def run_experiment(task_type: str,
                   dataset_size: str,
                   rdb_data_path: str,
                   task_data_path: str,
                   experiment_config: Dict) -> Dict:
    """
    Run a single experiment and return results.
    
    Args:
        rdb_data_path: Path to the RDB data directory
        task_data_path: Path to the task data directory
        experiment_config: Configuration for the experiment
        
    Returns:
        Dictionary containing experiment results and metadata
    """
    task_type = task_type.lower()
    task_name = Path(task_data_path).name
    dataset_name = Path(rdb_data_path).name
    dataset_size = dataset_size.lower()
    
    print(f"\n{'='*60}")
    print(f"Running {task_type} experiment: {dataset_name}/{task_name} ({dataset_size})")
    print(f"RDB path: {rdb_data_path}")
    print(f"Task path: {task_data_path}")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    
    try:
        # Build configs without mutating the shared experiment_config
        dfs_config = experiment_config.get('dfs_config', {})
        model_config = dict(experiment_config.get('model_config', {}))
        model_config.setdefault('task_type', task_type)
        custom_model_class = experiment_config.get('custom_model_class')

        # Run the experiment using the CustomLimiX flow
        preds, metrics = train_and_predict(
            rdb_data_path=rdb_data_path,
            task_data_path=task_data_path,
            enable_dfs=experiment_config.get('enable_dfs', False),
            dfs_config=dfs_config,
            model_config=model_config,
            eval_metrics=experiment_config.get('eval_metrics', ['mae']),
            batch_size=experiment_config.get('batch_size', 5000),
            custom_model_class=custom_model_class,
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Prepare results
        result = {
            'dataset_name': dataset_name,
            'task_name': task_name,
            'task_type': task_type,
            'dataset_size': dataset_size,
            'rdb_data_path': rdb_data_path,
            'task_data_path': task_data_path,
            'status': 'success',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'metrics': metrics,
            'num_predictions': len(preds) if preds is not None else 0,
            'config': experiment_config,
            'error': None
        }
        
        print(f"✓ Experiment completed successfully!")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Predictions: {result['num_predictions']}")
        if metrics:
            print(f"  Metrics: {metrics}")
        
        return result
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        print(f"✗ Experiment failed!")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Error: {error_msg}")
        
        result = {
            'dataset_name': dataset_name,
            'task_name': task_name,
            'task_type': task_type,
            'dataset_size': dataset_size,
            'rdb_data_path': rdb_data_path,
            'task_data_path': task_data_path,
            'status': 'failed',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'metrics': None,
            'num_predictions': 0,
            'config': experiment_config,
            'error': error_msg,
            'error_traceback': error_traceback
        }
        
        return result


def save_results(results: List[Dict], output_dir: str = "experiment_results"):
    """Save experiment results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as JSON
    json_file = output_path / f"regression_experiments_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create summary CSV
    summary_data = []
    for result in results:
        summary_row = {
            'dataset_name': result['dataset_name'],
            'task_name': result['task_name'],
            'task_type': result.get('task_type', ''),
            'dataset_size': result.get('dataset_size', ''),
            'status': result['status'],
            'duration_seconds': result['duration_seconds'],
            'num_predictions': result['num_predictions'],
        }
        
        # Add metrics to summary
        if result['metrics']:
            for metric_name, metric_value in result['metrics'].items():
                summary_row[f'metric_{metric_name}'] = metric_value
        
        if result['error']:
            summary_row['error'] = result['error']
            
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    csv_file = output_path / f"regression_experiments_summary_{timestamp}.csv"
    summary_df.to_csv(csv_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved:")
    print(f"  Detailed results: {json_file}")
    print(f"  Summary CSV: {csv_file}")
    print(f"{'='*60}")
    
    return json_file, csv_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MultiTabFM experiments across TabPFN datasets.")
    parser.add_argument(
        "--dataset-size",
        choices=["all", "small", "large"],
        default="all",
        help="Limit experiments to datasets classified as small or large. "
             "'small' currently includes rel-trial, rel-event, and rel-f1.",
    )
    return parser.parse_args()


def main():
    """Main function to run all configured experiments."""
    print("🚀 Starting automated MultiTabFM experiments")
    print(f"Timestamp: {datetime.now()}")
    dataset_size_filter = "small"
    
    # Specify which task types to run. Extend this list to include "classification" when needed.
    task_types_to_run = ['regression']

    # Gather tasks per type
    tasks_to_run: List[Tuple[str, str, str, str]] = []
    if 'regression' in task_types_to_run:
        for rdb, task in find_regression_tasks():
            dataset_name = Path(rdb).name
            if dataset_size_matches(dataset_name, dataset_size_filter):
                dataset_size = classify_dataset_size(dataset_name)
                tasks_to_run.append(('regression', dataset_size, rdb, task))
    if 'classification' in task_types_to_run:
        for rdb, task in find_classification_tasks():
            dataset_name = Path(rdb).name
            if dataset_size_matches(dataset_name, dataset_size_filter):
                dataset_size = classify_dataset_size(dataset_name)
                tasks_to_run.append(('classification', dataset_size, rdb, task))

    if not tasks_to_run:
        print("No tasks found for the configured task types and dataset size filter!")
        return

    print(f"\n🎯 Will run {len(tasks_to_run)} experiments for task types: {', '.join(task_types_to_run)}")
    if dataset_size_filter != 'all':
        print(f"  Dataset size filter: {dataset_size_filter}")

    # Run experiments
    results = []
    successful_experiments = 0
    failed_experiments = 0
    
    for i, (task_type, dataset_size, rdb_data_path, task_data_path) in enumerate(tasks_to_run, 1):
        print(f"\n📊 Experiment {i}/{len(tasks_to_run)} [{task_type}/{dataset_size}]")

        experiment_config = build_experiment_config(task_type)
        result = run_experiment(task_type, dataset_size, rdb_data_path, task_data_path, experiment_config)
        results.append(result)
        
        if result['status'] == 'success':
            successful_experiments += 1
        else:
            failed_experiments += 1
    
    # Save results
    json_file, csv_file = save_results(results)
    
    # Print final summary
    print(f"\n🏁 All experiments completed!")
    print(f"  Total experiments: {len(tasks_to_run)}")
    print(f"  Successful: {successful_experiments}")
    print(f"  Failed: {failed_experiments}")
    print(f"  Success rate: {successful_experiments/len(tasks_to_run)*100:.1f}%")
    
    if failed_experiments > 0:
        print(f"\n❌ Failed experiments:")
        for result in results:
            if result['status'] == 'failed':
                print(f"  - {result['dataset_name']}/{result['task_name']}: {result['error']}")


if __name__ == "__main__":
    main()