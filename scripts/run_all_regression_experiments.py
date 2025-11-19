#!/usr/bin/env python3
"""
Script to automatically run regression experiments on all datasets in tabpfn_data.

This script:
1. Scans all datasets under /root/autodl-tmp/tabpfn_data
2. Checks metadata.yaml files to identify regression tasks
3. Runs experiments on all discovered regression tasks
4. Saves results to a comprehensive report
"""

import os
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


def find_regression_tasks(base_path: str = "/root/autodl-tmp/tabpfn_data") -> List[Tuple[str, str]]:
    """
    Scan all datasets and find regression tasks.
    
    Returns:
        List of tuples (rdb_data_path, task_data_path) for regression tasks
    """
    regression_tasks = []
    base_path = Path(base_path)
    
    print(f"Scanning for regression tasks in {base_path}...")
    
    # Iterate through all dataset directories
    for dataset_dir in base_path.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        print(f"  Checking dataset: {dataset_name}")
        
        # Look for task directories within each dataset
        for task_dir in dataset_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            metadata_file = task_dir / "metadata.yaml"
            if not metadata_file.exists():
                continue
                
            try:
                # Read metadata to check task type
                with open(metadata_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                task_type = metadata.get('task_type', '').lower()
                task_name = metadata.get('task_name', task_dir.name)
                
                if task_type == 'regression':
                    rdb_data_path = str(dataset_dir)
                    task_data_path = str(task_dir)
                    regression_tasks.append((rdb_data_path, task_data_path))
                    print(f"    ✓ Found regression task: {task_name}")
                else:
                    print(f"    - Skipping {task_name} (task_type: {task_type})")
                    
            except Exception as e:
                print(f"    ! Error reading {metadata_file}: {e}")
                continue
    
    print(f"\nFound {len(regression_tasks)} regression tasks total.")
    return regression_tasks


def run_experiment(rdb_data_path: str, task_data_path: str, 
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
    task_name = Path(task_data_path).name
    dataset_name = Path(rdb_data_path).name
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {dataset_name}/{task_name}")
    print(f"RDB path: {rdb_data_path}")
    print(f"Task path: {task_data_path}")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    
    try:
        # Run the experiment
        preds, metrics = train_and_predict(
            rdb_data_path=rdb_data_path,
            task_data_path=task_data_path,
            enable_dfs=experiment_config.get('enable_dfs', False),
            dfs_config=experiment_config.get('dfs_config', {}),
            model_config=experiment_config.get('model_config', {}),
            eval_metrics=experiment_config.get('eval_metrics', ['mae']),
            batch_size=experiment_config.get('batch_size', 5000)
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Prepare results
        result = {
            'dataset_name': dataset_name,
            'task_name': task_name,
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


def main():
    """Main function to run all regression experiments."""
    print("🚀 Starting automated regression experiments")
    print(f"Timestamp: {datetime.now()}")
    
    # Configuration for experiments
    model_storage_path = "/root/autodl-tmp/autogluon_models"  # Keep AutoGluon artifacts off the project disk
    experiment_config = {
        'enable_dfs': True,  # Set to True if you want to enable deep feature synthesis
        'dfs_config': {
            "max_depth": 2,
            "agg_primitives": ["max", "min", "mean", "count", "mode", "std"],
            "engine": "dfs2sql"
        },
        'model_config': {
            "predictor_kwargs": {
                "path": model_storage_path,
            },
            "hyperparameters": {
                "TABPFNV2": {
                    "n_estimators": 8,
                }
            },
        },
        'eval_metrics': ['mae'],
        'batch_size': 5000  # Smaller batch size to avoid CUDA memory issues
    }
    
    # Find all regression tasks
    regression_tasks = find_regression_tasks()
    
    if not regression_tasks:
        print("No regression tasks found!")
        return
    
    print(f"\n🎯 Will run {len(regression_tasks)} regression experiments")
    
    # Run experiments
    results = []
    successful_experiments = 0
    failed_experiments = 0
    
    for i, (rdb_data_path, task_data_path) in enumerate(regression_tasks, 1):
        print(f"\n📊 Experiment {i}/{len(regression_tasks)}")
        
        result = run_experiment(rdb_data_path, task_data_path, experiment_config)
        results.append(result)
        
        if result['status'] == 'success':
            successful_experiments += 1
        else:
            failed_experiments += 1
    
    # Save results
    json_file, csv_file = save_results(results)
    
    # Print final summary
    print(f"\n🏁 All experiments completed!")
    print(f"  Total experiments: {len(regression_tasks)}")
    print(f"  Successful: {successful_experiments}")
    print(f"  Failed: {failed_experiments}")
    print(f"  Success rate: {successful_experiments/len(regression_tasks)*100:.1f}%")
    
    if failed_experiments > 0:
        print(f"\n❌ Failed experiments:")
        for result in results:
            if result['status'] == 'failed':
                print(f"  - {result['dataset_name']}/{result['task_name']}: {result['error']}")


if __name__ == "__main__":
    main()