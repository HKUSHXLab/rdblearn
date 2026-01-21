#!/usr/bin/env python3
"""
Pre-download all relbench and 4dbinfer datasets in parallel.

This script loads one task from each unique dataset to trigger the download.
Since all tasks within a dataset share the same underlying data, downloading
one task per dataset is sufficient to cache all data.

Usage:
    python scripts/predownload_datasets.py
    python scripts/predownload_datasets.py --max-workers 4
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

from loguru import logger


# Define one task per dataset to trigger download
# Format: (adapter_type, dataset_name, task_name)
DATASETS_TO_DOWNLOAD = [
    # RelBench datasets (6 datasets)
    ("relbench", "rel-amazon", "item-churn"),
    ("relbench", "rel-avito", "ad-ctr"),
    ("relbench", "rel-event", "user-attendance"),
    ("relbench", "rel-hm", "item-sales"),
    ("relbench", "rel-stack", "post-votes"),
    ("relbench", "rel-trial", "site-success"),
    # 4DBInfer datasets (4 datasets)
    ("4dbinfer", "outbrain-small", "ctr"),
    ("4dbinfer", "amazon", "churn"),
    ("4dbinfer", "retailrocket", "cvr"),
    ("4dbinfer", "stackexchange", "churn"),
]


def download_dataset(args: Tuple[str, str, str]) -> Tuple[str, str, bool, str]:
    """
    Download a single dataset by loading a task.
    
    Args:
        args: Tuple of (adapter_type, dataset_name, task_name)
    
    Returns:
        Tuple of (adapter_type, dataset_name, success, message)
    """
    adapter_type, dataset_name, task_name = args
    
    try:
        # Import inside function for multiprocessing
        from rdblearn.datasets import RDBDataset
        
        logger.info(f"Downloading {adapter_type}::{dataset_name}::{task_name}...")
        
        if adapter_type == "relbench":
            dataset = RDBDataset.from_relbench(dataset_name)
        elif adapter_type == "4dbinfer":
            dataset = RDBDataset.from_4dbinfer(dataset_name)
        else:
            return (adapter_type, dataset_name, False, f"Unknown adapter type: {adapter_type}")
        
        # Verify the task exists
        if task_name not in dataset.tasks:
            return (adapter_type, dataset_name, False, 
                    f"Task '{task_name}' not found. Available: {list(dataset.tasks.keys())}")
        
        task = dataset.tasks[task_name]
        
        # Access data to ensure it's fully loaded
        train_shape = task.train_df.shape
        test_shape = task.test_df.shape
        
        msg = f"Downloaded successfully. Train: {train_shape}, Test: {test_shape}"
        logger.success(f"{adapter_type}::{dataset_name}::{task_name} - {msg}")
        
        return (adapter_type, dataset_name, True, msg)
        
    except Exception as e:
        error_msg = f"Failed: {str(e)}"
        logger.error(f"{adapter_type}::{dataset_name}::{task_name} - {error_msg}")
        return (adapter_type, dataset_name, False, error_msg)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download all relbench and 4dbinfer datasets in parallel"
    )
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=None,
        help="Maximum number of parallel workers (default: number of CPUs)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
    
    logger.info(f"Starting parallel download of {len(DATASETS_TO_DOWNLOAD)} datasets...")
    if args.max_workers:
        logger.info(f"Using {args.max_workers} parallel workers")
    
    results = []
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all download tasks
        future_to_dataset = {
            executor.submit(download_dataset, dataset_info): dataset_info
            for dataset_info in DATASETS_TO_DOWNLOAD
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_dataset):
            dataset_info = future_to_dataset[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                adapter_type, dataset_name, task_name = dataset_info
                results.append((adapter_type, dataset_name, False, f"Exception: {str(e)}"))
    
    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    success_count = 0
    failure_count = 0
    
    for adapter_type, dataset_name, success, message in results:
        status = "✓" if success else "✗"
        print(f"{status} {adapter_type}::{dataset_name}: {message}")
        if success:
            success_count += 1
        else:
            failure_count += 1
    
    print("=" * 60)
    print(f"Total: {len(results)} | Success: {success_count} | Failed: {failure_count}")
    print("=" * 60)
    
    # Return non-zero exit code if any downloads failed
    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()