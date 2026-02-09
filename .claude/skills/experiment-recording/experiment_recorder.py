"""
Main experiment recording logic.

This module contains the core functions for:
1. Creating detailed tables from WandB runs
2. Creating target tables (best config per Dataset+Task)
3. Computing comparison tables with deltas
4. Running the full experiment recording workflow
"""

import pandas as pd
from datetime import datetime
from typing import Optional, Tuple

from wandb_client import WandBClient, fetch_sweep_as_detailed_table
from sheets_client import SheetsClient
from constants import MAIN_SHEET_NAME


def create_target_table(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target table by selecting best config per (Dataset, Task) based on dev_metric.

    For each unique (Dataset, Task) combination:
    - If Metric='AUC' (Direction='up'): pick row with MAX dev_metric
    - If Metric='MAE' (Direction='down'): pick row with MIN dev_metric

    Args:
        detailed_df: DataFrame with all runs

    Returns:
        DataFrame with one row per (Dataset, Task), containing the best config
    """
    if detailed_df.empty:
        return pd.DataFrame()

    target_rows = []

    # Group by Dataset and Task
    for (dataset, task), group in detailed_df.groupby(['Dataset', 'Task']):
        # Check direction (all rows in group should have same direction)
        direction = group['Direction'].iloc[0]

        if direction == 'up':
            # Higher is better - pick max dev_metric
            best_idx = group['dev_metric'].idxmax()
        else:
            # Lower is better - pick min dev_metric
            best_idx = group['dev_metric'].idxmin()

        best_row = group.loc[best_idx].copy()
        target_rows.append(best_row)

    target_df = pd.DataFrame(target_rows)

    # Sort by Dataset, Task
    target_df = target_df.sort_values(['Dataset', 'Task']).reset_index(drop=True)

    return target_df


def compute_comparison(
    experiment_target: pd.DataFrame,
    main_target: pd.DataFrame
) -> Tuple[pd.DataFrame, float]:
    """
    Compute comparison table between experiment and main baseline.

    Args:
        experiment_target: Target table from experiment
        main_target: Target table from main baseline

    Returns:
        Tuple of (comparison_df, improvement_rate)
        - comparison_df: DataFrame with experiment data + delta_test column
        - improvement_rate: Percentage of tasks that improved
    """
    if experiment_target.empty:
        return pd.DataFrame(), 0.0

    # Create comparison dataframe starting from experiment target
    comparison_df = experiment_target.copy()

    # Add delta_test column
    comparison_df['delta_test'] = None

    improved_count = 0
    total_count = 0

    for idx, row in comparison_df.iterrows():
        dataset = row['Dataset']
        task = row['Task']
        direction = row['Direction']
        exp_test = row['test_metric']

        # Find matching row in main baseline
        main_row = main_target[
            (main_target['Dataset'] == dataset) &
            (main_target['Task'] == task)
        ]

        if main_row.empty:
            # No baseline for this task
            comparison_df.at[idx, 'delta_test'] = None
            continue

        main_test = main_row.iloc[0]['test_metric']
        delta = exp_test - main_test

        comparison_df.at[idx, 'delta_test'] = delta

        total_count += 1

        # Check if improved
        if direction == 'up':
            # Higher is better
            if delta > 0:
                improved_count += 1
        else:
            # Lower is better
            if delta < 0:
                improved_count += 1

    # Calculate improvement rate
    improvement_rate = (improved_count / total_count * 100) if total_count > 0 else 0.0

    # Reorder columns to ensure test_r2 comes after delta_test
    desired_order = ['Dataset', 'Task', 'Metric', 'Direction', 'DFS_Depth',
                     'model_name', 'dev_metric', 'test_metric', 'delta_test', 'test_r2']
    # Only include columns that exist in the DataFrame
    final_columns = [col for col in desired_order if col in comparison_df.columns]
    comparison_df = comparison_df[final_columns]

    return comparison_df, improvement_rate


def run_experiment_recording(
    main_source: str,
    experiment_sweep_id: str,
    experiment_name: str,
    experiment_type: str,
    main_is_sweep: bool = False
) -> str:
    """
    Run the full experiment recording workflow.

    Args:
        main_source: Either a sweep_id (if main_is_sweep=True) or sheet_name
        experiment_sweep_id: Sweep ID for the experiment
        experiment_name: Descriptive name for the experiment
        experiment_type: "Feature" or "Bug"
        main_is_sweep: If True, main_source is a sweep_id; otherwise it's a sheet name

    Returns:
        URL to the created Google Sheet tab
    """
    print(f"\n{'='*60}")
    print(f"Recording Experiment: {experiment_name}")
    print(f"Type: {experiment_type}")
    print(f"{'='*60}\n")

    # Initialize clients
    wandb_client = WandBClient()
    sheets_client = SheetsClient()
    sheets_client.connect()

    # Step 1: Get main baseline target table
    print("Step 1: Loading main baseline...")
    if main_is_sweep:
        print(f"  Fetching from sweep: {main_source}")
        main_detailed = wandb_client.fetch_sweep_runs(main_source)
        if main_detailed.empty:
            raise ValueError(f"No finished runs found in main sweep: {main_source}")
        main_target = create_target_table(main_detailed)
    else:
        print(f"  Reading from sheet: {main_source}")
        main_target = sheets_client.read_target_table(main_source)
        if main_target.empty:
            raise ValueError(f"Main sheet '{main_source}' is empty or not found")

    print(f"  Main baseline has {len(main_target)} tasks")

    # Step 2: Fetch experiment data
    print(f"\nStep 2: Fetching experiment data from sweep: {experiment_sweep_id}")
    experiment_detailed = wandb_client.fetch_sweep_runs(experiment_sweep_id)
    if experiment_detailed.empty:
        raise ValueError(f"No finished runs found in experiment sweep: {experiment_sweep_id}")
    print(f"  Found {len(experiment_detailed)} finished runs")

    # Step 3: Create experiment target table
    print("\nStep 3: Creating target table (best config per task)...")
    experiment_target = create_target_table(experiment_detailed)
    print(f"  Target table has {len(experiment_target)} tasks")

    # Step 4: Compute comparison
    print("\nStep 4: Computing comparison with main baseline...")
    comparison_df, improvement_rate = compute_comparison(experiment_target, main_target)
    print(f"  Improvement rate: {improvement_rate:.1f}%")

    # Step 5: Write to Google Sheets
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    sheet_name = f"[{experiment_type}]{experiment_name}_{timestamp}"
    print(f"\nStep 5: Writing to Google Sheets: {sheet_name}")

    sheets_client.write_comparison_sheet(
        sheet_name,
        comparison_df,
        improvement_rate,
        detailed_df=experiment_detailed,
        sweep_id=experiment_sweep_id
    )

    # Get URL
    sheet_url = sheets_client.get_sheet_url(sheet_name)

    print(f"\n{'='*60}")
    print(f"Experiment recording complete!")
    print(f"Sheet: {sheet_name}")
    print(f"URL: {sheet_url}")
    print(f"Improvement Rate: {improvement_rate:.1f}%")
    print(f"{'='*60}\n")

    return sheet_url


def update_main_baseline(sweep_id: str) -> str:
    """
    Update the Main baseline sheet with results from a sweep.

    Args:
        sweep_id: WandB sweep ID to use as new baseline

    Returns:
        URL to the Main sheet
    """
    print(f"\n{'='*60}")
    print(f"Updating Main Baseline")
    print(f"{'='*60}\n")

    # Initialize clients
    wandb_client = WandBClient()
    sheets_client = SheetsClient()
    sheets_client.connect()

    # Fetch sweep data
    print(f"Fetching data from sweep: {sweep_id}")
    detailed_df = wandb_client.fetch_sweep_runs(sweep_id)
    if detailed_df.empty:
        raise ValueError(f"No finished runs found in sweep: {sweep_id}")
    print(f"  Found {len(detailed_df)} finished runs")

    # Create target table
    print("\nCreating target table...")
    target_df = create_target_table(detailed_df)
    print(f"  Target table has {len(target_df)} tasks")

    # Update main sheet
    print(f"\nUpdating Main sheet...")
    sheets_client.update_main_sheet(target_df, detailed_df=detailed_df, sweep_id=sweep_id)

    # Get URL
    sheet_url = sheets_client.get_sheet_url(MAIN_SHEET_NAME)

    print(f"\n{'='*60}")
    print(f"Main baseline updated!")
    print(f"URL: {sheet_url}")
    print(f"{'='*60}\n")

    return sheet_url


if __name__ == "__main__":
    # Example usage (for testing)
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python experiment_recorder.py record <main_sheet> <exp_sweep> <name> <type>")
        print("  python experiment_recorder.py update-main <sweep_id>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "record":
        if len(sys.argv) < 6:
            print("Usage: python experiment_recorder.py record <main_sheet> <exp_sweep> <name> <type>")
            sys.exit(1)
        main_sheet = sys.argv[2]
        exp_sweep = sys.argv[3]
        name = sys.argv[4]
        exp_type = sys.argv[5]
        run_experiment_recording(main_sheet, exp_sweep, name, exp_type, main_is_sweep=False)

    elif command == "update-main":
        if len(sys.argv) < 3:
            print("Usage: python experiment_recorder.py update-main <sweep_id>")
            sys.exit(1)
        sweep_id = sys.argv[2]
        update_main_baseline(sweep_id)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
