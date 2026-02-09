---
description: Record experiment results from WandB and compare against baseline in Google Sheets
user-invocable: true
---

# Experiment Recording Skill

This skill records ML experiment results from WandB and creates a comparison table in Google Sheets.

## What It Does

1. Fetches experiment runs from WandB using sweep ID
2. Creates a target table (best config per Dataset+Task based on dev_metric)
3. Compares against Main baseline and computes deltas
4. Writes comparison table to Google Sheets with improvement rate

## User Inputs Required

When invoked, ask the user for:

1. **Main baseline source**: Either:
   - `main_sheet_name`: Name of existing sheet with baseline (default: "Main")
   - `main_sweep_id`: WandB sweep ID to use as baseline

2. **Experiment sweep_id**: The WandB sweep ID for this experiment

3. **Experiment name**: Descriptive name (e.g., "TemporalDiff", "NewFeature")

4. **Type**: Either "Feature" or "Bug"

## Output

Creates a new Google Sheet tab named: `[Type]ExperimentName_YYYYMMDD_HHMM`

The sheet contains a comparison table with columns:
- Dataset, Task, Metric, Direction, DFS_Depth, model_name
- dev_metric, test_metric
- delta_test (difference from baseline)
- Improvement Rate at the bottom

## How to Execute

After collecting user inputs, run the Python script:

```python
import sys
sys.path.insert(0, '/root/yl_project/multitabfm/.claude/skills/experiment-recording')

from experiment_recorder import run_experiment_recording

# If using sheet name as baseline:
url = run_experiment_recording(
    main_source="Main",           # or user-provided sheet name
    experiment_sweep_id="abc123", # user-provided sweep ID
    experiment_name="MyExperiment",
    experiment_type="Feature",    # or "Bug"
    main_is_sweep=False
)

# If using sweep ID as baseline:
url = run_experiment_recording(
    main_source="xyz789",         # main sweep ID
    experiment_sweep_id="abc123",
    experiment_name="MyExperiment",
    experiment_type="Feature",
    main_is_sweep=True
)

print(f"Results written to: {url}")
```

## Prerequisites

- WandB must be logged in (`wandb login`)
- Google Sheets credentials bundled in `credentials.json`
- Target spreadsheet shared with service account

## Example Interaction

User: /experiment-recording