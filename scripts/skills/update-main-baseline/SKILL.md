---
description: Update the Main baseline sheet with results from a WandB sweep
user-invocable: true
---

# Update Main Baseline Skill

This skill updates the "Main" baseline sheet in Google Sheets with results from a WandB sweep.

## What It Does

1. Fetches all finished runs from a WandB sweep
2. Creates a target table (best config per Dataset+Task based on dev_metric)
3. Overwrites the "Main" sheet in Google Sheets with the new baseline

## User Input Required

When invoked, ask the user for:

1. **Sweep ID**: The WandB sweep ID to use as the new Main baseline

## Output

Updates the "Main" sheet in Google Sheets with the target table.

## How to Execute

After collecting the sweep ID from the user, run:

```python
import sys
sys.path.insert(0, '/root/yl_project/multitabfm/.claude/skills/experiment-recording')

from experiment_recorder import update_main_baseline

url = update_main_baseline(sweep_id="user_provided_sweep_id")

print(f"Main baseline updated: {url}")
```

## Prerequisites

- WandB must be logged in (`wandb login`)
- Google Sheets credentials bundled in experiment-recording skill
- Target spreadsheet shared with service account

## Example Interaction

User: /update-main-baseline