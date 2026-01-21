"""
Constants and configuration for experiment recording.
"""

# =============================================================================
# Name Mapping Dictionaries (for standardizing display names)
# =============================================================================

RELBENCH_DATASET_MAP = {
    'rel-amazon': 'Rel-Amazon',
    'rel-avito': 'Rel-Avito',
    'rel-event': 'Rel-Event',
    'rel-hm': 'Rel-Hm',
    'rel-stack': 'Rel-Stack',
    'rel-trial': 'Rel-Trial'
}

RELBENCH_TASK_MAP = {
    'item-churn': 'Item-Churn',
    'user-churn': 'User-Churn',
    'user-clicks': 'User-Clicks',
    'user-visits': 'User-Visits',
    'user-ignore': 'User-Ignore',
    'user-repeat': 'User-Repeat',
    'user-badge': 'User-Badge',
    'user-engagement': 'User-Engagement',
    'study-outcome': 'Study-Outcome',
    'item-ltv': 'Item-Ltv',
    'user-ltv': 'User-Ltv',
    'ad-ctr': 'Ad-Ctr',
    'user-attendance': 'User-Attendance',
    'item-sales': 'Item-Sales',
    'post-votes': 'Post-Votes',
    'site-success': 'Site-Success',
    'study-adverse': 'Study-Adverse'
}

DB4INFER_DATASET_MAP = {
    'amazon': 'Amazon',
    'outbrain-small': 'Outbrain',
    'retailrocket': 'Retailrocket',
    'stackexchange': 'StackExchange'
}

DB4INFER_TASK_MAP = {
    'churn': 'Churn',
    'ctr': 'CTR-100K',
    'cvr': 'CVR',
    'upvote': 'post-upvote'
}

MODEL_MAP = {
    'tabpfn_v2': 'TabPFN_V2',
    'tabpfn_v2.5': 'TabPFN_V2.5',
    'limix': 'LimiX'
}

# =============================================================================
# WandB Configuration
# =============================================================================

WANDB_ENTITY = "tgif"
WANDB_PROJECT = "rdblearn-scripts"

# =============================================================================
# Google Sheets Configuration
# =============================================================================

SPREADSHEET_ID = "1DhPk9H-MUc5RhkdfmqRmr2txBB2ZQuv4R5v_3gkK0OE"

# Main baseline sheet name
MAIN_SHEET_NAME = "Main"

# =============================================================================
# Column Definitions
# =============================================================================

DETAILED_COLUMNS = [
    'Adapter', 'Dataset', 'Task', 'Metric', 'Direction',
    'DFS_Depth', 'model_name', 'dev_metric', 'test_metric'
]

COMPARISON_COLUMNS = [
    'Dataset', 'Task', 'Metric', 'Direction',
    'DFS_Depth', 'model_name', 'dev_metric', 'test_metric', 'delta_test'
]
