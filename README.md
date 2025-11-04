# MultiTabFM

MultiTabFM is a lightweight multi-table feature modeling framework that pairs Deep Feature Synthesis (DFS) with AutoGluon for end-to-end tabular modeling on relational data.

It is designed to:
- Provide a compact, friendly API for end-to-end modeling on relational data.
- Generate features from a relational database (RDB) using DFS (via the `fastdfs` package).
- Train and predict with AutoGluon Tabular using sensible defaults.
- Offer both a one-call API for quick results and a step-by-step workflow for more control.

## What it Supports

Out of the box, MultiTabFM provides:
- **DFS-based Feature Generation**: Automatically creates features from multiple tables using Deep Feature Synthesis.
- **Automated Model Training**: Uses AutoGluon to train tabular models automatically.
- **One-Call API**: A single function `train_and_predict` for an end-to-end workflow.
- **Step-by-Step Interface**: Functions for loading data, generating features, and training models separately.
- **Evaluation Utilities**: Simple tools for calculating common metrics like accuracy, roc_auc, and logloss.

## Installation

Install in editable mode from within the repository folder:

```bash
git clone https://github.com/dglai/multitabfm.git
cd multitabfm
pip install -e .
```

## How to Use

There are two primary ways to use `multitabfm`: the one-call API for simplicity, and the step-by-step API for greater control and introspection.

### Data Layout

Both approaches expect a specific data layout:
- A directory for the relational database (`rdb_data_path`), readable by `fastdfs.load_rdb`.
- A directory for the task data (`task_data_path`), containing:
  - `train.pqt` and `test.pqt`: Parquet files with training and testing data.
  - `metadata.yaml`: A file describing the dataset, keys, and target for the modeling task.

#### `rdb_data_path` Structure

The `rdb_data_path` is the root directory for your relational dataset. It must contain:
1.  **`metadata.yaml`**: This file defines the schema of the entire relational database. It specifies the tables, their columns, data types, and relationships.
2.  **Table Data Files**: These are the actual data files (e.g., `users.parquet`, `items.csv`) for each table listed in the `metadata.yaml`. The `source` field in the metadata for each table points to its corresponding data file within the directory.

An example `metadata.yaml` inside `rdb_data_path` might look like this:
```yaml
dataset_name: my_relational_db
tables:
  - name: users
    source: users.parquet
    format: parquet
    columns:
      - name: user_id
        dtype: primary_key
      - name: signup_date
        dtype: datetime_t
  - name: transactions
    source: transactions.csv
    format: csv
    columns:
      - name: transaction_id
        dtype: primary_key
      - name: user_id
        dtype: foreign_key
        link_to: users.user_id
      - name: amount
        dtype: float_t
```

#### `task_data_path` Structure

The `task_data_path` contains the data and metadata for the specific prediction task.

Example of `metadata.yaml` in `task_data_path`:
```yaml
dataset_name: rel-event
key_mappings:
  - user_id: users.user_id
  - item_id: items.item_id
time_column: interaction_time
target_column: label
```

### 1. One-Call API (Quick Start)

For a straightforward, end-to-end process, use the `train_and_predict` function. It handles data loading, feature generation, model training, and prediction in a single call.

```python
from multitabfm import train_and_predict

proba, metrics = train_and_predict(
    rdb_data_path="data/rel-event",                 # path used by fastdfs.load_rdb
    task_data_path="data/rel-event/user-ignore",    # must contain train.pqt, test.pqt, metadata.yaml
    eval_metrics=["accuracy", "auroc"],
)

print(proba)
print(metrics)
```

### 2. Step-by-Step Usage

For more granular control, you can use the core components of the package individually. This is useful for debugging, custom feature engineering, or integrating parts of the workflow into a larger system.

```python
import pandas as pd
from multitabfm import (
    MultiTabFM,
    load_dataset,
    prepare_target_dataframes,
    generate_features,
)

# 1) Load raw task data, metadata, and RDB
train_data, test_data, metadata, rdb = load_dataset(
    rdb_data_path="data/rel-event",
    task_data_path="data/rel-event/user-ignore",
)

# 2) Prepare target dataframes (ID + time + label)
train_df, test_df = prepare_target_dataframes(train_data, test_data, metadata)

# 3) Parse metadata for DFS
key_mappings = {k: v for d in metadata["key_mappings"] for k, v in d.items()}
time_column = metadata["time_column"]
target_column = metadata["target_column"]

# 4) Generate features
train_features = generate_features(train_df, rdb, key_mappings, time_column)
test_features = generate_features(test_df, rdb, key_mappings, time_column)

# 5) Train and predict
engine = MultiTabFM()
engine.fit(train_features, label_column=target_column)
proba = engine.predict_proba(test_features)

# 6) Evaluate (optional if labels exist in test_features)
if target_column in test_features.columns:
    metrics = engine.evaluate(
        labels=test_features[target_column],
        proba=proba,
        metrics=["accuracy", "roc_auc", "logloss"],
    )
    print(metrics)
```

## License

MIT

## Changelog

v0.1.0
- Initial release with DFS feature generation, AutoGluon integration, and basic metrics