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

## Internal Pipeline Details

This section provides a detailed explanation of how the `train_and_predict()` method processes data internally.

### Complete Pipeline Flow

#### **Stage 1: Load Data**
**Input:** `rdb_data_path`, `task_data_path`

**Process:**
- Load train/test datasets from files
- Load metadata (target column, time column, task type, metrics)
- Optional: Sample training data if `max_samples` is set

**Output:**
- `train_data`, `test_data` (raw DataFrames)
- `metadata` (dict with configuration)
- `rdb` (relational database object)

---

#### **Stage 2: Split Features and Labels**
**Input:** `train_data`, `test_data`

**Process:**
```python
X_train = train_data.drop(columns=[target_column])
Y_train = train_data[target_column]
X_test = test_data.drop(columns=[target_column])
Y_test = test_data[target_column]
```

**Output:**
- `X_train`, `X_test`: Feature DataFrames (without target)
- `Y_train`, `Y_test`: Label Series (target values only)

---

#### **Stage 3: Apply DFS (Deep Feature Synthesis) - Optional**

##### **What DFS Does:**
DFS automatically generates new features by:
- **Aggregating** related table data (e.g., "count of user's past purchases")
- **Joining** information across tables using foreign keys
- **Creating temporal features** from timestamp columns (year, month, day, hour, dayofweek)

##### **Process:**

**Step 3.1 - Prepare Data for DFS**
```python
train_for_dfs, test_for_dfs = prepare_for_dfs(X_train, X_test, key_mappings, time_column)
```
- **Removes array-valued columns** (lists, arrays) that would break database joins
- **Keeps key columns** (foreign keys) and time columns needed for joins

**Step 3.2 - Generate DFS Features**
```python
train_dfs = generate_features(train_for_dfs, rdb, key_mappings, time_column, pipeline, dfs_config)
test_dfs = generate_features(test_for_dfs, rdb, key_mappings, time_column, pipeline, dfs_config)
```
- Applies DFS pipeline to compute aggregations across tables
- Example: If user table and order table exist, creates features like:
  - `COUNT(orders)` per user
  - `AVG(order_amount)` per user
  - `MAX(order_date)` per user

**Step 3.3 - Add DFS Features Back**
```python
train_features = add_dfs_features(X_train, train_dfs)
test_features = add_dfs_features(X_test, test_dfs)
```
- Combines original features with new DFS-generated features
- Avoids duplicates (only adds new columns)

**Effect:**
- `train_features`, `test_features` now have **MORE COLUMNS** (original + DFS features)
- Example: 10 original columns → 10 + 50 DFS features = 60 total columns

---

#### **Stage 4: Feature and Label Transformation**

##### **Step 4.1 - Feature Transformation**
```python
X_train_transformed, X_test_transformed = ag_transform(
    train_features.reset_index(drop=True),
    test_features.reset_index(drop=True)
)
```

**What `ag_transform` does internally:**

**Sub-step A: Convert Categoricals to Numeric**
- Finds all `object`/`category` dtype columns
- Applies `LabelEncoder` to convert strings to integers
- Example: ["red", "blue", "green"] → [0, 1, 2]
- **Effect:** ML models can process the data (they need numbers, not strings)

**Sub-step B: AutoGluon Feature Generation**
- **Datetime features:** Extracts year, month, day, hour, dayofweek from datetime columns
- **Missing value handling:** Imputes missing values
- **Feature engineering:** Creates interaction features, polynomial features, etc.
- **Effect:** Richer feature representation for better model performance

**Output:** `X_train_transformed`, `X_test_transformed` with:
- Categorical columns converted to numeric
- Datetime columns expanded into multiple numeric features
- Missing values handled
- Same number or MORE columns than input

---

##### **Step 4.2 - Label Transformation**
```python
y_train_transformed, y_test_transformed = ag_label_transform(
    Y_train.reset_index(drop=True),
    Y_test.reset_index(drop=True)
)
```

**What `ag_label_transform` does:**

1. **Infers problem type:**
   - Classification (binary/multiclass) or Regression
   
2. **Creates LabelCleaner** fitted on `y_train`:
   - For classification: Encodes string labels to integers
     - Example: ["cat", "dog", "bird"] → [0, 1, 2]
   - For regression: May normalize values

3. **Transforms BOTH y_train and y_test** using the SAME cleaner:
   - **Critical:** Uses the same encoding learned from `y_train`
   - Ensures test labels match training label encoding

**Effect:**
- `y_train_transformed`, `y_test_transformed` are consistently encoded
- Both labels use the same encoding, ensuring correct evaluation

---

##### **Step 4.3 - Combine Features and Labels**
```python
train_data = pd.concat([X_train_transformed, y_train_transformed], axis=1)
test_data = pd.concat([X_test_transformed, y_test_transformed], axis=1)
```

**Effect:** Creates complete datasets ready for training/testing

---

#### **Stage 5: Train Model**
```python
X_test = test_data.drop(columns=[target_column])
Y_test = test_data[target_column]

self.fit(train_data, label_column=target_column, task_type=task_type, eval_metric=eval_metrics[0])
```

- Model trains on **transformed** features and labels
- Learns patterns from the engineered features

---

#### **Stage 6: Predict**
- **Classification:** Returns probability distributions via `predict_proba()`
- **Regression:** Returns continuous predictions via `predict()`
- Uses the **transformed** test features

---

#### **Stage 7: Evaluate**
- Compares **transformed** `Y_test` with predictions
- Computes metrics (accuracy, RMSE, etc.)
- Both labels and predictions are on the same transformed scale

---

### Summary of Key Stages

**Stage 3 (DFS):**
- **Purpose:** Generate rich features from relational data
- **Effect:** Creates aggregation features (counts, averages, etc.) from related tables
- **Result:** More features = better pattern recognition

**Stage 4 (Transformation):**
- **Purpose:** Prepare data for ML models
- **Effect:**
  - Categoricals → Numbers (models need numbers)
  - Datetime → Multiple numeric features (year, month, etc.)
  - Labels consistently encoded (train and test match)
- **Result:** Clean, numeric, consistently-encoded data ready for modeling

## License

MIT

## Changelog

v0.1.0
- Initial release with DFS feature generation, AutoGluon integration, and basic metrics