# RDBLearn 🚀

> Relational Database Learning with Foundation Models.

---

## 📑 Table of Contents

- [Introduction](#-introduction)
- [Installation](#️-installation)
- [Usage](#-usage)
- [Core API Reference](#-core-api-reference)
- [License](#-license)

---

## 🎯 Introduction

**RDBLearn** is a framework designed to apply single-table foundation models to multi-table relational database tasks. It automates the process of flattening relational data into a single feature-rich table using Deep Feature Synthesis (DFS) and then leverages powerful single-table estimators (like TabPFN) for prediction.

### Core Components

* 🔧 **FastDFS** - Efficient Deep Feature Synthesis for automated multi-table flattening.
* 🤖 **RDBLearn Estimators** - Scikit-learn compatible `RDBLearnClassifier` and `RDBLearnRegressor` that integrate DFS and single-table models.
* ⚡ **Foundation Models** - Seamless integration with TabPFN and other foundation models for single table prediction tasks.

---

## ⚙️ Installation

```bash
pip install rdblearn
```

Or install from source:

```bash
git clone https://github.com/your-username/rdblearn.git
cd rdblearn
pip install -e .
```

---

## 🚀 Usage

### Basic Example (RelBench rel-avito)

RDBLearn includes two features enabled by default that improve prediction quality:

- **Target History Augmentation** (`enable_target_augmentation`): Injects the full training data (`X` and `y`) as a history table into the RDB before downsampling, allowing DFS to derive entity-level aggregate features from historical target values (e.g., mean past CTR per ad). Temporal cutoffs are respected to prevent data leakage. Requires `cutoff_time_column` to be provided.
- **Temporal Difference Features** (`temporal_diff`): Converts absolute epoch-time columns produced by DFS into relative temporal differences from the cutoff time (i.e., `cutoff_time - epochtime`), so the model sees how recently events occurred rather than raw timestamps.

```python
from rdblearn.datasets import RDBDataset
from rdblearn.estimator import RDBLearnRegressor
from tabpfn import TabPFNRegressor

# 1. Load RelBench dataset and task
dataset = RDBDataset.from_relbench("rel-avito")
task = dataset.tasks["ad-ctr"]

# 2. Initialize the estimator with a base model (e.g., TabPFN)
#    Both enable_target_augmentation and temporal_diff are enabled by default.
reg = RDBLearnRegressor(
    base_estimator=TabPFNRegressor(device="cpu"), # or "cuda"
    config={
        "dfs": {"max_depth": 2},
        "enable_target_augmentation": True,
        "temporal_diff": {"enabled": True},
        "max_train_samples": 1000
    }
)

# 3. Fit on relational data
X_train = task.train_df.drop(columns=[task.metadata.target_col])
y_train = task.train_df[task.metadata.target_col]

reg.fit(
    X=X_train,
    y=y_train,
    rdb=dataset.rdb,
    key_mappings=task.metadata.key_mappings,
    cutoff_time_column=task.metadata.time_col
)

# 4. Predict
X_test = task.test_df.drop(columns=[task.metadata.target_col])
predictions = reg.predict(X=X_test)
```

See `examples/` for more detailed usage.

---

## Core API Reference

### `RDBDataset`
The central class for managing relational data and task-specific tables.

- **`from_relbench(dataset_name: str) -> RDBDataset`**: Load a dataset from the RelBench benchmark.
- **`from_4dbinfer(dataset_name: str) -> RDBDataset`**: Load a dataset from the 4DBInfer benchmark.
- **`save(path: str)`**: Save the RDB and all associated tasks to disk.
- **`load(path: str) -> RDBDataset`**: Load a previously saved dataset from disk.

### `RDBLearnClassifier` / `RDBLearnRegressor`
Scikit-learn compatible estimators for relational learning.

- **`__init__(base_estimator, config: Optional[dict] = None)`**:
    - `base_estimator`: A single-table estimator (e.g., `TabPFNClassifier`, `AutoGluonClassifier`).
    - `config`: Optional dictionary to override default DFS or sampling settings. Key options:
        - `dfs`: DFS configuration (e.g., `{"max_depth": 2}`).
        - `max_train_samples` (int, default 10000): Maximum training samples before downsampling.
        - `stratified_sampling` (bool, default False): Use stratified sampling for classification tasks.
        - `enable_target_augmentation` (bool, default True): Augment the RDB with the full training target history table, enabling DFS to derive entity-level target aggregate features (e.g., entity mean). Requires `cutoff_time_column` to be set during `fit`.
        - `temporal_diff` (dict or TemporalDiffConfig, default `{"enabled": True}`): Convert DFS-generated epoch-time columns into temporal difference features relative to the cutoff time. Supports `enabled` (bool) and `exclude_columns` (list of column names to skip).
        - `predict_batch_size` (int, default 5000): Batch size for prediction.
- **`fit(X, y, rdb, key_mappings, cutoff_time_column=None, **kwargs)`**:
    - `X`: Training features (DataFrame).
    - `y`: Training labels (Series).
    - `rdb`: The relational database context (`fastdfs.RDB`).
    - `key_mappings`: Dictionary mapping columns in `X` to `table.primary_key` in the RDB.
    - `cutoff_time_column`: Optional column name in `X` representing the time of the observation.
- **`predict(X, rdb=None, **kwargs)`**:
    - `X`: Test features.
    - `rdb`: Optional RDB context (uses the one from `fit` if not provided).
- **`predict_proba(X, rdb=None, **kwargs)`**: (Classifier only) Predict class probabilities.

### `TaskMetadata`
Data structure containing task-specific information.
- `key_mappings`: Dict[str, str]
- `target_col`: str
- `time_col`: Optional[str]
- `task_type`: Optional[str]
- `evaluation_metric`: Optional[str]

### LimiX Integration
`rdblearn.utils` provides wrappers to adapt LimiX predictors into scikit-learn compatible estimators.

- **`LimiXWrapperClassifier(predictor)`**: Wrapper for classification tasks.
    - `predictor`: An initialized `LimiXPredictor` instance.
    - `fit(X, y)`: Stores training data for in-context inference.
    - `predict(X)`: Returns class labels.
    - `predict_proba(X)`: Returns class probabilities.

- **`LimiXWrapperRegressor(predictor)`**: Wrapper for regression tasks.
    - `predictor`: An initialized `LimiXPredictor` instance.
    - `fit(X, y)`: Stores training data.
    - `predict(X)`: Returns predicted values.

**Note**: You must install LimiX separately and provide an initialized `LimiXPredictor` to these wrappers.

---

## 📜 License

This project is licensed under the MIT License.
