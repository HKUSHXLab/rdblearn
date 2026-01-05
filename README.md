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

### Basic Example (RelBench rel-f1)

```python
from rdblearn.datasets import RDBDataset
from rdblearn.estimator import RDBLearnClassifier
from tabpfn import TabPFNClassifier

# 1. Load RelBench dataset and task
dataset = RDBDataset.from_relbench("rel-f1")
task = dataset.tasks["driver-dnf"]

# 2. Initialize the estimator with a base model (e.g., TabPFN)
clf = RDBLearnClassifier(
    base_estimator=TabPFNClassifier(device="cpu"), # or "cuda"
    config={
        "dfs": {"max_depth": 2},
        "max_train_samples": 1000
    }
)

# 3. Fit on relational data
X_train = task.train_df.drop(columns=[task.metadata.target_col])
y_train = task.train_df[task.metadata.target_col]

clf.fit(
    X=X_train, 
    y=y_train, 
    rdb=dataset.rdb,
    key_mappings=task.metadata.key_mappings,
    cutoff_time_column=task.metadata.time_col
)

# 4. Predict
X_test = task.test_df.drop(columns=[task.metadata.target_col])
predictions = clf.predict(X=X_test)
```

See `examples/` for more detailed usage.

---

## � Core API Reference

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
    - `config`: Optional dictionary to override default DFS or sampling settings.
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

---

## �📜 License

This project is licensed under the MIT License.
