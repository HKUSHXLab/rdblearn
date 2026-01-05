# RDBLearn 🚀

> Relational Database Learning with Foundation Models.

---

## 📑 Table of Contents

- [Introduction](#-introduction)
- [Installation](#️-installation)
- [Usage](#-usage)
- [License](#-license)

---

## 🎯 Introduction

**RDBLearn** is a framework designed to apply single-table foundation models to multi-table relational database tasks. It automates the process of flattening relational data into a single feature-rich table using Deep Feature Synthesis (DFS) and then leverages powerful single-table estimators (like TabPFN) for prediction.

### Core Components

🔧 **FastDFS** - Efficient Deep Feature Synthesis for automated multi-table flattening.
🤖 **RDBLearn Estimators** - Scikit-learn compatible `RDBLearnClassifier` and `RDBLearnRegressor` that integrate DFS and single-table models.
⚡ **Foundation Models** - Seamless integration with TabPFN and other foundation models for single table prediction tasks.

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

## 📜 License

This project is licensed under the MIT License.
