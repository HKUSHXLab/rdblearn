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
🤖 **RDBLearnEstimator** - A scikit-learn compatible estimator that integrates DFS and single-table models.
⚡ **Foundation Models** - Seamless integration with TabPFN and other foundation models.

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

### Basic Example

```python
from rdblearn.estimator import RDBLearnEstimator
from tabpfn import TabPFNClassifier

# Initialize the estimator with a base model (e.g., TabPFN)
clf = RDBLearnEstimator(
    base_estimator=TabPFNClassifier(),
    config={
        "dfs": {"max_depth": 2},
        "max_train_samples": 1000
    }
)

# Fit on relational data (EntitySet)
clf.fit(X=es, y=y)

# Predict
predictions = clf.predict(X=es_test)
```

See `examples/` for more detailed usage.

---

## 📜 License

This project is licensed under the MIT License.
