# MultiTabFM 🚀

> A benchmark framework for evaluating single-table prediction foundation models on multi-table tasks through automated table flattening.

---

## 📑 Table of Contents

- [Introduction](#-introduction)
- [Baseline Results](#-baseline-results)
- [Setup & Installation](#️-setup--installation)
- [Getting Datasets](#-getting-datasets)
- [Running Baselines](#-running-baselines)
- [License](#-license)

---

## 🎯 Introduction

**MultiTabFM** is a benchmarking framework designed to evaluate how well single-table prediction foundation models perform on multi-table relational tasks. It bridges the gap between relational databases and single-table models by automatically flattening multi-table data into a single feature-rich table.

### Core Components

🔧 **FastDFS** - Deep Feature Synthesis for automated multi-table flattening  
🤖 **Foundation Models** - Support for TabPFN, Limix, and other single-table models  
⚡ **AutoGluon Integration** - Available for debugging and baseline comparison  
🔄 **Flexible APIs** - One-call simplicity or step-by-step control  

### What It Does

MultiTabFM solves the challenge of applying powerful single-table foundation models (like TabPFN, Limix) to relational databases:

1. **Takes** multi-table relational data (users, transactions, products, etc.)
2. **Flattens** tables into a single feature-rich table using Deep Feature Synthesis
3. **Evaluates** foundation model performance on the flattened representation
4. **Benchmarks** results across different models and datasets

### Supported Tasks

✅ **Binary Classification** - Predict binary outcomes (churn, fraud, etc.)  
✅ **Regression** - Predict continuous values (sales, ratings, etc.)  

### Use Cases

- Benchmarking foundation models on relational prediction tasks
- Establishing baselines for multi-table datasets (RelBench, 4DBInfer)
- Research on table flattening strategies for foundation models
- Quick prototyping with TabPFN/Limix on relational data

---

## 📊 Baseline Results

### RelBench Datasets

Performance on [RelBench](https://github.com/snap-stanford/relbench) benchmark datasets:

#### Classification Tasks:
| Size         | Dataset    | Task            | KumoRFM | AG_Transform+TabPFN_V2 + Fastdfs | AG_Transform+TabPFN_V2.5 + Fastdfs | AG_Transform+Limix+ Fastdfs |
|--------------|------------|-----------------|---------|----------------------------------|------------------------------------|-----------------------------|
| larger size  | rel-amazon | user-churn      | 0.6729  | 0.6813                           | 0.6766                             | 0.6551                      |
|              | rel-amazon | item-churn      | 0.7993  | 0.81497                          | 0.80749                            | 0.8066                      |
|              | rel-avito  | user-visits     | 0.6485  | 0.6575                           | 0.652061                           | 0.64956                     |
|              | rel-avito  | user-clicks     | 0.6411  | 0.68281                          | 0.64359                            | 0.6534                      |
|              | rel-hm     | user-churn      | 0.6771  | 0.68427                          | 0.67882                            | 0.67607                     |
|              | rel-stack  | user-engagement | 0.8709  | 0.88486                          | 0.89614                            | 0.8928                      |
|              | rel-stack  | user-badge      | 0.8     | 0.8596                           | 0.85472                            | 0.7925                      |
|              |            |                 |         |                                  |                                    |                             |
| Smaller size | rel-trial  | study-outcome   | 0.7079  | 0.71809                          | 0.71921                            | 0.72087                     |
|              | rel-event  | user-repeat     | 0.7608  | 0.76393                          | 0.7667                             | 0.7692                      |
|              | rel-event  | user-ignore     | 0.892   | 0.8424                           | 0.83672                            | 0.84026                     |
|              | rel-f1     | driver-dnf      | 0.8241  | 0.7291                           | 0.7182                             | 0.7235                      |
|              | rel-f1     | driver-top3     | 0.9107  | 0.7979                           | 0.78746                            | 0.8015                      |

#### Regression tasks:
| Size         | Dataset    | Task            | KumoRFM | AG_Transform+TabPFN_V2 + Fastdfs | AG_Transform+TabPFN_V2.5 + Fastdfs | AG_Transform+Limix+ Fastdfs |
|--------------|------------|-----------------|---------|----------------------------------|------------------------------------|-----------------------------|
| larger size  | rel-amazon | user-ltv        | 16.161  | 14.971                           | 15.23                              | 24.2473                     |
|              | rel-amazon | item-ltv        | 55.254  | 50.1958                          | 67.7773                            | 239.5674                    |
|              | rel-avito  | ad-ctr          | 0.035   | 0.04243                          | 0.04391                            | 0.0445                      |
|              | rel-hm     | item-sales      | 0.04    | 0.064732                         | 0.06178                            | 0.08379                     |
|              | rel-stack  | post-votes      | 0.065   | 0.06766                          | 0.06829                            | 0.12729                     |
|              |            |                 |         |                                  |                                    |                             |
| Smaller size | rel-trial  | study-adverse   | 58.231  | 46.3368                          | 45.788                             | 46.364                      |
|              | rel-trial  | site-success    | 0.417   | 0.4365                           | 0.4405                             | (bug)                       |
|              | rel-event  | user-attendance | 0.264   | 0.241                            | 0.23595                            | 0.3852                      |
|              | rel-f1     | driver-position | 2.747   | 4.067                            | 4.067                              | 3.9696                      |


### 4DBInfer Datasets

Performance on [4DBInfer](https://github.com/awslabs/multi-table-benchmark) benchmark datasets:

| Dataset/Task    | Amazon/Churn   | Amazon/Rating-100K | Retailrocket/CTR-100K | Outbrain/CTR-100K | StackExchange/user-churn | StackExchange/post-upvote |
|-----------------|----------------|--------------------|-----------------------|-------------------|--------------------------|---------------------------|
|                 | Metric         | AUC ↑              | RMSE ↓                | AUC ↑             | AUC ↑                    | AUC ↑                     | AUC ↑      |
| Single Table    | MLP            | 0.5                | 1.0561                | 0.5               | 0.5105                   | 0.5                       | 0.5079     |
|                 | DeepFM         | 0.5                | 1.0553                | 0.5               | 0.5107                   | 0.4964                    | 0.5078     |
|                 | FT-Transformer | 0.5                | 1.0558                | 0.49247           | 0.52016                  | 0.4998                    | 0.5124     |
|                 | XGBoost        | 0.5614             | 1.058                 | 0.5               | 0.5                      | 0.5084                    | 0.4968     |
|                 | AutoGluon      | 0.6085             | 1.0603                | 0.5096            | 0.4965                   | 0.5                       | 0.5081     |
| Single Table    | TABPFN_V2      | 0.5421             | 1.037653              | 0.50374           | 0.51042                  | 0.60441                   | 0.50939    |
| Simple Join     | MLP            | 0.5                | 1.057                 | 0.5097            | 0.4891                   | 0.6024                    | 0.8745     |
|                 | DeepFM         | 0.5                | 1.0585                | 0.4933            | 0.5109                   | 0.5984                    | 0.8764     |
|                 | FT-Transformer | 0.5                | 1.0574                | 0.4917            | 0.52033                  | 0.6319                    | 0.867      |
|                 | XGBoost        | 0.5614             | 1.055                 | 0.5               | 0.5                      | 0.582                     | 0.8669     |
|                 | AutoGluon      | 0.6085             | 1.0501                | 0.5096            | 0.4969                   | 0.7501                    | 0.8687     |
| DFS             | MLP            | 0.6854             | 0.8621                | 0.7269            | 0.5456                   | 0.8331                    | 0.8783     |
|                 | DeepFM         | 0.6764             | 0.8673                | 0.7319            | 0.5289                   | 0.8212                    | 0.8821     |
|                 | FT-Transformer | 0.684              | 0.855                 | 0.72797           | 0.53595                  | 0.8376                    | 0.8749     |
|                 | XGBoost        | 0.6922             | 0.8543                | 0.7195            | 0.5421                   | 0.8251                    | 0.8675     |
|                 | AutoGluon      | 0.7291             | 0.9829                | 0.7343            | 0.5494                   | 0.8396                    | 0.8849     |
|                 | TABPFN_V2      | 0.74904            | 1.00452               | 0.741908          | 0.5076                   | 0.85106                   | 0.5102/... |
| Row2Node (R2G)  | SAGE           | 0.7571             | 0.9745                | 0.847             | 0.6239                   | 0.8558                    | 0.8861     |
|                 | GAT            | 0.7622             | 0.9703                | 0.8284±0.0000     | 0.6146                   | 0.8645                    | 0.8853     |
|                 | PNA            | 0.7645             | 0.9657                | 0.83665           | 0.6249                   | 0.8664                    | 0.8896     |
|                 | HGT            | 0.773              | 0.9669                | 0.84953           | 0.626                    | 0.867                     | 0.8817     |
| Row2Node++ (ER) | SAGE           | 0.7314             | 0.9696                | 0.8091            | 0.6271                   | 0.8485                    | 0.6798     |
|                 | GAT            | 0.7192             | 0.9682                | 0.7536            | 0.6308                   | 0.8528                    | 0.6883     |
|                 | PNA            | 0.7157             | 0.9564                | 0.84274           | 0.6322                   | 0.8657                    | 0.7045     |
|                 | HGT            | 0.6864             | 0.9671                | 0.83416           | 0.6323                   | 0.856                     | 0.6603     |

> 💡 **Note:** Baseline results will be updated as experiments are completed. To reproduce results, see [Running Baselines](#-running-baselines).

---

## ⚙️ Setup & Installation

### Requirements

- Python 3.8+
- pip or conda package manager

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/dglai/multitabfm.git
cd multitabfm
```

2. **Install in editable mode:**
```bash
pip install -e .
```

3. **Verify installation:**
```python
from multitabfm import train_and_predict
print("MultiTabFM installed successfully!")
```

---

## 📦 Getting Datasets

### Where to Download

- **RelBench Datasets:** Follow instructions at [RelBench Repository](https://github.com/snap-stanford/relbench)
- **4DBInfer Datasets:** Follow instructions at [4DBInfer Repository](https://github.com/microsoft/4DBInfer)

### Understanding Data Format

MultiTabFM expects two main directories for each task:

#### 1. RDB Data Directory (`rdb_data_path`)

Contains the **relational database** with multiple tables and their relationships.

**Structure:**
```
rdb_data_path/
├── metadata.yaml          # Schema definition
├── users.parquet          # Table 1
├── transactions.csv       # Table 2
└── items.parquet          # Table 3
```

**Example `metadata.yaml`:**
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

**Key Points:**
- `metadata.yaml` defines all tables, columns, and relationships
- Each table's `source` points to its data file
- Primary keys and foreign keys establish table relationships

#### 2. Task Data Directory (`task_data_path`)

Contains the **specific prediction task** data.

**Structure:**
```
task_data_path/
├── metadata.yaml          # Task configuration
├── train.pqt              # Training data
└── test.pqt               # Test data
```

**Example `metadata.yaml`:**
```yaml
dataset_name: rel-event
key_mappings:
  - user_id: users.user_id
  - item_id: items.item_id
time_column: interaction_time
target_column: label
```

**Key Points:**
- `train.pqt` and `test.pqt` are Parquet files with train/test splits
- `key_mappings` links task data to RDB tables
- `time_column` enables temporal feature generation
- `target_column` is what you're predicting

---

## 🚀 Running Baselines

MultiTabFM provides two ways to run experiments: a **One-Call API** for quick benchmarking and a **Step-by-Step API** for fine-grained control.

### Quick Start: One-Call API

The simplest way to benchmark foundation models on multi-table data:

```python
from multitabfm import train_and_predict

# Run end-to-end pipeline with foundation model
proba, metrics = train_and_predict(
    rdb_data_path="data/rel-event",              # Relational database
    task_data_path="data/rel-event/user-ignore", # Task-specific data
    eval_metrics=["accuracy", "auroc"],          # Metrics to compute
    model="tabpfn",                              # or "limix", "autogluon"
)

print("Predictions:", proba)
print("Metrics:", metrics)
```

**What happens internally:**

1. **Load Data** → Reads RDB + train/test files from both directories
2. **Feature Generation (FastDFS)** → Flattens multi-table data:
   - Joins related tables using foreign keys
   - Creates aggregation features (COUNT, AVG, MAX, MIN, etc.)
   - Generates temporal features from datetime columns
   - Outputs: Single enriched feature table
3. **Transform (AutoGluon-based)** → Prepares data for the model:
   - Uses AutoGluon's feature preprocessing utilities
   - Encodes categorical variables → numeric
   - Extracts datetime components (year, month, day, etc.)
   - Handles missing values
   - Normalizes features if needed
4. **Train** → Fits selected foundation model (TabPFN/Limix) or AutoGluon
5. **Predict** → Generates predictions on test set
6. **Evaluate** → Computes specified metrics

**Result:** Multi-table relational data → Single-table prediction (via flattening)

### Advanced: Step-by-Step API

For custom workflows, research experiments, or integration into larger pipelines:

```python
from multitabfm import (
    MultiTabFM,
    load_dataset,
    prepare_target_dataframes,
    generate_features,
)

# Step 1: Load raw data
train_data, test_data, metadata, rdb = load_dataset(
    rdb_data_path="data/rel-event",
    task_data_path="data/rel-event/user-ignore",
)

# Step 2: Prepare target dataframes (ID + time + label)
train_df, test_df = prepare_target_dataframes(train_data, test_data, metadata)

# Step 3: Parse metadata for DFS
key_mappings = {k: v for d in metadata["key_mappings"] for k, v in d.items()}
time_column = metadata["time_column"]
target_column = metadata["target_column"]

# Step 4: Generate features using FastDFS (multi-table → single-table)
train_features = generate_features(train_df, rdb, key_mappings, time_column)
test_features = generate_features(test_df, rdb, key_mappings, time_column)

# Step 5: Train foundation model
engine = MultiTabFM(model="tabpfn")  # or "limix", "autogluon"
engine.fit(train_features, label_column=target_column)

# Step 6: Predict
proba = engine.predict_proba(test_features)

# Step 7: Evaluate (optional)
if target_column in test_features.columns:
    metrics = engine.evaluate(
        labels=test_features[target_column],
        proba=proba,
        metrics=["accuracy", "roc_auc", "logloss"],
    )
    print(metrics)
```

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│ MULTITABFM PIPELINE: Multi-Table → Single-Table Prediction          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. LOAD DATA                                                       │
│     ├─ Read train.pqt & test.pqt (task data)                       │
│     ├─ Load metadata.yaml (task config)                            │
│     └─ Load relational database from rdb_data_path                 │
│                                                                     │
│  2. FASTDFS: MULTI-TABLE FLATTENING  ★★★                           │
│     ┌──────────────────────────────────────────────────┐           │
│     │  Input: Multiple related tables                  │           │
│     │    • users (user_id, country, signup_date)       │           │
│     │    • transactions (transaction_id, user_id, $)   │           │
│     │    • products (product_id, category, price)      │           │
│     │                                                   │           │
│     │  FastDFS Operations:                             │           │
│     │    ✓ Join tables via foreign keys                │           │
│     │    ✓ Aggregate: COUNT, SUM, AVG, MAX, MIN        │           │
│     │    ✓ Temporal: year, month, day, dayofweek       │           │
│     │    ✓ Create cross-table features                 │           │
│     │                                                   │           │
│     │  Output: Single flattened table                  │           │
│     │    • user_id + 50+ generated features            │           │
│     └──────────────────────────────────────────────────┘           │
│                                                                     │
│  3. FEATURE TRANSFORMATION (AutoGluon-based) ★                     │
│     ┌──────────────────────────────────────────────────┐           │
│     │  Uses AutoGluon's feature preprocessing:         │           │
│     │    ✓ Categorical encoding (Label/One-Hot)        │           │
│     │    ✓ Datetime decomposition (Y/M/D/H/DoW)        │           │
│     │    ✓ Missing value imputation                    │           │
│     │    ✓ Feature type inference                      │           │
│     │    ✓ Normalization/scaling (if needed)           │           │
│     │                                                   │           │
│     │  Note: This step uses AutoGluon's FeatureMetadata│           │
│     │  and preprocessing pipeline regardless of which  │           │
│     │  model (TabPFN/Limix/AutoGluon) is used for      │           │
│     │  final training.                                 │           │
│     └──────────────────────────────────────────────────┘           │
│                                                                     │
│  4. MODEL TRAINING (Foundation Model)                              │
│     ├─ TabPFN: Transformer-based tabular model                    │
│     ├─ Limix: Linear mixed model                                  │
│     └─ AutoGluon: Ensemble (for debugging/comparison)             │
│                                                                     │
│  5. PREDICTION & EVALUATION                                        │
│     ├─ Generate predictions on flattened test set                 │
│     └─ Compute metrics (accuracy, ROC-AUC, RMSE, etc.)           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Important Note on Feature Transformation:**  
While MultiTabFM supports multiple foundation models (TabPFN, Limix), the feature transformation step currently relies on **AutoGluon's preprocessing utilities**. This means:

- ✓ **AutoGluon's FeatureMetadata** is used to infer feature types
- ✓ **AutoGluon's data preprocessors** handle categorical encoding, datetime feature extraction, and missing value imputation
- ✓ This preprocessing occurs **before** passing data to the final model (whether TabPFN, Limix, or AutoGluon itself)
- ⚠️ Future versions may support model-specific preprocessing pipelines

This design ensures consistent feature preprocessing across all models, making benchmark comparisons fair and reproducible.

### FastDFS: The Core Flattening Engine

**FastDFS (Fast Deep Feature Synthesis)** is the engine that converts multi-table relational data into a single feature-rich table suitable for foundation models.

**Example Transformation:**

**Input - Multi-Table Database:**
```
Table: users (3 columns)
├─ user_id (PK)
├─ signup_date
└─ country

Table: transactions (4 columns)
├─ transaction_id (PK)
├─ user_id (FK → users)
├─ amount
└─ timestamp
```

**Output - Flattened Single Table:**
```
user_id | signup_date | country | COUNT(txns) | AVG(amount) | MAX(amount) | 
        | SUM(amount) | days_since_signup | txn_frequency | ... (50+ features)
```

**Generated Features Include:**
- **Aggregations:** COUNT, SUM, AVG, MIN, MAX of related table columns
- **Temporal:** Time-based features (year, month, day, hour, dayofweek)
- **Derived:** Time since events, ratios, frequencies
- **Cross-table:** Features combining information from multiple tables

**Why This Matters:**  
Foundation models like TabPFN and Limix are designed for single-table input. FastDFS enables these powerful models to handle relational data by intelligently flattening the schema while preserving relational information through feature engineering.

### Supported Models

| Model | Type | Best For | Notes |
|-------|------|----------|-------|
| **TabPFN** | Foundation Model | Small-medium datasets | Transformer-based, no training needed |
| **Limix** | Foundation Model | Linear relationships | Fast, interpretable |
| **AutoGluon** | Ensemble | Debugging/Baseline | Multiple models, auto-tuning |

---

## 📄 License

MIT License - see LICENSE file for details.

## 📝 Changelog

### v0.1.0
- Initial release with FastDFS-based table flattening
- Support for TabPFN, Limix, and AutoGluon
- One-call and step-by-step APIs
- Binary classification and regression tasks
- Evaluation utilities for benchmarking

---

**Questions or Issues?** Open an issue on [GitHub](https://github.com/dglai/multitabfm/issues)

**Want to Contribute?** Pull requests are welcome! 🎉