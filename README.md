# MultiTabFM

A simplified multi-table feature modeling framework with AutoGluon integration for automated machine learning on relational data.

## Overview

MultiTabFM provides an easy-to-use interface for:
- Multi-table feature engineering using Deep Feature Synthesis (DFS)
- Automated model training with AutoGluon
- End-to-end prediction pipelines
- Model evaluation with common metrics

## Installation

### Standard Installation

```bash
pip install multitabfm
```

### Development Installation

```bash
git clone https://github.com/your-username/multitabfm.git
cd multitabfm
pip install -e .
```

## Quick Start

### Basic Usage

```python
from multitabfm import MultiTabFM

# Initialize the framework
model = MultiTabFM()

# End-to-end training and prediction
predictions, metrics = model.train_and_predict(
    rdb_data_path="data/rel-event",
    task_data_path="data/rel-event/user-ignore",
    eval_metrics=["accuracy", "roc_auc"]
)

print(f"Predictions shape: {predictions.shape}")
print(f"Metrics: {metrics}")
```

### Step-by-Step Usage

```python
import pandas as pd
from multitabfm import MultiTabFM, load_dataset, prepare_target_dataframes, generate_features

# 1. Load your data
train_data, test_data, metadata, rdb = load_dataset(
    rdb_data_path="data/rel-event",
    task_data_path="data/rel-event/user-ignore"
)

# 2. Prepare target dataframes
train_df, test_df = prepare_target_dataframes(train_data, test_data, metadata)

# 3. Extract metadata
key_mappings = {k: v for d in metadata['key_mappings'] for k, v in d.items()}
time_column = metadata['time_column']
target_column = metadata['target_column']

# 4. Generate features using DFS
train_features = generate_features(train_df, rdb, key_mappings, time_column)
test_features = generate_features(test_df, rdb, key_mappings, time_column)

# 5. Train model
model = MultiTabFM()
model.fit(train_features, label_column=target_column)

# 6. Make predictions
predictions = model.predict_proba(test_features)

# 7. Evaluate (if test labels available)
if target_column in test_features.columns:
    metrics = model.evaluate(
        test_features[target_column], 
        predictions, 
        metrics=["accuracy", "roc_auc", "logloss"]
    )
    print(f"Evaluation metrics: {metrics}")
```

## API Reference

### Core Classes

#### `MultiTabFM`

Main class for multi-table feature modeling.

**Methods:**
- [`fit(train_features, label_column, model_config=None)`](multitabfm/core.py:18): Train the model
- [`predict_proba(test_features)`](multitabfm/core.py:22): Get prediction probabilities
- [`evaluate(labels, proba, metrics=None)`](multitabfm/core.py:26): Evaluate predictions
- [`train_and_predict(rdb_data_path, task_data_path, ...)`](multitabfm/core.py:30): End-to-end pipeline

#### `AGAdapter`

AutoGluon TabularPredictor adapter for model training and prediction.

### Utility Functions

#### [`load_dataset(rdb_data_path, task_data_path)`](multitabfm/utils.py:10)

Load RDB data and task data from specified paths.

**Returns:** `(train_df, test_df, metadata, rdb)`

#### [`prepare_target_dataframes(train_data, test_data, metadata)`](multitabfm/utils.py:35)

Prepare target dataframes with ID, time, and label columns for DFS.

**Returns:** `(train_df, test_df)`

#### [`generate_features(target_df, rdb, key_mappings, time_column, dfs_config=None)`](multitabfm/feature_engineer.py:6)

Generate DFS features using fastdfs.

**Returns:** Feature-augmented DataFrame

#### [`get_default_configs()`](multitabfm/utils.py:63)

Get default DFS and model configurations.

**Returns:** `(dfs_config, model_config)`

## Dependencies

### Core Dependencies
- pandas
- numpy
- PyYAML
- scikit-learn
- fastdfs (from git)
- autogluon.tabular

## License

This project is licensed under the MIT License.

## Changelog

### v0.1.0
- Initial release
- Basic multi-table feature modeling support
- AutoGluon integration
- Core evaluation metrics