# multitabfm — current package design and interfaces

This document describes the implemented interfaces of `multitabfm`, a small framework for multi-table (relational) feature generation with DFS and automated tabular modeling via AutoGluon.

## High-level goals

- Provide a compact, friendly API for end-to-end modeling on relational data
- Generate features from an RDB using DFS (via the local `fastdfs` package)
- Train and predict with AutoGluon Tabular using sensible defaults
- Offer both a one-call API and a step-by-step workflow

## What’s implemented

The package exposes the following top-level imports (see `multitabfm/__init__.py`):

- MultiTabFM — core orchestrator class
- AGAdapter, ModelPredictor — model layer abstractions (AGAdapter is the default)
- load_dataset, prepare_target_dataframes, get_default_configs — utilities
- generate_features — DFS wrapper powered by `fastdfs.api.compute_dfs_features`
- train_and_predict — convenience function mirroring the class method

### Data expectations

- rdb_data_path: directory consumable by `fastdfs.load_rdb`
- task_data_path: directory containing `train.pqt`, `test.pqt`, and `metadata.yaml`
- metadata.yaml keys:
  - key_mappings: list[dict] mapping local id columns to table.column (e.g., `user_id: users.user_id`)
  - time_column: cutoff time column name
  - target_column: binary label column name

## Public API (as implemented)

### Class: MultiTabFM (in `core.py`)

- __init__(dfs_config: dict | None = None, model_config: dict | None = None)
  - Stores optional default configs; initializes an AutoGluon-based adapter (AGAdapter)

- fit(train_features: pd.DataFrame, label_column: str, model_config: dict | None = None) -> None
  - Trains the internal adapter. `train_features` must include the label column.

- predict_proba(test_features: pd.DataFrame) -> np.ndarray
  - Returns 1D probabilities (positive-class when binary) from the trained adapter.

- evaluate(labels: pd.Series, proba: np.ndarray | pd.DataFrame, metrics: list[str] | None = None) -> dict
  - Computes metrics via `evaluation.compute_metrics` (supports: accuracy, roc_auc, logloss).

- train_and_predict(rdb_data_path: str, task_data_path: str, *, dfs_config: dict | None = None, model_config: dict | None = None, eval_metrics: list[str] | None = None) -> tuple[np.ndarray | pd.DataFrame, dict | None]
  - End-to-end: loads data (utils.load_dataset), prepares target tables, generates features for train/test, trains, predicts, and optionally evaluates if labels exist in test features and `eval_metrics` is provided.

### Convenience function (in `api.py`)

- train_and_predict(...)
  - Thin wrapper around `MultiTabFM.train_and_predict` with the same signature. Returns `(proba, metrics)`.

### Feature engineering (in `feature_engineer.py`)

- generate_features(target_df: pd.DataFrame, rdb: Any, key_mappings: dict, time_column: str, dfs_config: dict | None = None) -> pd.DataFrame
  - Calls `fastdfs.api.compute_dfs_features` with the provided inputs and optional overrides.

### Utilities (in `utils.py`)

- load_dataset(rdb_data_path: str, task_data_path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict, Any]
  - Loads the RDB (`fastdfs.load_rdb`), `train.pqt`, `test.pqt`, and `metadata.yaml`.

- prepare_target_dataframes(train_data: pd.DataFrame, test_data: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, pd.DataFrame]
  - Constructs ID+time (+label) target tables for DFS using metadata.

- get_default_configs() -> tuple[dict, dict]
  - Provides default DFS and model configs aligned with the repository’s AutoGluon usage.

### Model layer (in `model.py`)

- ModelPredictor — abstract base with `fit(...)` and `predict_proba(...)` methods
- AGAdapter — AutoGluon TabularPredictor wrapper
  - fit(X: pd.DataFrame, label_column: str, config: dict | None = None)
    - Expects `X` to include the label column. Sets up a feature generator and trains `TabularPredictor` with default hyperparameters (TABPFNV2).
  - predict_proba(X: pd.DataFrame) -> np.ndarray
    - Returns 1D probabilities; for binary classification, uses the positive class.

### Evaluation (in `evaluation.py`)

- compute_metrics(labels: pd.Series, proba: np.ndarray | pd.DataFrame, metrics: list[str] | None = None) -> dict
  - Supports: accuracy, roc_auc (if available), logloss (if available). Coerces `proba` to a 1D array.

## Data flow (step-by-step)

1) Load data and metadata: `train_data, test_data, metadata, rdb = load_dataset(rdb_data_path, task_data_path)`
2) Prepare target tables: `train_df, test_df = prepare_target_dataframes(train_data, test_data, metadata)`
3) Extract DFS parameters: `key_mappings`, `time_column`, `target_column`
4) Generate features: `train_features = generate_features(...); test_features = generate_features(...)`
5) Train and predict: `engine.fit(...); proba = engine.predict_proba(test_features)`
6) Optional evaluation: `engine.evaluate(labels, proba, metrics=[...])`

## Notes and limitations

- The current implementation focuses on AutoGluon for modeling (AGAdapter). Save/load and caching utilities are not implemented.
- `ModelPredictor` is minimal and meant as a base for future adapters.
- `generate_features` requires a valid `rdb` from `fastdfs.load_rdb` and a correct metadata mapping.

## Configuration examples

Default configs are provided by `get_default_configs()`:

```yaml
# DFS (dict example)
max_depth: 3
engine: dfs2sql
agg_primitives:
  - max
  - min
  - mean
  - count
  - mode
```

Model defaults (TABPFNV2 with sensible limits) are applied internally; you can pass overrides via `model_config`.

## Module layout

- multitabfm.api — convenience API (`train_and_predict`)
- multitabfm.core — `MultiTabFM` orchestrator
- multitabfm.feature_engineer — DFS wrapper
- multitabfm.model — model abstractions and AutoGluon adapter
- multitabfm.utils — I/O and configuration helpers
- multitabfm.evaluation — metrics

## Future directions

- Additional adapters (e.g., sklearn, LightGBM) and model artifact persistence
- Feature caching/persistence and richer DFS controls
- Expanded metrics and task-type support

## Done criteria (for this iteration)

- End-to-end path-based API (`train_and_predict`) works on provided dataset layout
- Step-by-step workflow functions match code and README examples
- Basic metrics available and tested on small samples

