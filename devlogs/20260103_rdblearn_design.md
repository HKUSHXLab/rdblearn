# RDBLearn Design Document
**Date:** 2026-01-03
**Status:** Draft

## 1. Overview

The goal of `rdblearn` is to provide a scikit-learn compatible interface for machine learning on relational databases (RDBs). Instead of requiring users to manually flatten tables, `rdblearn` estimators accept a target DataFrame (`X`) and an RDB object, automatically handling feature engineering via Deep Feature Synthesis (DFS) and feature preprocessing.

This design moves away from the monolithic `train_and_predict` workflow of `multitabfm` towards a more flexible, composable API.

## 2. Core Abstractions

### 2.1 Estimators

We will provide two main classes: `RDBLearnClassifier` and `RDBLearnRegressor`.

**Signature:**

```python
from fastdfs import RDB, DFSConfig

class RDBLearnEstimator:
    def __init__(
        self, 
        base_estimator, 
        dfs_config: Optional[Union[dict, DFSConfig]] = None,
        prep_config: Optional[dict] = None
    ):
        """
        Args:
            base_estimator: An instance of a scikit-learn compatible estimator 
                            (e.g., TabPFNClassifier, TabPFNRegressor).
            dfs_config: Configuration for fastdfs. If None, defaults to:
                        {
                            "max_depth": 3,
                            "agg_primitives": ["max", "min", "mean", "count", "mode", "std"],
                            "engine": "dfs2sql"
                        }
            prep_config: Configuration for feature preprocessing. If None, defaults to:
                        {
                            "enable_datetime_features": True,
                            "enable_raw_text_features": False,
                            "enable_text_special_features": False,
                            "enable_text_ngram_features": False,
                        }
                        Note: Categorical columns are automatically label-encoded before AG generation.
        """
        pass

    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        rdb: RDB,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str] = None,
        **kwargs
    ):
        """
        1. **Downsampling**: Check `max_samples` in config. If X is larger, sample X and y *before* DFS to save time.
        2. **RDB Transformation**: Apply standard RDB transforms to clean and prepare the database.
           Pipeline:
           - `HandleDummyTable()`: Remove tables with no columns.
           - `FillMissingPrimaryKey()`: Fill missing PKs to ensure join integrity.
           - `RDBTransformWrapper(FeaturizeDatetime(features=["year", "month", "hour"]))`: Extract datetime features.
           - `RDBTransformWrapper(FilterColumn(drop_dtypes=["text"]))`: Drop text columns (not supported by TabPFN).
           - `RDBTransformWrapper(CanonicalizeTypes())`: Ensure consistent types.
           *Store the transformed RDB in `self.rdb_`.*
        3. **Feature Augmentation**: Call `fastdfs.compute_dfs_features(self.rdb_, X, key_mappings, cutoff_time_column, config=self.dfs_config)`.
        4. **Preprocessing**: Initialize and fit the feature preprocessor (AutoGluon generator). 
           - Store the fitted preprocessor in `self.preprocessor_`.
           - Transform the features.
        5. **Model Training**: Call `base_estimator.fit(X_transformed, y, **kwargs)`.
        6. **State Persistence**: Store `key_mappings` and `cutoff_time_column` (if needed for validation) and the fitted `self.preprocessor_`.
        """
        pass

    def predict(
        self, 
        X: pd.DataFrame, 
        rdb: Optional[RDB] = None,
        **kwargs
    ):
        """
        1. **Validation**: Validate that X has the same schema as the X passed to fit (minus the label).
        2. **RDB Selection**: 
           - If `rdb` is None, use `self.rdb_` (stored from fit).
           - If `rdb` is provided, apply the same RDB transforms as in fit to create a new transformed RDB.
        3. **Feature Augmentation**: Call `fastdfs.compute_dfs_features(selected_rdb, X, self.key_mappings, self.cutoff_time_column, ...)` using stored mappings.
        4. **Preprocessing**: Transform features using the *stored* `self.preprocessor_` from `fit`.
        5. **Prediction**: Call `base_estimator.predict(X_transformed, **kwargs)`.
        """
        pass
```

`RDBLearnClassifier` will add `predict_proba(X, rdb=None, **kwargs)`.

**Note for Regressors:**
For `RDBLearnRegressor`, the `output_type` (e.g., for probabilistic regression) is determined by the user when calling `predict(..., output_type='quantiles')`. `rdblearn` does not modify or handle the output format; it simply passes `**kwargs` (including `output_type`) to `base_estimator.predict()` and returns the result.

### 2.2 Dataset Adapters

To facilitate benchmarking and usage with standard datasets (RelBench, 4DBInfer), we introduce `RDBDataset`.

```python
@dataclass
class Task:
    name: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    val_df: Optional[pd.DataFrame]
    metadata: dict  # Contains key_mappings, time_col, target_col

class RDBDataset:
    def __init__(self, rdb: RDB, tasks: List[Task]):
        self.rdb = rdb
        self.tasks = {t.name: t for t in tasks}

    @classmethod
    def from_relbench(cls, dataset_name: str):
        # Uses fastdfs.adapter.relbench to load RDB
        # Loads tasks using relbench library
        pass

    @classmethod
    def from_4dbinfer(cls, dataset_name: str):
        # Uses fastdfs.adapter.dbinfer
        pass
```

### 2.3 Default Model Configurations

To ensure consistent performance, we provide default configurations for the base estimators (`TabPFNClassifier` and `TabPFNRegressor`). These defaults align with the settings used in `multitabfm.model.CustomTabPFN`.

We will expose these as constants in `rdblearn.constants`:

```python
TABPFN_DEFAULT_CONFIG = {
    "n_estimators": 8,
    "ignore_pretraining_limits": True,  # Allow datasets larger than 10K samples
    "device": "cuda",
    "n_preprocessing_jobs": -1
}
```

**Usage:**

```python
from rdblearn.constants import TABPFN_DEFAULT_CONFIG
from tabpfn import TabPFNClassifier, TabPFNRegressor

# User can easily apply defaults
clf = TabPFNClassifier(**TABPFN_DEFAULT_CONFIG)
reg = TabPFNRegressor(**TABPFN_DEFAULT_CONFIG)
```

## 3. User Workflow (Mocking Example)

Here is how a user would use `rdblearn` with a RelBench dataset.

```python
from rdblearn.datasets import RDBDataset
from rdblearn.models import RDBLearnClassifier
from tabpfn import TabPFNClassifier
from fastdfs import DFSConfig

# 1. Load Dataset
dataset = RDBDataset.from_relbench("rel-stack")
task = dataset.tasks["user-churn"]

# 2. Initialize Model
# User brings their own base estimator
base_model = TabPFNClassifier(device='cuda', N_ensemble_configurations=32)

clf = RDBLearnClassifier(
    base_estimator=base_model,
    dfs_config=DFSConfig(max_depth=2, engine="dfs2sql")
)

# 3. Train
# X_train does not contain the label. y_train is passed separately.
X_train = task.train_df.drop(columns=[task.metadata["target_col"]])
y_train = task.train_df[task.metadata["target_col"]]

clf.fit(
    X=X_train, 
    y=y_train, 
    rdb=dataset.rdb,
    key_mappings=task.metadata["key_mappings"],
    cutoff_time_column=task.metadata["time_col"]
)

# 4. Predict
X_test = task.test_df.drop(columns=[task.metadata["target_col"]])
y_test = task.test_df[task.metadata["target_col"]]

y_pred = clf.predict(
    X=X_test
    # rdb is optional, uses the one from fit if omitted
)

# 5. Evaluate (User code)
from sklearn.metrics import roc_auc_score
print(f"AUC: {roc_auc_score(y_test, y_pred)}")
```

## 4. Implementation Plan

### Phase 1: Core Structure & Migration
1.  **Create `rdblearn` package structure**:
    *   `rdblearn/estimator.py`: `RDBLearnClassifier`, `RDBLearnRegressor`.
    *   `rdblearn/preprocessing.py`: Logic for AG transforms.
    *   `rdblearn/datasets.py`: `RDBDataset`, `Task`.
2.  **Migrate Feature Engineering**:
    *   Move logic from `multitabfm.feature_engineer` to `rdblearn.preprocessing`.
    *   Specifically `generate_features` (which calls `fastdfs`) and `ag_transform`.
    *   Ensure `ag_transform` state (e.g., category mappings, fill values) is stored in the estimator during `fit` and applied in `predict`.
3.  **Implement Estimators**:
    *   Implement `fit`, `predict`, `predict_proba`.
    *   Add downsampling logic in `fit` (reuse `multitabfm.core.MultiTabFM` sampling logic if applicable, or simplify).
    *   Ensure `base_estimator` is treated as a black box as much as possible, though we assume it follows sklearn API.

### Phase 2: Dataset Adapters
1.  **RelBench Adapter**:
    *   Implement `RDBDataset.from_relbench`.
    *   Requires `relbench` dependency.
    *   Map RelBench task structure to `rdblearn.Task`.
2.  **4DBInfer Adapter**:
    *   Implement `RDBDataset.from_4dbinfer`.
    *   Reuse existing loading logic from `multitabfm.utils.load_dataset` but adapt to new structure.

### Phase 3: Cleanup & Examples
1.  Create a new example script `examples/rdblearn_relbench.py` matching the mock above.
2.  Deprecate `multitabfm` package contents or keep them as legacy until fully replaced.
3.  Update `pyproject.toml` to include `rdblearn` (or rename project).

## 5. Code Reuse Strategy

*   **`multitabfm.feature_engineer.py`**:
    *   `generate_features`: This is essentially the first step of `fit`/`predict`. We will call `fastdfs.compute_dfs_features` directly in `rdblearn`, but we might need the logic that prepares the `dfs_config`.
    *   `ag_transform`, `ag_label_transform`: **CRITICAL**. This logic needs to be encapsulated in a stateful preprocessor class (e.g., `TabularPreprocessor`) so we can `fit` it on train and `transform` on test. Currently, it might be stateless or loosely coupled.
*   **`multitabfm.model.CustomTabPFN`**:
    *   The new `RDBEstimator` takes an *instance* of a model. Users can pass a `TabPFNClassifier` directly. We might not need `CustomTabPFN` anymore if standard `TabPFN` classes are sufficient. If `CustomTabPFN` added value (e.g., handling regression as classification, specific ensemble logic), we should document that users should configure their `TabPFN` instance accordingly before passing it.
*   **`multitabfm.utils`**:
    *   `load_dataset`: Logic for reading `metadata.yaml` and parquet files will be moved to `RDBDataset` loaders.
