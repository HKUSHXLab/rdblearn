# Dev Plan: RDBLearn Target History Augmentation

**Date:** 2026-01-22
**Status:** Planned

## 1. Objective

Enable `RDBLearnEstimator` to automatically augment the Relational Database (RDB) with the full training data (`X` and `y`) as a "history" table. This allows the model to learn from historical target values (e.g., past user ratings) while respecting cutoff times (temporal consistency). This augmentation should use the *full* provided dataset before any downsampling occurs.

## 2. Changes

### 2.1 Configuration (`rdblearn/config.py`)

Update `RDBLearnConfig` to include:
- `enable_target_augmentation: bool = False`

### 2.2 Estimator (`rdblearn/estimator.py`)

#### `RDBLearnEstimator._prepare_rdb`
Update signature to accept training context:
```python
def _prepare_rdb(
    self, 
    rdb: RDB, 
    X: Optional[pd.DataFrame] = None, 
    y: Optional[pd.Series] = None,
    key_mappings: Optional[Dict[str, str]] = None,
    cutoff_time_column: Optional[str] = None
) -> RDB:
```

**Logic:**
1. **Check Condition**: If `self.config.enable_target_augmentation` is True:
    - If `cutoff_time_column` is None: Log warning/debug "Target augmentation requires cutoff_time_column. Skipping." and proceed without augmentation.
    - If `X` and `y` are provided:
        - Construct "history" table:
            - Concatenate `X` and `y`.
            - Table Name: "target_history" (or similar).
            - Time Column: `cutoff_time_column`.
            - Foreign Keys: Infer from `key_mappings`.
                - Iterate `key_mappings`: `col_in_X` -> `table.col_in_RDB`.
                - Foreign Key: `(col_in_X, table, col_in_RDB)`.
        - Augment RDB: `rdb.add_table(...)`.
2. **Transform**: Apply `RDBTransformPipeline` to the (potentially augmented) RDB.
3. Return prepared RDB.

#### `RDBLearnEstimator.fit`
Reorder pipeline:
1. **Ensure keys are strings**.
2. **Prepare RDB (Augmentation + Transform)**: 
   - Call `self._prepare_rdb(rdb, X, y, key_mappings, cutoff_time_column)`.
   - This ensures `self.rdb_` contains the "target_history" table derived from the full `X, y`.
3. **Downsampling**:
   - Check `self.config.max_train_samples`.
   - If needed, downsample `X` and `y` -> `X_sampled`, `y_sampled`.
4. **Feature Augmentation (DFS)**:
   - Call `fastdfs.compute_dfs_features(self.rdb_, X_sampled, ...)`
   - DFS will use `self.rdb_` (which has full history) to generate features for `X_sampled`.
5. **Preprocessing**.
6. **Model Training**.

### 2.3 Tests (`tests/test_rdblearn_estimator.py`)

Add test case `test_fit_with_target_augmentation`:
- Enable `enable_target_augmentation`.
- Provide `cutoff_time_column`.
- Mock RDB and Data.
- Call `fit`.
- Assertions:
    - Verify `self.rdb_` contains "target_history" (or whatever name we choose).
    - Verify generated features (if possible, or at least that no error occurs).
    - Verify that if `cutoff_time_column` is missing, augmentation is skipped.

## 3. Implementation Details

- **Foreign Keys**: `key_mappings` is `{"target_col": "rdb_table.rdb_col"}`.
    - We need to parse `"rdb_table.rdb_col"` to split into table and column.
- **Table Name Clash**: Ensure "target_history" doesn't clash with existing tables. Maybe make it configurable or use a distinct name. Default `target_history`.

## 4. Verification

Run `pytest tests/test_rdblearn_estimator.py`.
