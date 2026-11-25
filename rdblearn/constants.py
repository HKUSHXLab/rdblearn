TABPFN_DEFAULT_CONFIG = {
    "n_estimators": 8,
    "ignore_pretraining_limits": True,  # Allow datasets larger than 10K samples
    "device": "cuda",
    "n_preprocessing_jobs": -1,
    # Keep the training context's key/value cache from `fit` so `predict*` calls only run the
    # test rows; TabPFN's default "fit_preprocessors" re-encodes the training rows on every call.
    "fit_mode": "fit_with_cache",
}

TABICL_DEFAULT_CONFIG = {
    "n_estimators": 8,
    "device": "cuda",
    # Same reason as TabPFN's fit_mode: build the training context's KV cache in `fit`
    # (TabICL's default is no cache, so every predict call re-encodes the training rows).
    "kv_cache": True,
}

RDBLEARN_DEFAULT_CONFIG = {
    "dfs": {
        "max_depth": 2,
        "agg_primitives": ["max", "min", "mean", "count", "mode", "std", "num_unique"],
        "engine": "dfs2sql"
    },
    "max_train_samples": 10000,
    "stratified_sampling": False,
    "predict_batch_size": 5000
}

TARGET_HISTORY_TABLE_NAME = "_RDBL_target_history"