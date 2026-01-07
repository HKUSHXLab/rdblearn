#!/usr/bin/env python3
"""
Test script for FastDFS pipeline.compute_features method with rel-trial dataset.

This script demonstrates:
1. Loading RDB dataset
2. Loading target table
3. Setting up DFSPipeline with transforms
4. Testing pipeline.compute_features method
5. Analyzing generated features
"""

import numpy as np
import pandas as pd

# Set environment variable to disable tqdm before importing fastdfs
import os
os.environ['TQDM_DISABLE'] = '1'

# Workaround PyTorch SDPA kernel issues on some GPUs / driver combos
# Force "math" implementation of scaled_dot_product_attention to avoid CUDA kernel crashes
# Ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
os.environ.setdefault("PYTORCH_SDP_KERNEL", "math")

# Optional: Enable synchronous CUDA error reporting while debugging
# Uncomment to get precise stacktraces from CUDA kernels
# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import fastdfs
from pathlib import Path
import sys
import warnings
from collections import Counter
from sklearn.metrics import roc_auc_score, mean_absolute_error
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularPredictor
from loguru import logger

from yaml_utils import process_yaml, load_yaml
from fastdfs.transform import RDBTransformWrapper, RDBTransformPipeline, HandleDummyTable, FeaturizeDatetime


# Suppress verbose logging from fastdfs - only show critical errors
logger.remove()
logger.add(sys.stderr, level="ERROR")

# Suppress tqdm progress bars by overriding the class
class SilentContext:
    """Context manager to suppress tqdm and other output"""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

# Monkey patch tqdm to be silent
import tqdm
original_tqdm_init = tqdm.tqdm.__init__
def silent_init(self, *args, **kwargs):
    kwargs['disable'] = True
    return original_tqdm_init(self, *args, **kwargs)

tqdm.tqdm.__init__ = silent_init

warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`")
warnings.filterwarnings(
    "ignore",
    message="The provided callable .* is currently using .*",
    category=FutureWarning,
    module="featuretools.computational_backends.feature_set_calculator"
)

def main(rdb_path, dfs_max_depth=0, target_tasks=None, target_task_type=None):
    """
    If target_tasks is specified, only the provided task will be evaluated
    """
    print("=" * 60)
    print("Training TabPFN-v2 on DFS features")
    print("=" * 60)

    # train_table_path = Path(task_path) / "train.pqt"
    # test_table_path = Path(task_path) / "test.pqt"


    metadata_path = Path(rdb_path) / "metadata.yaml"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    try:
        rdb = fastdfs.load_rdb(rdb_path)
        print(f"✅ RDB loaded successfully")
        print(f"   - Tables: {list(rdb.tables.keys())}")
        print()
    except Exception as e:
        print(f"❌ Failed to load RDB: {e}")
        return

    task_datas = process_yaml(metadata_path)
    for task_data in task_datas:
        
        key_mappings = task_data.get('key_mappings')
        print(f"key_mappings: {key_mappings}")
        target_column = task_data.get('target_column')
        cutoff_time_column = task_data.get('time_column')
        task_type = task_data.get('task_type')
        task_name = task_data.get('task_name')

        if (target_tasks is not None and task_name in target_tasks) or (target_tasks is None and task_type == target_task_type):

            # input(f"Press Enter to continue for task: {task_name}...")
            task_folder = Path(rdb_path) / task_name
            train_table_path = task_folder / "train.pqt"
            test_table_path = task_folder / "test.pqt"

            # Step 2: Load target table
            print("Step 2: Loading target table...")
            try:
                train_df = pd.read_parquet(train_table_path)
                test_df = pd.read_parquet(test_table_path)

                # Uniformly sample
                # if len(train_df) > 10000:
                #     # INSERT_YOUR_CODE
                #     print(f"Loaded large train_df with shape: {train_df.shape}")
                #     print(f"Loaded test_df with shape: {test_df.shape}")
                #     print(f"Columns in train_df: {list(train_df.columns)}")
                #     train_df = train_df.sample(n=10000, random_state=42).reset_index(drop=True)

            except Exception as e:
                print(f"❌ Failed to load target table: {e}")
                return

            # train_df["tmp_index"] = range(len(train_df))
            # test_df["tmp_index"] = range(len(test_df))

            #train_df = train_df.sort_values(by=cutoff_time_column).reset_index(drop=True)
            #test_df = test_df.sort_values(by=cutoff_time_column).reset_index(drop=True)

            X_train, Y_train = train_df.drop(columns=[target_column]), train_df[target_column]
            X_test, Y_test = test_df.drop(columns=[target_column]), test_df[target_column]

            # Step 4: Test basic DFS without transforms
            print("Step 4: Testing basic DFS feature computation...")
            #@try:
            # Use basic compute_dfs_features function

            dfs_train_features = None
            dfs_test_features = None

            # dfs_train_features = fastdfs.compute_dfs_features(
            #     rdb=rdb,
            #     target_dataframe=X_train, #[['timestamp', 'nct_id']],
            #     key_mappings=key_mappings,
            #     cutoff_time_column=cutoff_time_column,
            #     config_overrides={"max_depth": dfs_max_depth, "engine": "dfs2sql"}
            # )

            # dfs_test_features = fastdfs.compute_dfs_features(
            #     rdb=rdb,
            #     target_dataframe=X_test, #[['timestamp', 'nct_id']],
            #     key_mappings=key_mappings,
            #     cutoff_time_column=cutoff_time_column,
            #     config_overrides={"max_depth": dfs_max_depth, "engine": "dfs2sql"}
            # )

            dfs_config = fastdfs.DFSConfig(
                max_depth=dfs_max_depth,
                agg_primitives=["max", "min", "mean", "count", "mode", "std"],
                engine="dfs2sql"
                #engine="featuretools"
            )

            # Create pipeline (without transforms for now)
            pipeline = fastdfs.DFSPipeline(
                transform_pipeline=None,  # No transforms for this test
                # transform_pipeline=RDBTransformPipeline([
                #     HandleDummyTable(),
                #     RDBTransformWrapper(FeaturizeDatetime(features=["year", "month", "day", "hour", "dayofweek"])),
                # ]),
                dfs_config=dfs_config
            )

            dfs_train_features = pipeline.compute_features(
                rdb=rdb,
                target_dataframe=X_train,
                key_mappings=key_mappings,
                cutoff_time_column=cutoff_time_column
            )

            dfs_test_features = pipeline.compute_features(
                rdb=rdb,
                target_dataframe=X_test,
                key_mappings=key_mappings,
                cutoff_time_column=cutoff_time_column
            )

            # # Check for <NA> (pd.NA) in each column of dfs_train_features
            # for col in dfs_train_features.columns:
            #     # Skip boolean columns to avoid TypeError with pd.NA
            #     if dfs_train_features[col].dtype == 'bool' or dfs_train_features[col].dtype == 'boolean':
            #         contains_na = "skipped (boolean column)"
            #     else:
            #         contains_na = dfs_train_features[col].isin([pd.NA]).any()
            #     print(f"Column '{col}' contains <NA>: {contains_na}")
    
            # Remove duplicated columns in train_features and test_features
            # dfs_train_features = dfs_train_features.loc[:, ~dfs_train_features.columns.duplicated()]
            # dfs_test_features = dfs_test_features.loc[:, ~dfs_test_features.columns.duplicated()]

            # dfs_train_features = dfs_train_features.replace({pd.NA: None})
            # dfs_test_features = dfs_test_features.replace({pd.NA: None})

            # for col in dfs_train_features.columns:
            #     # Skip boolean columns to avoid TypeError with pd.NA
            #     if dfs_train_features[col].dtype == 'bool' or dfs_train_features[col].dtype == 'boolean':
            #         contains_na = "skipped (boolean column)"
            #     else:
            #         contains_na = dfs_train_features[col].isin([pd.NA]).any()
            #     print(f"Column '{col}' contains <NA>: {contains_na}")

            if dfs_train_features is None:
                print("\nUsing target table features")
                train_features = X_train
                test_features = X_test

            else:
                print("\nUsing DFS features")
                train_features = dfs_train_features
                test_features = dfs_test_features

                col_to_drop_list = []
                for col in train_features.columns:
                    if type(train_features[col].iloc[0]) == np.ndarray:
                        col_to_drop_list.append(col)
                        print(f"Dropping column {col} because it is a numpy array")
                train_features = train_features.drop(columns=col_to_drop_list)
                test_features = test_features.drop(columns=col_to_drop_list)

            # print("\n\n\n")
            # for col in X_train:
            #     print(col, X_train[col].dtype)

            # print("\n\n\n")
            # for col in train_features:
            #     print(col, train_features[col].dtype)
            

            print(f"   - Train features: {len(train_features.columns)}")
            # col_counts = Counter(train_features.columns)
            # duplicate_names = [col for col, cnt in col_counts.items() if cnt > 1]
            # if duplicate_names:
            #     print("⚠️ Columns with duplicate names detected:")
            #     for col in duplicate_names:
            #         # Get all indices that have this column name
            #         indices = [i for i, c in enumerate(train_features.columns) if c == col]
            #         dtypes = [train_features.iloc[:, i].dtype for i in indices]
            #         print(f"   - Column '{col}' appears {len(indices)} times with dtypes: {dtypes}")
            # else:
            #     print(f"   - All generated columns: {[col for col in train_features.columns]}")
            # print(f"   - Sample new features: {[col for col in train_features.columns if col not in X_train.columns]}")
            # print()

            
            final_train_df = pd.concat([train_features, Y_train], axis=1)
            # for col in final_train_df.columns:
            #     print(col, final_train_df[col].dtype, final_train_df[col].shape)
            #     print(final_train_df[col].head(3))
            #     print(f"{type(final_train_df[col].iloc[0])=}")

            feature_generator = AutoMLPipelineFeatureGenerator(
                enable_datetime_features=True,
                enable_raw_text_features=False,
                enable_text_special_features=False,
                enable_text_ngram_features=False, 
            )

            predictor = TabularPredictor(
                label=target_column,
                # problem_type=target_task_type,
            ).fit(
                train_data=final_train_df,
                feature_generator=feature_generator,
                    hyperparameters={
                        "TABPFNV2": { 
                            "random_state": 42,
                            "n_jobs": -1,
                            "n_estimators": 1,
                            "inference_config": {
                                "SUBSAMPLE_SAMPLES": 10000,
                            },
                            "ignore_pretraining_limits": True,
                            "ag.max_rows": 10000000,            # AG raises an error if the input is larger than this value
                            "ag.max_memory_usage_ratio": None,  # disabled memory check of AG
                        }
                    },
                # no ensembling:
                num_bag_folds=0,       # ensure no bagging
                num_bag_sets=None,
                num_stack_levels=0,    # ensure no stacking
                raise_on_model_failure=True,  # Set to True for debugging (see full traceback), False to skip failed models
            )

            # Predict in batches of 5000 and concatenate
            # batch_size = 5000
            # proba_batches = []
            # num_rows = len(test_features)
            # for start in range(0, num_rows, batch_size):
            #     end = min(start + batch_size, num_rows)
            #     batch = test_features.iloc[start:end]
            #     if target_task_type == "classification":
            #         proba_batch = predictor.predict_proba(batch)
            #     else:
            #         proba_batch = predictor.predict(batch)
            #     proba_batches.append(proba_batch)
            # # Concatenate all batches

            # proba = pd.concat(proba_batches, axis=0).reset_index(drop=True)

            # predict the whole test set at once
            if target_task_type == "classification":
                proba = predictor.predict_proba(test_features)
            else:
                proba = predictor.predict(test_features)

            # X_trans = predictor.transform_features(X_train)

            # print(X_trans.columns[:20])
            if target_task_type == "classification":
                if proba.shape[1] == 2:
                    print("Binary classification")
                    roc_auc = roc_auc_score(Y_test, proba.iloc[:, 1])
                else:
                    print("Multi-class classification")
                    roc_auc = roc_auc_score(Y_test, proba, multi_class='ovr')
                print(f"roc_auc: {roc_auc*100:.2f}")
            else:
                print("Regression")
                
                mae = mean_absolute_error(Y_test, proba)
                print(f"mae: {mae:.2f}")
            
            print("\n\n\n")



if __name__ == "__main__":
    # Set up paths
    # rdb_path = "/root/linjie/rel-data/rel-trial"
    # rdb_path = "/root/linjie/rel-data/rel-event"
    # rdb_path = "/root/linjie/rel-data/rel-f1"

    # rdb_path = "/root/linjie/rel-data/rel-avito"

    # rdb_path = "/root/linjie/rel-data/rel-amazon"
    rdb_path = "/root/yanlin/tabular-chat-predictor/data/rel-amazon"

    # rdb_path = "/root/linjie/rel-data/rel-hm"

    # rdb_path = "/root/linjie/rel-data/rel-stack"

    max_depth = 3

    target_task_type = "classification"
    # target_task_type = "regression"

    main(rdb_path, dfs_max_depth=max_depth, target_task_type=target_task_type)
