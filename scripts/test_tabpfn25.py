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
import argparse
import numpy as np
import pandas as pd
# Set environment variable to disable tqdm before importing fastdfs
import os
os.environ['TQDM_DISABLE'] = '1'

import fastdfs
from pathlib import Path
import sys
import warnings
from collections import Counter
from sklearn.metrics import roc_auc_score, mean_absolute_error
#from autogluon.features.generators import AutoMLPipelineFeatureGenerator
#from autogluon.tabular import TabularPredictor
from tabpfn import TabPFNClassifier, TabPFNRegressor
from loguru import logger

from yaml_utils import process_yaml, load_yaml
from fastdfs.transform import RDBTransformWrapper, RDBTransformPipeline, HandleDummyTable, FeaturizeDatetime

# Suppress verbose logging from fastdfs - only show critical errors
logger.remove()
logger.add(sys.stderr, level="ERROR")

# Suppress tqdm progress bars by overriding the class
# class SilentContext:
#     """Context manager to suppress tqdm and other output"""
#     def __enter__(self):
#         return self
#     def __exit__(self, *args):
#         pass

# Monkey patch tqdm to be silent
import tqdm
# original_tqdm_init = tqdm.tqdm.__init__
# def silent_init(self, *args, **kwargs):
#     kwargs['disable'] = True
#     return original_tqdm_init(self, *args, **kwargs)
#tqdm.tqdm.__init__ = silent_init

# warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`")
# warnings.filterwarnings(
#     "ignore",
#     message="The provided callable .* is currently using .*",
#     category=FutureWarning,
#     module="featuretools.computational_backends.feature_set_calculator"
# )

def main(rdb_path, dfs_max_depth=0, target_tasks=None, target_task_type=None, n_ensemble=8, subsample_samples=10000, cyclic=False):
    """
    If target_tasks is specified, only the provided task will be evaluated
    """
    print("=" * 60)
    print("Training TabPFN-v2 on DFS features")
    print("=" * 60)

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
        target_column = task_data.get('target_column')
        cutoff_time_column = task_data.get('time_column')
        task_type = task_data.get('task_type')
        task_name = task_data.get('task_name')

        if (target_tasks is not None and task_name in target_tasks) or (target_tasks is None and task_type == target_task_type):
            print(f"Processing task: {task_name}")
            print(f"key_mappings: {key_mappings}")
            print(f"target_column: {target_column}")
            print(f"cutoff_time_column: {cutoff_time_column}")
            print(f"task_type: {task_type}")
            print(f"task_name: {task_name}")
            print()

            # input(f"Press Enter to continue for task: {task_name}...")
            task_folder = Path(rdb_path) / task_name
            train_table_path = task_folder / "train.pqt"
            test_table_path = task_folder / "test.pqt"

            # Step 2: Load target table
            print("Step 2: Loading target table...")
            try:
                train_df = pd.read_parquet(train_table_path)
                print(f"Loaded train_df with {len(train_df)} datapoints")
                test_df = pd.read_parquet(test_table_path)
                print(f"Loaded test_df with {len(test_df)} datapoints")

                # downsample train_df for faster DFS
                SUBSAMPLE_SAMPLES_BEFORE_DFS = 50000
                if len(train_df) > SUBSAMPLE_SAMPLES_BEFORE_DFS:
                    print(f"Sampling train_df to {SUBSAMPLE_SAMPLES_BEFORE_DFS} datapoints")
                    train_df = train_df.sample(n=SUBSAMPLE_SAMPLES_BEFORE_DFS, random_state=42).reset_index(drop=True)

            except Exception as e:
                print(f"❌ Failed to load target table: {e}")
                return
            
            # print(f"train_df.head(3):\n{train_df.head(3)}")
            # raise Exception("Stop here")

            X_train, Y_train = train_df.drop(columns=[target_column]), train_df[target_column]
            X_test, Y_test = test_df.drop(columns=[target_column]), test_df[target_column]

            # Step 4: Test basic DFS without transforms
            print("Step 4: DFS feature computation...")

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

            # Create pipeline
            pipeline = fastdfs.DFSPipeline(
                # transform_pipeline=None,  # No transforms for this test
                transform_pipeline=RDBTransformPipeline([
                    HandleDummyTable(),
                    #RDBTransformWrapper(FeaturizeDatetime(features=["year", "month", "day", "hour", "dayofweek"], cyclic=cyclic)),
                    RDBTransformWrapper(FeaturizeDatetime(features=["year", "month", "day", "hour", "dayofweek"])), #, cyclic=cyclic)),
                ]),
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
            dfs_train_features = dfs_train_features.loc[:, ~dfs_train_features.columns.duplicated()]
            dfs_test_features = dfs_test_features.loc[:, ~dfs_test_features.columns.duplicated()]

            dfs_train_features = dfs_train_features.replace({pd.NA: None})
            dfs_test_features = dfs_test_features.replace({pd.NA: None})

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
                    elif train_features[col].dtype == 'datetime64[ns]':
                        col_to_drop_list.append(col)
                        print(f"Dropping column {col} because it is a datetime column")
                train_features = train_features.drop(columns=col_to_drop_list)
                test_features = test_features.drop(columns=col_to_drop_list)

            # Print column names and their dtypes in train_features
            # print("\nColumn names and dtypes in train_features:")
            # for col in train_features.columns:
            #     print(f" - {col}: {train_features[col].dtype}")

            # Combine train_features and test_features for preprocessing
            combined_features = pd.concat([train_features, test_features], axis=0, ignore_index=True)
            
            # Convert categorical columns to integer codes using sklearn's LabelEncoder
            from sklearn.preprocessing import LabelEncoder

            for col in combined_features.columns:
                # Detect categorical columns: 'object', 'category', or pandas string
                if combined_features[col].dtype == 'object' or str(combined_features[col].dtype) == 'category' or pd.api.types.is_string_dtype(combined_features[col]):
                    le = LabelEncoder()
                    try:
                        # Fill NA because LabelEncoder does not accept NAs; use string '<NA>'
                        combined_features[col] = le.fit_transform(combined_features[col].astype(str).fillna('<NA>'))
                    except Exception as e:
                        print(f"Could not label encode column '{col}': {e}")
        
            # TODO@remove all columns whose type is not supported by TabPFN
            # Delete the 'timestamp' column in combined_features if it exists
            if 'timestamp' in combined_features.columns:
                combined_features = combined_features.drop(columns=['timestamp'])
                print("Dropped 'timestamp' column from combined_features. TabPFN doesn't support timestamp column.")
            else:
                print("'timestamp' column not present in combined_features (nothing to drop).")
            # Verify that 'timestamp' has been removed
            if 'timestamp' in combined_features.columns:
                print("⚠️ Warning: 'timestamp' column still present after attempted drop!")
            else:
                print("✅ 'timestamp' column successfully removed from combined_features.")

            # Split back into train_features and test_features with the original row counts
            train_len = len(train_features)
            test_len = len(test_features)
            train_features = combined_features.iloc[:train_len, :].reset_index(drop=True)
            test_features = combined_features.iloc[train_len:, :].reset_index(drop=True)
            # print("\n\n\n")
            # for col in train_features:
            #     print(col, train_features[col].dtype)
            #     print(train_features[col].head(3))

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

            # final_train_df = pd.concat([train_features, Y_train], axis=1)
            # for col in final_train_df.columns:
            #     print(col, final_train_df[col].dtype, final_train_df[col].shape)
            #     print(final_train_df[col].head(3))
            #     print(f"{type(final_train_df[col].iloc[0])=}")

            # feature_generator = AutoMLPipelineFeatureGenerator(
            #     enable_datetime_features=True,
            #     enable_raw_text_features=False,
            #     enable_text_special_features=False,
            #     enable_text_ngram_features=False, 
            # )

            params = { 
                "random_state": 42,
                "n_preprocessing_jobs": -1,
                "n_estimators": n_ensemble,
                "inference_config": {
                    "SUBSAMPLE_SAMPLES": subsample_samples,
                },
                "ignore_pretraining_limits": True,
                "model_path":"/root/.cache/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
            }
            predictor = TabPFNClassifier(**params)

            predictor.fit(train_features, Y_train)

            # Predict in batches of 5000 and concatenate
            batch_size = 5000
            proba_batches = []
            num_rows = len(test_features)

            for start in tqdm.tqdm(range(0, num_rows, batch_size), desc="Predicting batches"):
                end = min(start + batch_size, num_rows)
                batch = test_features.iloc[start:end]
                if target_task_type == "classification":
                    proba_batch = predictor.predict_proba(batch)
                else:
                    proba_batch = predictor.predict(batch)
                proba_batches.append(proba_batch)

            proba = np.concatenate(proba_batches, axis=0)

            print(proba.shape)
            # print(X_trans.columns[:20])
            if target_task_type == "classification":
                if proba.shape[1] == 2:
                    print(f"{task_name}: Binary classification")
                    roc_auc = roc_auc_score(Y_test, proba[:, 1])
                else:
                    print(f"{task_name}: Multi-class classification")
                    roc_auc = roc_auc_score(Y_test, proba, multi_class='ovr')
                print(f"roc_auc: {roc_auc*100:.2f}")
            else:
                print(f"{task_name}: Regression")
                
                mae = mean_absolute_error(Y_test, proba)
                print(f"mae: {mae:.2f}")
            
            print("\n\n\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run TabPFN with DFS on a relational database")

    parser.add_argument('--rdb_path', type=str, required=True,
                        help="Path to the relational database (e.g., /root/linjie/rel-data/rel-stack)")
    parser.add_argument('--n_ensemble', type=int, default=8,
                        help="Number of estimators/ensemble models for TabPFN (default: 8)")
    parser.add_argument('--subsample_samples', type=int, default=10000,
                        help="Subsample size for TabPFN inference (default: 10000)")
    parser.add_argument('--dfs_depth', type=int, default=3,
                        help="DFS max depth (default: 3)")
    parser.add_argument('--target_task_type', choices=['classification', 'regression'], default='classification',
                        help="Target task type (classification or regression, default: classification)")
    parser.add_argument('--cyclic', action='store_true',
                        help="Whether to use cyclic datetime features (default: False)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    """
    /root/linjie/rel-data/rel-stack
    /root/linjie/rel-data/rel-trial
    /root/linjie/rel-data/rel-event
    /root/linjie/rel-data/rel-f1

    /root/linjie/rel-data/rel-avito
    /root/linjie/rel-data/rel-amazon
    /root/linjie/rel-data/rel-hm
    /root/linjie/rel-data/rel-stack
    """
    
    # If you want to print the configuration for debugging:
    print(f"Running with arguments: {vars(args)}")

    # Pass arguments into your main function and wherever else necessary.
    # Make sure your main and inner logic uses these variables instead of hardcoded values.
    main(
        rdb_path=args.rdb_path,
        dfs_max_depth=args.dfs_depth,
        target_task_type=args.target_task_type,
        n_ensemble=args.n_ensemble,
        subsample_samples=args.subsample_samples,
        cyclic=args.cyclic
    )
