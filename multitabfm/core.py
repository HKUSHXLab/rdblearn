import os
import tempfile
from typing import Optional, List, Tuple, Union
import pandas as pd
import numpy as np
import fastdfs
from datetime import datetime
from fastdfs.transform import RDBTransformWrapper, RDBTransformPipeline, HandleDummyTable, FeaturizeDatetime, FillMissingPrimaryKey, ResolveSelfLoops
from fastdfs.dfs import DFSConfig
import logging

from .feature_engineer import generate_features, prepare_for_dfs, add_dfs_features, ag_transform, ag_label_transform
from .model import CustomTabPFN
from .evaluation import compute_metrics
from .utils import load_dataset

logger = logging.getLogger(__name__)

class MultiTabFM:
    
    def __init__(self, dfs_config: Optional[dict] = None, model_config: Optional[dict] = None, batch_size: int = 5000, max_samples: int = 10000, stratified_sampling: bool = False):
        self.model_config = (model_config or {}).copy()
        self.dfs_config = dfs_config or {}
        self.stratified_sampling = stratified_sampling
        
        # Extract batch_size from model_config if present, otherwise use argument
        if 'batch_size' in self.model_config:
            self.batch_size = self.model_config.pop('batch_size')
        else:
            self.batch_size = batch_size
        if "max_samples" in self.model_config:
            self.max_samples = self.model_config.pop("max_samples")
        else:
            self.max_samples = max_samples
            
        self.custom_model_class = self.model_config.pop('custom_model_class', None)
        self.model = None

    def _batch_predict_proba(self, X: pd.DataFrame, batch_size: int) -> pd.DataFrame:
        """Batch-wise probability prediction. Returns DataFrame."""
        proba_batches = []
        num_rows = len(X)
        
        for start in range(0, num_rows, batch_size):
            end = min(start + batch_size, num_rows)
            batch = X.iloc[start:end]
            proba_batch = self.model.predict_proba(batch)
            proba_batches.append(proba_batch)
        
        # Concatenate all batches
        if len(proba_batches) == 1:
            return proba_batches[0]
        else:
            return pd.concat(proba_batches, axis=0).reset_index(drop=True)

    def _batch_predict(self, X: pd.DataFrame, batch_size: int) -> pd.Series:
        """Batch-wise value prediction. Returns Series."""
        pred_batches = []
        num_rows = len(X)
        
        for start in range(0, num_rows, batch_size):
            end = min(start + batch_size, num_rows)
            batch = X.iloc[start:end]
            pred_batch = self.model.predict(batch)
            pred_batches.append(pred_batch)
        
        # Concatenate all batches
        if len(pred_batches) == 1:
            return pred_batches[0]
        else:
            return pd.concat(pred_batches, axis=0).reset_index(drop=True)

    def fit(self, train_features: pd.DataFrame, label_column: str, task_type: Optional[str] = None, eval_metric: Optional[str] = None) -> None:
        """Fit the model on feature-augmented training data."""
        # Separate features and labels
        features = train_features.drop(columns=[label_column])
        labels = train_features[label_column]
        
        # Add task_type to model config
        model_config = self.model_config.copy()
        model_config['task_type'] = task_type
        model_config['eval_metric'] = eval_metric
        
        # Use custom model class if provided, otherwise default to CustomTabPFN
        model_class = self.custom_model_class or CustomTabPFN
        
        # Initialize and fit the model
        self.model = model_class(**model_config)
        self.model.fit(X=features, y=labels)

    def predict_proba(self, test_features: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities using batch processing."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._batch_predict_proba(test_features, self.batch_size)

    def predict(self, test_features: pd.DataFrame) -> pd.Series:
        """Predict target values using batch processing."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._batch_predict(test_features, self.batch_size)

    def evaluate(self, labels: pd.Series, preds_or_proba: Union[pd.DataFrame, np.ndarray, pd.Series], metrics: Optional[List[str]] = None) -> dict:
        """Evaluate predictions against true labels for classification or regression."""
        return compute_metrics(labels, preds_or_proba, metrics)

    def train_and_predict(self,
                         rdb_data_path: str,
                         task_data_path: str) -> Tuple[Union[pd.DataFrame, np.ndarray, pd.Series], Optional[dict], dict]:
        """Main API: End-to-end training and prediction from data paths.
        
        Args:
            rdb_data_path: Path to RDB data directory (e.g., "data/rel-event")
            task_data_path: Path to task data directory (e.g., "data/rel-event/user-ignore")
            
        Returns:
            Tuple of (predictions, metrics, timing_info)
            - predictions: Model predictions
            - metrics: Evaluation metrics (if available)
            - timing_info: Dict with 'fit_seconds' and 'predict_seconds'
        """
        with tempfile.NamedTemporaryFile(suffix=".db") as temp_db_file:
            self.dfs_config['engine_path'] = temp_db_file.name
            
            return self._train_and_predict_internal(
                rdb_data_path,
                task_data_path
            )
        
    def _train_and_predict_internal(self,
                         rdb_data_path: str,
                         task_data_path: str) -> Tuple[Union[pd.DataFrame, np.ndarray, pd.Series], Optional[dict], dict]:
        """Internal method for training and prediction with timing.
        """
        
        # Load dataset
        train_data, test_data, metadata, rdb = load_dataset(rdb_data_path, task_data_path)
        load_dataset_end = datetime.now()
        
        # Parse metadata
        key_mappings = {k: v for d in metadata['key_mappings'] for k, v in d.items()}
        time_column = metadata['time_column']
        target_column = metadata['target_column']
        task_type = metadata.get('task_type')
        metric = metadata.get('evaluation_metric')
        eval_metrics = [metric] if metric else None

        # Sampling target_table if max_samples is set
        # NOTE: without this subsampling, DFS on training set will be slow and TabPFN will also be slow
        # even if we turn on subsampling in TabPFN itself.
        if len(train_data) > self.max_samples:
            train_data = self._downsample_training_set(
                train_data, 
                target_column, 
                task_type, 
                self.max_samples, 
                self.stratified_sampling
            )

        # Prepare target dataframes
        X_train, Y_train = train_data.drop(columns=[target_column]), train_data[target_column]
        X_test, Y_test = test_data.drop(columns=[target_column]), test_data[target_column]

        # Configure DFS pipeline
        effective_config = DFSConfig()
        if self.dfs_config:
            for key, value in self.dfs_config.items():
                if hasattr(effective_config, key):
                    setattr(effective_config, key, value)

        # Apply DFS feature generation if enabled
        if self.dfs_config:
            pipeline = fastdfs.DFSPipeline(
                transform_pipeline=RDBTransformPipeline([
                    HandleDummyTable(),
                    FillMissingPrimaryKey(),
                    ResolveSelfLoops(),
                    RDBTransformWrapper(FeaturizeDatetime(features=["year", "month", "day", "hour", "dayofweek"])),
                ]),
                dfs_config=effective_config
            )
            # Prepare clean data for DFS (removes array-valued columns)
            train_for_dfs, test_for_dfs = prepare_for_dfs(X_train, X_test, key_mappings, time_column)
            
            # Generate DFS features
            train_dfs = generate_features(train_for_dfs, rdb, key_mappings, time_column, pipeline)
            test_dfs = generate_features(test_for_dfs, rdb, key_mappings, time_column, pipeline)
            
            # Add DFS features to original data
            train_features = add_dfs_features(X_train, train_dfs)
            test_features = add_dfs_features(X_test, test_dfs)
        else:
            train_features = X_train
            test_features = X_test
        
        # Apply ag_transform for feature engineering (categorical conversion, datetime, etc.)
        X_train_transformed, X_test_transformed = ag_transform(
            train_features.reset_index(drop=True),
            test_features.reset_index(drop=True)
        )
        
        # Apply label transformation to both train and test labels
        # Right now if "normalization_regression=True", this includes label normalization for 4DBInfer. Need to disable normalization for RelBench.
        print(f"Y_test NaNs before transform: {Y_test.isna().sum()}")
        
        # If task_type is explicitly regression, pass it to override auto-inference
        # which might incorrectly infer multiclass for integer regression targets
        ag_problem_type = "regression" if task_type == "regression" else None
        
        y_train_transformed, y_test_transformed = ag_label_transform(
            Y_train.reset_index(drop=True),
            Y_test.reset_index(drop=True),
            normalize_regression = False,
            problem_type=ag_problem_type
        )
        print(f"y_test_transformed NaNs after transform: {y_test_transformed.isna().sum()}")

        # Step 3: Combine features and target
        train_data = pd.concat([X_train_transformed, y_train_transformed], axis=1)

        # 2. Train model (data already normalized) - TIMED
        fit_start = datetime.now()
        dfs_seconds = (fit_start - load_dataset_end).total_seconds()
        
        self.fit(train_data, label_column=target_column, task_type=task_type,
                eval_metric=eval_metrics[0] if eval_metrics else None)
        fit_end = datetime.now()
        fit_seconds = (fit_end - fit_start).total_seconds()

        # 3. Predict (predictions will be in normalized space) - TIMED
        predict_start = datetime.now()
        if task_type == "regression":
            preds_or_proba = self.predict(X_test_transformed)
        else:
            preds_or_proba = self.predict_proba(X_test_transformed)
        
        if isinstance(preds_or_proba, pd.DataFrame):
             print(f"preds_or_proba NaNs: {preds_or_proba.isna().sum().sum()}")
        else:
             print(f"preds_or_proba NaNs: {pd.Series(preds_or_proba).isna().sum()}")

        predict_end = datetime.now()
        predict_seconds = (predict_end - predict_start).total_seconds()
        
        # Calculate time to first batch prediction
        import math
        num_batches = math.ceil(len(X_test_transformed) / self.batch_size)
        time_to_first_prediction = predict_seconds / num_batches if num_batches > 0 else 0

        # 4. Evaluate on normalized scale (following tab2graph's approach)
        metrics = None
        if eval_metrics and y_test_transformed is not None:
            print("\nEvaluating on normalized scale (tab2graph approach)...")
            metrics = self.evaluate(y_test_transformed, preds_or_proba, metrics=eval_metrics)
            print(f"Metrics (on normalized data): {metrics}")

        # Prepare timing information
        timing_info = {
            'fit_seconds': fit_seconds,
            'predict_seconds': predict_seconds,
            'total_fit_predict_seconds': fit_seconds + predict_seconds,
            'dfs_seconds': dfs_seconds,
            'time_to_first_prediction': time_to_first_prediction,
            'num_batches': num_batches
        }
        
        print(f"\nTiming: fit={fit_seconds:.2f}s, predict={predict_seconds:.2f}s, total={timing_info['total_fit_predict_seconds']:.2f}s")
        print(f"DFS time: {dfs_seconds:.2f}s, Time to first prediction: {time_to_first_prediction:.4f}s (batches: {num_batches})")

        return preds_or_proba, metrics, timing_info
    
    def _downsample_training_set(
        self,
        data: pd.DataFrame,
        target_column: str,
        task_type: str,
        max_samples: int,
        stratified_sampling: bool = False
    ) -> pd.DataFrame:
        """Downsample data to max_samples while preserving class balance for classification tasks.

        Args:
            data: Training DataFrame containing features and target
            target_column: Name of the target column
            task_type: Type of task ("regression" or "classification")
            max_samples: Maximum number of samples to retain
            stratified_sampling: If True, maintain class balance. If False, sample randomly
                                 but ensure at least one sample per class for classification tasks.

        Returns:
            Downsampled DataFrame
        """
        if len(data) <= max_samples:
            return data

        logger.info(f"Downsampling training set from {len(data)} to {max_samples} samples.")
        
        X = data.drop(columns=[target_column])
        y = data[target_column].values

        if task_type == "regression":
            # For regression tasks, we can just sample randomly
            idx = np.random.choice(len(X), max_samples, replace=False)
            return data.iloc[idx].reset_index(drop=True)

        # For classification tasks, ensure at least one sample per class regardless of stratified_sampling
        if not stratified_sampling:
            # Even with random sampling, ensure at least one sample per class
            unique_labels = np.unique(y)
            
            # First, pick one sample from each class
            selected_indices = []
            for label in unique_labels:
                class_indices = np.where(y == label)[0]
                if len(class_indices) > 0:
                    selected_idx = np.random.choice(class_indices, 1)[0]
                    selected_indices.append(selected_idx)

            # Then randomly sample the rest
            remaining_samples = max_samples - len(selected_indices)
            if remaining_samples > 0:
                # Create mask to exclude already selected indices
                mask = np.ones(len(X), dtype=bool)
                mask[selected_indices] = False
                eligible_indices = np.where(mask)[0]

                if len(eligible_indices) > 0:
                    additional_indices = np.random.choice(
                        eligible_indices,
                        min(remaining_samples, len(eligible_indices)),
                        replace=False
                    )
                    selected_indices.extend(additional_indices)

            # Shuffle to avoid having all samples of one class together
            np.random.shuffle(selected_indices)
            idx = np.array(selected_indices)

            # Log the class distribution in the sample
            sampled_dist = np.unique(y[idx], return_counts=True)
            logger.info(f"Random sample class distribution: {dict(zip(*sampled_dist))}")

            return data.iloc[idx].reset_index(drop=True)

        else:
            # Get the unique labels and their counts
            unique_labels, label_counts = np.unique(y, return_counts=True)
            logger.info(f"Original label distribution: {dict(zip(unique_labels, label_counts))}")

            # Calculate samples per class, ensuring all classes have at least one sample
            n_classes = len(unique_labels)
            # Determine samples per class with a minimum of 1
            samples_per_class = max(1, max_samples // n_classes)

            # Sample from each class
            balanced_indices = []
            remaining_indices = []

            for label in unique_labels:
                class_indices = np.where(y == label)[0]
                if len(class_indices) == 0:
                    continue

                # If we have fewer samples than samples_per_class, take all of them (no replacement)
                if len(class_indices) <= samples_per_class:
                    balanced_indices.extend(class_indices)
                else:
                    # Sample without replacement
                    sampled_indices = np.random.choice(class_indices, samples_per_class, replace=False)
                    balanced_indices.extend(sampled_indices)

                    # Store unused indices for potential additional sampling
                    mask = np.ones(len(class_indices), dtype=bool)
                    mask[np.isin(class_indices, sampled_indices)] = False
                    remaining_indices.extend(class_indices[mask])

            # If we haven't reached max_samples, sample from remaining indices
            samples_needed = max_samples - len(balanced_indices)
            if samples_needed > 0 and len(remaining_indices) > 0:
                # Sample without replacement if possible, otherwise with replacement
                additional_samples = np.random.choice(
                    remaining_indices,
                    min(samples_needed, len(remaining_indices)),
                    replace=False
                )
                balanced_indices.extend(additional_samples)
                logger.info(f"Added {len(additional_samples)} additional samples to balance the dataset")

            # Shuffle the indices to avoid having all samples of one class together
            np.random.shuffle(balanced_indices)

            # Limit to max_samples in case we somehow exceeded it
            balanced_indices = balanced_indices[:max_samples]
            idx = np.array(balanced_indices)

            # Log the new distribution
            new_label_counts = np.unique(y[idx], return_counts=True)
            logger.info(f"Balanced label distribution: {dict(zip(*new_label_counts))}")

            return data.iloc[idx].reset_index(drop=True)
