"""Example of loading a pre-trained model for prediction.

This shows how to load a saved AutoGluon TabularPredictor and use it for prediction
on new data, after applying the same feature engineering steps.
"""
from autogluon.tabular import TabularPredictor
from multitabfm.utils import load_dataset
from multitabfm.feature_engineer import generate_features
from sklearn.metrics import roc_auc_score
import pandas as pd

def main():
    """Load a model and predict."""
    # --- Configuration ---
    # Paths to your data
    rdb_data_path = "/root/yanlin/tabular-chat-predictor/data/rel-amazon"
    task_data_path = "/root/yanlin/tabular-chat-predictor/data/rel-amazon/user-churn"
    
    # This is the model path from your traceback.
    # You might need to update it if you have a different model saved from another run.
    saved_model_path = "/root/yanlin/multitabfm/AutogluonModels/ag-20251103_094324"
    
    # DFS configuration should match what was used for training
    dfs_config = {
        "max_depth": 3,
        "agg_primitives":["max", "min", "mean", "count", "mode", "std"],
        "engine": "dfs2sql"
    }

    # 1. Load data to get the test set and necessary metadata
    print("Loading dataset...")
    # We only need test_data, metadata, and rdb for prediction
    _, test_data, metadata, rdb = load_dataset(rdb_data_path, task_data_path)
    
    # 2. Parse metadata to get column names
    key_mappings = {k: v for d in metadata['key_mappings'] for k, v in d.items()}
    time_column = metadata['time_column']
    target_column = metadata['target_column']

    # 3. Prepare test features
    # The feature generation process must be the same as used for training
    print("Generating test features...")
    X_test, Y_test = test_data.drop(columns=[target_column]), test_data[target_column]
    test_features = generate_features(X_test, rdb, key_mappings, time_column, dfs_config)

    # 4. Load the pre-trained AutoGluon model
    print(f"Loading model from: {saved_model_path}")
    try:
        predictor = TabularPredictor.load(saved_model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the path to the saved model is correct and the model files are not corrupted.")
        return

    # --- FOR DEBUGGING: Use a small subset of the data ---
    print("DEBUG: Using a small subset of test features for prediction (10 rows).")
    # test_features_subset = test_features.head(10)

    # 5. Make predictions on the test set
    print("Making predictions...")
    # Predict in batches of 5000 and concatenate

    batch_size = 5000
    proba_batches = []
    num_rows = len(test_features)
    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)
        batch = test_features.iloc[start:end]
        proba_batch = predictor.predict_proba(batch)
        proba_batches.append(proba_batch)
    # Concatenate all batches
    predictions = pd.concat(proba_batches, axis=0).reset_index(drop=True)
    
    print("\nPredictions (probabilities):")
    print(predictions.head())

    # 6. (Optional) Evaluate predictions against true labels
    auroc = roc_auc_score(Y_test, predictions)
    print(f"\nAUROC on test set: {auroc:.4f}")


if __name__ == "__main__":
    main()
