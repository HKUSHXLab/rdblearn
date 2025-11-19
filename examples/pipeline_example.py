"""Simple example using the updated multitabfm API.

This shows how easy it is to run experiments with just data paths.
"""
from multitabfm.api import train_and_predict

def main():
    """Using the convenience function."""
    dfs_config = {
        "max_depth": 2,
        "agg_primitives":["max", "min", "mean", "count", "mode", "std"],
        "engine": "dfs2sql"
    }
    model_storage_path = "/root/autodl-tmp/autogluon_models"  # Keep AutoGluon artifacts off the project disk
    model_config = {
            "predictor_kwargs": {
                "path": model_storage_path,
            },
            "hyperparameters": {
                "TABPFNV2": {
                    "n_estimators": 8,
                }
            },
        }
    preds, metrics = train_and_predict(
        rdb_data_path="/root/autodl-tmp/tabpfn_data/rel-amazon",
        task_data_path="/root/autodl-tmp/tabpfn_data/rel-amazon/item-ltv",
        enable_dfs=True,
        dfs_config=dfs_config,
        model_config=model_config,
        eval_metrics=["mae"],
        batch_size=5000  # Smaller batch size to avoid CUDA memory issues
    )
    
    print(f"Results: {metrics}")


if __name__ == "__main__":
    main()
