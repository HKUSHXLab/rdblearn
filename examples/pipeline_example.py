"""Simple example using the updated multitabfm API.

This shows how easy it is to run experiments with just data paths.
"""
from multitabfm.api import train_and_predict

def main():
    """Using the convenience function."""
    dfs_config = {
        "max_depth": 3,
        "agg_primitives":["max", "min", "mean", "count", "mode", "std"],
        "engine": "dfs2sql"
    }
    model_config = {
            "hyperparameters": {
                "TABPFNV2": {
                    "n_estimators": 2,
                }
            },
        }
    preds, metrics = train_and_predict(
        rdb_data_path="/root/yanlin/tabular-chat-predictor/data/rel-amazon",
        task_data_path="/root/yanlin/tabular-chat-predictor/data/rel-amazon/user-churn",
        enable_dfs=True,
        dfs_config=dfs_config,
        model_config=model_config,
        eval_metrics=["auroc"]
    )
    
    print(f"Results: {metrics}")


if __name__ == "__main__":
    main()
