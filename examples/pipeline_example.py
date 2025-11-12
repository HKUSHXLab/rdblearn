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
                    "n_estimators": 8,
                }
            },
        }
    preds, metrics = train_and_predict(
        rdb_data_path="/root/yanlin/tabular-chat-predictor/data/rel-f1-post-dfs-3",
        task_data_path="/root/yanlin/tabular-chat-predictor/data/rel-f1-post-dfs-3/driver-position",
        enable_dfs=False,
        dfs_config=dfs_config,
        model_config=model_config,
        eval_metrics=["mae"]
    )
    
    print(f"Results: {metrics}")


if __name__ == "__main__":
    main()
