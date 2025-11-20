"""Simple example using the updated multitabfm API with CustomTabPFN.

This shows how easy it is to run experiments with just data paths using a custom model.
"""
from multitabfm.api import train_and_predict
from multitabfm.model import CustomTabPFN

def main():
    """Using the convenience function."""
    dfs_config = {
        "max_depth": 3,
        "agg_primitives":["max", "min", "mean", "count", "mode", "std"],
        "engine": "dfs2sql"
    }
    model_storage_path = "/root/autodl-tmp/autogluon_models"  # Keep AutoGluon artifacts off the project disk
    # Configuration for CustomTabPFN
    model_config = {
        "model_path": "/root/autodl-tmp/tabpfn_2_5",  # Use TabPFN v2.5
        "task_type": "regression"  # Will be automatically inferred if not specified
    }
    preds, metrics = train_and_predict(
        rdb_data_path="/root/autodl-tmp/tabpfn_data/rel-event",
        task_data_path="/root/autodl-tmp/tabpfn_data/rel-event/user-attendance",
        enable_dfs=False,
        dfs_config=dfs_config,
        model_config=model_config,
        eval_metrics=["mae"],
        batch_size=5000,  # Smaller batch size to avoid CUDA memory issues
        custom_model_class=CustomTabPFN  # Use the custom TabPFN model
    )
    
    print(f"Results: {metrics}")


if __name__ == "__main__":
    main()
