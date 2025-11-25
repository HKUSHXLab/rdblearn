"""Simple example showing how to run MultiTabFM with CustomLimiX."""

from multitabfm.api import train_and_predict
from multitabfm.model import CustomLimiX,CustomTabPFN

def main():
    """Train + predict with LimiX using the convenience API."""
    dfs_config = {
        "max_depth": 3,
        "agg_primitives":["max", "min", "mean", "count", "mode","std"],
        "engine": "dfs2sql"
    }
    # model_storage_path = "/root/autodl-tmp/autogluon_models"  # Keep AutoGluon artifacts off the project disk
    # Configuration for CustomTabPFN
    model_config = {
        # "model_path": "/root/autodl-tmp/tabpfn_2_5/",  # Use TabPFN v2.5
        "model_path": "/root/autodl-tmp/limix/cache/LimiX-16M.ckpt",
        "task_type": "regression",  # Will be automatically inferred if not specified
        # "max_samples": 1000  # Randomly sample up to 50K samples for training
    }
    preds, metrics = train_and_predict(
        rdb_data_path="/root/autodl-tmp/tabpfn_data/rel-f1-post-dfs-3",
        # rdb_data_path="/root/autodl-tmp/tabpfn_data/rel-event",
        task_data_path="/root/autodl-tmp/tabpfn_data/rel-f1-post-dfs-3/driver-dnf",
        # task_data_path="/root/autodl-tmp/tabpfn_data/rel-event/user-ignore",
        enable_dfs=False,
        dfs_config=dfs_config,
        model_config=model_config,
        eval_metrics=["mae"],
        batch_size=2000,  # Reduced batch size for datasets with few features
        custom_model_class=CustomLimiX,
    )
    # Configuration for CustomLimiX
    # model_config = {
    #     "model_path": "/root/autodl-tmp/limix/cache/LimiX-16M.ckpt",
    #     "task_type": "classification",
        # All other CustomLimiX kwargs (e.g., inference_config) can also be passed if needed.
    # }
    print(f"Results: {metrics}")


if __name__ == "__main__":
    main()
