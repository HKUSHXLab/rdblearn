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
    model_config = {
        "model_path": "/root/autodl-tmp/limix/cache/LimiX-16M.ckpt",
        "max_samples": 1000,  # Randomly sample up to 1000 samples for training
        "batch_size": 2000,  # Reduced batch size for datasets with few features
        "custom_model_class": CustomLimiX,
    }
    preds, metrics = train_and_predict(
        rdb_data_path="/root/autodl-tmp/tabpfn_data/rel-f1-post-dfs-3",
        task_data_path="/root/autodl-tmp/tabpfn_data/rel-f1-post-dfs-3/driver-dnf",
        # dfs_config=dfs_config,
        model_config=model_config,
    )
    print(f"Results: {metrics}")


if __name__ == "__main__":
    main()
