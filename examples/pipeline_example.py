"""Simple example showing how to run MultiTabFM with CustomLimiX."""

import json
from pathlib import Path
from multitabfm.api import train_and_predict
from multitabfm.model import CustomLimiX, CustomTabPFN,CustomMLP

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def main():
    """Train + predict with LimiX using the convenience API."""
    dfs_config = load_config("/root/yl_project/multitabfm/config/dfs/dfs_default.json")
    model_config = load_config("/root/yl_project/multitabfm/config/model/tabpfn_v2_classification_default.json")
    # model_config = load_config("/root/yl_project/multitabfm/config/model/mlp_default.json")
    
    # Map model_name to class
    model_map = {
        "tabpfn": CustomTabPFN,
        "limix": CustomLimiX,
        "mlp": CustomMLP
    }
    
    model_name = model_config["model_name"]
    if model_name in model_map:
        model_config["custom_model_class"] = model_map[model_name]
        del model_config["model_name"]
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Supported: {list(model_map.keys())}")

    preds, metrics = train_and_predict(
        rdb_data_path="/root/autodl-tmp/4dbinfer/stackexchange",
        task_data_path="/root/autodl-tmp/4dbinfer/stackexchange/upvote",
        dfs_config=dfs_config,
        model_config=model_config,
    )
    print(f"Results: {metrics}")


if __name__ == "__main__":
    main()
