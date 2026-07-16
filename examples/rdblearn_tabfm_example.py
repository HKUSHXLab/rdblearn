import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from rdblearn.datasets import RDBDataset
from rdblearn.estimator import RDBLearnClassifier
from rdblearn.utils import TabFMWrapperClassifier

from loguru import logger
logger.enable("rdblearn")

# --- TabFM Setup ---
# TabFM (google/tabfm-1.0.0-pytorch) is not a hard dependency of rdblearn.
# Install it alongside the pinned torch build with:
#   pip install "jaxtyping<0.3" "typeguard<3" absl-py safetensors
#   pip install tabfm --no-deps
#
# Weights: either let the wrapper download from Hugging Face (checkpoint_path=None),
# or pre-download the repo once and point TABFM_CHECKPOINT at its root:
#   from huggingface_hub import snapshot_download
#   snapshot_download("google/tabfm-1.0.0-pytorch", local_dir="/path/to/tabfm-1.0.0-pytorch")
# The wrapper auto-converts model.safetensors -> pytorch_model.bin on first use
# (the tabfm==1.0.0 wheel expects the .bin file).
TABFM_CHECKPOINT = os.environ.get("TABFM_CHECKPOINT")  # None -> download from HF


def main():
    # 1. Load Dataset
    print("Loading 'rel-f1' dataset...")
    dataset = RDBDataset.from_relbench("rel-f1")

    task_name = "driver-dnf"
    if task_name not in dataset.tasks:
        raise ValueError(f"Task '{task_name}' not found. Available: {list(dataset.tasks.keys())}")

    task = dataset.tasks[task_name]
    print(f"Loaded task: {task.name}")
    print(f"Train shape: {task.train_df.shape}")
    print(f"Test shape: {task.test_df.shape}")

    # 2. Check for GPU (TabFM is a 1.6B-parameter model; CUDA strongly recommended)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 3. Build the TabFM base estimator
    #    - chunk_size="auto" keeps wide DFS feature sets within 32 GB VRAM;
    #    - dtype="bfloat16" halves activation memory for very wide/deep feature sets.
    base_estimator = TabFMWrapperClassifier(
        checkpoint_path=TABFM_CHECKPOINT,
        device=device,
        n_estimators=8,
        chunk_size="auto",
    )

    # 4. Wrap with RDBLearn (DFS feature synthesis + preprocessing)
    clf = RDBLearnClassifier(
        base_estimator=base_estimator,
        config={
            "dfs": {"max_depth": 2},
            "max_train_samples": 10000,
        },
    )

    # 5. Fit on relational data
    target_col = task.metadata.target_col
    X_train = task.train_df.drop(columns=[target_col])
    y_train = task.train_df[target_col]

    clf.fit(
        X=X_train,
        y=y_train,
        rdb=dataset.rdb,
        key_mappings=task.metadata.key_mappings,
        cutoff_time_column=task.metadata.time_col,
    )

    # 6. Evaluate
    X_test = task.test_df.drop(columns=[target_col])
    y_test = task.test_df[target_col]
    proba = clf.predict_proba(X=X_test)
    auc = roc_auc_score(np.asarray(y_test).astype(int), np.asarray(proba)[:, 1])
    print(f"Test ROC AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
