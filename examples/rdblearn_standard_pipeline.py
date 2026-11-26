"""Standard RDBLearn pipeline on a RelBench task.

One DFS at depth max(hops) with target augmentation and temporal differences,
optional hub + period extras, a (hop x foundation model) sweep on validation,
optional hard_pool / tfidf / hurdle, a refit of the best pair on train +
validation, and the test score.

    python examples/rdblearn_standard_pipeline.py --dataset rel-f1 --task driver-dnf
    python examples/rdblearn_standard_pipeline.py --dataset rel-trial --task study-outcome \\
        --hub --period --text-encoder tfidf --context-strategy hard_pool
"""
import argparse
import json

import numpy as np
from loguru import logger

from rdblearn import RDBDataset, StandardPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-trial")
parser.add_argument("--task", default="study-outcome")
parser.add_argument("--hops", type=int, nargs="+", default=[2, 3, 4])
parser.add_argument("--models", nargs="+", default=["tabpfn-v3.5"])
parser.add_argument("--device", default="cuda")
parser.add_argument("--seed", type=int, default=0, help="Seeds training-set downsampling / hard_pool.")
parser.add_argument("--hub", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--period", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--text-encoder", choices=["tfidf", "none"], default="tfidf")
parser.add_argument(
    "--context-strategy",
    choices=["hard_pool", "random", "none"],
    default="hard_pool",
)
parser.add_argument("--recency-window", type=int, default=800_000)
parser.add_argument("--hurdle", choices=["auto", "none"], default="auto")
parser.add_argument("--hurdle-zero-threshold", type=float, default=0.05)
parser.add_argument("--output", default=None, help="Optional JSON file for the results.")
args = parser.parse_args()

logger.enable("rdblearn")
np.random.seed(args.seed)

dataset = RDBDataset.from_relbench(args.dataset, for_task=args.task)
task = dataset.tasks[args.task]
md = task.metadata
task_type = "regression" if "regression" in (md.task_type or "") else "classification"
task_id = f"relbench::{args.dataset}::{args.task}"

text_encoder = None if args.text_encoder == "none" else args.text_encoder
hurdle = None if args.hurdle == "none" or task_type != "regression" else args.hurdle
context = None if args.context_strategy in ("none", "random") else args.context_strategy

pipeline = StandardPipeline(
    task_type=task_type,
    hops=args.hops,
    models=args.models,
    metric="roc_auc" if task_type == "classification" else "mae",
    model_params={"device": args.device},
    hub=args.hub,
    period=args.period,
    task_id=task_id,
    text_encoder=text_encoder,
    context_strategy=context,
    recency_window=args.recency_window,
    hurdle=hurdle,
    hurdle_zero_threshold=args.hurdle_zero_threshold,
    seed=args.seed,
)
pipeline.fit(
    task.train_df, task.val_df, md.target_col,
    rdb=dataset.rdb, key_mappings=md.key_mappings, cutoff_time_column=md.time_col,
    task_id=task_id,
)
test_score = pipeline.score(task.test_df)

print(pipeline.results_.to_string(index=False))
print(f"Best: {pipeline.best_}")
print(f"Test {pipeline.metric} (refit on train + val): {test_score:.4f}")

if args.output:
    with open(args.output, "w") as f:
        json.dump({
            "dataset": args.dataset, "task": args.task, "metric": pipeline.metric,
            "task_id": task_id,
            "hub": args.hub, "period": args.period,
            "text_encoder": text_encoder,
            "context_strategy": context,
            "recency_window": args.recency_window,
            "hurdle": hurdle,
            "dfs_seconds": pipeline.dfs_seconds_, "refit_seconds": pipeline.refit_seconds_,
            "sweep": pipeline.results_.to_dict(orient="records"),
            "best": pipeline.best_, "test_score": test_score,
            "n_extra_columns": len(getattr(pipeline, "extra_columns_", [])),
        }, f, indent=2)
