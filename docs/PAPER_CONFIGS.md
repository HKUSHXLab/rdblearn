# Paper Experiment Configurations

This page documents the RDBLearn settings used in the published papers. Config keys refer to the `RDBLearnClassifier` / `RDBLearnRegressor` `config` argument (and the tabular `base_estimator`).

| Paper | Venue | Package / pipeline |
| --- | --- | --- |
| [No Need to Train Your RDB Foundation Model](https://arxiv.org/abs/2602.13697) | ICML 2026 | Original RDBLearn (JUICE + single-table ICL) |
| [RDBLearn: Simple In-Context Prediction Over Relational Databases](https://arxiv.org/abs/2602.18495) | arXiv (package / toolkit paper) | Companion documentation for the RDBLearn toolkit |
| [Parameter-Free Encoders Remain Viable for RDB Foundation Models](https://arxiv.org/abs/2607.05476) | 2nd ICML Workshop on Foundation Models for Structured Data, 2026 | RDBLearn **v1.1** ([release](https://github.com/HKUSHXLab/rdblearn/releases/tag/v1.1)) |

---

## 1. ICML 2026 — *No Need to Train Your RDB Foundation Model*

Source: paper Appendix F.1 and Section 6.

### Shared settings (all benchmarks / tasks)

| Setting | Value |
| --- | --- |
| Encoder depth $H$ (`dfs.max_depth`) | Chosen from $\{2, 3\}$ on the **dev / validation** split |
| Train ICL sample size (`max_train_samples`) | Random downsample to **10,000** when larger |
| Temporal features (`temporal_diff`) | **Enabled** (absolute timestamps → relative differences to cutoff) |
| Text features | **Disabled** (`enable_raw_text_features` / special / $n$-gram all off) |
| Task-specific hyperparameter search | **None** beyond choosing $H$ and base model on val |
| Hardware | Single NVIDIA 4080 (32 GB) |

### Aggregation primitives (JUICE / DFS)

Fixed across all datasets (no per-task retuning):

| Column type | Aggregators |
| --- | --- |
| Continuous | `sum`, `mean`, `mode`, `min`, `max`, `std` |
| Categorical | `count`, `mode` |

Activation along meta-path aggregation was linear ($\sigma = \mathrm{id}$).

### Base tabular ICL models (chosen on val)

One of:

1. **TabPFN-v2** — checkpoints `tabpfn-v2-classification/regression-finetuned-zk73skhh`
2. **TabPFN-v2.5** — checkpoints `tabpfn-v2.5-classification/regression-default`
3. **LimiX** — checkpoint `LimiX-16M`

Ablations (paper Figure 7) also vary $H$ and these base models; performance is stable across combinations.

### Approximate `config` sketch

```python
from tabpfn import TabPFNClassifier  # or TabPFNRegressor / LimiX wrapper

config = {
    "dfs": {
        "max_depth": 2,  # or 3; selected on validation
        "agg_primitives": ["sum", "mean", "mode", "min", "max", "std"],
        "engine": "dfs2sql",
    },
    "max_train_samples": 10000,
    "temporal_diff": {"enabled": True},
    "ag_config": {
        "enable_datetime_features": True,
        "enable_raw_text_features": False,
        "enable_text_special_features": False,
        "enable_text_ngram_features": False,
    },
}

# Example base estimator (classification):
base_estimator = TabPFNClassifier(
    # TabPFN-v2.5 default checkpoint in current package releases;
    # ICML runs also evaluated TabPFN-v2 and LimiX.
)
```

### Benchmarks reported with this setup

- RelBench-v1 classification & regression (excluding `rel-event`, `rel-f1` for leakage concerns)
- RelBench-v2 (excluding text-heavy `rel-arxiv` and non-public `rel-mimic` in the ICML draft)
- 4DBInfer classification subset (Amazon churn, Outbrain CTR, RetailRocket CVR, StackExchange upvote / churn)

---

## 2. Package paper — *RDBLearn: Simple In-Context Prediction Over Relational Databases*

Source: paper Section 5 (“RDBLearn configuration”). Companion toolkit paper for the open-source package; cites the ICML work for the theoretical JUICE framing.

### Shared settings

| Setting | Value |
| --- | --- |
| Encoder depth $H$ (`dfs.max_depth`) | Chosen from $\{2, 3, 4\}$ on the **validation** split |
| Tabular ICL backend | Chosen on val from TabPFN-v2, TabPFN-v2.5, LimiX (see below) |
| Fit / ICL sample limit | **10,000** per backend; uniform random downsample if larger |
| Preprocessing | AutoGluon-style tabular preprocessing (impute missing values, normalize features) |
| Other hyperparameter search | **None** beyond depth and backend selection on val |
| Hardware | Single NVIDIA 4090 (32 GB) |

### Base tabular ICL backends

| Backend | Checkpoints | Fit limit |
| --- | --- | --- |
| TabPFN-v2 | `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` / `tabpfn-v2-regressor.ckpt` | 10k |
| TabPFN-v2.5 | `tabpfn-v2.5-classifier-v2.5_default.ckpt` / `tabpfn-v2.5-regressor-v2.5_default.ckpt` | 10k |
| LimiX | `LimiX-16M` | 10k |

### Approximate `config` sketch

```python
config = {
    "dfs": {
        "max_depth": 2,  # or 3 or 4; selected on validation
        "engine": "dfs2sql",
    },
    "max_train_samples": 10000,
    # AutoGluon-style preprocessing is applied by default in the package pipeline.
}
```

### Benchmarks reported with this setup

- RelBench classification and regression
- 4DBInfer classification (heterogeneous GNN / Transformer supervised baselines + Griffin)

---

## 3. ICML Workshop 2026 — *Parameter-Free Encoders Remain Viable for RDB Foundation Models*

Source: paper Appendix A (RDBLearn **v1.1** updates) and Appendix B.

Reported tables label the pipeline as **RDBLearn (v1.1)**. Changes relative to the ICML paper:

### Updates in v1.1

| Component | Change |
| --- | --- |
| Encoder depth $H$ (`dfs.max_depth`) | Chosen from $\{2, 3, 4\}$ (via validation when available) |
| Aggregations | Broader set: original primitives **plus** continuous **25th / 75th quantiles** and categorical **discrete entropy**; optional **encode categoricals as numeric** (changes which aggregators apply) |
| Aggregation selection | When a validation set exists, pick between the **original** and **expanded** aggregator sets using val performance |
| Base models added | **TabICL-v2**, **TabPFN-v3** (in addition to earlier TabPFN / LimiX options) |
| SQL feature merge | Chunk-based merging for large feature sets (efficiency only; not a modeling hyperparameter) |

### Default when **no** official validation set

| Setting | Value |
| --- | --- |
| Base predictor | **TabPFN-v2.5** |
| Encoder depth $H$ (`dfs.max_depth`) | Selected from $\{2, 3, 4\}$ |
| Aggregations | **Expanded** set (quantiles + entropy, etc.) |

### Other notes from the workshop paper

- Same high-level pipeline as the ICML paper (parameter-free relational encoder + frozen single-table FM); **no** task-specific retuning beyond the v1.1 selection rules above.
- Text columns remain ignored.
- Neighborhood **labels are not** injected into the encoder (target-history / label-as-feature encoder inputs left for future work).
- Benchmarks: RelBench-v1 / v2, 4DBInfer, and SALT multiclass tasks.

### Approximate `config` sketch (no-val default)

```python
config = {
    "dfs": {
        "max_depth": 2,  # or 3 or 4; selected from {2, 3, 4}
        # Expanded aggregators (illustrative; exact primitive names depend on fastdfs version):
        # original + quantile_25, quantile_75, entropy, ...
        "engine": "dfs2sql",
    },
    "max_train_samples": 10000,
    "temporal_diff": {"enabled": True},
    "encode_categorical_as_float": False,  # optionally True; selected via val when available
    "ag_config": {
        "enable_raw_text_features": False,
        "enable_text_special_features": False,
        "enable_text_ngram_features": False,
    },
}
```

---

## Quick comparison

| | ICML 2026 | Package paper | Workshop 2026 (v1.1) |
| --- | --- | --- | --- |
| Depth $H$ | $\{2, 3\}$ | $\{2, 3, 4\}$  | $\{2, 3, 4\}$ |
| Aggregators | Fixed original set | Package defaults (DFS primitives) | Original **vs** expanded (val); expanded default if no val |
| Base models | TabPFN-v2 / v2.5, LimiX | TabPFN-v2 / v2.5, LimiX | + TabICL-v2, TabPFN-v3 |
| Train samples | ≤ 10k | ≤ 10k | ≤ 10k |
| Temporal diffs | On | On | On |
| Text features | Off | Off | Off |
