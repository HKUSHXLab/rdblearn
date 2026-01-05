import pandas as pd
import types

import multitabfm.feature_engineer as fe


def test_generate_features_monkeypatched(monkeypatch):
    # fake compute_dfs_features returns input with an extra feature column
    def _fake_compute_dfs_features(rdb, target_dataframe, key_mappings, cutoff_time_column, config_overrides):
        df = target_dataframe.copy()
        df["feat_const"] = 1.0
        return df

    monkeypatch.setattr(fe, "compute_dfs_features", _fake_compute_dfs_features)

    target_df = pd.DataFrame({"user_id": [1, 2], "ts": [100, 200], "label": [0, 1]})
    rdb = object()  # any non-None placeholder
    key_mappings = {"user_id": "users.user_id"}

    out = fe.generate_features(target_df, rdb, key_mappings, time_column="ts", dfs_config=None)
    assert "feat_const" in out.columns
    assert len(out) == 2
