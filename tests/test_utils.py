import pandas as pd

from multitabfm.utils import prepare_target_dataframes


def test_prepare_target_dataframes_basic():
    train = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "item_id": [10, 20, 30],
            "ts": [100, 200, 300],
            "label": [0, 1, 0],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": [4, 5],
            "item_id": [40, 50],
            "ts": [400, 500],
            "label": [1, 0],
        }
    )
    metadata = {
        "key_mappings": [{"user_id": "users.user_id"}, {"item_id": "items.item_id"}],
        "time_column": "ts",
        "target_column": "label",
    }

    train_df, test_df = prepare_target_dataframes(train, test, metadata)

    assert list(train_df.columns) == ["user_id", "item_id", "ts", "label"]
    assert list(test_df.columns) == ["user_id", "item_id", "ts", "label"]
    assert len(train_df) == 3 and len(test_df) == 2
