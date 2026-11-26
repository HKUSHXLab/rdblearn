"""Tests for hub / period / text_encoder / hard_pool wiring on StandardPipeline."""
import unittest

import numpy as np
import pandas as pd
from fastdfs.api import create_rdb
from sklearn.base import BaseEstimator, ClassifierMixin

from rdblearn.pipeline import StandardPipeline
from rdblearn.text import AnchorTextEncoder, merge_transform_text, parse_entity_key


class FirstFeatureClassifier(BaseEstimator, ClassifierMixin):
    """Uses the first numeric column; ignores string/text columns (e.g. __raw_text)."""

    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y))
        return self

    @staticmethod
    def _numeric_matrix(X):
        if hasattr(X, "select_dtypes"):
            num = X.select_dtypes(include=[np.number])
            if num.shape[1] == 0:
                raise ValueError("no numeric columns")
            return np.nan_to_num(num.to_numpy(dtype=float))
        return np.nan_to_num(np.asarray(X, dtype=float))

    def predict_proba(self, X):
        x = self._numeric_matrix(X)[:, 0]
        p = 1 / (1 + np.exp(-(x - x.mean()) / (x.std() + 1e-9)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def make_shop(n_users=40, seed=0):
    rng = np.random.default_rng(seed)
    users = pd.DataFrame({
        "user_id": [str(i) for i in range(n_users)],
        "age": rng.integers(18, 80, n_users),
        "bio": [f"user {i} likes coffee" for i in range(n_users)],
    })
    products = pd.DataFrame({
        "product_id": [str(i) for i in range(15)],
        "price": rng.random(15) * 50,
    })
    orders = pd.DataFrame({
        "order_id": [str(i) for i in range(300)],
        "user_id": [str(i) for i in rng.integers(0, n_users, 300)],
        "product_id": [str(i) for i in rng.integers(0, 15, 300)],
        "amount": rng.random(300) * 100,
        "ts": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 200, 300), unit="D"),
    })
    rdb = create_rdb(
        name="shop",
        tables={"users": users, "products": products, "orders": orders},
        primary_keys={"users": "user_id", "products": "product_id", "orders": "order_id"},
        foreign_keys=[
            ("orders", "user_id", "users", "user_id"),
            ("orders", "product_id", "products", "product_id"),
        ],
        time_columns={"orders": "ts"},
    )

    def split(cutoff, n):
        ids = rng.integers(0, n_users, n)
        return pd.DataFrame({
            "user_id": [str(i) for i in ids],
            "cutoff": pd.Timestamp(cutoff),
            "label": (rng.random(n) < 0.4).astype(int),
        })

    train = pd.concat([split("2023-05-01", 50), split("2023-07-01", 50)], ignore_index=True)
    return rdb, train, split("2023-08-01", 40), split("2023-09-01", 40)


class TestTextHelpers(unittest.TestCase):
    def test_merge_transform_text(self):
        out = merge_transform_text({"device": "cpu"})
        self.assertEqual(out["device"], "cpu")
        self.assertTrue(out["inference_config"]["TRANSFORM_TEXT"])

    def test_anchor_text_encoder_attaches_string_cols(self):
        rdb, train, _, _ = make_shop()
        enc = AnchorTextEncoder().fit(rdb, train, {"user_id": "users.user_id"})
        feats = pd.DataFrame({"x": np.arange(len(train), dtype=float)})
        out = enc.transform(rdb, feats, train)
        self.assertTrue(any(c.endswith("__raw_text") for c in out.columns))
        text_col = [c for c in out.columns if c.endswith("__raw_text")][0]
        self.assertTrue(pd.api.types.is_string_dtype(out[text_col]))


class TestPipelineExtras(unittest.TestCase):
    def test_rejects_bad_knobs(self):
        with self.assertRaises(ValueError):
            StandardPipeline("classification", text_encoder="hash")
        with self.assertRaises(ValueError):
            StandardPipeline("classification", context_strategy="soft_pool")
        with self.assertRaises(ValueError):
            StandardPipeline("classification", hurdle="always")

    def test_hub_adds_peer_columns(self):
        rdb, train, val, test = make_shop()
        pipe = StandardPipeline(
            "classification",
            hops=(2,),
            models={"m": lambda: FirstFeatureClassifier()},
            hub=True,
            period=False,
        )
        pipe.fit(train, val, "label", rdb, {"user_id": "users.user_id"}, "cutoff")
        self.assertTrue(any(str(c).startswith("MEAN(") for c in pipe.extra_columns_))
        score = pipe.score(test)
        self.assertTrue(0.0 <= score <= 1.0)

    def test_period_without_task_id_is_skipped(self):
        rdb, train, val, test = make_shop()
        pipe = StandardPipeline(
            "classification",
            hops=(2,),
            models={"m": lambda: FirstFeatureClassifier()},
            period=True,
            task_id=None,
        )
        pipe.fit(train, val, "label", rdb, {"user_id": "users.user_id"}, "cutoff")
        self.assertFalse(any(str(c).startswith("sql__") for c in pipe.extra_columns_))
        pipe.score(test)

    def test_text_encoder_adds_raw_text_and_sets_transform_flag(self):
        rdb, train, val, test = make_shop()
        pipe = StandardPipeline(
            "classification",
            hops=(2,),
            models={"m": lambda: FirstFeatureClassifier()},
            text_encoder="tfidf",
        )
        self.assertTrue(pipe._effective_model_params()["inference_config"]["TRANSFORM_TEXT"])
        pipe.fit(train, val, "label", rdb, {"user_id": "users.user_id"}, "cutoff")
        self.assertTrue(any(c.endswith("__raw_text") for c in pipe.extra_columns_))
        pipe.score(test)

    def test_hard_pool_with_tiny_window_on_mock_model(self):
        rdb, train, val, test = make_shop()
        recorded = {}

        class RecordingClassifier(FirstFeatureClassifier):
            def __init__(self, **kwargs):
                recorded["kwargs"] = kwargs

            def fit(self, X, y):
                recorded["n_fit"] = len(X)
                return super().fit(X, y)

        pipe = StandardPipeline(
            "classification",
            hops=(2,),
            models={"m": lambda **kw: RecordingClassifier(**kw)},
            context_strategy="hard_pool",
            recency_window=20,  # M=20, K=5; train has 100 rows
            seed=0,
        )
        pipe.fit(train, val, "label", rdb, {"user_id": "users.user_id"}, "cutoff")
        self.assertEqual(recorded["kwargs"]["n_estimators"], 8)
        self.assertIn("SUBSAMPLE_SAMPLES", recorded["kwargs"]["inference_config"])
        self.assertLessEqual(recorded["n_fit"], 20)
        pipe.score(test)


if __name__ == "__main__":
    unittest.main()
