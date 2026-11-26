"""Max-depth DFS cache: one deep DFS, shallower hops reuse it."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from fastdfs.api import create_rdb
from sklearn.base import BaseEstimator, ClassifierMixin

from rdblearn.dfs_depth_cache import DEPTH_CACHE, DFS_CACHE_MAX_DEPTH
from rdblearn.relbench_v1 import RelBenchV1Estimator


class FirstFeatureClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y))
        return self

    @staticmethod
    def _num(X):
        if hasattr(X, "select_dtypes"):
            num = X.select_dtypes(include=[np.number])
            return np.nan_to_num(num.to_numpy(dtype=float))
        return np.nan_to_num(np.asarray(X, dtype=float))

    def predict_proba(self, X):
        x = self._num(X)[:, 0]
        p = 1 / (1 + np.exp(-(x - x.mean()) / (x.std() + 1e-9)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def make_shop(n_users=30, seed=0):
    rng = np.random.default_rng(seed)
    users = pd.DataFrame({
        "user_id": [str(i) for i in range(n_users)],
        "age": rng.integers(18, 80, n_users),
    })
    products = pd.DataFrame({
        "product_id": [str(i) for i in range(10)],
        "price": rng.random(10) * 50,
    })
    orders = pd.DataFrame({
        "order_id": [str(i) for i in range(200)],
        "user_id": [str(i) for i in rng.integers(0, n_users, 200)],
        "product_id": [str(i) for i in rng.integers(0, 10, 200)],
        "amount": rng.random(200) * 100,
        "ts": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 200, 200), unit="D"),
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

    train = split("2023-07-01", 80)
    val = split("2023-08-01", 40)
    return rdb, train, val


class TestDFSDepthCache(unittest.TestCase):
    def setUp(self):
        DEPTH_CACHE.clear()

    def tearDown(self):
        DEPTH_CACHE.clear()

    def test_shallower_hops_reuse_one_deep_dfs(self):
        rdb, train, val = make_shop()
        db_scope = object()  # stand-in for RelBench Database identity
        keys = {"user_id": "users.user_id"}

        def fit_depth(d):
            est = RelBenchV1Estimator(
                "classification",
                max_depth=d,
                cache_max_depth=DFS_CACHE_MAX_DEPTH,
                model="tabpfn-v3.5",
                hub=False,
                period=False,
                text_encoder=None,
                context_strategy=None,
                hurdle=None,
                model_params={},
            )
            # Use a tiny mock TFM via patching TFM_FACTORIES would be heavier;
            # instead swap model factories by constructing with a custom path:
            from rdblearn import relbench_v1 as m

            orig = m.TFM_FACTORIES

            class Fake:
                def __init__(self, **kwargs):
                    pass

                def fit(self, X, y):
                    self.classes_ = np.unique(np.asarray(y))
                    return self

                def predict_proba(self, X):
                    n = len(X)
                    return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])

                def predict(self, X):
                    return np.zeros(len(X), dtype=int)

            try:
                m.TFM_FACTORIES = {"tabpfn-v3.5": (lambda **k: Fake(), lambda **k: Fake())}
                est.model_name = "tabpfn-v3.5"
                return est.fit(
                    train, "label", rdb, keys, "cutoff", db=db_scope, train_key="train"
                )
            finally:
                m.TFM_FACTORIES = orig

        self.assertEqual(DEPTH_CACHE.matrix_computations, 0)
        e2 = fit_depth(2)
        self.assertEqual(DEPTH_CACHE.matrix_computations, 1)
        e3 = fit_depth(3)
        self.assertEqual(DEPTH_CACHE.matrix_computations, 1)
        e4 = fit_depth(4)
        self.assertEqual(DEPTH_CACHE.matrix_computations, 1)

        # Hop slices are nested: deeper has at least as many DFS+extra cols planned.
        self.assertLessEqual(
            len(e2.bundle_.hop_columns[2]), len(e3.bundle_.hop_columns[3])
        )
        self.assertLessEqual(
            len(e3.bundle_.hop_columns[3]), len(e4.bundle_.hop_columns[4])
        )
        # Shared bundle / featurizer session
        self.assertIs(e2.bundle_, e3.bundle_)
        self.assertIs(e3.featurizer_, e4.featurizer_)

        X_val = val.drop(columns=["label"])
        p2 = e2.predict_proba(X_val)
        p4 = e4.predict_proba(X_val)
        self.assertEqual(p2.shape, (len(val), 2))
        self.assertEqual(p4.shape, (len(val), 2))

    def test_new_db_scope_recomputes_dfs(self):
        rdb, train, _ = make_shop()
        keys = {"user_id": "users.user_id"}
        from rdblearn import relbench_v1 as m

        class Fake:
            def __init__(self, **kwargs):
                pass

            def fit(self, X, y):
                self.classes_ = np.unique(np.asarray(y))
                return self

            def predict_proba(self, X):
                n = len(X)
                return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

        orig = m.TFM_FACTORIES
        m.TFM_FACTORIES = {"tabpfn-v3.5": (lambda **k: Fake(), lambda **k: Fake())}
        try:
            for scope in (object(), object()):
                est = RelBenchV1Estimator(
                    "classification",
                    max_depth=2,
                    hub=False,
                    period=False,
                    text_encoder=None,
                    context_strategy=None,
                    hurdle=None,
                )
                est.fit(
                    train, "label", rdb, keys, "cutoff", db=scope, train_key="train"
                )
        finally:
            m.TFM_FACTORIES = orig
        self.assertEqual(DEPTH_CACHE.matrix_computations, 2)


if __name__ == "__main__":
    unittest.main()
