import functools
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from fastdfs.api import create_rdb

from rdblearn.config import RDBLearnConfig
from rdblearn.constants import TABPFN_DEFAULT_CONFIG
from rdblearn.estimator import RDBLearnClassifier, RDBLearnRegressor

try:
    import torch
    from tabpfn import TabPFNClassifier, TabPFNRegressor
except ImportError:  # pragma: no cover
    TabPFNClassifier = TabPFNRegressor = None

N_TRAIN = 120
N_TEST = 30
PREDICT_BATCH_SIZE = 10


class TestTabPFNDefaultConfig(unittest.TestCase):
    def test_default_config_builds_kv_cache_in_fit(self):
        self.assertEqual(TABPFN_DEFAULT_CONFIG["fit_mode"], "fit_with_cache")

    @unittest.skipIf(TabPFNClassifier is None, "tabpfn is not installed")
    def test_default_config_is_accepted_by_tabpfn(self):
        for cls in (TabPFNClassifier, TabPFNRegressor):
            est = cls(**TABPFN_DEFAULT_CONFIG)
            self.assertEqual(est.get_params()["fit_mode"], "fit_with_cache")


@unittest.skipIf(TabPFNClassifier is None, "tabpfn is not installed")
class TestTabPFNKVCacheReuse(unittest.TestCase):
    """Fit real TabPFN models through RDBLearn and check what `predict*` feeds the transformer."""

    def setUp(self):
        rng = np.random.default_rng(0)
        users = pd.DataFrame({
            "user_id": [str(i) for i in range(N_TRAIN)],
            "age": rng.integers(18, 80, N_TRAIN),
        })
        tx = pd.DataFrame({
            "tx_id": range(600),
            "user_id": [str(i) for i in rng.integers(0, N_TRAIN, 600)],
            "amount": rng.random(600) * 100,
            "timestamp": pd.date_range("2023-01-01", periods=600, freq="h"),
        })
        self.rdb = create_rdb(
            name="kv_cache_db",
            tables={"users": users, "transactions": tx},
            primary_keys={"users": "user_id", "transactions": "tx_id"},
            foreign_keys=[("transactions", "user_id", "users", "user_id")],
            time_columns={"transactions": "timestamp"},
        )
        self.X_train = pd.DataFrame({
            "user_id": [str(i) for i in range(N_TRAIN)],
            "cutoff_time": pd.Timestamp("2023-02-01"),
        })
        self.y_cls = pd.Series(rng.integers(0, 2, N_TRAIN), name="target")
        self.y_reg = pd.Series(rng.random(N_TRAIN), name="target")
        self.X_test = pd.DataFrame({
            "user_id": [str(i) for i in range(N_TEST)],
            "cutoff_time": pd.Timestamp("2023-02-01"),
        })
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _fit(self, rdblearn_cls, tabpfn_cls, y, fit_mode):
        params = {**TABPFN_DEFAULT_CONFIG, "n_estimators": 2, "device": self.device}
        if fit_mode is not None:
            params["fit_mode"] = fit_mode
        model = rdblearn_cls(
            base_estimator=tabpfn_cls(**params),
            config=RDBLearnConfig(predict_batch_size=PREDICT_BATCH_SIZE),
        )
        try:
            model.fit(
                self.X_train, y, rdb=self.rdb,
                key_mappings={"user_id": "users.user_id"}, cutoff_time_column="cutoff_time",
            )
        except Exception as e:  # noqa: BLE001
            if "download" in str(e).lower() or "license" in str(e).lower():
                self.skipTest(f"TabPFN checkpoint unavailable: {e}")
            raise
        return model

    def _rows_seen_by_transformer(self, tabpfn_model, predict):
        """Run `predict` and return the row count of every transformer forward call."""
        arch = type(tabpfn_model.models_[0])
        rows = []
        original_forward = arch.forward

        # TabPFN inspects forward's signature to decide which kwargs to pass, so keep it intact.
        @functools.wraps(original_forward)
        def spy(module, x, *args, **kwargs):
            rows.append(int(x.shape[0]))
            return original_forward(module, x, *args, **kwargs)

        with mock.patch.object(arch, "forward", spy):
            out = predict()
        self.assertTrue(rows, "transformer was never called during predict")
        return out, rows

    def _check(self, rdblearn_cls, tabpfn_cls, y, method):
        cached = self._fit(rdblearn_cls, tabpfn_cls, y, fit_mode=None)
        executor = cached.base_estimator.executor_
        self.assertTrue(getattr(executor, "kv_caches", None), f"fit did not build a KV cache ({type(executor).__name__})")
        out, rows = self._rows_seen_by_transformer(
            cached.base_estimator, lambda: getattr(cached, method)(self.X_test)
        )
        self.assertEqual(len(out), N_TEST)
        # Each predict batch only runs its own test rows against the cached training context.
        self.assertLessEqual(max(rows), PREDICT_BATCH_SIZE)

        # Control: TabPFN's own default re-encodes all training rows on every predict batch.
        uncached = self._fit(rdblearn_cls, tabpfn_cls, y, fit_mode="fit_preprocessors")
        _, rows = self._rows_seen_by_transformer(
            uncached.base_estimator, lambda: getattr(uncached, method)(self.X_test)
        )
        self.assertGreaterEqual(min(rows), N_TRAIN)

    def test_classifier_predict_proba_reuses_kv_cache(self):
        self._check(RDBLearnClassifier, TabPFNClassifier, self.y_cls, "predict_proba")

    def test_regressor_predict_reuses_kv_cache(self):
        self._check(RDBLearnRegressor, TabPFNRegressor, self.y_reg, "predict")


if __name__ == "__main__":
    unittest.main()
