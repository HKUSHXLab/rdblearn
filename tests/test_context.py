import unittest

import numpy as np
import pandas as pd

from rdblearn.context import (
    hard_pool_sizes,
    hard_pool_subsample_indices,
    prepare_hard_pool_fit,
)


class TestHardPool(unittest.TestCase):
    def test_sizes_from_recency_window(self):
        self.assertEqual(hard_pool_sizes(800_000), (200_000, 800_000))
        self.assertEqual(hard_pool_sizes(100), (25, 100))

    def test_subsample_none_when_n_le_m(self):
        t = np.arange(10)
        self.assertIsNone(
            hard_pool_subsample_indices(t, K=2, M=10, n_estimators=4, seed=0)
        )

    def test_subsample_from_most_recent_pool(self):
        # timestamps: rows 0..9 oldest -> newest would be high values
        t = np.arange(20)
        bags = hard_pool_subsample_indices(t, K=3, M=5, n_estimators=4, seed=0)
        self.assertEqual(len(bags), 4)
        pool = set(np.argsort(-t, kind="stable")[:5])
        for bag in bags:
            self.assertEqual(len(bag), 3)
            self.assertEqual(len(set(bag)), 3)  # replace=False
            self.assertTrue(set(bag).issubset(pool))

    def test_prepare_hard_pool_fit_union_and_overrides(self):
        n = 30
        X = pd.DataFrame({"a": np.arange(n, dtype=float)})
        y = pd.Series(np.arange(n))
        t = np.arange(n)
        X2, y2, overrides = prepare_hard_pool_fit(
            X, y, t, recency_window=12, n_estimators=4, seed=1
        )
        self.assertEqual(overrides["n_estimators"], 4)
        self.assertIn("SUBSAMPLE_SAMPLES", overrides["inference_config"])
        # M=12, K=3; union size <= 12
        self.assertLessEqual(len(X2), 12)
        self.assertEqual(len(X2), len(y2))
        bags = overrides["inference_config"]["SUBSAMPLE_SAMPLES"]
        self.assertEqual(len(bags), 4)
        for bag in bags:
            self.assertEqual(len(bag), 3)
            self.assertTrue(max(bag) < len(X2))


if __name__ == "__main__":
    unittest.main()
