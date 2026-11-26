"""Tests for the RelArena RelBench v1 model plugin (hop-only space)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd


class TestRelBenchV1Estimator(unittest.TestCase):
    def test_rejects_bad_knobs(self):
        from rdblearn.relbench_v1 import RelBenchV1Estimator

        with self.assertRaises(ValueError):
            RelBenchV1Estimator("classification", text_encoder="hash")
        with self.assertRaises(ValueError):
            RelBenchV1Estimator("classification", model="nope")


class TestRelArenaModelRegistration(unittest.TestCase):
    def test_space_is_hop_only_default_first(self):
        from rdblearn.relarena_model import RELBENCH_V1_SPACE, RDBLearnRelBenchV1Model
        from relarena_core.registry import registry

        self.assertEqual(RELBENCH_V1_SPACE.default_overrides, {"max_depth": 2})
        self.assertEqual(
            RELBENCH_V1_SPACE.fixed_grid,
            [{"max_depth": 2}, {"max_depth": 3}, {"max_depth": 4}],
        )
        self.assertIs(registry.get("rdblearn-relbench-v1-2026-11-26"), RDBLearnRelBenchV1Model)
        self.assertIs(registry.get("rdblearn-relbench-v1-latest"), RDBLearnRelBenchV1Model)
        self.assertEqual(registry.kind("rdblearn-relbench-v1-latest"), "model")
        self.assertIs(registry.search_space("rdblearn-relbench-v1-2026-11-26"), RELBENCH_V1_SPACE)

    def test_fit_uses_config_depth_once(self):
        from relbench.base import TaskType
        from relarena_core import RunIdentity

        from rdblearn.relarena_model import RDBLearnRelBenchV1Model

        calls = []

        class FakeEst:
            classes_ = np.array([0, 1])

            def __init__(self, task_type, **kwargs):
                kwargs = {"task_type": task_type, **kwargs}
                calls.append(kwargs)

            def fit(self, *args, **kwargs):
                return self

            def predict_proba(self, X):
                return np.column_stack([np.full(len(X), 0.4), np.full(len(X), 0.6)])

        train = SimpleNamespace(
            df=pd.DataFrame(
                {
                    "driver_id": ["0", "1"],
                    "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                    "y": [0, 1],
                }
            )
        )
        drivers = SimpleNamespace(
            df=pd.DataFrame({"driver_id": ["0", "1"]}),
            pkey_col="driver_id",
            fkey_col_to_pkey_table={},
            time_col=None,
        )
        db = SimpleNamespace(table_dict={"drivers": drivers})
        task = SimpleNamespace(
            task_type=TaskType.BINARY_CLASSIFICATION,
            target_col="y",
            entity_col="driver_id",
            entity_table="drivers",
            time_col="date",
        )

        with mock.patch("rdblearn.relarena_model.RelBenchV1Estimator", FakeEst), mock.patch(
            "rdblearn.relarena_model.database_to_rdb", return_value="rdb"
        ), mock.patch(
            "rdblearn.relarena_model.key_mappings_for_task",
            return_value={"driver_id": "drivers.driver_id"},
        ):
            model = RDBLearnRelBenchV1Model(
                {"max_depth": 3},
                run_identity=RunIdentity("rel-f1", "fp", "driver-dnf", "tfp"),
            )
            model.fit(task, db, train, None, seed=0)
            pred = model.predict(
                task,
                db,
                SimpleNamespace(
                    df=pd.DataFrame(
                        {
                            "driver_id": ["0"],
                            "date": pd.to_datetime(["2020-03-01"]),
                        }
                    )
                ),
            )

        self.assertEqual(calls[0]["max_depth"], 3)
        self.assertEqual(calls[0]["model"], "tabpfn-v3.5")
        self.assertEqual(calls[0]["task_id"], "relbench::rel-f1::driver-dnf")
        self.assertEqual(pred.shape, (1,))
        np.testing.assert_allclose(pred, [0.6])


if __name__ == "__main__":
    unittest.main()
