import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import fastdfs
from fastdfs import DFSConfig
from fastdfs.api import create_rdb
from fastdfs.dfs import DFSSession
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from rdblearn.config import RDBLearnConfig
from rdblearn.constants import RDBLEARN_DEFAULT_CONFIG
from rdblearn.estimator import RDBLearnClassifier, RDBLearnRegressor
from rdblearn.pipeline import StandardPipeline, planned_dfs_columns

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    HAS_CUDA = False


class FirstFeatureClassifier(BaseEstimator, ClassifierMixin):
    """Deterministic stand-in for a foundation model: scores rows by one feature column."""

    def __init__(self, column_index=0):
        self.column_index = column_index

    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict_proba(self, X):
        x = np.nan_to_num(np.asarray(X, dtype=float)[:, self.column_index % X.shape[1]])
        p = 1 / (1 + np.exp(-(x - x.mean()) / (x.std() + 1e-9)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ReversedFeatureClassifier(FirstFeatureClassifier):
    def predict_proba(self, X):
        return super().predict_proba(X)[:, ::-1]


class ConstantClassifier(FirstFeatureClassifier):
    """Validation ROC AUC of exactly 0.5."""

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


class OutOfMemoryClassifier(FirstFeatureClassifier):
    """Raises CUDA out-of-memory when fit on more than ``max_rows`` rows."""

    def __init__(self, column_index=0, max_rows=0, reverse=False):
        super().__init__(column_index)
        self.max_rows = max_rows
        self.reverse = reverse

    def fit(self, X, y):
        if len(X) > self.max_rows:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 80.00 GiB")
        return super().fit(X, y)

    def predict_proba(self, X):
        proba = super().predict_proba(X)
        return proba[:, ::-1] if self.reverse else proba


class BrokenClassifier(FirstFeatureClassifier):
    def fit(self, X, y):
        raise ValueError("not an out-of-memory error")


class RecordingRegressor(BaseEstimator, RegressorMixin):
    """Records the ``output_type`` it is asked to predict."""

    def fit(self, X, y):
        return self

    def predict(self, X, output_type=None):
        self.output_type = output_type
        return np.zeros(len(X))


def make_relational_data(n_users=60, seed=0):
    """users <- orders -> products <- reviews: deep enough for depth-3 and depth-4 features."""
    rng = np.random.default_rng(seed)
    users = pd.DataFrame({"user_id": [str(i) for i in range(n_users)], "age": rng.integers(18, 80, n_users)})
    products = pd.DataFrame({"product_id": [str(i) for i in range(20)], "price": rng.random(20) * 50})
    orders = pd.DataFrame({
        "order_id": [str(i) for i in range(600)],
        "user_id": [str(i) for i in rng.integers(0, n_users, 600)],
        "product_id": [str(i) for i in rng.integers(0, 20, 600)],
        "amount": rng.random(600) * 100,
        "channel": rng.choice(["web", "app", "store"], 600),
        "ts": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 300, 600), unit="D"),
    })
    reviews = pd.DataFrame({
        "review_id": [str(i) for i in range(400)],
        "product_id": [str(i) for i in rng.integers(0, 20, 400)],
        "stars": rng.integers(1, 6, 400).astype(float),
        "ts": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 300, 400), unit="D"),
    })
    rdb = create_rdb(
        name="shop",
        tables={"users": users, "products": products, "orders": orders, "reviews": reviews},
        primary_keys={"users": "user_id", "products": "product_id", "orders": "order_id", "reviews": "review_id"},
        foreign_keys=[
            ("orders", "user_id", "users", "user_id"),
            ("orders", "product_id", "products", "product_id"),
            ("reviews", "product_id", "products", "product_id"),
        ],
        time_columns={"orders": "ts", "reviews": "ts"},
    )

    def split(cutoff, n):
        ids = rng.integers(0, n_users, n)
        return pd.DataFrame({
            "user_id": [str(i) for i in ids],
            "cutoff": pd.Timestamp(cutoff),
            "label": (rng.random(n) < 0.4).astype(int),
        })

    train = pd.concat([split("2023-06-01", 80), split("2023-08-01", 80)], ignore_index=True)
    return rdb, train, split("2023-09-01", 60), split("2023-10-01", 60)


class TestDFSSessionReuse(unittest.TestCase):
    def setUp(self):
        self.rdb, self.train, self.val, self.test = make_relational_data()
        self.kw = dict(rdb=self.rdb, key_mappings={"user_id": "users.user_id"}, cutoff_time_column="cutoff")

    def test_fit_and_predict_share_one_dfs_plan(self):
        clf = RDBLearnClassifier(base_estimator=FirstFeatureClassifier())
        clf.fit(self.train.drop(columns=["label"]), self.train["label"], **self.kw)
        session = clf.dfs_session_
        clf.predict_proba(self.val.drop(columns=["label"]))
        clf.predict_proba(self.test.drop(columns=["label"]))
        self.assertIs(clf.dfs_session_, session)
        self.assertEqual(session._n_plan_calls, 1)

    def test_session_features_match_one_shot_dfs(self):
        clf = RDBLearnClassifier(base_estimator=FirstFeatureClassifier())
        clf.fit(self.train.drop(columns=["label"]), self.train["label"], **self.kw)
        X = self.val.drop(columns=["label"])
        via_session = clf._dfs_for_prediction(X, None)
        one_shot = fastdfs.compute_dfs_features(
            clf.rdb_, X, key_mappings=clf.key_mappings_, cutoff_time_column="cutoff", config=clf._dfs_config()
        )
        pd.testing.assert_frame_equal(via_session, one_shot[via_session.columns], check_dtype=False)

    def test_session_and_one_shot_predictions_match(self):
        class OneShotDFSClassifier(RDBLearnClassifier):
            def _compute_dfs(self, X, rdb=None):
                return fastdfs.compute_dfs_features(
                    rdb if rdb is not None else self.rdb_, X, key_mappings=self.key_mappings_,
                    cutoff_time_column=self.cutoff_time_column_, config=self._dfs_config(),
                )

        predictions = []
        for cls in (RDBLearnClassifier, OneShotDFSClassifier):
            clf = cls(base_estimator=FirstFeatureClassifier())
            clf.fit(self.train.drop(columns=["label"]), self.train["label"], **self.kw)
            predictions.append([
                clf.predict_proba(split.drop(columns=["label"])) for split in (self.val, self.test)
            ])
        for session_proba, one_shot_proba in zip(*predictions):
            np.testing.assert_allclose(session_proba, one_shot_proba)

    def test_pickled_estimator_reopens_session(self):
        clf = RDBLearnClassifier(base_estimator=FirstFeatureClassifier())
        clf.fit(self.train.drop(columns=["label"]), self.train["label"], **self.kw)
        X = self.test.drop(columns=["label"])
        expected = clf.predict_proba(X)
        restored = pickle.loads(pickle.dumps(clf))
        np.testing.assert_allclose(restored.predict_proba(X), expected)


class TestHopSelection(unittest.TestCase):
    def test_hop_columns_from_deep_dfs_match_direct_dfs(self):
        rdb, train, _, _ = make_relational_data()
        X = train.drop(columns=["label"])
        km = {"user_id": "users.user_id"}

        def cfg(depth):
            return DFSConfig(**{**RDBLEARN_DEFAULT_CONFIG["dfs"], "max_depth": depth})

        deep = fastdfs.compute_dfs_features(rdb, X, key_mappings=km, cutoff_time_column="cutoff", config=cfg(4))
        n_deep = len(deep.columns) - len(X.columns)
        self.assertTrue(any("NUM_UNIQUE(" in c for c in deep.columns))
        for depth in (2, 3):
            direct = fastdfs.compute_dfs_features(rdb, X, key_mappings=km, cutoff_time_column="cutoff", config=cfg(depth))
            planned = set(planned_dfs_columns(rdb, X, km, "cutoff", cfg(depth)))
            selected = [c for c in deep.columns if c in planned]
            self.assertEqual(sorted(selected), sorted(set(direct.columns) - set(X.columns)))
            self.assertLess(len(selected), n_deep)
            pd.testing.assert_frame_equal(deep[selected], direct[selected], check_dtype=False)


class TestStandardPipeline(unittest.TestCase):
    def test_sweep_selects_best_and_refits_on_train_val(self):
        rdb, train, val, test = make_relational_data()
        pipe = StandardPipeline(
            task_type="classification",
            hops=(2, 3, 4),
            models={"col0": lambda: FirstFeatureClassifier(0), "col5": lambda: FirstFeatureClassifier(5)},
        )
        pipe.fit(train, val, "label", rdb, {"user_id": "users.user_id"}, "cutoff")

        self.assertEqual(len(pipe.results_), 6)
        self.assertEqual(set(pipe.results_["hop"]), {2, 3, 4})
        n_features = pipe.results_.groupby("hop")["n_features"].first()
        self.assertTrue(n_features.is_monotonic_increasing)
        best = pipe.results_.loc[pipe.results_["val_roc_auc"].idxmax()]
        self.assertEqual((pipe.best_["hop"], pipe.best_["model"]), (best["hop"], best["model"]))

        final = pipe.featurizer_
        self.assertEqual(final.config.dfs.max_depth, pipe.best_["hop"])
        self.assertTrue(final.config.enable_target_augmentation)
        self.assertTrue(final.config.temporal_diff.enabled)
        self.assertEqual(len(final.history_df_), len(train) + len(val))
        self.assertNotIsInstance(pipe.model_, (RDBLearnClassifier, RDBLearnRegressor))
        self.assertIsInstance(pipe.model_.estimator, FirstFeatureClassifier)
        score = pipe.score(test)
        self.assertTrue(0.0 <= score <= 1.0)
        self.assertEqual(pipe.featurizer_.dfs_session_._n_plan_calls, 1)

    def test_unknown_model_or_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            StandardPipeline("classification", metric="f2")
        with self.assertRaises(ValueError):
            StandardPipeline("classification", models=("tabpfn-v9",))._model_factories()
        with self.assertRaises(ValueError):
            StandardPipeline("regression", regression_output_type="mode")

    def test_regression_output_type(self):
        self.assertEqual(StandardPipeline("regression")._config_for_depth(2).regression_output_type, "median")
        pipe = StandardPipeline("regression", metric="rmse", regression_output_type="mean")
        self.assertEqual(pipe._config_for_depth(2).regression_output_type, "mean")
        config = RDBLearnConfig(**{**RDBLEARN_DEFAULT_CONFIG, "regression_output_type": "mean"})
        self.assertEqual(StandardPipeline("regression", config=config)._config_for_depth(2).regression_output_type, "mean")

    def test_out_of_memory_candidates_are_skipped(self):
        rdb, train, val, test = make_relational_data()
        pipe = StandardPipeline(
            task_type="classification",
            hops=(2, 3),
            models={"oom": lambda: OutOfMemoryClassifier(), "col0": lambda: FirstFeatureClassifier(0)},
        )
        pipe.fit(train, val, "label", rdb, {"user_id": "users.user_id"}, "cutoff")

        oom = pipe.results_[pipe.results_["model"] == "oom"]
        self.assertEqual(len(oom), 2)
        self.assertTrue(oom["val_roc_auc"].isna().all())
        self.assertTrue(oom["error"].str.contains("OutOfMemoryError").all())
        self.assertEqual(pipe.best_["model"], "col0")
        self.assertTrue(0.0 <= pipe.score(test) <= 1.0)

    def test_refit_out_of_memory_falls_back_to_next_best(self):
        rdb, train, val, test = make_relational_data()
        n_train = len(train)
        # "a" and "b" score 1 - each other's ROC AUC, so one of them ranks at or above the
        # constant model's 0.5; both fit on train but run out of memory on train + val.
        pipe = StandardPipeline(
            task_type="classification",
            hops=(2,),
            models={
                "a": lambda: OutOfMemoryClassifier(0, max_rows=n_train),
                "b": lambda: OutOfMemoryClassifier(0, max_rows=n_train, reverse=True),
                "constant": lambda: ConstantClassifier(),
            },
        )
        pipe.fit(train, val, "label", rdb, {"user_id": "users.user_id"}, "cutoff")

        self.assertFalse(pipe.results_["val_roc_auc"].isna().any())
        self.assertNotEqual(pipe.results_["val_roc_auc"].idxmax(), 2)
        self.assertEqual(pipe.best_["model"], "constant")
        self.assertEqual(len(pipe.featurizer_.history_df_), len(train) + len(val))

    def test_every_candidate_out_of_memory_raises(self):
        rdb, train, val, _ = make_relational_data()
        pipe = StandardPipeline("classification", hops=(2,), models={"oom": lambda: OutOfMemoryClassifier()})
        with self.assertRaisesRegex(RuntimeError, "Every candidate ran out of memory"):
            pipe.fit(train, val, "label", rdb, {"user_id": "users.user_id"}, "cutoff")

    def test_other_errors_are_not_swallowed(self):
        rdb, train, val, _ = make_relational_data()
        pipe = StandardPipeline("classification", hops=(2,), models={"bad": lambda: BrokenClassifier()})
        with self.assertRaises(ValueError):
            pipe.fit(train, val, "label", rdb, {"user_id": "users.user_id"}, "cutoff")

    def test_does_not_construct_an_rdblearn_estimator(self):
        rdb, train, val, _ = make_relational_data()
        pipe = StandardPipeline(
            "classification", hops=(2,), models={"col0": lambda: FirstFeatureClassifier(0)},
        )
        with patch.object(RDBLearnClassifier, "__init__", side_effect=AssertionError("no classifier")), \
             patch.object(RDBLearnRegressor, "__init__", side_effect=AssertionError("no regressor")):
            pipe.fit(train, val, "label", rdb, {"user_id": "users.user_id"}, "cutoff")
        self.assertIsInstance(pipe.featurizer_.dfs_session_, DFSSession)

    def test_sweep_session_is_not_the_refit_session(self):
        rdb, train, val, test = make_relational_data()
        seen = []
        original = DFSSession.compute

        def record(self, frame):
            seen.append(id(self))
            return original(self, frame)

        pipe = StandardPipeline(
            "classification", hops=(2, 3), models={"col0": lambda: FirstFeatureClassifier(0)},
        )
        with patch.object(DFSSession, "compute", record):
            pipe.fit(train, val, "label", rdb, {"user_id": "users.user_id"}, "cutoff")
            pipe.score(test)
        # train + val share the sweep session; the refit and the test score share another.
        self.assertEqual(seen[0], seen[1])
        self.assertNotEqual(seen[0], seen[2])
        self.assertTrue(all(i == seen[2] for i in seen[2:]))

    def test_regression_output_type_reaches_the_model(self):
        rdb, train, val, test = make_relational_data()
        pipe = StandardPipeline(
            "regression", hops=(2,), models={"rec": RecordingRegressor},
            regression_output_type="mean",
        )
        pipe.fit(train, val, "label", rdb, {"user_id": "users.user_id"}, "cutoff")
        pipe.score(test)
        self.assertEqual(pipe.model_.estimator.output_type, "mean")


@unittest.skipUnless(HAS_CUDA, "needs a GPU and foundation-model weights")
class TestStandardPipelineWithFoundationModels(unittest.TestCase):
    def test_default_models_tabpfn_v35_and_kumo_medium(self):
        rdb, train, val, test = make_relational_data()
        pipe = StandardPipeline(
            task_type="classification", hops=(2, 3),
            model_params={"n_estimators": 1},
        )
        pipe.fit(train, val, "label", rdb, {"user_id": "users.user_id"}, "cutoff")
        self.assertEqual(set(pipe.results_["model"]), {"tabpfn-v3.5", "kumo-tabular-medium"})
        self.assertTrue(pipe.results_["val_roc_auc"].between(0, 1).all())
        self.assertTrue(0.0 <= pipe.score(test) <= 1.0)


if __name__ == "__main__":
    unittest.main()
