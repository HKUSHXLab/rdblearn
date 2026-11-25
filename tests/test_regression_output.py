import unittest

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from rdblearn.config import RDBLearnConfig
from rdblearn.estimator import RDBLearnRegressor
from test_standard_pipeline import HAS_CUDA, make_relational_data


class DistributionRegressor(BaseEstimator, RegressorMixin):
    """Returns 1.0 as its mean and 2.0 as its median."""

    def fit(self, X, y):
        return self

    def predict(self, X, output_type="mean"):
        return np.full(len(X), {"mean": 1.0, "median": 2.0}[output_type])


def fit_regressor(base_estimator, **config):
    rdb, train, val, test = make_relational_data()
    reg = RDBLearnRegressor(base_estimator=base_estimator, config=RDBLearnConfig(**config))
    reg.fit(train.drop(columns=["label"]), train["label"].astype(float), rdb=rdb,
            key_mappings={"user_id": "users.user_id"}, cutoff_time_column="cutoff")
    return reg, test.drop(columns=["label"])


class TestRegressionOutputType(unittest.TestCase):
    def test_median_by_default(self):
        reg, X = fit_regressor(DistributionRegressor())
        np.testing.assert_array_equal(reg.predict(X), 2.0)

    def test_explicit_output_type_and_opt_out(self):
        reg, X = fit_regressor(DistributionRegressor())
        np.testing.assert_array_equal(reg.predict(X, output_type="mean"), 1.0)
        reg, X = fit_regressor(DistributionRegressor(), regression_output_type=None)
        np.testing.assert_array_equal(reg.predict(X), 1.0)

    def test_median_applies_to_every_predict_batch(self):
        reg, X = fit_regressor(DistributionRegressor(), predict_batch_size=7)
        np.testing.assert_array_equal(reg.predict(X), 2.0)


@unittest.skipUnless(HAS_CUDA, "needs a GPU and foundation-model weights")
class TestFoundationModelRegressorsUseMedian(unittest.TestCase):
    def check(self, base_estimator):
        reg, X = fit_regressor(base_estimator)
        X_t = reg.preprocessor_.transform(reg._dfs_for_prediction(X, None))
        median = np.asarray(reg.base_estimator.predict(X_t, output_type="median"))
        mean = np.asarray(reg.base_estimator.predict(X_t, output_type="mean"))
        np.testing.assert_allclose(reg.predict(X), median, rtol=1e-5)
        self.assertFalse(np.allclose(median, mean))

    def test_tabpfn_v35(self):
        from rdblearn.utils import make_tabpfn_regressor

        self.check(make_tabpfn_regressor("v3.5", n_estimators=1))

    def test_tabicl_v2(self):
        from rdblearn.utils import make_tabicl_regressor

        self.check(make_tabicl_regressor(n_estimators=1))

    def test_kumo_tabular_medium(self):
        from rdblearn.utils import KumoTabularWrapperRegressor

        self.check(KumoTabularWrapperRegressor(size="medium", n_estimators=1))


if __name__ == "__main__":
    unittest.main()
