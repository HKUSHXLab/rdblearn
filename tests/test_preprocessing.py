import unittest
import pandas as pd
import numpy as np
from rdblearn.preprocessing import TemporalDiffTransformer, NANO_TO_TIME_UNIT
from rdblearn.config import TemporalDiffConfig

from loguru import logger
logger.enable("rdblearn")


class TestTemporalDiffTransformer(unittest.TestCase):
    """Tests for TemporalDiffTransformer class."""

    def test_fit_detects_epochtime_columns(self):
        """Test that fit() correctly identifies columns containing '_epochtime'."""
        config = TemporalDiffConfig(enabled=True, time_unit="days")
        transformer = TemporalDiffTransformer(config)

        df = pd.DataFrame({
            'feature_epochtime': [1000, 2000],
            'other_epochtime': [3000, 4000],
            'regular_col': [1, 2]
        })

        transformer.fit(df)

        self.assertTrue(transformer.is_fitted_)
        self.assertEqual(len(transformer.timestamp_columns_), 2)
        self.assertIn('feature_epochtime', transformer.timestamp_columns_)
        self.assertIn('other_epochtime', transformer.timestamp_columns_)
        self.assertNotIn('regular_col', transformer.timestamp_columns_)

    def test_transform_computes_time_diff(self):
        """Test that transform correctly computes time differences."""
        config = TemporalDiffConfig(enabled=True, time_unit="days")
        transformer = TemporalDiffTransformer(config)

        # Create data with epochtime column (nanoseconds)
        one_day_ns = int(1e9 * 86400)
        df = pd.DataFrame({
            'tx_epochtime': [one_day_ns, 2 * one_day_ns],  # Day 1, Day 2
            'other_col': [10, 20]
        })
        cutoff_time = pd.Series([3 * one_day_ns, 4 * one_day_ns])  # Day 3, Day 4

        transformer.fit(df)
        result = transformer.transform(df, cutoff_time)

        # Original timestamp column should be removed
        self.assertNotIn('tx_epochtime', result.columns)
        # New feature column should exist
        self.assertIn('tx_epochtime.days_since', result.columns)
        # Check computed values: (3-1)=2, (4-2)=2 days
        np.testing.assert_array_almost_equal(
            result['tx_epochtime.days_since'].values, [2.0, 2.0]
        )
        # Other columns preserved
        self.assertIn('other_col', result.columns)

    def test_fit_respects_exclude_columns(self):
        """Test that excluded columns are not transformed."""
        config = TemporalDiffConfig(
            enabled=True,
            time_unit="days",
            exclude_columns=['feature_epochtime']
        )
        transformer = TemporalDiffTransformer(config)

        df = pd.DataFrame({
            'feature_epochtime': [1000, 2000],
            'other_epochtime': [3000, 4000],
        })

        transformer.fit(df)

        self.assertEqual(len(transformer.timestamp_columns_), 1)
        self.assertNotIn('feature_epochtime', transformer.timestamp_columns_)
        self.assertIn('other_epochtime', transformer.timestamp_columns_)

    def test_fit_no_timestamp_columns(self):
        """Test fit() when no timestamp columns exist."""
        config = TemporalDiffConfig(enabled=True, time_unit="days")
        transformer = TemporalDiffTransformer(config)

        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        transformer.fit(df)

        self.assertTrue(transformer.is_fitted_)
        self.assertEqual(len(transformer.timestamp_columns_), 0)

if __name__ == '__main__':
    unittest.main()
