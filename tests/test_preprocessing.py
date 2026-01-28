import unittest
import pandas as pd
import numpy as np
from rdblearn.preprocessing import TemporalDiffTransformer
from rdblearn.config import TemporalDiffConfig

from loguru import logger
logger.enable("rdblearn")


class TestTemporalDiffTransformer(unittest.TestCase):
    """Tests for TemporalDiffTransformer class."""

    def test_fit_detects_epochtime_columns(self):
        """Test that fit() correctly identifies columns containing '_epochtime'."""
        config = TemporalDiffConfig(enabled=True)
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
        config = TemporalDiffConfig(enabled=True)
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
        self.assertIn('tx_epochtime_diff', result.columns)
        # Check computed values in nanoseconds: (3-1)*one_day_ns, (4-2)*one_day_ns
        np.testing.assert_array_almost_equal(
            result['tx_epochtime_diff'].values, [2 * one_day_ns, 2 * one_day_ns]
        )
        # Other columns preserved
        self.assertIn('other_col', result.columns)

    def test_fit_respects_exclude_columns(self):
        """Test that excluded columns are not transformed."""
        config = TemporalDiffConfig(
            enabled=True,
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
        config = TemporalDiffConfig(enabled=True)
        transformer = TemporalDiffTransformer(config)

        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        transformer.fit(df)

        self.assertTrue(transformer.is_fitted_)
        self.assertEqual(len(transformer.timestamp_columns_), 0)

    def test_datetime_diff_feature_dtype(self):
        """Test that datetime diff features are converted to float64 type."""
        config = TemporalDiffConfig(enabled=True)
        transformer = TemporalDiffTransformer(config)

        # Create data with epochtime column (int64 nanoseconds as returned by fastdfs)
        one_day_ns = int(1e9 * 86400)
        df = pd.DataFrame({
            'tx_epochtime': [one_day_ns, 2 * one_day_ns],
            'user_epochtime': [5 * one_day_ns, 6 * one_day_ns],
            'amount': [100.5, 200.5]
        })

        # Cutoff time as datetime64 (as passed from X[cutoff_time_column])
        cutoff_time = pd.Series(pd.to_datetime(['2023-01-10', '2023-01-15']))

        transformer.fit(df)
        result = transformer.transform(df, cutoff_time)

        # Verify that all generated diff features are float64 type
        self.assertIn('tx_epochtime_diff', result.columns)
        self.assertIn('user_epochtime_diff', result.columns)

        # Check dtype is float64 (following RDBColumnDType.float_t practice)
        self.assertEqual(result['tx_epochtime_diff'].dtype, np.float64)
        self.assertEqual(result['user_epochtime_diff'].dtype, np.float64)

        # Verify original timestamp columns are removed
        self.assertNotIn('tx_epochtime', result.columns)
        self.assertNotIn('user_epochtime', result.columns)

        # Verify non-timestamp columns are preserved
        self.assertIn('amount', result.columns)
        self.assertEqual(result['amount'].dtype, np.float64)

if __name__ == '__main__':
    unittest.main()
