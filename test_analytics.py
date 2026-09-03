import unittest

import pandas as pd

from analytics import calculate_metrics, cost_sensitivity, prepare_backtest


class AnalyticsTests(unittest.TestCase):
    def test_cost_adjustment_and_metrics(self):
        data = pd.DataFrame(
            {
                "date": ["2026-01-02", "2026-01-05"],
                "strategy_return": [0.01, -0.005],
                "turnover": [0.5, 0.2],
            }
        )
        result = prepare_backtest(data, transaction_cost_bps=10)
        self.assertFalse(result.blocking_errors)
        self.assertEqual(result.data["net_return"].round(6).tolist(), [0.0095, -0.0052])
        metrics = calculate_metrics(pd.Series([0.10, -0.05]))
        self.assertAlmostEqual(metrics["total_return"], 0.045)
        self.assertAlmostEqual(metrics["win_rate"], 0.5)

    def test_duplicate_dates_fail(self):
        data = pd.DataFrame(
            {
                "date": ["2026-01-02", "2026-01-02"],
                "strategy_return": [0.01, 0.02],
            }
        )
        result = prepare_backtest(data, transaction_cost_bps=10)
        self.assertEqual(result.verdict, "Fail")
        self.assertIn("duplicate dates", result.blocking_errors[0])

    def test_missing_turnover_is_transparent(self):
        data = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=252, freq="B"),
                "strategy_return": [0.001] * 252,
            }
        )
        result = prepare_backtest(data, transaction_cost_bps=10)
        cost_check = next(c for c in result.checks if c.name == "Transaction costs")
        self.assertEqual(cost_check.status, "Warning")
        self.assertEqual(result.data["net_return"].tolist(), data["strategy_return"].tolist())

    def test_cost_sensitivity_declines_with_cost(self):
        data = pd.DataFrame(
            {"strategy_return": [0.01, 0.01], "turnover": [1.0, 1.0]}
        )
        result = cost_sensitivity(data, [0, 10, 50])
        self.assertTrue(result["Total return"].is_monotonic_decreasing)

    def test_empty_dataset_fails_cleanly(self):
        data = pd.DataFrame(columns=["date", "strategy_return"])
        result = prepare_backtest(data, transaction_cost_bps=10)
        self.assertEqual(result.verdict, "Fail")
        self.assertIn("no data rows", result.blocking_errors[0])

    def test_sample_file_produces_a_complete_audit(self):
        data = pd.read_csv("sample_backtest.csv")
        result = prepare_backtest(data, transaction_cost_bps=10)
        metrics = calculate_metrics(result.data["net_return"])

        self.assertFalse(result.blocking_errors)
        self.assertEqual(len(result.checks), 7)
        self.assertLess(metrics["total_return"], 0.011)


if __name__ == "__main__":
    unittest.main()
