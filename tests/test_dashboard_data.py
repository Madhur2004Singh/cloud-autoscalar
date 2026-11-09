"""Unit tests for Dashboard Data Loader, Transformation, and Visualization."""

from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
import plotly.graph_objects as go

from dashboard.charts import (
    create_autoscaling_cpu_chart,
    create_autoscaling_overload_chart,
    create_autoscaling_server_usage_chart,
    create_infrastructure_telemetry_chart,
    create_prediction_chart,
    create_prediction_error_chart,
    create_strategy_comparison_timeline,
    create_workload_chart,
)
from dashboard.data_loader import (
    PROJECT_ROOT,
    SCENARIOS,
    STRATEGIES,
    compute_prediction_metrics,
    compute_workload_summary_metrics,
    get_project_root,
    load_autoscaling_decisions,
    load_autoscaling_metrics,
    load_autoscaling_summary,
    load_infrastructure_metrics,
    load_prediction_data,
    load_workload_data,
)


class TestDashboardDataLoader(unittest.TestCase):
    """Test suite for dashboard data loading and metric extraction functions."""

    def test_get_project_root(self):
        root = get_project_root()
        self.assertIsInstance(root, Path)
        self.assertTrue((root / "data").exists())

    def test_load_workload_data_all(self):
        df = load_workload_data()
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 480)
        expected_cols = {"minute", "timestamp", "viewers", "scenario", "event_active"}
        self.assertTrue(expected_cols.issubset(df.columns))

    def test_load_workload_data_per_scenario(self):
        for scenario in SCENARIOS:
            df = load_workload_data(scenario=scenario)
            self.assertEqual(len(df), 120)
            self.assertTrue((df["scenario"] == scenario).all())

    def test_load_workload_data_invalid_scenario(self):
        with self.assertRaises(ValueError):
            load_workload_data(scenario="NonExistentScenario")

    def test_compute_workload_summary_metrics(self):
        df = load_workload_data(scenario="Normal Traffic")
        metrics = compute_workload_summary_metrics(df)
        self.assertIn("peak_viewers", metrics)
        self.assertIn("avg_viewers", metrics)
        self.assertIn("total_viewers", metrics)
        self.assertIn("min_viewers", metrics)
        self.assertIn("duration_minutes", metrics)
        self.assertGreater(metrics["peak_viewers"], metrics["min_viewers"])
        self.assertEqual(metrics["duration_minutes"], 120)

    def test_compute_workload_summary_metrics_empty(self):
        metrics = compute_workload_summary_metrics(pd.DataFrame())
        self.assertEqual(metrics["peak_viewers"], 0)
        self.assertEqual(metrics["total_viewers"], 0)

    def test_load_prediction_data_all(self):
        df = load_prediction_data()
        self.assertEqual(len(df), 88)
        expected_cols = {
            "minute",
            "scenario",
            "actual_viewers",
            "predicted_viewers",
            "baseline_viewers",
            "error_rf",
            "error_baseline",
        }
        self.assertTrue(expected_cols.issubset(df.columns))

    def test_load_prediction_data_per_scenario(self):
        for scenario in SCENARIOS:
            df = load_prediction_data(scenario=scenario)
            self.assertEqual(len(df), 22)
            self.assertTrue((df["scenario"] == scenario).all())

    def test_load_prediction_data_invalid_scenario(self):
        with self.assertRaises(ValueError):
            load_prediction_data(scenario="NonExistentScenario")

    def test_compute_prediction_metrics(self):
        df = load_prediction_data(scenario="Normal Traffic")
        metrics = compute_prediction_metrics(df)
        self.assertGreater(metrics["rf_mae"], 0.0)
        self.assertGreater(metrics["rf_rmse"], 0.0)
        self.assertGreater(metrics["base_mae"], 0.0)
        self.assertGreater(metrics["base_rmse"], 0.0)
        self.assertIn("mae_reduction_pct", metrics)
        self.assertIn("rmse_reduction_pct", metrics)

    def test_compute_prediction_metrics_empty(self):
        metrics = compute_prediction_metrics(pd.DataFrame())
        self.assertEqual(metrics["rf_mae"], 0.0)
        self.assertEqual(metrics["mae_reduction_pct"], 0.0)

    def test_load_autoscaling_summary(self):
        df = load_autoscaling_summary()
        self.assertEqual(len(df), 8)
        expected_cols = {
            "scenario",
            "strategy",
            "total_overload_intervals",
            "overload_percentage",
            "average_cpu_utilization",
            "maximum_cpu_utilization",
            "average_active_servers",
            "maximum_active_servers",
            "total_server_minutes",
            "scale_up_operations",
            "scale_down_operations",
            "total_scaling_operations",
        }
        self.assertTrue(expected_cols.issubset(df.columns))

    def test_load_autoscaling_metrics_filtering(self):
        df_normal_reactive = load_autoscaling_metrics(
            scenario="Normal Traffic", strategy="Reactive"
        )
        self.assertEqual(len(df_normal_reactive), 120)
        self.assertTrue((df_normal_reactive["scenario"] == "Normal Traffic").all())
        self.assertTrue((df_normal_reactive["strategy"] == "Reactive").all())

        df_all = load_autoscaling_metrics()
        self.assertEqual(len(df_all), 960)

    def test_load_autoscaling_metrics_invalid_filter(self):
        with self.assertRaises(ValueError):
            load_autoscaling_metrics(scenario="InvalidScenario")
        with self.assertRaises(ValueError):
            load_autoscaling_metrics(strategy="InvalidStrategy")

    def test_load_autoscaling_decisions(self):
        df = load_autoscaling_decisions(scenario="Normal Traffic", strategy="Predictive")
        self.assertEqual(len(df), 120)
        self.assertIn("action", df.columns)
        self.assertIn("reason", df.columns)

    def test_load_infrastructure_metrics(self):
        df = load_infrastructure_metrics(scenario="Peak Broadcasting Event")
        self.assertEqual(len(df), 120)
        self.assertIn("average_cpu_utilization", df.columns)
        self.assertIn("overload_status", df.columns)

    def test_missing_file_raises_error(self):
        with patch("dashboard.data_loader.DATA_RAW_DIR", Path("/non_existent_dir")):
            with self.assertRaises(FileNotFoundError):
                # When combined_path does not exist and individual files also do not exist
                load_workload_data.__wrapped__()


class TestDashboardCharts(unittest.TestCase):
    """Test suite for Plotly chart creation routines."""

    def setUp(self):
        self.scenario = "Normal Traffic"
        self.df_workload = load_workload_data(scenario=self.scenario)
        self.df_pred = load_prediction_data(scenario=self.scenario)
        self.df_summary = load_autoscaling_summary()
        self.df_metrics = load_autoscaling_metrics(scenario=self.scenario)

    def test_create_workload_chart(self):
        fig = create_workload_chart(self.df_workload, self.scenario)
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)

    def test_create_prediction_chart(self):
        fig = create_prediction_chart(self.df_pred, self.scenario)
        self.assertIsInstance(fig, go.Figure)
        # Should have Actual, RF, and Baseline traces (3 traces)
        self.assertEqual(len(fig.data), 3)

    def test_create_prediction_error_chart(self):
        fig = create_prediction_error_chart(self.df_pred, self.scenario)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)

    def test_create_autoscaling_summary_charts(self):
        fig_overload = create_autoscaling_overload_chart(self.df_summary)
        self.assertIsInstance(fig_overload, go.Figure)

        fig_cpu = create_autoscaling_cpu_chart(self.df_summary)
        self.assertIsInstance(fig_cpu, go.Figure)

        fig_srv = create_autoscaling_server_usage_chart(self.df_summary)
        self.assertIsInstance(fig_srv, go.Figure)

    def test_create_strategy_comparison_timeline(self):
        fig = create_strategy_comparison_timeline(self.df_metrics, self.scenario)
        self.assertIsInstance(fig, go.Figure)

    def test_create_infrastructure_telemetry_chart(self):
        df_sub = self.df_metrics[self.df_metrics["strategy"] == "Reactive"]
        fig = create_infrastructure_telemetry_chart(df_sub, self.scenario, "Reactive")
        self.assertIsInstance(fig, go.Figure)


if __name__ == "__main__":
    unittest.main()
