"""Unit tests for the synthetic workload generator module."""

import unittest
import numpy as np
import pandas as pd

from simulation.workload_generator import (
    DEFAULT_DURATION_MINUTES,
    DEFAULT_MAX_VIEWERS,
    DEFAULT_MIN_VIEWERS,
    SCENARIO_ENDING,
    SCENARIO_NORMAL,
    SCENARIO_PEAK,
    SCENARIO_SPIKE,
    generate_all_scenarios,
    generate_event_ending_traffic,
    generate_normal_traffic,
    generate_peak_event_traffic,
    generate_sudden_spike_traffic,
)


class TestWorkloadGenerator(unittest.TestCase):
    """Test suite covering Stage A synthetic workload generator requirements."""

    def setUp(self) -> None:
        """Initialize default scenarios for tests."""
        self.scenarios = generate_all_scenarios(random_seed=42)
        self.expected_columns = [
            "minute",
            "timestamp",
            "viewers",
            "scenario",
            "event_active",
        ]

    # Test 1: Every scenario generates exactly 120 observations with default configuration.
    def test_default_observation_count(self) -> None:
        """Verify each scenario produces exactly 120 observations by default."""
        for name, df in self.scenarios.items():
            with self.subTest(scenario=name):
                self.assertEqual(
                    len(df),
                    DEFAULT_DURATION_MINUTES,
                    f"Scenario '{name}' generated {len(df)} rows; expected {DEFAULT_DURATION_MINUTES}",
                )

    # Test 2: All viewer counts are non-negative.
    def test_viewers_non_negative(self) -> None:
        """Verify viewer counts are always greater than or equal to 0."""
        for name, df in self.scenarios.items():
            with self.subTest(scenario=name):
                min_val = df["viewers"].min()
                self.assertGreaterEqual(
                    min_val,
                    DEFAULT_MIN_VIEWERS,
                    f"Scenario '{name}' has negative viewer count: {min_val}",
                )

    # Test 3: No viewer count exceeds the configured maximum.
    def test_viewers_do_not_exceed_max(self) -> None:
        """Verify viewer counts never exceed the configured maximum threshold."""
        for name, df in self.scenarios.items():
            with self.subTest(scenario=name):
                max_val = df["viewers"].max()
                self.assertLessEqual(
                    max_val,
                    DEFAULT_MAX_VIEWERS,
                    f"Scenario '{name}' exceeded max viewers: {max_val} > {DEFAULT_MAX_VIEWERS}",
                )

    # Test 4: Viewer counts are integers.
    def test_viewers_are_integers(self) -> None:
        """Verify the 'viewers' column contains integer values."""
        for name, df in self.scenarios.items():
            with self.subTest(scenario=name):
                self.assertTrue(
                    np.issubdtype(df["viewers"].dtype, np.integer),
                    f"Scenario '{name}' viewers dtype is {df['viewers'].dtype}, expected integer",
                )

    # Test 5: Minute values are sequential.
    def test_minutes_are_sequential(self) -> None:
        """Verify minute values start at 0, increment by 1, and end at duration - 1."""
        for name, df in self.scenarios.items():
            with self.subTest(scenario=name):
                expected_minutes = list(range(DEFAULT_DURATION_MINUTES))
                self.assertListEqual(
                    df["minute"].tolist(),
                    expected_minutes,
                    f"Scenario '{name}' minutes are not sequential from 0 to {DEFAULT_DURATION_MINUTES - 1}",
                )

    # Test 6: All required columns are present.
    def test_required_columns_present(self) -> None:
        """Verify all mandatory columns exist in every generated scenario DataFrame."""
        for name, df in self.scenarios.items():
            with self.subTest(scenario=name):
                for col in self.expected_columns:
                    self.assertIn(
                        col,
                        df.columns,
                        f"Scenario '{name}' is missing required column: '{col}'",
                    )
                self.assertEqual(
                    list(df.columns),
                    self.expected_columns,
                    f"Scenario '{name}' column order does not match standard specification",
                )

    # Test 7: Every scenario has the correct scenario label.
    def test_correct_scenario_labels(self) -> None:
        """Verify each scenario DataFrame has its expected constant label."""
        expected_labels = {
            "normal": SCENARIO_NORMAL,
            "sudden_spike": SCENARIO_SPIKE,
            "peak_event": SCENARIO_PEAK,
            "event_ending": SCENARIO_ENDING,
        }
        for key, expected_label in expected_labels.items():
            df = self.scenarios[key]
            unique_labels = df["scenario"].unique()
            self.assertEqual(
                len(unique_labels),
                1,
                f"Scenario '{key}' contains multiple scenario labels: {unique_labels}",
            )
            self.assertEqual(
                unique_labels[0],
                expected_label,
                f"Scenario '{key}' has label '{unique_labels[0]}'; expected '{expected_label}'",
            )

    # Test 8: Identical random seeds produce identical results.
    def test_reproducibility_with_identical_seed(self) -> None:
        """Verify that identical random seeds yield identical DataFrame values."""
        df1 = generate_normal_traffic(random_seed=123)
        df2 = generate_normal_traffic(random_seed=123)
        pd.testing.assert_frame_equal(
            df1, df2, check_dtype=True, obj="Normal traffic identical seed check"
        )

        spike1 = generate_sudden_spike_traffic(random_seed=999)
        spike2 = generate_sudden_spike_traffic(random_seed=999)
        pd.testing.assert_frame_equal(
            spike1,
            spike2,
            check_dtype=True,
            obj="Sudden spike identical seed check",
        )

    # Test 9: Different seeds can produce different results.
    def test_different_seeds_produce_variation(self) -> None:
        """Verify that different random seeds produce varying noise in viewer counts."""
        df_seed_a = generate_normal_traffic(random_seed=42)
        df_seed_b = generate_normal_traffic(random_seed=101)
        self.assertFalse(
            df_seed_a["viewers"].equals(df_seed_b["viewers"]),
            "Different seeds unexpectedly produced identical viewer series",
        )

    # Test 10: The sudden-spike scenario exhibits a meaningful increase around its configured spike time.
    def test_sudden_spike_meaningful_increase(self) -> None:
        """Verify a sharp, meaningful increase occurs near the configured spike time."""
        spike_time = 60
        spike_magnitude = 25000
        df = generate_sudden_spike_traffic(
            spike_time=spike_time,
            spike_magnitude=spike_magnitude,
            spike_sigma=5.0,
            random_seed=42,
        )

        # Baseline window well before spike (minute 10 to 30)
        baseline_mean = df.loc[
            (df["minute"] >= 10) & (df["minute"] <= 30), "viewers"
        ].mean()

        # Traffic at the exact spike peak (minute 60)
        peak_traffic = df.loc[df["minute"] == spike_time, "viewers"].values[0]

        # The spike peak should exceed baseline by at least 80% of spike magnitude
        spike_increase = peak_traffic - baseline_mean
        self.assertGreater(
            spike_increase,
            spike_magnitude * 0.8,
            f"Spike increase ({spike_increase}) is less than 80% of magnitude ({spike_magnitude})",
        )

        # Confirm event_active is True around spike time
        event_active_at_peak = df.loc[
            df["minute"] == spike_time, "event_active"
        ].values[0]
        self.assertTrue(
            event_active_at_peak,
            "event_active must be True at configured spike_time",
        )

    # Test 11: The peak-event scenario reaches a higher peak than its starting traffic.
    def test_peak_event_reaches_higher_peak(self) -> None:
        """Verify peak-event scenario significantly rises from starting viewership."""
        initial_viewers = 2000
        peak_viewers = 55000
        df = generate_peak_event_traffic(
            initial_viewers=initial_viewers,
            peak_viewers=peak_viewers,
            peak_start=50,
            peak_end=70,
            random_seed=42,
        )

        start_viewers = df.loc[df["minute"] == 0, "viewers"].values[0]
        max_viewers = df["viewers"].max()

        self.assertGreater(
            max_viewers,
            start_viewers * 10,
            f"Peak viewers ({max_viewers}) should be much higher than starting ({start_viewers})",
        )
        # Peak should be close to configured peak_viewers (within 5% margin for noise)
        self.assertAlmostEqual(max_viewers, peak_viewers, delta=peak_viewers * 0.05)

        # Confirm event_active is True during peak window and False at beginning
        self.assertFalse(df.loc[df["minute"] == 0, "event_active"].values[0])
        self.assertTrue(df.loc[df["minute"] == 60, "event_active"].values[0])

    # Test 12: The event-ending scenario finishes with fewer viewers than it starts with.
    def test_event_ending_finishes_with_fewer_viewers(self) -> None:
        """Verify viewership at the end is substantially lower than starting viewership."""
        df = generate_event_ending_traffic(
            initial_viewers=50000,
            final_viewers=3000,
            hold_duration=15,
            random_seed=42,
        )

        start_viewers = df.loc[df["minute"] == 0, "viewers"].values[0]
        end_viewers = df.loc[
            df["minute"] == DEFAULT_DURATION_MINUTES - 1, "viewers"
        ].values[0]

        self.assertLess(
            end_viewers,
            start_viewers,
            f"Final viewers ({end_viewers}) must be less than start ({start_viewers})",
        )
        self.assertLess(
            end_viewers,
            start_viewers * 0.15,
            f"Final viewers ({end_viewers}) should drop to under 15% of starting ({start_viewers})",
        )

    # Additional tests for parameter customization and invalid input validation
    def test_custom_duration_and_interval(self) -> None:
        """Verify scenario generation with custom duration and step interval."""
        custom_df = generate_normal_traffic(duration_minutes=60, time_interval=2)
        self.assertEqual(len(custom_df), 30)
        self.assertEqual(custom_df["minute"].iloc[-1], 58)

    def test_invalid_configurations_raise_value_error(self) -> None:
        """Verify invalid configuration arguments trigger descriptive ValueErrors."""
        with self.assertRaises(ValueError):
            generate_normal_traffic(duration_minutes=-10)

        with self.assertRaises(ValueError):
            generate_normal_traffic(time_interval=0)

        with self.assertRaises(ValueError):
            generate_normal_traffic(min_viewers=-1)

        with self.assertRaises(ValueError):
            generate_normal_traffic(min_viewers=5000, max_viewers=2000)

        with self.assertRaises(ValueError):
            generate_normal_traffic(
                initial_viewers=10000, min_viewers=0, max_viewers=5000
            )

        with self.assertRaises(ValueError):
            generate_sudden_spike_traffic(spike_sigma=-1.0)

        with self.assertRaises(ValueError):
            generate_peak_event_traffic(peak_viewers=1000, initial_viewers=2000)

        with self.assertRaises(ValueError):
            generate_peak_event_traffic(peak_start=80, peak_end=40)

        with self.assertRaises(ValueError):
            generate_event_ending_traffic(
                initial_viewers=1000, final_viewers=5000
            )

    def test_event_active_dtype_is_bool(self) -> None:
        """Verify event_active column has boolean dtype across all scenarios."""
        for name, df in self.scenarios.items():
            with self.subTest(scenario=name):
                self.assertEqual(
                    df["event_active"].dtype,
                    bool,
                    f"Scenario '{name}' event_active dtype is not bool",
                )


if __name__ == "__main__":
    unittest.main()
