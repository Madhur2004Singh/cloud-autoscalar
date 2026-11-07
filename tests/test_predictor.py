"""Unit tests for the Stage B ML-Based Workload Prediction Engine."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from models.predictor import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTION_HORIZON,
    FEATURE_COLUMNS,
    MIN_HISTORICAL_OBSERVATIONS,
    engineer_features,
    load_model,
    predict_next_viewers,
    prepare_training_data,
    save_model,
    train_random_forest,
)
from simulation.workload_generator import generate_all_scenarios
from train_model import run_sudden_spike_stress_evaluation


class TestPredictor(unittest.TestCase):
    """Test suite covering Stage B ML prediction engine requirements."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate scenarios and prepare train/test splits once for the suite."""
        cls.raw_scenarios = generate_all_scenarios(duration_minutes=120, random_seed=42)
        cls.horizon = DEFAULT_PREDICTION_HORIZON
        (
            cls.X_train,
            cls.y_train,
            cls.X_test,
            cls.y_test,
            cls.meta_train,
            cls.meta_test,
        ) = prepare_training_data(
            cls.raw_scenarios, prediction_horizon=cls.horizon, train_ratio=0.8
        )

    # Test 1: Correct feature columns.
    def test_correct_feature_columns(self) -> None:
        """Verify feature engineering produces all 9 expected feature columns."""
        sample_df = self.raw_scenarios["normal"]
        features_df = engineer_features(sample_df, prediction_horizon=self.horizon)

        for col in FEATURE_COLUMNS:
            self.assertIn(
                col,
                features_df.columns,
                f"Engineered features missing expected column: '{col}'",
            )

        self.assertEqual(
            list(self.X_train.columns),
            FEATURE_COLUMNS,
            "X_train feature columns do not match expected FEATURE_COLUMNS specification.",
        )

    # Test 2: Correct target alignment for the 5-minute prediction horizon.
    def test_target_alignment(self) -> None:
        """Verify target at minute t exactly equals viewers at minute t + horizon."""
        sample_df = self.raw_scenarios["peak_event"].copy()
        feat_df = engineer_features(sample_df, prediction_horizon=self.horizon)

        # For rows where target is not NaN, check exact alignment with future viewers
        valid_rows = feat_df.dropna(subset=["target"])
        for idx, row in valid_rows.iterrows():
            current_minute = int(row["minute"])
            expected_target_minute = current_minute + self.horizon
            actual_future_viewers = sample_df.loc[
                sample_df["minute"] == expected_target_minute, "viewers"
            ].values[0]
            self.assertEqual(
                row["target"],
                actual_future_viewers,
                f"Target at minute {current_minute} ({row['target']}) does not equal "
                f"viewers at minute {expected_target_minute} ({actual_future_viewers})",
            )

        # Test with a custom prediction horizon
        custom_horizon = 3
        custom_feat = engineer_features(sample_df, prediction_horizon=custom_horizon)
        val_custom = custom_feat.dropna(subset=["target"]).iloc[0]
        cur_min = int(val_custom["minute"])
        future_viewers = sample_df.loc[
            sample_df["minute"] == cur_min + custom_horizon, "viewers"
        ].values[0]
        self.assertEqual(val_custom["target"], future_viewers)

    # Test 3: No mixing of features across scenario boundaries.
    def test_no_mixing_across_scenario_boundaries(self) -> None:
        """Verify scenario boundaries prevent historical feature leakage."""
        df_a = pd.DataFrame(
            {
                "minute": range(10),
                "viewers": [10000 + i * 100 for i in range(10)],
                "event_active": [False] * 10,
                "scenario": ["ScenarioA"] * 10,
            }
        )
        df_b = pd.DataFrame(
            {
                "minute": range(10),
                "viewers": [50 + i * 2 for i in range(10)],
                "event_active": [False] * 10,
                "scenario": ["ScenarioB"] * 10,
            }
        )

        # Process individually vs via prepare_training_data
        feat_b_alone = engineer_features(df_b, prediction_horizon=2)
        X_tr, _, _, _, meta_tr, _ = prepare_training_data(
            [df_a, df_b], prediction_horizon=2, train_ratio=0.8
        )

        # Check Scenario B features in combined output match individual processing
        b_mask = meta_tr["scenario"] == "ScenarioB"
        combined_b_features = X_tr.loc[b_mask].reset_index(drop=True)

        expected_b_features = (
            feat_b_alone.dropna(subset=FEATURE_COLUMNS + ["target"])[FEATURE_COLUMNS]
            .iloc[: len(combined_b_features)]
            .reset_index(drop=True)
        )

        pd.testing.assert_frame_equal(
            combined_b_features,
            expected_b_features,
            obj="Scenario boundary isolation check",
        )

    # Test 4: No missing values in final training feature matrix.
    def test_no_missing_values(self) -> None:
        """Verify X_train, y_train, X_test, y_test contain zero NaN values."""
        self.assertEqual(
            self.X_train.isna().sum().sum(),
            0,
            "X_train contains unexpected NaN values.",
        )
        self.assertEqual(
            self.y_train.isna().sum(),
            0,
            "y_train contains unexpected NaN values.",
        )
        self.assertEqual(
            self.X_test.isna().sum().sum(),
            0,
            "X_test contains unexpected NaN values.",
        )
        self.assertEqual(
            self.y_test.isna().sum(),
            0,
            "y_test contains unexpected NaN values.",
        )

    # Test 5: Chronological train/test splitting and Purge Condition.
    def test_chronological_splitting(self) -> None:
        """Verify that training observations strictly precede test observations per scenario."""
        scenarios = self.meta_train["scenario"].unique()
        for sc in scenarios:
            train_minutes = self.meta_train.loc[
                self.meta_train["scenario"] == sc, "minute"
            ]
            test_minutes = self.meta_test.loc[
                self.meta_test["scenario"] == sc, "minute"
            ]

            max_train_min = train_minutes.max()
            min_test_min = test_minutes.min()

            self.assertLess(
                max_train_min,
                min_test_min,
                f"Scenario '{sc}' violated chronological split: "
                f"max train minute ({max_train_min}) >= min test minute ({min_test_min})",
            )

    def test_purged_split_condition(self) -> None:
        """Explicitly verify that max(training target) < min(test feature) for all scenarios."""
        scenarios = self.meta_train["scenario"].unique()
        for sc in scenarios:
            train_sub = self.meta_train[self.meta_train["scenario"] == sc]
            test_sub = self.meta_test[self.meta_test["scenario"] == sc]

            max_train_target = train_sub["target_minute"].max()
            min_test_feature = test_sub["minute"].min()

            self.assertLess(
                max_train_target,
                min_test_feature,
                f"Purge condition failed for '{sc}': "
                f"max train target minute ({max_train_target}) is not strictly less than "
                f"min test feature minute ({min_test_feature})!",
            )

            if "timestamp" in train_sub.columns and "target_timestamp" in train_sub.columns:
                max_train_target_ts = train_sub["target_timestamp"].max()
                min_test_feature_ts = test_sub["timestamp"].min()
                self.assertLess(
                    max_train_target_ts,
                    min_test_feature_ts,
                    f"Timestamp purge condition failed for '{sc}': "
                    f"max train target timestamp ({max_train_target_ts}) >= "
                    f"min test feature timestamp ({min_test_feature_ts})",
                )

    # Test 6: Correct model training.
    def test_model_training(self) -> None:
        """Verify model trains and produces predictions matching test size."""
        model = train_random_forest(
            self.X_train,
            self.y_train,
            n_estimators=50,  # lightweight for quick testing
            random_state=42,
        )
        self.assertIsInstance(model, RandomForestRegressor)
        self.assertTrue(hasattr(model, "estimators_"))
        self.assertEqual(len(model.estimators_), 50)

        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

    # Test 7: Successful model saving and loading.
    def test_model_save_and_load(self) -> None:
        """Verify model can be saved to disk and loaded back with identical predictions."""
        model = train_random_forest(
            self.X_train, self.y_train, n_estimators=30, random_state=42
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "test_model.joblib"
            save_model(model, filepath=tmp_path, prediction_horizon=self.horizon)
            self.assertTrue(tmp_path.is_file())

            loaded_model, meta = load_model(tmp_path)
            self.assertEqual(meta["prediction_horizon"], self.horizon)
            self.assertEqual(meta["feature_names"], FEATURE_COLUMNS)

            orig_preds = model.predict(self.X_test)
            loaded_preds = loaded_model.predict(self.X_test)
            np.testing.assert_array_almost_equal(orig_preds, loaded_preds)

    # Test 8: Prediction API returns a valid non-negative integer.
    def test_prediction_api_valid_output(self) -> None:
        """Verify predict_next_viewers returns a non-negative integer."""
        sample_history = self.raw_scenarios["normal"].iloc[:15].copy()
        pred = predict_next_viewers(sample_history, model_artifact_path=DEFAULT_MODEL_PATH)

        self.assertIsInstance(
            pred,
            (int, np.integer),
            f"Prediction should be an integer, got {type(pred)}",
        )
        self.assertGreaterEqual(pred, 0, f"Predicted viewers must be non-negative, got {pred}")

    # Test 9: Prediction API rejects insufficient or non-consecutive historical input.
    def test_prediction_api_insufficient_history(self) -> None:
        """Verify predict_next_viewers raises ValueError when history has fewer than 6 observations."""
        too_short_history = pd.DataFrame(
            {
                "minute": [0, 1, 2],
                "viewers": [1000, 1100, 1200],
                "event_active": [False, False, False],
            }
        )
        with self.assertRaises(ValueError) as ctx:
            predict_next_viewers(too_short_history, model_artifact_path=DEFAULT_MODEL_PATH)
        self.assertIn("Insufficient historical observations", str(ctx.exception))

        # Test missing required column
        missing_col_history = pd.DataFrame(
            {"minute": range(10), "wrong_col": [100] * 10}
        )
        with self.assertRaises(ValueError):
            predict_next_viewers(missing_col_history, model_artifact_path=DEFAULT_MODEL_PATH)

    def test_prediction_api_rejects_non_consecutive_history(self) -> None:
        """Verify predict_next_viewers raises ValueError when minute sequence has gaps."""
        non_consecutive_history = pd.DataFrame(
            {
                "minute": [10, 20, 30, 40, 50, 60],
                "viewers": [2000, 2100, 2200, 2300, 2400, 2500],
                "event_active": [False] * 6,
            }
        )
        with self.assertRaises(ValueError) as ctx:
            predict_next_viewers(non_consecutive_history, model_artifact_path=DEFAULT_MODEL_PATH)
        self.assertIn("consecutive", str(ctx.exception).lower())

    def test_prediction_api_handles_missing_event_active(self) -> None:
        """Verify predict_next_viewers operates cleanly when event_active column is omitted."""
        history_no_event = pd.DataFrame(
            {
                "minute": list(range(10, 18)),
                "viewers": [3000 + i * 50 for i in range(8)],
            }
        )
        pred = predict_next_viewers(history_no_event, model_artifact_path=DEFAULT_MODEL_PATH)
        self.assertIsInstance(pred, (int, np.integer))
        self.assertGreaterEqual(pred, 0)

    # Test 10: Changing future viewer values does not change historical features at earlier prediction time.
    def test_no_future_data_leakage(self) -> None:
        """Verify that altering future observations does not affect features at time t."""
        base_df = pd.DataFrame(
            {
                "minute": range(12),
                "viewers": [2000 + i * 50 for i in range(12)],
                "event_active": [False] * 12,
            }
        )

        # Baseline features at minute 5
        feat_base = engineer_features(base_df, include_target=False)
        features_at_min_5_original = feat_base.loc[feat_base["minute"] == 5, FEATURE_COLUMNS].to_dict()

        # Create modified future: massively change viewers at minutes 6 through 11
        modified_df = base_df.copy()
        modified_df.loc[modified_df["minute"] >= 6, "viewers"] = [999999, 888888, 777777, 666666, 555555, 444444]
        modified_df.loc[modified_df["minute"] >= 6, "event_active"] = True

        feat_modified = engineer_features(modified_df, include_target=False)
        features_at_min_5_modified = feat_modified.loc[feat_modified["minute"] == 5, FEATURE_COLUMNS].to_dict()

        self.assertEqual(
            features_at_min_5_original,
            features_at_min_5_modified,
            "Altering future data leaked into and altered historical features at minute 5!",
        )

    # Test 11: Sudden-Spike Stress Evaluation execution.
    def test_sudden_spike_stress_evaluation(self) -> None:
        """Verify sudden-spike stress evaluation executes and returns valid metrics."""
        res = run_sudden_spike_stress_evaluation(self.raw_scenarios, prediction_horizon=self.horizon)
        self.assertIn("rf_mae", res)
        self.assertIn("base_mae", res)
        self.assertIn("rf_rmse", res)
        self.assertIn("base_rmse", res)
        self.assertEqual(res["n_samples"], 26)  # minutes 45 to 70 inclusive
        self.assertGreater(res["rf_mae"], 0)
        self.assertGreater(res["base_mae"], 0)


if __name__ == "__main__":
    unittest.main()
