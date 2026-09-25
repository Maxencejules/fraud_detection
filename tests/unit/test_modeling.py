from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import DecisionThresholds
from fraud_detection.evaluation import (
    binary_metrics,
    fit_logit_calibration,
    precision_recall_table,
    reliability_table,
    temporal_split,
)
from fraud_detection.features import FEATURE_COLUMNS
from fraud_detection.modeling import EnsembleScorer, LogitCalibrator, to_matrix
from fraud_detection.trainer import MODEL_CODE_PATH


class TestLogitCalibrator:
    def test_identity_leaves_probabilities_unchanged(self) -> None:
        scores = np.array([0.001, 0.2, 0.5, 0.97])
        np.testing.assert_allclose(LogitCalibrator.identity()(scores), scores, rtol=1e-9)

    def test_preserves_ranking(self) -> None:
        scores = np.random.default_rng(0).random(1_000)
        calibrated = LogitCalibrator(slope=0.7, intercept=-1.3)(scores)
        assert (np.argsort(calibrated, kind="stable") == np.argsort(scores, kind="stable")).all()
        assert len(np.unique(calibrated)) == len(np.unique(scores))

    def test_json_round_trip(self) -> None:
        calibrator = LogitCalibrator(slope=0.98, intercept=0.27)
        assert LogitCalibrator.from_json(calibrator.to_json()) == calibrator

    def test_rejects_unknown_format(self) -> None:
        with pytest.raises(ValueError, match="unsupported calibrator"):
            LogitCalibrator.from_json('{"x": [0, 1], "y": [0, 1]}')

    @pytest.mark.parametrize(("slope", "intercept"), [(0.0, 0.0), (-1.0, 0.0), (float("nan"), 0.0)])
    def test_rejects_non_monotone_or_invalid_parameters(
        self, slope: float, intercept: float
    ) -> None:
        with pytest.raises(ValueError, match="calibrator"):
            LogitCalibrator(slope, intercept)


class TestEnsembleScorer:
    def test_probabilities_are_bounded(self, tiny_scorer: EnsembleScorer) -> None:
        features = np.random.default_rng(1).random((50, len(FEATURE_COLUMNS)))
        probabilities = tiny_scorer.predict_proba(features)
        assert probabilities.shape == (50,)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

    def test_rejects_wrong_shape(self, tiny_scorer: EnsembleScorer) -> None:
        with pytest.raises(ValueError, match="matrix"):
            tiny_scorer.predict_proba(np.zeros((3, 4)))

    def test_native_artifacts_round_trip(self, tiny_scorer: EnsembleScorer, tmp_path: Path) -> None:
        features = np.random.default_rng(2).random((20, len(FEATURE_COLUMNS)))

        restored = EnsembleScorer.load(tiny_scorer.save(tmp_path), FEATURE_COLUMNS, n_threads=1)

        np.testing.assert_allclose(
            restored.predict_proba(features), tiny_scorer.predict_proba(features)
        )

    def test_to_matrix_orders_columns_and_converts_booleans(self) -> None:
        frame = pd.DataFrame({c: [1] for c in reversed(FEATURE_COLUMNS)}).assign(card_present=True)
        matrix = to_matrix(frame, FEATURE_COLUMNS)
        assert matrix.dtype == np.float64
        assert matrix.shape == (1, len(FEATURE_COLUMNS))

    def test_to_matrix_reports_missing_columns(self) -> None:
        with pytest.raises(ValueError, match="missing feature columns"):
            to_matrix(pd.DataFrame({"amount": [1.0]}), FEATURE_COLUMNS)


def test_fraud_model_loads_through_mlflow_models_from_code(
    tiny_scorer: EnsembleScorer, tmp_path: Path
) -> None:
    model_path = tmp_path / "model"
    mlflow.pyfunc.save_model(
        str(model_path),
        python_model=str(MODEL_CODE_PATH),
        artifacts=tiny_scorer.save(tmp_path / "artifacts"),
        model_config={"feature_columns": list(FEATURE_COLUMNS), "n_threads": 1},
        pip_requirements=["numpy"],
    )
    frame = pd.DataFrame(
        np.random.default_rng(3).random((5, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS)
    )

    loaded = mlflow.pyfunc.load_model(str(model_path))

    np.testing.assert_allclose(
        loaded.predict(frame), tiny_scorer.predict_proba(to_matrix(frame, FEATURE_COLUMNS))
    )


def _labelled_frame(n: int = 1_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "event_time": np.sort(rng.uniform(0, 30 * 86_400, n)),
            "label": (rng.random(n) < 0.1).astype(int),
        }
    )


class TestTemporalSplit:
    def test_splits_are_ordered_in_time(self) -> None:
        split = temporal_split(
            _labelled_frame(), warmup_days=0, validation_fraction=0.2, test_fraction=0.2
        )
        assert split.train["event_time"].max() <= split.validation["event_time"].min()
        assert split.validation["event_time"].max() <= split.test["event_time"].min()
        assert (len(split.train), len(split.validation), len(split.test)) == (600, 200, 200)

    def test_warmup_period_is_dropped(self) -> None:
        split = temporal_split(
            _labelled_frame(), warmup_days=10, validation_fraction=0.2, test_fraction=0.2
        )
        assert split.train["event_time"].min() >= 10 * 86_400

    def test_requires_both_classes_in_every_split(self) -> None:
        frame = _labelled_frame().assign(label=0)
        with pytest.raises(ValueError, match="both classes"):
            temporal_split(frame, warmup_days=0, validation_fraction=0.2, test_fraction=0.2)


class TestCalibrationAndMetrics:
    def test_logit_calibration_recovers_a_known_distortion(self) -> None:
        rng = np.random.default_rng(0)
        true_probability = rng.beta(0.5, 8.0, 50_000)
        labels = (rng.random(50_000) < true_probability).astype(int)
        # Scores that are systematically over-confident: logit scaled by 2, shifted by +1.
        logits = np.log(true_probability / (1 - true_probability))
        scores = 1 / (1 + np.exp(-(2 * logits + 1)))

        calibrator = fit_logit_calibration(scores, labels)

        assert calibrator.slope == pytest.approx(0.5, abs=0.05)
        assert calibrator.intercept == pytest.approx(-0.5, abs=0.1)

    def test_uninformative_scores_fall_back_to_identity(self) -> None:
        rng = np.random.default_rng(1)
        scores = rng.random(2_000)
        labels = (rng.random(2_000) < 0.1).astype(int)
        labels[np.argsort(scores)[-200:]] = 0  # anti-correlated: a negative slope

        assert fit_logit_calibration(scores, labels) == LogitCalibrator.identity()

    def test_binary_metrics_at_operating_points(self) -> None:
        y = np.array([0, 0, 0, 1, 1])
        p = np.array([0.01, 0.2, 0.95, 0.5, 0.99])

        metrics = binary_metrics(y, p, DecisionThresholds(review=0.1, block=0.9))

        assert metrics["base_rate"] == pytest.approx(0.4)
        assert metrics["review_precision"] == pytest.approx(2 / 4)  # 4 flagged, 2 are fraud
        assert metrics["review_recall"] == 1.0
        assert metrics["block_precision"] == pytest.approx(1 / 2)
        assert metrics["block_flag_rate"] == pytest.approx(2 / 5)
        assert 0.0 <= metrics["pr_auc"] <= 1.0

    def test_tables(self) -> None:
        rng = np.random.default_rng(4)
        p = rng.random(500)
        y = (rng.random(500) < p).astype(int)

        assert len(reliability_table(y, p, bins=5)) == 5
        assert set(precision_recall_table(y, p, max_points=50).columns) == {
            "threshold",
            "precision",
            "recall",
        }


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    from sklearn.metrics import average_precision_score

    from fraud_detection.evaluation import bootstrap_interval

    rng = np.random.default_rng(5)
    groups = rng.integers(0, 300, 3_000)
    p = rng.random(3_000)
    y = (rng.random(3_000) < p * 0.3).astype(int)

    low, high = bootstrap_interval(y, p, groups, resamples=100)

    assert low < average_precision_score(y, p) < high
    assert 0.0 <= low < high <= 1.0
