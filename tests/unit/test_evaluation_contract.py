"""Independent small cases for chronology, ranking and cluster uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import DecisionThresholds
from fraud_detection.evaluation import (
    binary_metrics,
    bootstrap_interval,
    reliability_table,
    temporal_split,
)
from fraud_detection.modeling import LogitCalibrator


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [f"tx-{i}" for i in range(72)],
            "event_time": np.repeat(np.arange(12, dtype=float), 6),
            "label": np.tile([0, 1], 36),
        }
    )


def test_equal_timestamps_never_straddle_temporal_boundaries() -> None:
    frame = _frame().sample(frac=1, random_state=4)
    split = temporal_split(frame, warmup_days=0, validation_fraction=0.2, test_fraction=0.2)
    assert split.train.event_time.max() < split.validation.event_time.min()
    assert split.validation.event_time.max() < split.test.event_time.min()
    observed = pd.concat([split.train, split.validation, split.test])
    assert sorted(observed.transaction_id) == sorted(frame.transaction_id)


@pytest.mark.parametrize(
    ("validation", "test", "warmup"),
    [
        (0.0, 0.2, 0.0),
        (-0.1, 0.2, 0.0),
        (0.6, 0.5, 0.0),
        (0.2, 0.2, -1.0),
        (float("nan"), 0.2, 0.0),
        (0.2, float("inf"), 0.0),
    ],
)
def test_temporal_split_rejects_invalid_period_settings(
    validation: float, test: float, warmup: float
) -> None:
    with pytest.raises(ValueError, match=r"fraction|warmup"):
        temporal_split(
            _frame(), warmup_days=warmup, validation_fraction=validation, test_fraction=test
        )


@pytest.mark.parametrize("bad_time", [float("nan"), float("inf"), -float("inf")])
def test_temporal_split_rejects_nonfinite_time(bad_time: float) -> None:
    frame = _frame()
    frame.loc[0, "event_time"] = bad_time
    with pytest.raises(ValueError, match=r"event_time|time"):
        temporal_split(frame, warmup_days=0, validation_fraction=0.2, test_fraction=0.2)


def test_temporal_split_rejects_repeated_transaction_identity() -> None:
    frame = _frame()
    frame.loc[71, "transaction_id"] = frame.loc[0, "transaction_id"]
    with pytest.raises(ValueError, match=r"transaction_id|duplicate"):
        temporal_split(frame, warmup_days=0, validation_fraction=0.2, test_fraction=0.2)


def test_temporal_split_rejects_nonbinary_labels() -> None:
    frame = _frame()
    frame["label"] = frame.label.astype(float)
    frame.loc[0, "label"] = 0.5
    with pytest.raises(ValueError, match=r"label|binary"):
        temporal_split(frame, warmup_days=0, validation_fraction=0.2, test_fraction=0.2)


def test_valid_tiny_probabilities_keep_their_ranking_and_decisions() -> None:
    metrics = binary_metrics(
        np.array([0, 1]), np.array([0.0, 1e-9]), DecisionThresholds(review=1e-10, block=0.9)
    )
    # Independent oracle: the single positive outranks the negative, and only it
    # crosses the review threshold. Clipping both values to 1e-7 creates a false tie.
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["review_precision"] == 1.0
    assert metrics["review_flag_rate"] == 0.5


def test_exact_endpoint_probabilities_have_zero_brier_error() -> None:
    metrics = binary_metrics(np.array([0, 1]), np.array([0.0, 1.0]), DecisionThresholds())
    assert metrics["brier"] == 0.0
    assert np.isfinite(metrics["log_loss"])


def test_identity_calibration_keeps_tiny_probabilities_and_endpoints() -> None:
    p = np.array([0.0, 1e-15, 1e-9, 0.4, 1.0])
    np.testing.assert_array_equal(LogitCalibrator.identity()(p), p)


def test_calibration_does_not_create_an_arbitrary_probability_floor() -> None:
    p = np.array([0.0, 1e-15, 1e-9, 0.4, 1.0])
    result = LogitCalibrator(0.5, -1.0)(p)
    assert np.isfinite(result).all()
    assert (np.diff(result) > 0).all()
    assert result[0] == 0.0
    assert result[-1] == 1.0


@pytest.mark.parametrize(
    "probability",
    [[-0.1, 0.7], [0.3, 1.1], [0.3, float("inf")], [0.3, float("nan")]],
)
def test_metrics_reject_invalid_probabilities(probability: list[float]) -> None:
    with pytest.raises(ValueError, match="probabilit"):
        binary_metrics(np.array([0, 1]), np.array(probability), DecisionThresholds())


@pytest.mark.parametrize("labels", [[0.9, 1.0], [0.0, 2.0], [0.0, 0.0]])
def test_metrics_reject_nonbinary_or_undefined_labels(labels: list[float]) -> None:
    with pytest.raises(ValueError, match=r"label|class|binary"):
        binary_metrics(np.array(labels), np.array([0.1, 0.9]), DecisionThresholds())


def test_reliability_report_retains_constant_score_rows() -> None:
    table = reliability_table(np.array([0, 1, 0]), np.array([0.2, 0.2, 0.2]))
    assert table.rows.sum() == 3
    assert len(table) == 1
    assert table.iloc[0].mean_probability == pytest.approx(0.2)
    assert table.iloc[0].observed_rate == pytest.approx(1 / 3)


def _independent_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Sum precision times incremental recall at each distinct threshold."""
    total_positives = int(labels.sum())
    previous = 0
    result = 0.0
    for threshold in sorted(set(scores), reverse=True):
        selected = scores >= threshold
        positives = int(labels[selected].sum())
        result += (positives - previous) / total_positives * positives / int(selected.sum())
        previous = positives
    return result


def test_bootstrap_resamples_whole_unequal_user_histories() -> None:
    y = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.7, 0.4, 0.9, 0.6, 0.5, 0.2, 0.5])
    groups = np.array(["b", "b", "a", "a", "a", "c", "c", "c", "c"])
    users = sorted(set(groups))
    rng = np.random.default_rng(19)
    estimates = []
    for _ in range(40):
        chosen = rng.choice(users, size=len(users), replace=True)
        # Repeated users duplicate every transaction, not just a random subset.
        indices = np.concatenate([np.flatnonzero(groups == user) for user in chosen])
        estimates.append(_independent_average_precision(y[indices], p[indices]))
    expected = np.quantile(estimates, [0.025, 0.975])
    actual = bootstrap_interval(y, p, groups, resamples=40, seed=19)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("resamples", [0, -1])
def test_bootstrap_rejects_invalid_resample_count(resamples: int) -> None:
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_interval(
            np.array([0, 1]), np.array([0.1, 0.9]), np.array(["a", "a"]), resamples=resamples
        )


@pytest.mark.parametrize("confidence", [0.0, 1.0, float("nan")])
def test_bootstrap_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_interval(
            np.array([0, 1]), np.array([0.1, 0.9]), np.array(["a", "a"]), confidence=confidence
        )


def test_bootstrap_reports_no_valid_replicates_explicitly() -> None:
    with pytest.raises(ValueError, match=r"valid|class"):
        bootstrap_interval(
            np.array([0, 0, 1, 1]),
            np.array([0.1, 0.2, 0.8, 0.9]),
            np.array(["a", "a", "b", "b"]),
            resamples=1,
            seed=0,
        )
