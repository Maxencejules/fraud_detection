"""Training-time evaluation: out-of-time splits, calibration fitting and metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from fraud_detection.config import DecisionThresholds
from fraud_detection.modeling import LogitCalibrator

DAY = 86_400.0


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    time_column: str = "event_time"

    def describe(self) -> dict[str, float]:
        description: dict[str, float] = {}
        for name, part in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            description[f"{name}_rows"] = float(len(part))
            description[f"{name}_fraud_rate"] = float(part["label"].mean())
            description[f"{name}_start"] = float(part[self.time_column].min())
            description[f"{name}_end"] = float(part[self.time_column].max())
        return description


def _binary_labels(labels: np.ndarray, *, both_classes: bool = True) -> np.ndarray:
    try:
        y = np.asarray(labels, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be a finite binary vector") from exc
    if y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be a nonempty finite binary vector containing only 0/1")
    if both_classes and np.unique(y).size != 2:
        raise ValueError("labels need both classes for ranking and calibration")
    return y.astype(int)


def _binary_data(
    labels: np.ndarray, probability: np.ndarray, *, both_classes: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    y = _binary_labels(labels, both_classes=both_classes)
    try:
        p = np.asarray(probability, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must be a finite vector in [0, 1]") from exc
    if p.ndim != 1 or len(p) != len(y):
        raise ValueError("probabilities and labels must be equally sized one-dimensional vectors")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    return y, p


def temporal_split(
    frame: pd.DataFrame,
    *,
    warmup_days: float,
    validation_fraction: float,
    test_fraction: float,
    time_column: str = "event_time",
) -> TemporalSplit:
    """Split by event time: train on the past, validate and test on the future.

    Random splits leak information across time (e.g. one fraud burst lands in both
    train and test), which inflates offline metrics. The first ``warmup_days`` are
    dropped because rolling windows are still filling up there. Fractions target row
    counts; every boundary moves left to keep its entire timestamp group in the later
    period. Actual fractions can therefore differ on coarse timestamps.
    """
    if not math.isfinite(warmup_days) or warmup_days < 0:
        raise ValueError("warmup_days must be finite and nonnegative")
    if (
        not math.isfinite(validation_fraction)
        or not math.isfinite(test_fraction)
        or validation_fraction <= 0
        or test_fraction <= 0
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError(
            "validation/test fractions must be positive, finite and sum to less than 1"
        )
    if time_column not in frame or "label" not in frame:
        raise ValueError(f"dataset needs {time_column!r} and 'label' columns")
    times = frame[time_column].to_numpy()
    if not pd.api.types.is_numeric_dtype(times) or not np.isfinite(times).all():
        raise ValueError(f"{time_column} must contain finite numeric event times")
    _binary_labels(frame["label"].to_numpy(), both_classes=False)
    if "transaction_id" in frame and (
        frame["transaction_id"].isna().any() or frame["transaction_id"].duplicated().any()
    ):
        raise ValueError("transaction_id must be nonmissing and unique; duplicate events leak")
    ordered = frame.sort_values(time_column, kind="stable")
    if warmup_days > 0:
        cutoff = ordered[time_column].min() + warmup_days * DAY
        if not math.isfinite(cutoff):
            raise ValueError("warmup cutoff must be finite")
        ordered = ordered[ordered[time_column] >= cutoff]
    n = len(ordered)
    n_test = round(n * test_fraction)
    n_validation = round(n * validation_fraction)
    n_train = n - n_validation - n_test
    if n_train <= 0 or n_validation <= 0 or n_test <= 0:
        raise ValueError("every temporal split needs rows and both classes")
    times = ordered[time_column].to_numpy()
    train_end = int(np.searchsorted(times, times[n_train], side="left"))
    validation_end = int(np.searchsorted(times, times[n_train + n_validation], side="left"))
    split = TemporalSplit(
        train=ordered.iloc[:train_end],
        validation=ordered.iloc[train_end:validation_end],
        test=ordered.iloc[validation_end:],
        time_column=time_column,
    )
    for name, part in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        if part["label"].nunique() < 2:
            raise ValueError(f"{name} split needs both classes; got {len(part)} rows")
    return split


def fit_logit_calibration(scores: np.ndarray, labels: np.ndarray) -> LogitCalibrator:
    """Fit Platt scaling on held-out scores (unregularised logistic regression)."""
    y, p = _binary_data(labels, scores)
    # Only fitting needs finite endpoint logits; preserve the original scores in
    # ranking metrics and in the endpoint-aware serving transformation.
    epsilon = np.finfo(np.float64).eps
    p = np.clip(p, epsilon, 1 - epsilon)
    logits = (np.log(p) - np.log1p(-p)).reshape(-1, 1)
    regression = LogisticRegression(C=1e6, max_iter=1_000).fit(logits, y)
    slope, intercept = float(regression.coef_[0][0]), float(regression.intercept_[0])
    if slope <= 0:  # degenerate fit (uninformative scores): keep the raw scale
        return LogitCalibrator.identity()
    return LogitCalibrator(slope, intercept)


def binary_metrics(
    y_true: np.ndarray, probability: np.ndarray, thresholds: DecisionThresholds
) -> dict[str, float]:
    """Ranking, calibration and operating-point metrics for a fraud score."""
    y, p = _binary_data(y_true, probability)
    fpr, tpr, _ = roc_curve(y, p)
    metrics = {
        "rows": float(len(y)),
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "recall_at_1pct_fpr": float(tpr[fpr <= 0.01].max(initial=0.0)),
    }
    for name, threshold in (("review", thresholds.review), ("block", thresholds.block)):
        flagged = p >= threshold
        true_positive = float(np.sum(flagged & (y == 1)))
        precision = true_positive / flagged.sum() if flagged.any() else 0.0
        recall = true_positive / y.sum() if y.any() else 0.0
        metrics[f"{name}_precision"] = precision
        metrics[f"{name}_recall"] = recall
        metrics[f"{name}_flag_rate"] = float(flagged.mean())
    return metrics


@dataclass(frozen=True)
class ClusterBootstrapResult:
    low: float
    high: float
    requested_resamples: int
    valid_resamples: int
    users: int
    confidence: float


def bootstrap_result(
    y_true: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    *,
    resamples: int = 200,
    confidence: float = 0.95,
    seed: int = 0,
) -> ClusterBootstrapResult:
    """Cluster-bootstrap confidence interval for PR-AUC.

    Fraud arrives in per-user bursts, so rows are not independent: resampling rows
    would understate the uncertainty. Whole users are resampled instead. Replicates
    with only one class are omitted, so the percentile interval is conditional on
    having both classes; the valid and requested counts are reported explicitly.
    """
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and strictly between 0 and 1")
    y, p = _binary_data(y_true, probability)
    group_values = np.asarray(groups)
    if group_values.ndim != 1 or len(group_values) != len(y) or pd.isna(group_values).any():
        raise ValueError("groups must be a nonmissing vector with one user per label")
    _, inverse = np.unique(group_values, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    members = np.split(order, boundaries)
    if len(members) < 2:
        raise ValueError("cluster uncertainty needs at least two users")
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(members), len(members))
        index = np.concatenate([members[g] for g in chosen])
        if 0 < y[index].sum() < len(index):
            scores.append(average_precision_score(y[index], p[index]))
    alpha = (1.0 - confidence) / 2
    if len(scores) < 2:
        raise ValueError("cluster interval needs at least two valid two-class replicates")
    low, high = np.quantile(scores, [alpha, 1.0 - alpha])
    return ClusterBootstrapResult(
        float(low), float(high), resamples, len(scores), len(members), confidence
    )


def bootstrap_interval(
    y_true: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    *,
    resamples: int = 200,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Compatibility wrapper for the bounds; ``bootstrap_result`` also reports sample counts."""
    result = bootstrap_result(
        y_true, probability, groups, resamples=resamples, confidence=confidence, seed=seed
    )
    return result.low, result.high


def reliability_table(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Observed fraud rate per predicted-probability bin (quantile bins)."""
    if bins < 1:
        raise ValueError("bins must be positive")
    y, p = _binary_data(y_true, probability, both_classes=False)
    frame = pd.DataFrame({"label": y, "probability": p})
    if np.unique(p).size == 1:
        frame["bin"] = 0
    else:
        frame["bin"] = pd.qcut(frame["probability"], q=bins, duplicates="drop")
    table = frame.groupby("bin", observed=True).agg(
        mean_probability=("probability", "mean"),
        observed_rate=("label", "mean"),
        rows=("label", "size"),
    )
    return table.reset_index(drop=True)


def precision_recall_table(
    y_true: np.ndarray, probability: np.ndarray, max_points: int = 200
) -> pd.DataFrame:
    if max_points < 1:
        raise ValueError("max_points must be positive")
    y, p = _binary_data(y_true, probability)
    precision, recall, thresholds = precision_recall_curve(y, p)
    table = pd.DataFrame(
        {"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]}
    )
    if len(table) > max_points:
        table = table.iloc[np.linspace(0, len(table) - 1, max_points).astype(int)]
    return table.reset_index(drop=True)
