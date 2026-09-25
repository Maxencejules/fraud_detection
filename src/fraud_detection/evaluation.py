"""Training-time evaluation: out-of-time splits, calibration fitting and metrics."""

from __future__ import annotations

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

    def describe(self) -> dict[str, float]:
        description: dict[str, float] = {}
        for name, part in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            description[f"{name}_rows"] = float(len(part))
            description[f"{name}_fraud_rate"] = float(part["label"].mean())
            description[f"{name}_start"] = float(part["event_time"].min())
            description[f"{name}_end"] = float(part["event_time"].max())
        return description


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
    dropped because rolling windows are still filling up there.
    """
    ordered = frame.sort_values(time_column, kind="stable")
    if warmup_days > 0:
        cutoff = ordered[time_column].min() + warmup_days * DAY
        ordered = ordered[ordered[time_column] >= cutoff]
    n = len(ordered)
    n_test = round(n * test_fraction)
    n_validation = round(n * validation_fraction)
    n_train = n - n_validation - n_test
    split = TemporalSplit(
        train=ordered.iloc[:n_train],
        validation=ordered.iloc[n_train : n_train + n_validation],
        test=ordered.iloc[n_train + n_validation :],
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
    p = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1 - 1e-7)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    regression = LogisticRegression(C=1e6, max_iter=1_000).fit(logits, np.asarray(labels))
    slope, intercept = float(regression.coef_[0][0]), float(regression.intercept_[0])
    if slope <= 0:  # degenerate fit (uninformative scores): keep the raw scale
        return LogitCalibrator.identity()
    return LogitCalibrator(slope, intercept)


def binary_metrics(
    y_true: np.ndarray, probability: np.ndarray, thresholds: DecisionThresholds
) -> dict[str, float]:
    """Ranking, calibration and operating-point metrics for a fraud score."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
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


def bootstrap_interval(
    y_true: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    *,
    resamples: int = 200,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Cluster-bootstrap confidence interval for PR-AUC.

    Fraud arrives in per-user bursts, so rows are not independent: resampling rows
    would understate the uncertainty. Whole users are resampled instead.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    members = np.split(order, boundaries)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(members), len(members))
        index = np.concatenate([members[g] for g in chosen])
        if 0 < y[index].sum() < len(index):
            scores.append(average_precision_score(y[index], p[index]))
    alpha = (1.0 - confidence) / 2
    low, high = np.quantile(scores, [alpha, 1.0 - alpha])
    return float(low), float(high)


def reliability_table(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Observed fraud rate per predicted-probability bin (quantile bins)."""
    frame = pd.DataFrame({"label": np.asarray(y_true, dtype=int), "probability": probability})
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
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    table = pd.DataFrame(
        {"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]}
    )
    if len(table) > max_points:
        table = table.iloc[np.linspace(0, len(table) - 1, max_points).astype(int)]
    return table.reset_index(drop=True)
