"""Bounded synthetic training/serialization evidence, without running any services."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Sequence
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import fakeredis
import numpy as np
import pandas as pd

from fraud_detection.bootstrap import replay
from fraud_detection.config import DecisionThresholds
from fraud_detection.evaluation import (
    TemporalSplit,
    binary_metrics,
    bootstrap_result,
    temporal_split,
)
from fraud_detection.features import FEATURE_COLUMNS, FeatureEngineer, FeatureStoreConfig
from fraud_detection.modeling import EnsembleScorer, to_matrix
from fraud_detection.simulation import DAY, SimulationConfig, TransactionSimulator
from fraud_detection.trainer import train_ensemble

DEFAULT_END = 1_767_225_600.0 + 40 * DAY


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")


def validate_offline(
    output: Path,
    *,
    users: int = 400,
    merchants: int = 120,
    days: float = 28,
    seed: int = 42,
    end: float = DEFAULT_END,
) -> dict[str, Any]:
    """Train twice; flipping evaluation labels must not affect any fitted component.

    Simulation and feature-store dataclasses are constructed explicitly, so runtime
    service environment variables cannot silently change the experiment.
    """
    if users < 10 or merchants < 10 or not np.isfinite(days) or days <= 5:
        raise ValueError("need at least 10 users/merchants and more than 5 history days")
    if not np.isfinite(end) or end <= days * DAY:
        raise ValueError("end must be a finite Unix timestamp after the history duration")
    output.mkdir(parents=True, exist_ok=True)
    simulation = SimulationConfig(seed=seed, n_users=users, n_merchants=merchants)
    feature_config = FeatureStoreConfig()
    frame = replay(
        FeatureEngineer(fakeredis.FakeRedis(), feature_config),
        TransactionSimulator(simulation, stream="history").events(end - days * DAY, end),
        batch_size=1_000,
    )
    data_bytes = _csv_bytes(frame)
    (output / "synthetic_features.csv").write_bytes(data_bytes)
    split = temporal_split(frame, warmup_days=5, validation_fraction=0.15, test_fraction=0.15)
    scorer, fit_info = train_ensemble(split, seed=seed, n_threads=1)
    features = to_matrix(split.test, FEATURE_COLUMNS)
    prediction = scorer.predict_proba(features)
    changed_labels = TemporalSplit(
        split.train, split.validation, split.test.assign(label=1 - split.test.label)
    )
    repeated, repeated_fit = train_ensemble(changed_labels, seed=seed, n_threads=1)
    repeated_prediction = repeated.predict_proba(features)
    np.testing.assert_array_equal(prediction, repeated_prediction)
    if fit_info != repeated_fit:
        raise RuntimeError("changing only evaluation labels changed the fitted components")

    artifacts = scorer.save(output / "native-model")
    restored = EnsembleScorer.load(artifacts, FEATURE_COLUMNS, n_threads=1)
    restored_prediction = restored.predict_proba(features)
    np.testing.assert_allclose(restored_prediction, prediction, rtol=0, atol=1e-12)

    labels = split.test.label.to_numpy()
    metrics = binary_metrics(labels, prediction, DecisionThresholds())
    interval = bootstrap_result(labels, prediction, split.test.user_id.to_numpy(), seed=seed)
    predicted_rows = split.test[["transaction_id", "user_id", "event_time", "label"]].assign(
        fraud_probability=prediction
    )
    prediction_bytes = _csv_bytes(predicted_rows)
    (output / "predictions.csv").write_bytes(prediction_bytes)
    try:
        xgboost_distribution = "xgboost-cpu"
        xgboost_version = version(xgboost_distribution)
    except PackageNotFoundError:
        xgboost_distribution = "xgboost"
        xgboost_version = version(xgboost_distribution)
    packages = {
        name: version(name)
        for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "fakeredis", "mlflow-skinny")
    }
    packages[xgboost_distribution] = xgboost_version
    root = Path(__file__).resolve().parents[1]
    source_paths = [
        Path("scripts/offline_validation.py"),
        *[
            Path("src/fraud_detection") / f"{name}.py"
            for name in (
                "bootstrap",
                "config",
                "evaluation",
                "features",
                "modeling",
                "schemas",
                "simulation",
                "trainer",
            )
        ],
    ]
    report: dict[str, Any] = {
        "data_kind": "synthetic simulator output; no real fraud evidence",
        "simulation": asdict(simulation),
        "feature_store": asdict(feature_config),
        "history": {"end": end, "days": days, "rows": len(frame), "warmup_days": 5},
        "features": list(FEATURE_COLUMNS),
        "split": split.describe(),
        "class_counts": {
            name: {"rows": len(part), "positives": int(part.label.sum())}
            for name, part in (
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            )
        },
        "fitted": fit_info,
        "metrics": metrics,
        "pr_auc_cluster_interval": asdict(interval),
        "checks": {
            "same_seed_and_changed_test_labels_max_abs_difference": float(
                np.abs(prediction - repeated_prediction).max(initial=0)
            ),
            "native_reload_max_abs_difference": float(
                np.abs(prediction - restored_prediction).max(initial=0)
            ),
            "native_reload_absolute_tolerance": 1e-12,
        },
        "sha256": {
            "dataset_csv_utf8_lf_float17g": hashlib.sha256(data_bytes).hexdigest(),
            "predictions_csv": hashlib.sha256(prediction_bytes).hexdigest(),
            "native_artifacts": {
                Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for path in artifacts.values()
            },
            "source_utf8_lf": {
                str(path.as_posix()): hashlib.sha256(
                    (root / path).read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest()
                for path in source_paths
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "training_threads": 1,
            "packages": packages,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/offline-validation"))
    parser.add_argument("--users", type=int, default=400)
    parser.add_argument("--merchants", type=int, default=120)
    parser.add_argument("--days", type=float, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--end", type=float, default=DEFAULT_END)
    args = parser.parse_args(argv)
    report = validate_offline(
        args.out,
        users=args.users,
        merchants=args.merchants,
        days=args.days,
        seed=args.seed,
        end=args.end,
    )
    print(json.dumps({"report": str(args.out / "report.json"), "checks": report["checks"]}))


if __name__ == "__main__":
    main()
