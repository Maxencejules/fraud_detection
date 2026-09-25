"""Train, evaluate, register and conditionally promote the fraud model.

Pipeline
--------
1. Out-of-time split of the feature dataset (train on the past, test on the future).
2. XGBoost and LightGBM with early stopping on the validation period.
3. Average the two probabilities and fit Platt (logit) scaling on validation.
4. Evaluate the calibrated ensemble on the untouched test period.
5. Log everything to MLflow and register a new model version.
6. Score the *current champion on the same test period*; move the ``champion`` alias
   to the new version only if it passes the quality gate and is not worse.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from fraud_detection import __version__
from fraud_detection.config import DecisionThresholds, TrainerSettings
from fraud_detection.evaluation import (
    TemporalSplit,
    binary_metrics,
    bootstrap_interval,
    fit_logit_calibration,
    precision_recall_table,
    reliability_table,
    temporal_split,
)
from fraud_detection.features import FEATURE_COLUMNS
from fraud_detection.modeling import EnsembleScorer, LogitCalibrator, to_matrix
from fraud_detection.observability import configure_logging
from fraud_detection.registry import (
    REFERENCE_ARTIFACT_DIR,
    REFERENCE_FILE,
    ModelRef,
    load_model,
    resolve_alias,
    wait_for_tracking_server,
)

logger = logging.getLogger("fraud_detection.trainer")

CHALLENGER_ALIAS = "challenger"
REFERENCE_ROWS = 10_000
# Logged via MLflow "models from code"; referenced by path so it is not executed here.
MODEL_CODE_PATH = Path(__file__).with_name("model_code.py")


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str


def decide_promotion(
    challenger_pr_auc: float,
    champion_pr_auc: float | None,
    *,
    min_pr_auc: float,
    min_improvement: float,
) -> PromotionDecision:
    """Champion/challenger rule. Both scores must come from the same test data."""
    if challenger_pr_auc < min_pr_auc:
        return PromotionDecision(
            False, f"PR-AUC {challenger_pr_auc:.4f} is below the quality gate {min_pr_auc:.4f}"
        )
    if champion_pr_auc is None:
        return PromotionDecision(True, "no comparable champion")
    if challenger_pr_auc >= champion_pr_auc + min_improvement:
        return PromotionDecision(
            True, f"PR-AUC {challenger_pr_auc:.4f} vs champion {champion_pr_auc:.4f}"
        )
    return PromotionDecision(
        False, f"PR-AUC {challenger_pr_auc:.4f} does not beat champion {champion_pr_auc:.4f}"
    )


def xgboost_params(seed: int) -> dict[str, Any]:
    return {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "early_stopping_rounds": 50,
        "random_state": seed,
        "n_jobs": -1,
    }


def lightgbm_params(seed: int) -> dict[str, Any]:
    return {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def train_ensemble(split: TemporalSplit, seed: int) -> tuple[EnsembleScorer, dict[str, float]]:
    """Fit both boosters with early stopping and calibrate their average on validation."""
    x_train, y_train = split.train[list(FEATURE_COLUMNS)].astype(float), split.train["label"]
    x_valid, y_valid = (
        split.validation[list(FEATURE_COLUMNS)].astype(float),
        split.validation["label"],
    )

    xgb_model = xgb.XGBClassifier(**xgboost_params(seed))
    xgb_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

    lgb_model = lgb.LGBMClassifier(**lightgbm_params(seed))
    lgb_model.fit(
        x_train,
        y_train,
        eval_X=(x_valid,),
        eval_y=(y_valid,),
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    uncalibrated = EnsembleScorer(
        xgb_model.get_booster(), lgb_model.booster_, LogitCalibrator.identity(), FEATURE_COLUMNS
    )
    valid_scores = uncalibrated.raw_scores(to_matrix(split.validation, FEATURE_COLUMNS))
    calibrator = fit_logit_calibration(valid_scores, split.validation["label"].to_numpy())
    scorer = EnsembleScorer(
        uncalibrated.xgb_booster, uncalibrated.lgb_booster, calibrator, FEATURE_COLUMNS
    )
    info = {
        "xgb_best_iteration": float(xgb_model.best_iteration),
        "lgb_best_iteration": float(lgb_model.best_iteration_),
        "calibration_slope": calibrator.slope,
        "calibration_intercept": calibrator.intercept,
    }
    return scorer, info


def feature_importance(scorer: EnsembleScorer) -> pd.DataFrame:
    xgb_gain = scorer.xgb_booster.get_score(importance_type="total_gain")
    lgb_gain = scorer.lgb_booster.feature_importance(importance_type="gain")
    frame = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "xgboost_gain": [xgb_gain.get(f, 0.0) for f in FEATURE_COLUMNS],
            "lightgbm_gain": lgb_gain,
        }
    )
    for column in ("xgboost_gain", "lightgbm_gain"):
        total = frame[column].sum()
        frame[column] = frame[column] / total if total else 0.0
    frame["mean_share"] = (frame["xgboost_gain"] + frame["lightgbm_gain"]) / 2
    return frame.sort_values("mean_share", ascending=False).reset_index(drop=True)


def _requirements() -> list[str]:
    packages = ("lightgbm", "mlflow-skinny", "numpy", "pandas")
    pins = [f"{name}=={package_version(name)}" for name in packages]
    xgboost_dist = "xgboost-cpu" if _installed("xgboost-cpu") else "xgboost"
    return [*pins, f"{xgboost_dist}=={package_version(xgboost_dist)}"]


def _installed(distribution: str) -> bool:
    try:
        package_version(distribution)
    except Exception:  # noqa: BLE001 - PackageNotFoundError or broken metadata
        return False
    return True


def _score_champion(champion: ModelRef, test: pd.DataFrame) -> np.ndarray | None:
    try:
        loaded = load_model(champion.uri, champion.version)
    except Exception:
        logger.exception("champion_load_failed", extra={"uri": champion.uri})
        return None
    if loaded.feature_columns != FEATURE_COLUMNS:
        logger.warning("champion_feature_contract_differs", extra={"uri": champion.uri})
        return None
    return loaded.scorer.predict_proba(to_matrix(test, FEATURE_COLUMNS))


def train_and_register(settings: TrainerSettings, client: MlflowClient) -> PromotionDecision:
    thresholds: DecisionThresholds = settings.thresholds()
    frame = pd.read_parquet(settings.data_path)
    split = temporal_split(
        frame,
        warmup_days=settings.warmup_days,
        validation_fraction=settings.validation_fraction,
        test_fraction=settings.test_fraction,
    )
    logger.info("dataset_split", extra=split.describe())

    scorer, fit_info = train_ensemble(split, settings.seed)
    x_test = to_matrix(split.test, FEATURE_COLUMNS)
    y_test = split.test["label"].to_numpy()
    test_probability = scorer.predict_proba(x_test)
    test_metrics = binary_metrics(y_test, test_probability, thresholds)
    test_metrics["pr_auc_ci_low"], test_metrics["pr_auc_ci_high"] = bootstrap_interval(
        y_test, test_probability, split.test["user_id"].to_numpy(), seed=settings.seed
    )
    uncalibrated_brier = binary_metrics(y_test, scorer.raw_scores(x_test), thresholds)["brier"]
    component_pr_auc = {
        "xgb_pr_auc": binary_metrics(
            y_test, scorer.xgb_booster.inplace_predict(x_test, validate_features=False), thresholds
        )["pr_auc"],
        "lgb_pr_auc": binary_metrics(
            y_test, np.asarray(scorer.lgb_booster.predict(x_test)), thresholds
        )["pr_auc"],
    }

    mlflow.set_experiment(settings.experiment_name)
    with mlflow.start_run(run_name="train-ensemble") as run, tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        mlflow.set_tags({"package_version": __version__, "model_type": "xgb+lgb, logit-calibrated"})
        mlflow.log_params(
            {
                "features": ",".join(FEATURE_COLUMNS),
                "data_path": str(settings.data_path),
                "warmup_days": settings.warmup_days,
                "threshold_review": thresholds.review,
                "threshold_block": thresholds.block,
                "seed": settings.seed,
                **{f"xgb_{k}": v for k, v in xgboost_params(settings.seed).items()},
                **{f"lgb_{k}": v for k, v in lightgbm_params(settings.seed).items()},
            }
        )
        mlflow.log_metrics(split.describe())
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in component_pr_auc.items()})
        mlflow.log_metrics({"test_brier_uncalibrated": uncalibrated_brier, **fit_info})

        feature_importance(scorer).to_csv(workdir / "feature_importance.csv", index=False)
        reliability_table(y_test, test_probability).to_csv(workdir / "reliability.csv", index=False)
        precision_recall_table(y_test, test_probability).to_csv(
            workdir / "precision_recall.csv", index=False
        )
        (workdir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
        for name in ("feature_importance.csv", "reliability.csv", "precision_recall.csv"):
            mlflow.log_artifact(str(workdir / name), artifact_path="evaluation")
        mlflow.log_artifact(str(workdir / "test_metrics.json"), artifact_path="evaluation")

        reference = split.test.assign(prediction=test_probability)
        if len(reference) > REFERENCE_ROWS:
            reference = reference.sample(REFERENCE_ROWS, random_state=settings.seed)
        reference_path = workdir / REFERENCE_FILE
        reference.sort_values("event_time").to_parquet(reference_path, index=False)
        mlflow.log_artifact(str(reference_path), artifact_path=REFERENCE_ARTIFACT_DIR)

        example = split.test[list(FEATURE_COLUMNS)].head(5)
        with warnings.catch_warnings():
            # Features are validated upstream and never missing, so MLflow's hint about
            # integer columns and missing values does not apply.
            warnings.filterwarnings("ignore", message=".*integer column", category=UserWarning)
            signature = infer_signature(
                example, scorer.predict_proba(to_matrix(example, FEATURE_COLUMNS))
            )
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=str(MODEL_CODE_PATH),
            artifacts=scorer.save(workdir / "artifacts"),
            model_config={"feature_columns": list(FEATURE_COLUMNS), "n_threads": 1},
            signature=signature,
            input_example=example.head(2),
            pip_requirements=_requirements(),
            registered_model_name=settings.model_name,
        )
        new_version = str(model_info.registered_model_version)

    champion = resolve_alias(client, settings.model_name, settings.model_alias)
    champion_pr_auc: float | None = None
    if champion is not None:
        champion_probability = _score_champion(champion, split.test)
        if champion_probability is not None:
            champion_pr_auc = binary_metrics(y_test, champion_probability, thresholds)["pr_auc"]

    decision = decide_promotion(
        test_metrics["pr_auc"],
        champion_pr_auc,
        min_pr_auc=settings.min_pr_auc,
        min_improvement=settings.min_improvement,
    )
    alias = settings.model_alias if decision.promote else CHALLENGER_ALIAS
    client.set_registered_model_alias(settings.model_name, alias, new_version)
    tags = {
        "test_pr_auc": f"{test_metrics['pr_auc']:.5f}",
        "champion_pr_auc_on_same_test": "n/a"
        if champion_pr_auc is None
        else f"{champion_pr_auc:.5f}",
        "promotion": "promoted" if decision.promote else "rejected",
        "promotion_reason": decision.reason,
        "training_run_id": run.info.run_id,
    }
    for key, value in tags.items():
        client.set_model_version_tag(settings.model_name, new_version, key, value)

    logger.info(
        "training_completed",
        extra={
            "version": new_version,
            "alias": alias,
            "reason": decision.reason,
            "run_id": run.info.run_id,
            **{k: round(v, 5) for k, v in test_metrics.items()},
        },
    )
    return decision


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and register the fraud model.")
    parser.add_argument(
        "--skip-if-champion",
        action="store_true",
        help="exit successfully without training when a champion model already exists",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = TrainerSettings()
    configure_logging("trainer", settings.log_level)
    wait_for_tracking_server(settings.mlflow_tracking_uri)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    if args.skip_if_champion:
        champion = resolve_alias(client, settings.model_name, settings.model_alias)
        if champion is not None:
            logger.info("training_skipped", extra={"champion_version": champion.version})
            return
    train_and_register(settings, client)


if __name__ == "__main__":
    main()
