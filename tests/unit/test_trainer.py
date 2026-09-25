from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import fakeredis
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from fraud_detection.bootstrap import run_bootstrap
from fraud_detection.config import BootstrapSettings, TrainerSettings
from fraud_detection.evaluation import temporal_split
from fraud_detection.features import FEATURE_COLUMNS
from fraud_detection.registry import _inside, download_reference, load_model, resolve_alias
from fraud_detection.simulation import DAY
from fraud_detection.trainer import (
    decide_promotion,
    main,
    train_and_register,
    train_ensemble,
    xgboost_params,
)

END = 1_767_225_600.0 + 40 * DAY


class TestPromotionRule:
    def test_first_model_passing_the_gate_is_promoted(self) -> None:
        decision = decide_promotion(0.7, None, min_pr_auc=0.3, min_improvement=0.0)
        assert decision.promote is True

    def test_quality_gate_blocks_weak_models(self) -> None:
        decision = decide_promotion(0.2, None, min_pr_auc=0.3, min_improvement=0.0)
        assert decision.promote is False
        assert "quality gate" in decision.reason

    def test_challenger_must_not_be_worse(self) -> None:
        assert decide_promotion(0.71, 0.70, min_pr_auc=0.3, min_improvement=0.0).promote
        assert not decide_promotion(0.69, 0.70, min_pr_auc=0.3, min_improvement=0.0).promote

    def test_minimum_improvement_is_respected(self) -> None:
        assert not decide_promotion(0.705, 0.70, min_pr_auc=0.3, min_improvement=0.01).promote

    def test_unevaluable_champion_defers_promotion(self) -> None:
        decision = decide_promotion(
            0.9,
            None,
            min_pr_auc=0.3,
            min_improvement=0.0,
            champion_error="champion version 1 could not be evaluated (OSError)",
        )
        assert decision.promote is False
        assert decision.outcome == "deferred"
        assert "could not be evaluated" in decision.reason

    def test_quality_gate_applies_before_deferral(self) -> None:
        decision = decide_promotion(
            0.2, None, min_pr_auc=0.3, min_improvement=0.0, champion_error="unavailable"
        )
        assert decision.outcome == "rejected"
        assert "quality gate" in decision.reason


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "features.parquet"
    settings = BootstrapSettings(
        sim_users=400, sim_merchants=120, history_days=28, output_path=path
    )
    run_bootstrap(settings, fakeredis.FakeRedis(), end=END)
    return path


@pytest.fixture
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = f"file://{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    return uri


def test_boosters_keep_only_their_early_stopped_trees(dataset: Path) -> None:
    split = temporal_split(
        pd.read_parquet(dataset), warmup_days=5, validation_fraction=0.15, test_fraction=0.15
    )

    scorer, info = train_ensemble(split, seed=42)

    best_rounds = int(info["xgb_best_iteration"]) + 1
    assert best_rounds < xgboost_params(42)["n_estimators"]  # early stopping did trigger
    assert scorer.xgb_booster.num_boosted_rounds() == best_rounds
    assert scorer.lgb_booster.current_iteration() >= int(info["lgb_best_iteration"])


def test_train_register_promote_and_compare(dataset: Path, tracking: str) -> None:
    settings = TrainerSettings(
        mlflow_tracking_uri=tracking, data_path=dataset, warmup_days=5, min_pr_auc=0.05
    )
    client = MlflowClient()

    first = train_and_register(settings, client)

    assert first.promote is True
    champion = resolve_alias(client, settings.model_name, settings.model_alias)
    assert champion is not None
    assert champion.version == "1"
    tags = client.get_model_version(settings.model_name, "1").tags
    assert tags["promotion"] == "promoted"

    # A retrain on the same data is compared with the champion on the same test period.
    second = train_and_register(settings, client)
    assert "vs champion" in second.reason or "does not beat" in second.reason
    comparison = client.get_model_version(settings.model_name, "2").tags
    assert comparison["champion_pr_auc_on_same_test"] != "n/a"

    loaded = load_model(champion.uri, champion.version)
    assert loaded.feature_columns == FEATURE_COLUMNS
    probabilities = loaded.scorer.predict_proba(np.zeros((2, len(FEATURE_COLUMNS))))
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    assert champion.run_id is not None
    reference = download_reference(champion.run_id)
    assert {"label", "prediction", *FEATURE_COLUMNS} <= set(reference.columns)


def test_loading_never_executes_code_from_the_registry(dataset: Path, tracking: str) -> None:
    settings = TrainerSettings(
        mlflow_tracking_uri=tracking, data_path=dataset, warmup_days=5, min_pr_auc=0.05
    )
    client = MlflowClient()
    train_and_register(settings, client)
    stored_code = list(Path(tracking.removeprefix("file://")).rglob("model_code.py"))
    assert stored_code  # the pyfunc wrapper's code is stored with the model
    for path in stored_code:  # simulate a tampered registry entry
        path.write_text('raise RuntimeError("code from the registry was executed")\n')

    champion = resolve_alias(client, settings.model_name, settings.model_alias)
    assert champion is not None
    loaded = load_model(champion.uri, champion.version)

    assert loaded.feature_columns == FEATURE_COLUMNS
    assert loaded.run_id is not None


def test_artifact_paths_cannot_escape_the_model_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        _inside(tmp_path, "../outside.json")
    assert _inside(tmp_path, "artifacts/xgboost.json").startswith(str(tmp_path.resolve()))


def test_champion_that_cannot_be_loaded_is_not_replaced(
    dataset: Path, tracking: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = TrainerSettings(
        mlflow_tracking_uri=tracking, data_path=dataset, warmup_days=5, min_pr_auc=0.05
    )
    client = MlflowClient()
    assert train_and_register(settings, client).promote is True

    def artifact_store_down(uri: str, version: str | None = None) -> NoReturn:
        raise OSError("artifact store unreachable")

    monkeypatch.setattr("fraud_detection.trainer.load_model", artifact_store_down)
    decision = train_and_register(settings, client)

    assert decision.outcome == "deferred"
    champion = resolve_alias(client, settings.model_name, settings.model_alias)
    challenger = resolve_alias(client, settings.model_name, "challenger")
    assert champion is not None
    assert champion.version == "1"
    assert challenger is not None
    assert challenger.version == "2"
    tags = client.get_model_version(settings.model_name, "2").tags
    assert tags["promotion"] == "deferred"
    assert "could not be evaluated (OSError)" in tags["promotion_reason"]


def test_rejected_models_are_parked_as_challenger(dataset: Path, tracking: str) -> None:
    settings = TrainerSettings(mlflow_tracking_uri=tracking, data_path=dataset, min_pr_auc=0.9999)
    client = MlflowClient()

    decision = train_and_register(settings, client)

    assert decision.promote is False
    assert resolve_alias(client, settings.model_name, settings.model_alias) is None
    challenger = resolve_alias(client, settings.model_name, "challenger")
    assert challenger is not None


def test_cli_skips_training_when_a_champion_exists(
    dataset: Path, tracking: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_PATH", str(dataset))
    monkeypatch.setenv("MIN_PR_AUC", "0.05")
    main([])
    main(["--skip-if-champion"])

    versions = MlflowClient().search_model_versions("name='fraud-detector'")
    assert len(versions) == 1
