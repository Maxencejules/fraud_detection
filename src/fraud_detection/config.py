"""Typed runtime configuration.

Every service reads its configuration from environment variables through one of the
settings classes below. Values are validated at start-up so that a misconfigured
container fails immediately with a readable error instead of misbehaving later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from fraud_detection.features import FeatureStoreConfig
from fraud_detection.schemas import Decision
from fraud_detection.simulation import SimulationConfig

Probability = Annotated[float, Field(gt=0.0, le=1.0)]


class DecisionThresholds(BaseModel):
    """Probability cut-offs that turn a calibrated fraud score into an action."""

    model_config = ConfigDict(frozen=True)

    review: Probability = 0.1
    block: Probability = 0.9

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.review >= self.block:
            raise ValueError(
                f"review threshold ({self.review}) must be below block threshold ({self.block})"
            )
        return self

    def decide(self, probability: float) -> Decision:
        if probability >= self.block:
            return Decision.BLOCK
        if probability >= self.review:
            return Decision.REVIEW
        return Decision.APPROVE


class _Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", frozen=True, case_sensitive=False)

    log_level: str = "INFO"


class _KafkaSettings(_Settings):
    kafka_bootstrap_servers: str = "localhost:9092"
    topic_raw: str = "transactions.raw"
    topic_features: str = "transactions.features"
    topic_decisions: str = "transactions.decisions"
    topic_dlq: str = "transactions.dlq"
    topic_partitions: int = Field(default=6, ge=1)
    topic_replication_factor: int = Field(default=1, ge=1)


class _RedisSettings(_Settings):
    redis_url: str = "redis://localhost:6379/0"


class _FeatureStoreSettings(_Settings):
    feature_key_prefix: str = "fd"
    label_delay_seconds: int = Field(default=2 * 86_400, ge=0)
    merchant_prior_rate: float = Field(default=0.01, gt=0.0, lt=1.0)
    merchant_prior_weight: float = Field(default=50.0, ge=0.0)

    def feature_store_config(self) -> FeatureStoreConfig:
        return FeatureStoreConfig(
            key_prefix=self.feature_key_prefix,
            label_delay_seconds=self.label_delay_seconds,
            merchant_prior_rate=self.merchant_prior_rate,
            merchant_prior_weight=self.merchant_prior_weight,
        )


class _SimulationSettings(_Settings):
    sim_seed: int = 42
    sim_users: int = Field(default=3_000, ge=10)
    sim_merchants: int = Field(default=600, ge=10)
    sim_tx_per_user_per_day: float = Field(default=1.5, gt=0.0)

    def simulation_config(self) -> SimulationConfig:
        return SimulationConfig(
            seed=self.sim_seed,
            n_users=self.sim_users,
            n_merchants=self.sim_merchants,
            mean_tx_per_user_per_day=self.sim_tx_per_user_per_day,
        )


class _MlflowSettings(_Settings):
    mlflow_tracking_uri: str = "http://localhost:5001"
    model_name: str = "fraud-detector"
    model_alias: str = "champion"


class _ThresholdSettings(_Settings):
    threshold_review: Probability = 0.1
    threshold_block: Probability = 0.9

    @model_validator(mode="after")
    def _validate_thresholds(self) -> Self:
        self.thresholds()  # raises if the pair is inconsistent
        return self

    def thresholds(self) -> DecisionThresholds:
        return DecisionThresholds(review=self.threshold_review, block=self.threshold_block)


class ProducerSettings(_KafkaSettings, _RedisSettings, _SimulationSettings, _FeatureStoreSettings):
    emit_rate_tps: float = Field(default=20.0, gt=0.0, le=10_000.0)
    metrics_port: int = Field(default=9107, ge=0)


class FeatureProcessorSettings(_KafkaSettings, _RedisSettings, _FeatureStoreSettings):
    consumer_group: str = "feature-processor"
    batch_size: int = Field(default=500, ge=1, le=10_000)
    # Max wait for a batch to fill; bounds the batching latency added per stage.
    poll_timeout_s: float = Field(default=0.05, gt=0.0)
    metrics_port: int = Field(default=9108, ge=0)


class ScorerSettings(_KafkaSettings):
    consumer_group: str = "scorer"
    predictor_url: str = "http://localhost:8000"
    predictor_timeout_s: float = Field(default=5.0, gt=0.0)
    batch_size: int = Field(default=200, ge=1, le=1_000)
    # Max wait for a batch to fill; bounds the batching latency added per stage.
    poll_timeout_s: float = Field(default=0.05, gt=0.0)
    max_backoff_s: float = Field(default=30.0, gt=0.0)
    metrics_port: int = Field(default=9110, ge=0)


class PredictorSettings(_MlflowSettings, _ThresholdSettings):
    # Explicit model URI (e.g. a local path) overrides the registry alias lookup.
    model_uri: str | None = None
    model_poll_interval_s: float = Field(default=30.0, ge=0.0)
    admin_token: SecretStr | None = None
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    max_batch_size: int = Field(default=1_000, ge=1, le=10_000)
    host: str = "0.0.0.0"  # noqa: S104 - the API is meant to be reachable inside the container network
    port: int = Field(default=8000, ge=1, le=65_535)
    access_log: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, data: object) -> object:
        if isinstance(data, dict):
            raw = data.get("cors_allow_origins")
            if isinstance(raw, str):
                origins = [o.strip() for o in raw.split(",") if o.strip()]
                data = {**data, "cors_allow_origins": origins}
            # Compose passes unset variables as empty strings; that must disable the admin
            # API rather than enable it with an empty token.
            if data.get("admin_token") == "":
                data = {**data, "admin_token": None}
        return data

    @model_validator(mode="after")
    def _strong_admin_token(self) -> Self:
        if self.admin_token is not None and len(self.admin_token.get_secret_value()) < 16:
            raise ValueError("ADMIN_TOKEN must be at least 16 characters (or unset to disable)")
        return self


class TrainerSettings(_MlflowSettings, _ThresholdSettings):
    data_path: Path = Path("data/features.parquet")
    experiment_name: str = "fraud-detection-training"
    warmup_days: float = Field(default=10.0, ge=0.0)
    validation_fraction: float = Field(default=0.15, gt=0.0, lt=0.5)
    test_fraction: float = Field(default=0.15, gt=0.0, lt=0.5)
    min_pr_auc: float = Field(default=0.3, ge=0.0, le=1.0)
    min_improvement: float = Field(default=0.0, ge=0.0)
    seed: int = 42


class MonitorSettings(_KafkaSettings, _MlflowSettings, _ThresholdSettings):
    consumer_group: str = "monitor"
    experiment_name: str = "fraud-monitoring"
    window_size: int = Field(default=2_000, ge=10)
    min_rows: int = Field(default=200, ge=10)
    report_interval_s: float = Field(default=300.0, gt=0.0)
    champion_refresh_s: float = Field(default=60.0, gt=0.0)
    reports_dir: Path = Path("reports")
    reports_keep: int = Field(default=20, ge=1)
    drift_share_threshold: float = Field(default=0.3, gt=0.0, le=1.0)
    metrics_port: int = Field(default=9109, ge=0)


class BootstrapSettings(_RedisSettings, _SimulationSettings, _FeatureStoreSettings):
    history_days: float = Field(default=45.0, gt=0.0)
    # End of the simulated history (Unix seconds). Unset: now. Pin it for reproducible runs.
    history_end: float | None = Field(default=None, gt=0.0)
    output_path: Path = Path("data/features.parquet")
    batch_size: int = Field(default=1_000, ge=1)

    @field_validator("history_end", mode="before")
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        return None if value == "" else value
