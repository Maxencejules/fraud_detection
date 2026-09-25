"""MLflow Model Registry helpers shared by the trainer, predictor and monitor."""

from __future__ import annotations

import logging
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

if TYPE_CHECKING:
    # Model code (XGBoost/LightGBM) is imported lazily: the monitor uses this module
    # without the serving/training dependencies installed.
    from fraud_detection.modeling import EnsembleScorer

REFERENCE_ARTIFACT_DIR = "reference"
REFERENCE_FILE = "reference.parquet"
_MISSING = {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE", "NOT_FOUND"}

logger = logging.getLogger("fraud_detection.registry")


@dataclass(frozen=True)
class ModelRef:
    """A loadable model: a registered version or an explicit URI."""

    name: str
    version: str
    run_id: str | None
    uri: str


@dataclass(frozen=True)
class LoadedModel:
    scorer: EnsembleScorer
    uri: str
    version: str
    run_id: str | None
    loaded_at: float

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.scorer.feature_columns


def resolve_alias(client: MlflowClient, name: str, alias: str) -> ModelRef | None:
    """Return the version an alias points to, or ``None`` if the model/alias is absent."""
    try:
        version = client.get_model_version_by_alias(name, alias)
    except MlflowException as exc:
        if exc.error_code in _MISSING:
            return None
        raise
    return ModelRef(
        name=name,
        version=str(version.version),
        run_id=version.run_id,
        uri=f"models:/{name}/{version.version}",
    )


def load_model(uri: str, version: str | None = None) -> LoadedModel:
    """Load a logged ``FraudModel`` and expose its fast NumPy scoring path."""
    from fraud_detection.modeling import FraudModel

    pyfunc_model = mlflow.pyfunc.load_model(uri)
    python_model = pyfunc_model.unwrap_python_model()
    if not isinstance(python_model, FraudModel) or python_model.scorer is None:
        raise TypeError(f"{uri} is not a fraud_detection FraudModel")
    return LoadedModel(
        scorer=python_model.scorer,
        uri=uri,
        version=version or uri,
        run_id=pyfunc_model.metadata.run_id,
        loaded_at=time.time(),
    )


def download_reference(run_id: str) -> pd.DataFrame:
    """Fetch the monitoring reference dataset logged with a training run."""
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=f"{REFERENCE_ARTIFACT_DIR}/{REFERENCE_FILE}", dst_path=tmp
        )
        return pd.read_parquet(Path(local))


def wait_for_tracking_server(tracking_uri: str, timeout_s: float = 120.0) -> None:
    """Block until an HTTP tracking server answers ``/health`` (no-op for local stores)."""
    if not tracking_uri.startswith(("http://", "https://")):
        return
    deadline = time.monotonic() + timeout_s
    url = tracking_uri.rstrip("/") + "/health"
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 - configured URL
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError) as exc:
            if time.monotonic() > deadline:
                raise TimeoutError(f"MLflow at {tracking_uri} not reachable: {exc}") from exc
            logger.info("waiting_for_mlflow", extra={"tracking_uri": tracking_uri})
        time.sleep(2)
