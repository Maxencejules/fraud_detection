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
from mlflow.models import Model
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
    """Load a logged fraud model from its native artifacts.

    Only data is read: the ``MLmodel`` YAML (safe loader), XGBoost JSON, LightGBM text
    and calibration JSON. The pyfunc wrapper's logged code is never imported and nothing
    is unpickled, so a tampered registry entry cannot execute code in this process; the
    scoring code always comes from the installed package.
    """
    from fraud_detection.modeling import EnsembleScorer

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=tmp))
        metadata = Model.load(str(local))
        flavor = metadata.flavors.get("python_function") or {}
        config = flavor.get("config") or {}
        try:
            artifacts = {
                name: _inside(local, spec["path"]) for name, spec in flavor["artifacts"].items()
            }
            feature_columns = [str(column) for column in config["feature_columns"]]
            n_threads = int(config["n_threads"]) if config.get("n_threads") else None
            scorer = EnsembleScorer.load(artifacts, feature_columns, n_threads)
        except (KeyError, TypeError, ValueError) as exc:
            raise TypeError(f"{uri} is not a fraud_detection model: {exc!r}") from exc
    return LoadedModel(
        scorer=scorer,
        uri=uri,
        version=version or uri,
        run_id=metadata.run_id,
        loaded_at=time.time(),
    )


def _inside(root: Path, relative: str) -> str:
    """Resolve an artifact path from ``MLmodel``, refusing anything outside the model."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"artifact path escapes the model directory: {relative}")
    return str(path)


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
