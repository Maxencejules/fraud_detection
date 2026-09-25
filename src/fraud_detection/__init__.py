"""Real-time card fraud detection pipeline."""

import logging
import os
from importlib.metadata import PackageNotFoundError, version

# Defaults applied before any service imports MLflow (explicit environment variables
# still win): route MLflow's logs through our JSON handler, keep container logs free of
# progress bars and import-time hints, and opt out of MLflow's usage telemetry.
for _name, _value in {
    "MLFLOW_CONFIGURE_LOGGING": "false",
    "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR": "false",
    "MLFLOW_DISABLE_AGENT_HINT": "1",
    "MLFLOW_DISABLE_TELEMETRY": "true",
    # Containers ship without git; MLflow would otherwise warn on every run.
    "GIT_PYTHON_REFRESH": "quiet",
}.items():
    os.environ.setdefault(_name, _value)

# MLflow logs schema-inference warnings about its own GenAI types at import time.
logging.getLogger("mlflow.types.type_hints").setLevel(logging.ERROR)

try:
    __version__ = version("fraud-detection")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
