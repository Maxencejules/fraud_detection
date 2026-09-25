"""The fraud model: a calibrated XGBoost + LightGBM ensemble packaged for MLflow.

Artifacts are stored in the libraries' native formats (XGBoost JSON, LightGBM text,
calibration parameters as JSON) rather than pickles, so they load across library versions
and never execute code on load. ``FraudModel`` is logged with MLflow's
"models from code" mechanism (see ``model_code.py``) so standard MLflow tooling can serve
it; this project's services never import that logged code (see ``registry.load_model``).

``FraudModel.predict`` returns *calibrated probabilities*: a score of 0.2 means roughly
a 20% chance of fraud, so decision thresholds are meaningful business settings.
"""

# No ``from __future__ import annotations``: MLflow inspects ``FraudModel.predict``
# type hints at class creation and cannot resolve stringified annotations.
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.pyfunc.model import PythonModel

ARTIFACT_XGB = "xgboost_model"
ARTIFACT_LGB = "lightgbm_model"
ARTIFACT_CALIBRATOR = "calibrator"
_EPSILON = 1e-7


@dataclass(frozen=True)
class LogitCalibrator:
    """Platt scaling on the logit: ``p' = sigmoid(slope * logit(p) + intercept)``.

    Strictly increasing for ``slope > 0``, so it never changes the ranking of
    transactions (PR-AUC and alert ordering are preserved) while correcting the
    probability scale. Isotonic regression was evaluated and rejected: its step
    function collapsed ~24k distinct test scores into 29 plateaus and lost ~0.02 PR-AUC.
    """

    slope: float = 1.0
    intercept: float = 0.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.slope) and math.isfinite(self.intercept)):
            raise ValueError("calibrator parameters must be finite")
        if self.slope <= 0:
            raise ValueError("calibrator slope must be positive to preserve ranking")

    @classmethod
    def identity(cls) -> Self:
        return cls(1.0, 0.0)

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(scores, dtype=np.float64), _EPSILON, 1.0 - _EPSILON)
        z = self.slope * np.log(p / (1.0 - p)) + self.intercept
        return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)

    def to_json(self) -> str:
        return json.dumps({"type": "logit", "slope": self.slope, "intercept": self.intercept})

    @classmethod
    def from_json(cls, text: str) -> Self:
        data = json.loads(text)
        if data.get("type") != "logit":
            raise ValueError(f"unsupported calibrator type: {data.get('type')!r}")
        return cls(float(data["slope"]), float(data["intercept"]))


class EnsembleScorer:
    """Averages XGBoost and LightGBM probabilities, then calibrates the average."""

    def __init__(
        self,
        xgb_booster: xgb.Booster,
        lgb_booster: lgb.Booster,
        calibrator: LogitCalibrator,
        feature_columns: Sequence[str],
        n_threads: int | None = None,
    ) -> None:
        self.xgb_booster = xgb_booster
        self.lgb_booster = lgb_booster
        self.calibrator = calibrator
        self.feature_columns = tuple(feature_columns)
        self.n_threads = n_threads
        if n_threads is not None:
            self.xgb_booster.set_param({"nthread": n_threads})

    def raw_scores(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"expected a (n, {len(self.feature_columns)}) matrix, got {features.shape}"
            )
        xgb_scores = self.xgb_booster.inplace_predict(features, validate_features=False)
        lgb_kwargs: dict[str, Any] = {}
        if self.n_threads is not None:
            lgb_kwargs["num_threads"] = self.n_threads
        lgb_scores = self.lgb_booster.predict(features, **lgb_kwargs)
        combined = np.asarray(xgb_scores, dtype=np.float64) + np.asarray(
            lgb_scores, dtype=np.float64
        )
        return combined / 2.0

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(np.clip(self.calibrator(self.raw_scores(features)), 0.0, 1.0))

    def save(self, directory: Path) -> dict[str, str]:
        """Write native artifacts to ``directory``; returns the MLflow artifact mapping."""
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            ARTIFACT_XGB: directory / "xgboost.json",
            ARTIFACT_LGB: directory / "lightgbm.txt",
            ARTIFACT_CALIBRATOR: directory / "calibrator.json",
        }
        self.xgb_booster.save_model(str(paths[ARTIFACT_XGB]))
        self.lgb_booster.save_model(str(paths[ARTIFACT_LGB]))
        paths[ARTIFACT_CALIBRATOR].write_text(self.calibrator.to_json(), encoding="utf-8")
        return {name: str(path) for name, path in paths.items()}

    @classmethod
    def load(
        cls,
        artifacts: Mapping[str, str],
        feature_columns: Sequence[str],
        n_threads: int | None = None,
    ) -> Self:
        xgb_booster = xgb.Booster()
        xgb_booster.load_model(artifacts[ARTIFACT_XGB])
        lgb_booster = lgb.Booster(model_file=artifacts[ARTIFACT_LGB])
        calibrator = LogitCalibrator.from_json(
            Path(artifacts[ARTIFACT_CALIBRATOR]).read_text(encoding="utf-8")
        )
        return cls(xgb_booster, lgb_booster, calibrator, feature_columns, n_threads)


def to_matrix(frame: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    """Select ``feature_columns`` in model order as a float64 matrix (bools become 0/1)."""
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float64)


class FraudModel(PythonModel):
    """MLflow pyfunc wrapper. ``predict`` returns calibrated fraud probabilities."""

    def __init__(self) -> None:
        self.scorer: EnsembleScorer | None = None

    def load_context(self, context: Any) -> None:
        config = context.model_config or {}
        self.scorer = EnsembleScorer.load(
            context.artifacts,
            feature_columns=config["feature_columns"],
            n_threads=config.get("n_threads"),
        )

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> np.ndarray:
        if self.scorer is None:
            raise RuntimeError("model artifacts are not loaded")
        return self.scorer.predict_proba(to_matrix(model_input, self.scorer.feature_columns))
