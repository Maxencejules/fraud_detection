# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - Unreleased

This release rewrites the pipeline as one installable package. Topics, Redis keys, the
model contract and the API all changed. To upgrade, delete the old stack with its data
(`docker compose down -v` from the old checkout), then run `docker compose up -d --build`.

### Breaking changes

- The predictor API moved under `/v1`:
  - `POST /predict` is now `POST /v1/predict`.
  - `POST /reload-model` is now `POST /v1/admin/reload`. It is disabled unless
    `ADMIN_TOKEN` is set.
- Prediction requests take 17 features instead of 15 (`seconds_since_last_tx` and
  `is_new_country` are new), and requests with unknown fields are rejected.
- The default decision thresholds changed from 0.4 / 0.75 to 0.1 / 0.9, because scores
  are now calibrated probabilities.
- The predictor finds its model through the `champion` registry alias. It no longer
  uses the deprecated `Production` stage.
- The dead-letter topic is now `transactions.dlq` (it was `transactions.raw.dlq`).
- The `consumer` service is now `feature-processor`.
- `compose.yaml` replaces `docker-compose.yml`, and host ports bind to `127.0.0.1`.
- Python 3.12 or newer is required.

### Added

- A population-based transaction simulator with account-takeover, card-testing and
  opportunistic fraud.
- A `bootstrap` job that replays simulated history through the production feature
  code. It builds the training set and warms Redis.
- A stream `scorer` service and the `transactions.decisions` topic.
- New endpoints: `POST /v1/predict/batch` and `GET /v1/model`. Every response carries
  a request ID.
- An XGBoost + LightGBM ensemble with Platt calibration, out-of-time evaluation with a
  bootstrap confidence interval, and champion/challenger promotion.
- Prometheus and a provisioned Grafana dashboard, enabled with the `observability`
  profile.
- Unit, dependency-boundary and integration tests.
- CI with linting, strict type checking, a vulnerability audit and an end-to-end Docker
  Compose job.
- An architecture document, a model card and an operations runbook.

### Fixed

- The MLflow server failed to start because its image had no PostgreSQL driver.
- Models were saved inside the training container instead of the MLflow artifact store,
  so the predictor could never load them.
- The consumer exited when its input topic did not exist yet.
- The monitor crashed on start-up with Evidently 0.7.
- The predictor returned class labels (0 or 1) instead of probabilities.
- Rolling windows never expired for active users.
- Training data was perfectly separable, which made the reported PR-AUC of 1.0
  meaningless. Separate code generated it, different from the serving features.

### Removed

- `requirements-dev.txt`, `pytest.ini`, each service's `requirements.txt` and
  `scripts/gen_training_data.py`. `pyproject.toml`, `uv.lock` and the simulator replace
  them.
- Generated files that had been committed to the repository: a drift report and runtime
  JSON.
