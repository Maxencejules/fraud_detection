# Operations runbook

How to run, observe and fix the stack. Commands assume the repository root; `make`
targets wrap the same commands (run `make help`).

## Services

| Service | Role | Host endpoint | Health |
|---|---|---|---|
| `predictor` | Scoring API | http://localhost:8000 (`/docs` for OpenAPI) | `/health` (live), `/ready` (model loaded) |
| `producer` | Simulated traffic | metrics :9107 | `/metrics` |
| `feature-processor` | Raw events to features | metrics :9108 | `/metrics` |
| `monitor` | Drift and quality reports | metrics :9109 | `/metrics` |
| `scorer` | Features to decisions | metrics :9110 | `/metrics` |
| `mlflow` | Tracking server and model registry | http://localhost:5001 | `/health` |
| `kafka` | Event log (KRaft, single broker) | localhost:9092 | broker API check |
| `redis` | Online feature store (AOF persistence) | localhost:6379 | `PING` |
| `postgres` | MLflow backend store | internal only | `pg_isready` |
| `bootstrap`, `trainer` | One-shot jobs run on `up` | – | exit code 0 |
| `prometheus`, `grafana` | `--profile observability` | :9090, http://localhost:3000 | – |
| `kafka-ui` | `--profile tools` | http://localhost:8080 | – |

All host ports bind to `127.0.0.1`.

## Everyday tasks

| Task | Command |
|---|---|
| Start everything | `docker compose up -d --build` |
| Start with dashboards | `docker compose --profile observability up -d --build` |
| Check status | `docker compose ps -a` |
| Verify end to end | `python3 scripts/smoke_test.py` |
| Follow logs (JSON lines) | `docker compose logs -f scorer` |
| Stop, keep data | `docker compose down` |
| Stop and delete all data | `docker compose down -v` |
| Change traffic rate | `EMIT_RATE_TPS=50 docker compose up -d producer` |

## Model lifecycle

**Retrain.** `make train` (or `docker compose run --rm trainer python -m fraud_detection.trainer`)
trains on the bootstrapped dataset, scores the current champion on the same test
period and moves the `champion` alias only if the new version is not worse. If the
champion cannot be evaluated, the alias stays where it is. The predictor picks up a new
champion within `MODEL_POLL_INTERVAL_S` (15 s in Compose) without a restart; its log
shows `model_loaded` with the new version.

**Roll back.** Point the alias at an earlier version; the predictor follows on its next
poll:

```bash
docker compose exec mlflow python -c "
from mlflow import MlflowClient
MlflowClient('http://localhost:5000').set_registered_model_alias('fraud-detector', 'champion', '1')"
```

**Force an immediate reload** (only when `ADMIN_TOKEN` is set, at least 16 characters):

```bash
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" http://localhost:8000/v1/admin/reload
```

**Rebuild the dataset and the online store.** `make bootstrap` re-simulates the
history with `--reset`, which deletes the feature store keys first. Then retrain.

## Key metrics and suggested alerts

| Signal | PromQL | Alert when |
|---|---|---|
| Decisions flowing | `sum(rate(fraud_scorer_decisions_total[5m]))` | 0 for 5 minutes while the producer runs |
| End-to-end latency | `histogram_quantile(0.95, sum by (le) (rate(fraud_scorer_end_to_end_seconds_bucket[5m])))` | > 0.5 s for 10 minutes |
| Predictor latency | `histogram_quantile(0.99, sum by (le) (rate(fraud_predictor_request_latency_seconds_bucket[5m])))` | > 0.05 s for 10 minutes |
| Model loaded | `fraud_predictor_model_loaded` | 0 |
| Failed model reloads | `increase(fraud_predictor_model_reloads_total{outcome="failed"}[15m])` | > 0 |
| Dead letters | `increase(fraud_features_dead_letters_total[15m]) + increase(fraud_scorer_dead_letters_total[15m])` | > 0 |
| Data drift | `fraud_monitor_drift_detected` | 1 for two consecutive reports |
| Live recall | `fraud_monitor_quality{metric="recall"}` | clearly below the model card's value |
| Monitor failures | `increase(fraud_monitor_errors_total[30m])` | > 0 |

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `/ready` returns 503 with `no model registered` | No champion yet: the trainer has not finished, or its model failed the quality gate. | `docker compose logs trainer`; look for `training_completed` and its `reason`. |
| Trainer logs `champion_evaluation_failed`; the new version has tag `promotion=deferred` | The current champion could not be loaded or scored, so the new version was not compared and not promoted. | Check `mlflow` health and its artifact volume, then run `make train` again, or compare the versions and move the alias yourself. |
| `/ready` shows `last_reload_error` | A newer champion failed to load, so the predictor still serves the version in `model_version` and retries on every poll. | Read the error, then check `mlflow` health or the model's feature contract. |
| `/ready` returns 503 with a load error | The artifact store is unreachable, or the model's feature contract differs from the service. | Check the `mlflow` health and `docker compose logs predictor` (`model_load_failed`). |
| Scorer logs `operation_retry` for `predictor` | The predictor is down or has no model. Events are held, not lost. | Fix the predictor; the scorer resumes automatically. |
| Scorer logs HTTP 4xx and exits | Predictor and scorer disagree on the API contract (version skew). | Deploy both from the same version. |
| Messages accumulate in `transactions.dlq` | Producers send invalid payloads. | Inspect them (Kafka UI, or `kafka-console-consumer` in the `kafka` container); every envelope records the stage, error and origin. Fix the source, then republish the corrected payloads to `transactions.raw`. |
| Feature processor restarts repeatedly | Redis unreachable beyond the retry budget. | `docker compose logs redis`; the batch was not committed, so nothing is lost. |
| MLflow API calls return 403 | The request's Host header is not in `--allowed-hosts`. | Add the hostname to the `mlflow` command in `compose.yaml`. |
| Drift detected right after start-up | The first minutes after a restart include backlog and cold users. | Compare with the next reports before acting; see the model card. |
| Monitor writes no reports | Fewer than `MONITOR_MIN_ROWS` decisions per window, or no champion. | Check `fraud_monitor_window_rows` and the monitor log (`report_skipped`). |

## Data and retention

| Data | Location | Retention |
|---|---|---|
| Kafka topics | `kafka` container | Broker defaults (7 days); not persisted across container re-creation. |
| Online features | `redis_data` volume (AOF) | Per-user state trimmed to 32 days of event time; per-merchant daily counters deleted after 63 days of event time; all keys also expire via TTL. |
| MLflow runs and models | `postgres_data`, `mlflow_artifacts` volumes | Until deleted. |
| Drift reports (HTML) | `reports` volume | Newest 20 kept. |
| Training dataset | `training_data` volume | Overwritten by each bootstrap. |
