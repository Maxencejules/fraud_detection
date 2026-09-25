"""End-to-end smoke test for a running Compose stack (standard library only).

Checks, in order, that:
  1. the predictor is live and ready with a registered champion model;
  2. the scoring API honours its contract (single, batch, validation errors);
  3. transactions flow producer -> feature processor -> scorer (counters increase);
  4. the monitor publishes a drift report and MLflow holds the champion alias.

Usage: python scripts/smoke_test.py [--timeout 300]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

PREDICTOR = "http://localhost:8000"
MLFLOW = "http://localhost:5001"
METRICS = {
    "producer": "http://localhost:9107/metrics",
    "feature-processor": "http://localhost:9108/metrics",
    "monitor": "http://localhost:9109/metrics",
    "scorer": "http://localhost:9110/metrics",
}
FEATURES: dict[str, Any] = {
    "amount": 42.5,
    "amount_log": 3.773,
    "amount_zscore": 0.3,
    "tx_count_1h": 1,
    "tx_count_24h": 3,
    "tx_sum_1h": 42.5,
    "tx_sum_24h": 101.2,
    "unique_merchants_24h": 2,
    "unique_countries_7d": 1,
    "seconds_since_last_tx": 5400.0,
    "is_new_country": False,
    "hour_of_day": 13,
    "day_of_week": 2,
    "is_weekend": False,
    "card_present": True,
    "merchant_fraud_rate_30d": 0.01,
    "user_chargeback_rate": 0.0,
}


class SmokeError(AssertionError):
    pass


def request(url: str, payload: Any = None) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def wait_until(description: str, check: Callable[[], bool], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            if check():
                return
        except (OSError, ValueError, KeyError) as exc:
            last_error = f" (last error: {exc})"
        time.sleep(3)
    raise SmokeError(f"timed out after {timeout:.0f}s waiting for {description}{last_error}")


def counter(url: str, metric: str) -> float:
    """Sum every sample of a Prometheus counter (all label sets)."""
    _, text = request(url)
    pattern = re.compile(rf"^{re.escape(metric)}(?:\{{[^}}]*\}})? ([0-9.eE+-]+)$", re.MULTILINE)
    return sum(float(value) for value in pattern.findall(text))


def check_api() -> str:
    status, body = request(f"{PREDICTOR}/v1/model")
    info = json.loads(body)
    if status != 200 or len(info["feature_columns"]) != len(FEATURES):
        raise SmokeError(f"/v1/model returned {status}: {body[:200]}")

    status, body = request(f"{PREDICTOR}/v1/predict", {"transaction_id": "smoke-1", **FEATURES})
    result = json.loads(body)
    if status != 200 or result["decision"] not in {"APPROVE", "REVIEW", "BLOCK"}:
        raise SmokeError(f"/v1/predict returned {status}: {body[:200]}")
    if not 0.0 <= result["fraud_probability"] <= 1.0:
        raise SmokeError(f"probability out of range: {result}")

    batch = {"items": [{"transaction_id": f"smoke-{i}", **FEATURES} for i in range(3)]}
    status, body = request(f"{PREDICTOR}/v1/predict/batch", batch)
    if status != 200 or len(json.loads(body)["items"]) != 3:
        raise SmokeError(f"/v1/predict/batch returned {status}: {body[:200]}")

    invalid = {"transaction_id": "smoke-bad", **FEATURES, "hour_of_day": 42}
    status, _ = request(f"{PREDICTOR}/v1/predict", invalid)
    if status != 422:
        raise SmokeError(f"invalid payload returned {status}, expected 422")
    decision, probability = result["decision"], result["fraud_probability"]
    return f"model v{info['version']}, decision={decision} p={probability}"


def champion_version() -> str:
    query = urllib.parse.urlencode({"name": "fraud-detector", "alias": "champion"})
    status, body = request(f"{MLFLOW}/api/2.0/mlflow/registered-models/alias?{query}")
    if status != 200:
        raise SmokeError(f"champion alias lookup returned {status}: {body[:200]}")
    return str(json.loads(body)["model_version"]["version"])


def monitoring_runs() -> int:
    status, body = request(
        f"{MLFLOW}/api/2.0/mlflow/experiments/get-by-name?experiment_name=fraud-monitoring"
    )
    if status != 200:
        return 0
    experiment_id = json.loads(body)["experiment"]["experiment_id"]
    _, runs = request(
        f"{MLFLOW}/api/2.0/mlflow/runs/search",
        {"experiment_ids": [experiment_id], "max_results": 10},
    )
    return len(json.loads(runs).get("runs", []))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per wait step")
    args = parser.parse_args()
    steps: list[tuple[str, Callable[[], str]]] = []

    def ready() -> str:
        live = f"{PREDICTOR}/health"
        wait_until("predictor liveness", lambda: request(live)[0] == 200, args.timeout)
        wait_until("a loaded model", lambda: request(f"{PREDICTOR}/ready")[0] == 200, args.timeout)
        return str(json.loads(request(f"{PREDICTOR}/ready")[1])["model_version"])

    def flowing() -> str:
        wait_until(
            "decisions on the stream",
            lambda: counter(METRICS["scorer"], "fraud_scorer_decisions_total") > 0,
            args.timeout,
        )
        before = counter(METRICS["scorer"], "fraud_scorer_decisions_total")
        wait_until(
            "decision counter to increase",
            lambda: counter(METRICS["scorer"], "fraud_scorer_decisions_total") > before,
            60,
        )
        produced = counter(METRICS["producer"], "fraud_producer_events_total")
        processed = counter(METRICS["feature-processor"], "fraud_features_processed_total")
        decided = counter(METRICS["scorer"], "fraud_scorer_decisions_total")
        return f"produced={produced:.0f} features={processed:.0f} decisions={decided:.0f}"

    def monitored() -> str:
        wait_until(
            "a drift report",
            lambda: counter(METRICS["monitor"], "fraud_monitor_reports_total") >= 1,
            args.timeout,
        )
        wait_until("a monitoring run in MLflow", lambda: monitoring_runs() >= 1, 60)
        reports = counter(METRICS["monitor"], "fraud_monitor_reports_total")
        return f"reports={reports:.0f}, champion v{champion_version()}"

    steps = [
        ("predictor ready", ready),
        ("scoring API contract", check_api),
        ("stream flowing", flowing),
        ("monitoring", monitored),
    ]
    for name, step in steps:
        started = time.monotonic()
        try:
            detail = step()
        except SmokeError as exc:
            print(f"FAIL  {name}: {exc}")
            return 1
        print(f"ok    {name} ({time.monotonic() - started:.1f}s): {detail}")
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
