import argparse
import glob
import json
import os
import time
import urllib.error
import urllib.request


def request_json(url: str, method: str = "GET", payload: dict | None = None) -> tuple[int, dict]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=5) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body)


def request_text(url: str) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=5) as response:
        body = response.read().decode("utf-8")
        return response.status, body


def wait_for_json(url: str, timeout: int, expected_status: int = 200) -> dict:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            status, payload = request_json(url)
            if status == expected_status:
                return payload
            last_error = f"unexpected status {status}"
        except Exception as exc:  # pragma: no cover - smoke path
            last_error = str(exc)
        time.sleep(2)

    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


def wait_for_report(reports_dir: str, start_ts: float, timeout: int) -> str:
    deadline = time.time() + timeout
    pattern = os.path.join(reports_dir, "drift_report_*.html")
    while time.time() < deadline:
        candidates = [
            path for path in glob.glob(pattern)
            if os.path.getmtime(path) >= start_ts
        ]
        if candidates:
            return max(candidates, key=os.path.getmtime)
        time.sleep(2)

    raise TimeoutError(f"Timed out waiting for report in {reports_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    started_at = time.time()

    health = wait_for_json(f"{args.base_url}/health", timeout=args.timeout)
    print(f"Health OK: model_loaded={health['model_loaded']} model_version={health['model_version']}")

    ready = wait_for_json(f"{args.base_url}/ready", timeout=args.timeout)
    print(f"Ready OK: model_version={ready['model_version']}")

    status, metrics = request_text(f"{args.base_url}/metrics")
    if status != 200 or "fraud_predict_requests_total" not in metrics:
        raise RuntimeError("Predictor metrics endpoint did not expose expected Prometheus counters")
    print("Metrics OK: predictor Prometheus endpoint exposed")

    payload = {
        "transaction_id": "smoke-test-tx",
        "amount": 100.0,
        "amount_log": 4.6151205168,
        "amount_zscore": 0.1,
        "tx_count_1h": 1,
        "tx_count_24h": 4,
        "tx_sum_1h": 100.0,
        "tx_sum_24h": 500.0,
        "unique_merchants_24h": 2,
        "unique_countries_7d": 1,
        "hour_of_day": 12,
        "day_of_week": 2,
        "is_weekend": False,
        "card_present": True,
        "merchant_fraud_rate_30d": 0.01,
        "user_chargeback_rate": 0.0,
    }
    status, prediction = request_json(f"{args.base_url}/predict", method="POST", payload=payload)
    if status != 200:
        raise RuntimeError("Predictor smoke request failed")
    print(
        "Predict OK: "
        f"decision={prediction['decision']} "
        f"probability={prediction['fraud_probability']} "
        f"latency_ms={prediction['latency_ms']}"
    )

    report_path = wait_for_report(args.reports_dir, started_at, timeout=args.timeout)
    print(f"Monitor OK: report generated at {report_path}")


if __name__ == "__main__":
    main()
