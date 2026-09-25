from __future__ import annotations

import json
import logging
import sys

import pytest
from pydantic import ValidationError

from fraud_detection.config import DecisionThresholds, PredictorSettings, ProducerSettings
from fraud_detection.observability import JsonFormatter, configure_logging
from fraud_detection.schemas import Decision


class TestDecisionThresholds:
    @pytest.mark.parametrize(
        ("probability", "decision"),
        [
            (0.0, Decision.APPROVE),
            (0.1, Decision.REVIEW),
            (0.89, Decision.REVIEW),
            (0.9, Decision.BLOCK),
        ],
    )
    def test_decide(self, probability: float, decision: Decision) -> None:
        assert DecisionThresholds(review=0.1, block=0.9).decide(probability) is decision

    def test_review_must_be_below_block(self) -> None:
        with pytest.raises(ValidationError, match="below block"):
            DecisionThresholds(review=0.9, block=0.5)

    def test_settings_reject_inconsistent_thresholds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("THRESHOLD_REVIEW", "0.95")
        with pytest.raises(ValidationError):
            PredictorSettings()


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMIT_RATE_TPS", "55")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    monkeypatch.setenv("SIM_USERS", "1234")

    settings = ProducerSettings()

    assert settings.emit_rate_tps == 55
    assert settings.kafka_bootstrap_servers == "kafka:29092"
    assert settings.simulation_config().n_users == 1234


def test_rate_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMIT_RATE_TPS", "0")
    with pytest.raises(ValidationError):
        ProducerSettings()


def test_cors_origins_accept_comma_separated_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://a.example, https://b.example")
    assert PredictorSettings().cors_allow_origins == ["https://a.example", "https://b.example"]


def test_admin_token_is_not_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "super-secret-admin-token")
    assert "super-secret-admin-token" not in repr(PredictorSettings())


def test_json_formatter_includes_extra_fields_and_exceptions() -> None:
    formatter = JsonFormatter("svc")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = logging.LogRecord(
            "x", logging.ERROR, __file__, 1, "event_name", (), exc_info=sys.exc_info()
        )
    record.rows = 3

    payload = json.loads(formatter.format(record))

    assert payload["service"] == "svc"
    assert payload["event"] == "event_name"
    assert payload["rows"] == 3
    assert "RuntimeError: boom" in payload["exc_info"]


def test_configure_logging_installs_a_single_json_handler(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = logging.getLogger()
    previous = (root.handlers[:], root.level)
    try:
        logger = configure_logging("unit", "DEBUG")
        configure_logging("unit", "DEBUG")
        logger.info("hello", extra={"answer": 42})
        assert len(root.handlers) == 1
        line = capsys.readouterr().out.strip().splitlines()[-1]
        assert json.loads(line)["answer"] == 42
    finally:
        root.handlers[:], _ = previous
        root.setLevel(previous[1])


def test_empty_admin_token_disables_the_admin_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "")
    assert PredictorSettings().admin_token is None


def test_short_admin_tokens_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "short")
    with pytest.raises(ValidationError, match="at least 16"):
        PredictorSettings()


def test_history_end_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    from fraud_detection.config import BootstrapSettings

    monkeypatch.setenv("HISTORY_END", "")
    assert BootstrapSettings().history_end is None
    monkeypatch.setenv("HISTORY_END", "1767225600")
    assert BootstrapSettings().history_end == 1_767_225_600.0
