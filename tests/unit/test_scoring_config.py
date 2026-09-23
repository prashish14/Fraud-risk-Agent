"""Tests for the YAML scoring config loader and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from fraud_risk_agent.scoring.config import (
    SignalRule,
    Thresholds,
    load_scoring_config,
)

# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_load_real_config(scoring_config_path: str) -> None:
    """The shipped scoring_rules_v1.yaml loads and validates."""
    config = load_scoring_config(scoring_config_path)
    assert config.version == "v1.0"
    assert len(config.signals) > 0


def test_version_tracked(scoring_config_path: str) -> None:
    """Config exposes the version string from the YAML."""
    config = load_scoring_config(scoring_config_path)
    assert config.version == "v1.0"


def test_thresholds_defaults(scoring_config_path: str) -> None:
    """Thresholds match the values declared in the YAML."""
    config = load_scoring_config(scoring_config_path)
    assert config.thresholds.low_max == 39
    assert config.thresholds.medium_max == 69
    assert config.thresholds.min_orders_for_rates == 5


def test_signal_fields_present(scoring_config_path: str) -> None:
    """Every signal has the required fields populated."""
    config = load_scoring_config(scoring_config_path)
    for name, rule in config.signals.items():
        assert rule.description, f"{name} missing description"
        assert rule.field, f"{name} missing field"
        assert rule.operator in {">", ">=", "==", "<", "<="}, f"{name} bad operator"
        assert 1 <= rule.points <= 50, f"{name} points out of range"


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_frozen_config_rejects_mutation(scoring_config_path: str) -> None:
    """ScoringConfig is frozen -- attribute assignment raises."""
    config = load_scoring_config(scoring_config_path)
    with pytest.raises(ValidationError):
        config.version = "v2.0"  # type: ignore[misc]


def test_frozen_thresholds_rejects_mutation() -> None:
    """Thresholds model is frozen."""
    t = Thresholds()
    with pytest.raises(ValidationError):
        t.low_max = 99  # type: ignore[misc]


def test_frozen_signal_rule_rejects_mutation() -> None:
    """SignalRule model is frozen."""
    rule = SignalRule(
        description="test",
        field="return_rate",
        operator=">",
        threshold=1.0,
        points=10,
    )
    with pytest.raises(ValidationError):
        rule.points = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_missing_version_rejected(tmp_path: Path) -> None:
    """YAML without a version key is rejected."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        textwrap.dedent("""\
        thresholds:
          low_max: 39
          medium_max: 69
          min_orders_for_rates: 5
        signals: {}
        """)
    )
    with pytest.raises(ValidationError, match="version"):
        load_scoring_config(str(bad))


def test_missing_signals_rejected(tmp_path: Path) -> None:
    """YAML without the signals key is rejected."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        textwrap.dedent("""\
        version: "v1.0"
        thresholds:
          low_max: 39
        """)
    )
    with pytest.raises(ValidationError, match="signals"):
        load_scoring_config(str(bad))


def test_points_out_of_range_rejected(tmp_path: Path) -> None:
    """Signal with points > 50 is rejected."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        textwrap.dedent("""\
        version: "v1.0"
        thresholds:
          low_max: 39
          medium_max: 69
          min_orders_for_rates: 5
        signals:
          over:
            description: "Too many points"
            field: return_rate
            operator: ">"
            threshold: 1.0
            points: 99
        """)
    )
    with pytest.raises(ValidationError, match="points"):
        load_scoring_config(str(bad))


def test_points_zero_rejected(tmp_path: Path) -> None:
    """Signal with points < 1 is rejected."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        textwrap.dedent("""\
        version: "v1.0"
        thresholds:
          low_max: 39
          medium_max: 69
          min_orders_for_rates: 5
        signals:
          zero:
            description: "Zero points"
            field: return_rate
            operator: ">"
            threshold: 1.0
            points: 0
        """)
    )
    with pytest.raises(ValidationError, match="points"):
        load_scoring_config(str(bad))


def test_file_not_found_raises() -> None:
    """Non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_scoring_config("/nonexistent/path.yaml")


# ---------------------------------------------------------------------------
# Minimal valid config round-trip
# ---------------------------------------------------------------------------


def test_minimal_valid_config(tmp_path: Path) -> None:
    """A minimal YAML with one signal loads successfully."""
    path = tmp_path / "mini.yaml"
    path.write_text(
        textwrap.dedent("""\
        version: "v0.1"
        thresholds:
          low_max: 30
          medium_max: 60
          min_orders_for_rates: 3
        signals:
          test_signal:
            description: "Test signal"
            field: return_rate
            operator: ">"
            threshold: 1.5
            points: 10
        """)
    )
    config = load_scoring_config(str(path))
    assert config.version == "v0.1"
    assert config.thresholds.low_max == 30
    assert "test_signal" in config.signals
    rule = config.signals["test_signal"]
    assert rule.points == 10
    assert rule.threshold == 1.5


def test_config_with_empty_signals(tmp_path: Path) -> None:
    """An empty signals dict is valid (no rules fire)."""
    path = tmp_path / "empty.yaml"
    path.write_text(
        textwrap.dedent("""\
        version: "v1.0"
        thresholds:
          low_max: 39
          medium_max: 69
          min_orders_for_rates: 5
        signals: {}
        """)
    )
    config = load_scoring_config(str(path))
    assert len(config.signals) == 0
