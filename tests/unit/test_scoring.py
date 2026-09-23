"""Tests for the deterministic scoring engine."""

from __future__ import annotations

import textwrap

import pytest

from fraud_risk_agent.scoring.config import ScoringConfig, load_scoring_config
from fraud_risk_agent.scoring.engine import FiredRule, ScoringResult, compute_score
from fraud_risk_agent.scoring.features import FeatureSet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ZERO_FEATURES = FeatureSet(
    return_rate=0.0,
    return_value_ratio=0.0,
    category_adjusted_rate=0.0,
    high_risk_reason_count=0,
    late_window_return_count=0,
    cancel_rate=0.0,
    cancel_after_ship_count=0,
    promo_cancel_detected=False,
    cancel_reorder_loop_count=0,
    linked_account_count=0,
    linked_abuse_score=0.0,
    total_orders=100,
)


def _make_config(
    signals_yaml: str,
    *,
    low_max: int = 39,
    medium_max: int = 69,
    min_orders: int = 5,
) -> ScoringConfig:
    """Build a ScoringConfig from inline YAML for one or more signals."""
    import yaml

    signals = yaml.safe_load(textwrap.dedent(signals_yaml))
    return ScoringConfig(
        version="test",
        thresholds={
            "low_max": low_max,
            "medium_max": medium_max,
            "min_orders_for_rates": min_orders,
        },
        signals=signals,
    )


def _features(**overrides: object) -> FeatureSet:
    """Return _ZERO_FEATURES with selected fields overridden."""
    base = {f: getattr(_ZERO_FEATURES, f) for f in _ZERO_FEATURES.__dataclass_fields__}
    base.update(overrides)
    return FeatureSet(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Core scoring tests
# ---------------------------------------------------------------------------


def test_all_zero_features_score_zero(scoring_config_path: str) -> None:
    """All-zero features produce score 0, low risk, no fired rules."""
    config = load_scoring_config(scoring_config_path)
    result = compute_score(_ZERO_FEATURES, config)
    assert result.score == 0
    assert result.risk_level == "low"
    assert result.fired_rules == []


def test_single_signal_fires() -> None:
    """A single signal fires and reports correct points + description."""
    config = _make_config("""\
        high_risk:
          description: "High risk reasons"
          field: high_risk_reason_count
          operator: ">="
          threshold: 3
          points: 30
    """)
    result = compute_score(_features(high_risk_reason_count=5), config)
    assert result.score == 30
    assert len(result.fired_rules) == 1
    assert result.fired_rules[0].signal_name == "high_risk"
    assert result.fired_rules[0].description == "High risk reasons"
    assert result.fired_rules[0].feature_value == 5
    assert result.fired_rules[0].threshold == 3


def test_multiple_signals_sum() -> None:
    """Multiple fired signals have their points summed."""
    config = _make_config("""\
        sig_a:
          description: "A"
          field: high_risk_reason_count
          operator: ">="
          threshold: 1
          points: 15
        sig_b:
          description: "B"
          field: late_window_return_count
          operator: ">="
          threshold: 1
          points: 10
    """)
    result = compute_score(
        _features(high_risk_reason_count=2, late_window_return_count=3),
        config,
    )
    assert result.score == 25
    assert len(result.fired_rules) == 2


def test_score_capped_at_100() -> None:
    """Total score is capped at 100 even when fired points exceed it."""
    config = _make_config("""\
        sig_a:
          description: "A"
          field: high_risk_reason_count
          operator: ">="
          threshold: 1
          points: 50
        sig_b:
          description: "B"
          field: late_window_return_count
          operator: ">="
          threshold: 1
          points: 50
        sig_c:
          description: "C"
          field: cancel_after_ship_count
          operator: ">="
          threshold: 1
          points: 30
    """)
    result = compute_score(
        _features(
            high_risk_reason_count=5,
            late_window_return_count=5,
            cancel_after_ship_count=5,
        ),
        config,
    )
    assert result.score == 100
    assert len(result.fired_rules) == 3


# ---------------------------------------------------------------------------
# Operator edge cases
# ---------------------------------------------------------------------------


def test_gt_at_threshold_does_not_fire() -> None:
    """Operator '>' does NOT fire when feature == threshold."""
    config = _make_config("""\
        test:
          description: "test"
          field: return_value_ratio
          operator: ">"
          threshold: 0.50
          points: 10
    """)
    result = compute_score(_features(return_value_ratio=0.50), config)
    assert result.score == 0
    assert result.fired_rules == []


def test_gt_just_above_threshold_fires() -> None:
    """Operator '>' fires when feature is just above threshold."""
    config = _make_config("""\
        test:
          description: "test"
          field: return_value_ratio
          operator: ">"
          threshold: 0.50
          points: 10
    """)
    result = compute_score(_features(return_value_ratio=0.51), config)
    assert result.score == 10


def test_gte_at_threshold_fires() -> None:
    """Operator '>=' fires when feature == threshold."""
    config = _make_config("""\
        test:
          description: "test"
          field: high_risk_reason_count
          operator: ">="
          threshold: 3
          points: 20
    """)
    result = compute_score(_features(high_risk_reason_count=3), config)
    assert result.score == 20


def test_eq_operator() -> None:
    """Operator '==' fires on exact match (bool case)."""
    config = _make_config("""\
        promo:
          description: "promo cancel"
          field: promo_cancel_detected
          operator: "=="
          threshold: true
          points: 20
    """)
    result = compute_score(_features(promo_cancel_detected=True), config)
    assert result.score == 20

    result_false = compute_score(_features(promo_cancel_detected=False), config)
    assert result_false.score == 0


def test_lt_operator() -> None:
    """Operator '<' fires when feature is below threshold."""
    config = _make_config("""\
        low_orders:
          description: "test lt"
          field: linked_account_count
          operator: "<"
          threshold: 2
          points: 5
    """)
    result = compute_score(_features(linked_account_count=1), config)
    assert result.score == 5

    result_at = compute_score(_features(linked_account_count=2), config)
    assert result_at.score == 0


def test_lte_operator() -> None:
    """Operator '<=' fires when feature is at or below threshold."""
    config = _make_config("""\
        lte_test:
          description: "test lte"
          field: linked_account_count
          operator: "<="
          threshold: 2
          points: 5
    """)
    result_at = compute_score(_features(linked_account_count=2), config)
    assert result_at.score == 5

    result_above = compute_score(_features(linked_account_count=3), config)
    assert result_above.score == 0


def test_unknown_operator_raises() -> None:
    """An unsupported operator raises ValueError."""
    import yaml

    data = yaml.safe_load(
        textwrap.dedent("""\
        version: "test"
        thresholds:
          low_max: 39
          medium_max: 69
          min_orders_for_rates: 5
        signals:
          bad:
            description: "bad op"
            field: return_value_ratio
            operator: "!="
            threshold: 1.0
            points: 10
    """)
    )
    config = ScoringConfig(**data)
    with pytest.raises(ValueError, match="Unknown operator"):
        compute_score(_features(return_value_ratio=2.0), config)


# ---------------------------------------------------------------------------
# requires_min_orders
# ---------------------------------------------------------------------------


def test_requires_min_orders_skips_when_too_few() -> None:
    """Signals with requires_min_orders skip when total_orders < min."""
    config = _make_config(
        """\
        rate:
          description: "return rate"
          field: return_rate
          operator: ">"
          threshold: 1.0
          points: 25
          requires_min_orders: true
    """,
        min_orders=5,
    )
    # Would fire if not skipped.
    result = compute_score(_features(return_rate=3.0, total_orders=4), config)
    assert result.score == 0
    assert result.fired_rules == []


def test_requires_min_orders_fires_when_enough() -> None:
    """Signals with requires_min_orders fire when total_orders >= min."""
    config = _make_config(
        """\
        rate:
          description: "return rate"
          field: return_rate
          operator: ">"
          threshold: 1.0
          points: 25
          requires_min_orders: true
    """,
        min_orders=5,
    )
    result = compute_score(_features(return_rate=3.0, total_orders=5), config)
    assert result.score == 25


# ---------------------------------------------------------------------------
# None feature handling
# ---------------------------------------------------------------------------


def test_none_feature_skipped_gracefully() -> None:
    """A feature that is None (e.g. return_rate) is silently skipped."""
    config = _make_config("""\
        rate:
          description: "return rate"
          field: return_rate
          operator: ">"
          threshold: 1.0
          points: 25
    """)
    result = compute_score(_features(return_rate=None), config)
    assert result.score == 0
    assert result.fired_rules == []


def test_none_and_valid_features_mixed() -> None:
    """None features are skipped while valid ones still fire."""
    config = _make_config("""\
        rate:
          description: "return rate"
          field: return_rate
          operator: ">"
          threshold: 1.0
          points: 25
        count:
          description: "risk reasons"
          field: high_risk_reason_count
          operator: ">="
          threshold: 3
          points: 30
    """)
    result = compute_score(
        _features(return_rate=None, high_risk_reason_count=5),
        config,
    )
    assert result.score == 30
    assert len(result.fired_rules) == 1
    assert result.fired_rules[0].signal_name == "count"


# ---------------------------------------------------------------------------
# Different config weights
# ---------------------------------------------------------------------------


def test_different_weights_different_scores() -> None:
    """Same features but different point values produce different scores."""
    config_low = _make_config("""\
        sig:
          description: "test"
          field: high_risk_reason_count
          operator: ">="
          threshold: 1
          points: 5
    """)
    config_high = _make_config("""\
        sig:
          description: "test"
          field: high_risk_reason_count
          operator: ">="
          threshold: 1
          points: 40
    """)
    features = _features(high_risk_reason_count=2)
    assert compute_score(features, config_low).score == 5
    assert compute_score(features, config_high).score == 40


# ---------------------------------------------------------------------------
# Risk-level band boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score_points", "expected_level"),
    [
        (0, "low"),
        (39, "low"),
        (40, "medium"),
        (69, "medium"),
        (70, "high"),
        (100, "high"),
    ],
    ids=["0-low", "39-low", "40-medium", "69-medium", "70-high", "100-high"],
)
def test_risk_level_bands(score_points: int, expected_level: str) -> None:
    """Risk level boundaries: <=39 low, 40-69 medium, >=70 high."""
    # Use a single signal whose points equal the desired score.
    # Cap at 50 (max allowed per signal) and add a second if needed.
    if score_points == 0:
        config = _make_config("""\
            sig:
              description: "test"
              field: high_risk_reason_count
              operator: ">="
              threshold: 99
              points: 10
        """)
        result = compute_score(_ZERO_FEATURES, config)
    elif score_points <= 50:
        config = _make_config(f"""\
            sig:
              description: "test"
              field: high_risk_reason_count
              operator: ">="
              threshold: 1
              points: {score_points}
        """)
        result = compute_score(_features(high_risk_reason_count=5), config)
    else:
        first = 50
        second = score_points - 50
        config = _make_config(f"""\
            sig_a:
              description: "A"
              field: high_risk_reason_count
              operator: ">="
              threshold: 1
              points: {first}
            sig_b:
              description: "B"
              field: late_window_return_count
              operator: ">="
              threshold: 1
              points: {second}
        """)
        result = compute_score(
            _features(high_risk_reason_count=5, late_window_return_count=5),
            config,
        )

    assert result.score == score_points
    assert result.risk_level == expected_level


# ---------------------------------------------------------------------------
# Integration with real YAML config
# ---------------------------------------------------------------------------


def test_real_config_all_signals_fire(scoring_config_path: str) -> None:
    """With extreme features every signal fires; score caps at 100."""
    config = load_scoring_config(scoring_config_path)
    extreme = FeatureSet(
        return_rate=10.0,
        return_value_ratio=0.99,
        category_adjusted_rate=10.0,
        high_risk_reason_count=50,
        late_window_return_count=50,
        cancel_rate=0.99,
        cancel_after_ship_count=50,
        promo_cancel_detected=True,
        cancel_reorder_loop_count=50,
        linked_account_count=50,
        linked_abuse_score=100.0,
        total_orders=100,
    )
    result = compute_score(extreme, config)
    assert result.score == 100
    assert result.risk_level == "high"
    # All 11 signals in the YAML should fire.
    assert len(result.fired_rules) == len(config.signals)


def test_scoring_result_is_frozen() -> None:
    """ScoringResult and FiredRule are immutable."""
    result = ScoringResult(score=50, risk_level="medium", fired_rules=[])
    with pytest.raises(AttributeError):
        result.score = 99  # type: ignore[misc]


def test_fired_rule_is_frozen() -> None:
    """FiredRule is immutable."""
    rule = FiredRule(
        signal_name="test",
        points=10,
        description="test",
        feature_value=1.0,
        threshold=0.5,
    )
    with pytest.raises(AttributeError):
        rule.points = 99  # type: ignore[misc]
