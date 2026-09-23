"""Tests for feature extraction."""

from __future__ import annotations

import pytest

from fraud_risk_agent.data.queries import (
    CancellationHistory,
    LinkedAccounts,
    OrderSummary,
    ReturnHistory,
)
from fraud_risk_agent.scoring.config import load_scoring_config
from fraud_risk_agent.scoring.features import FeatureSet, extract_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_returns(**overrides: object) -> ReturnHistory:
    defaults: dict[str, object] = {
        "customer_id": "cust-001",
        "repeat_return_count_30d": 1,
        "repeat_return_count_90d": 3,
        "repeat_return_count_365d": 5,
        "return_rate": 0.1,
        "return_value_ratio": 0.2,
        "category_adjusted_rate": 0.15,
        "high_risk_reason_count": 0,
        "late_window_return_count": 0,
        "total_orders": 20,
        "total_returns": 2,
    }
    defaults.update(overrides)
    return ReturnHistory(**defaults)  # type: ignore[arg-type]


def _sample_cancellations(**overrides: object) -> CancellationHistory:
    defaults: dict[str, object] = {
        "customer_id": "cust-001",
        "cancel_count_30d": 0,
        "cancel_count_90d": 1,
        "cancel_rate": 0.05,
        "cancel_after_ship_count": 0,
        "promo_cancel_detected": False,
        "cancel_reorder_loop_count": 0,
        "total_orders": 20,
        "total_cancellations": 1,
    }
    defaults.update(overrides)
    return CancellationHistory(**defaults)  # type: ignore[arg-type]


def _sample_orders(**overrides: object) -> OrderSummary:
    defaults: dict[str, object] = {
        "customer_id": "cust-001",
        "total_orders": 20,
        "total_spend": 1500.0,
        "account_age_days": 365,
        "top_categories": ["electronics", "clothing"],
    }
    defaults.update(overrides)
    return OrderSummary(**defaults)  # type: ignore[arg-type]


def _sample_linked(**overrides: object) -> LinkedAccounts:
    defaults: dict[str, object] = {
        "customer_id": "cust-001",
        "linked_account_count": 1,
        "linked_abuse_score": 0.0,
        "linked_by": ["address"],
    }
    defaults.update(overrides)
    return LinkedAccounts(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_features_basic() -> None:
    """extract_features maps dataclass fields into a FeatureSet."""
    fs = extract_features(
        returns=_sample_returns(return_rate=0.3),
        cancellations=_sample_cancellations(cancel_rate=0.1),
        orders=_sample_orders(total_orders=50),
        linked=_sample_linked(linked_account_count=2),
    )
    assert fs.return_rate == 0.3
    assert fs.cancel_rate == 0.1
    assert fs.total_orders == 50
    assert fs.linked_account_count == 2


def test_extract_features_preserves_none() -> None:
    """None values pass through from the query results."""
    fs = extract_features(
        returns=_sample_returns(return_rate=None, category_adjusted_rate=None),
        cancellations=_sample_cancellations(cancel_rate=None),
        orders=_sample_orders(total_orders=2),
        linked=_sample_linked(),
    )
    assert fs.return_rate is None
    assert fs.category_adjusted_rate is None
    assert fs.cancel_rate is None


def test_all_zero_inputs() -> None:
    """All-zero/None inputs produce a valid FeatureSet."""
    fs = extract_features(
        returns=ReturnHistory(
            customer_id="cust-zero",
            repeat_return_count_30d=0,
            repeat_return_count_90d=0,
            repeat_return_count_365d=0,
            return_rate=None,
            return_value_ratio=0.0,
            category_adjusted_rate=None,
            high_risk_reason_count=0,
            late_window_return_count=0,
            total_orders=0,
            total_returns=0,
        ),
        cancellations=CancellationHistory(
            customer_id="cust-zero",
            cancel_count_30d=0,
            cancel_count_90d=0,
            cancel_rate=None,
            cancel_after_ship_count=0,
            promo_cancel_detected=False,
            cancel_reorder_loop_count=0,
            total_orders=0,
            total_cancellations=0,
        ),
        orders=OrderSummary(
            customer_id="cust-zero",
            total_orders=0,
            total_spend=0.0,
            account_age_days=0,
        ),
        linked=LinkedAccounts(
            customer_id="cust-zero",
            linked_account_count=0,
            linked_abuse_score=0.0,
        ),
    )
    assert fs.total_orders == 0
    assert fs.return_rate is None
    assert fs.return_value_ratio == 0.0


def test_feature_set_is_frozen() -> None:
    """FeatureSet is immutable."""
    fs = extract_features(
        returns=_sample_returns(),
        cancellations=_sample_cancellations(),
        orders=_sample_orders(),
        linked=_sample_linked(),
    )
    with pytest.raises(AttributeError):
        fs.total_orders = 999  # type: ignore[misc]


def test_feature_fields_match_yaml_signals(scoring_config_path: str) -> None:
    """Every YAML signal 'field' references a real FeatureSet attribute."""
    config = load_scoring_config(scoring_config_path)
    feature_fields = {f.name for f in FeatureSet.__dataclass_fields__.values()}
    for name, rule in config.signals.items():
        assert rule.field in feature_fields, (
            f"Signal {name!r} references field {rule.field!r} which is not on FeatureSet"
        )
