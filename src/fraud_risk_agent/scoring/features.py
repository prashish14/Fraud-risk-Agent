"""Feature extraction -- transforms raw query results into a flat FeatureSet."""

from __future__ import annotations

from dataclasses import dataclass

from fraud_risk_agent.data.queries import (
    CancellationHistory,
    LinkedAccounts,
    OrderSummary,
    ReturnHistory,
)


@dataclass(frozen=True)
class FeatureSet:
    """Flat feature vector used by the scoring engine.

    Every field name matches a ``field`` value in the scoring YAML config.
    """

    # Return signals
    return_rate: float | None
    return_value_ratio: float
    category_adjusted_rate: float | None
    high_risk_reason_count: int
    late_window_return_count: int

    # Cancellation signals
    cancel_rate: float | None
    cancel_after_ship_count: int
    promo_cancel_detected: bool
    cancel_reorder_loop_count: int

    # Identity signals
    linked_account_count: int
    linked_abuse_score: float

    # Context
    total_orders: int


def extract_features(
    returns: ReturnHistory,
    cancellations: CancellationHistory,
    orders: OrderSummary,
    linked: LinkedAccounts,
) -> FeatureSet:
    """Build a :class:`FeatureSet` from the four query-result dataclasses.

    Pure function -- no I/O, no side effects.
    """
    return FeatureSet(
        # Return signals
        return_rate=returns.return_rate,
        return_value_ratio=returns.return_value_ratio,
        category_adjusted_rate=returns.category_adjusted_rate,
        high_risk_reason_count=returns.high_risk_reason_count,
        late_window_return_count=returns.late_window_return_count,
        # Cancellation signals
        cancel_rate=cancellations.cancel_rate,
        cancel_after_ship_count=cancellations.cancel_after_ship_count,
        promo_cancel_detected=cancellations.promo_cancel_detected,
        cancel_reorder_loop_count=cancellations.cancel_reorder_loop_count,
        # Identity signals
        linked_account_count=linked.linked_account_count,
        linked_abuse_score=linked.linked_abuse_score,
        # Context
        total_orders=orders.total_orders,
    )
