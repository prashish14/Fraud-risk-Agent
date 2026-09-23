"""Parameterized read-only SQL queries against the e-commerce replica.

Each function takes an ``AsyncSession`` and returns a typed dataclass.
All queries use ``text()`` with bound parameters -- never f-strings.

Assumed replica schema:
- orders (id, customer_id, order_date, total_amount, status, promo_code, created_at)
- order_items (id, order_id, sku, category, quantity, price)
- returns (id, order_id, customer_id, return_date, reason_code,
           inspection_result, refund_amount, return_window_end)
- cancellations (id, order_id, customer_id, cancel_date,
                  order_status_at_cancel, had_promo)
- customers (id, email, phone, shipping_address_hash, payment_token_hash,
             device_id, created_at)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class ReturnHistory:
    """Aggregated return behaviour for a single customer."""

    customer_id: str
    repeat_return_count_30d: int
    repeat_return_count_90d: int
    repeat_return_count_365d: int
    return_rate: float | None  # None if < 5 orders
    return_value_ratio: float
    category_adjusted_rate: float | None
    high_risk_reason_count: int
    late_window_return_count: int
    total_orders: int
    total_returns: int


@dataclass(frozen=True)
class CancellationHistory:
    """Aggregated cancellation behaviour for a single customer."""

    customer_id: str
    cancel_count_30d: int
    cancel_count_90d: int
    cancel_rate: float | None  # None if < 5 orders
    cancel_after_ship_count: int
    promo_cancel_detected: bool
    cancel_reorder_loop_count: int
    total_orders: int
    total_cancellations: int


@dataclass(frozen=True)
class OrderSummary:
    """High-level order profile for a customer."""

    customer_id: str
    total_orders: int
    total_spend: float
    account_age_days: int
    top_categories: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LinkedAccounts:
    """Accounts linked to this customer by shared identifiers."""

    customer_id: str
    linked_account_count: int
    linked_abuse_score: float
    linked_by: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Query timeout applied to every execution
# ---------------------------------------------------------------------------
_EXEC_OPTIONS: dict[str, int] = {"timeout": 5000}


# ---------------------------------------------------------------------------
# Return history
# ---------------------------------------------------------------------------

_RETURN_HISTORY_SQL = text("""
    WITH customer_orders AS (
        SELECT count(*) AS total_orders,
               coalesce(sum(total_amount), 0) AS total_spend
        FROM orders
        WHERE customer_id = :customer_id
    ),
    customer_returns AS (
        SELECT count(*) AS total_returns,
               coalesce(sum(refund_amount), 0) AS total_refund,
               count(*) FILTER (
                   WHERE return_date >= now() - interval '30 days'
               ) AS cnt_30d,
               count(*) FILTER (
                   WHERE return_date >= now() - interval '90 days'
               ) AS cnt_90d,
               count(*) FILTER (
                   WHERE return_date >= now() - interval '365 days'
               ) AS cnt_365d,
               count(*) FILTER (
                   WHERE reason_code IN (
                       'fraud', 'not_as_described', 'unauthorized'
                   )
               ) AS high_risk_reason_count,
               count(*) FILTER (
                   WHERE return_window_end IS NOT NULL
                     AND return_date > return_window_end
                         - (return_window_end - o.order_date) * 0.1
               ) AS late_window_return_count
        FROM returns r
        JOIN orders o ON o.id = r.order_id
        WHERE r.customer_id = :customer_id
    ),
    category_baseline AS (
        SELECT oi.category,
               avg(cat_rate.rate) AS baseline_rate
        FROM order_items oi
        JOIN orders o ON o.id = oi.order_id
        LEFT JOIN LATERAL (
            SELECT count(r2.id)::float
                   / nullif(count(DISTINCT o2.id), 0) AS rate
            FROM orders o2
            JOIN order_items oi2 ON oi2.order_id = o2.id
            LEFT JOIN returns r2 ON r2.order_id = o2.id
            WHERE oi2.category = oi.category
        ) cat_rate ON true
        WHERE o.customer_id = :customer_id
        GROUP BY oi.category
    )
    SELECT co.total_orders,
           cr.total_returns,
           cr.cnt_30d  AS repeat_return_count_30d,
           cr.cnt_90d  AS repeat_return_count_90d,
           cr.cnt_365d AS repeat_return_count_365d,
           CASE WHEN co.total_orders >= 5
                THEN cr.total_returns::float / co.total_orders
                ELSE NULL
           END AS return_rate,
           CASE WHEN co.total_spend > 0
                THEN cr.total_refund::float / co.total_spend
                ELSE 0
           END AS return_value_ratio,
           CASE WHEN co.total_orders >= 5
                THEN (
                    SELECT cr.total_returns::float / co.total_orders
                           - coalesce(avg(cb.baseline_rate), 0)
                    FROM category_baseline cb
                )
                ELSE NULL
           END AS category_adjusted_rate,
           cr.high_risk_reason_count,
           cr.late_window_return_count
    FROM customer_orders co
    CROSS JOIN customer_returns cr
""")


async def fetch_return_history(
    session: AsyncSession,
    customer_id: str,
) -> ReturnHistory:
    """Fetch aggregated return history for a customer from the replica."""
    result = await session.execute(
        _RETURN_HISTORY_SQL,
        {"customer_id": customer_id},
        execution_options=_EXEC_OPTIONS,
    )
    row = result.mappings().first()
    if row is None:
        return ReturnHistory(
            customer_id=customer_id,
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
        )
    return ReturnHistory(
        customer_id=customer_id,
        repeat_return_count_30d=int(row["repeat_return_count_30d"]),
        repeat_return_count_90d=int(row["repeat_return_count_90d"]),
        repeat_return_count_365d=int(row["repeat_return_count_365d"]),
        return_rate=float(row["return_rate"]) if row["return_rate"] is not None else None,
        return_value_ratio=float(row["return_value_ratio"]),
        category_adjusted_rate=(
            float(row["category_adjusted_rate"])
            if row["category_adjusted_rate"] is not None
            else None
        ),
        high_risk_reason_count=int(row["high_risk_reason_count"]),
        late_window_return_count=int(row["late_window_return_count"]),
        total_orders=int(row["total_orders"]),
        total_returns=int(row["total_returns"]),
    )


# ---------------------------------------------------------------------------
# Cancellation history
# ---------------------------------------------------------------------------

_CANCELLATION_HISTORY_SQL = text("""
    WITH customer_orders AS (
        SELECT count(*) AS total_orders
        FROM orders
        WHERE customer_id = :customer_id
    ),
    customer_cancellations AS (
        SELECT count(*) AS total_cancellations,
               count(*) FILTER (
                   WHERE cancel_date >= now() - interval '30 days'
               ) AS cnt_30d,
               count(*) FILTER (
                   WHERE cancel_date >= now() - interval '90 days'
               ) AS cnt_90d,
               count(*) FILTER (
                   WHERE order_status_at_cancel = 'shipped'
               ) AS cancel_after_ship_count,
               bool_or(had_promo) AS promo_cancel_detected
        FROM cancellations
        WHERE customer_id = :customer_id
    ),
    cancel_reorder AS (
        SELECT count(*) AS loop_count
        FROM cancellations c
        JOIN orders cancelled_order ON cancelled_order.id = c.order_id
        JOIN order_items ci ON ci.order_id = cancelled_order.id
        JOIN orders reorder ON reorder.customer_id = c.customer_id
                           AND reorder.order_date > c.cancel_date
                           AND reorder.order_date < c.cancel_date + interval '24 hours'
                           AND reorder.id != cancelled_order.id
        JOIN order_items ri ON ri.order_id = reorder.id AND ri.sku = ci.sku
        WHERE c.customer_id = :customer_id
    )
    SELECT co.total_orders,
           cc.total_cancellations,
           cc.cnt_30d  AS cancel_count_30d,
           cc.cnt_90d  AS cancel_count_90d,
           CASE WHEN co.total_orders >= 5
                THEN cc.total_cancellations::float / co.total_orders
                ELSE NULL
           END AS cancel_rate,
           cc.cancel_after_ship_count,
           coalesce(cc.promo_cancel_detected, false) AS promo_cancel_detected,
           cr.loop_count AS cancel_reorder_loop_count
    FROM customer_orders co
    CROSS JOIN customer_cancellations cc
    CROSS JOIN cancel_reorder cr
""")


async def fetch_cancellation_history(
    session: AsyncSession,
    customer_id: str,
) -> CancellationHistory:
    """Fetch aggregated cancellation history for a customer from the replica."""
    result = await session.execute(
        _CANCELLATION_HISTORY_SQL,
        {"customer_id": customer_id},
        execution_options=_EXEC_OPTIONS,
    )
    row = result.mappings().first()
    if row is None:
        return CancellationHistory(
            customer_id=customer_id,
            cancel_count_30d=0,
            cancel_count_90d=0,
            cancel_rate=None,
            cancel_after_ship_count=0,
            promo_cancel_detected=False,
            cancel_reorder_loop_count=0,
            total_orders=0,
            total_cancellations=0,
        )
    return CancellationHistory(
        customer_id=customer_id,
        cancel_count_30d=int(row["cancel_count_30d"]),
        cancel_count_90d=int(row["cancel_count_90d"]),
        cancel_rate=float(row["cancel_rate"]) if row["cancel_rate"] is not None else None,
        cancel_after_ship_count=int(row["cancel_after_ship_count"]),
        promo_cancel_detected=bool(row["promo_cancel_detected"]),
        cancel_reorder_loop_count=int(row["cancel_reorder_loop_count"]),
        total_orders=int(row["total_orders"]),
        total_cancellations=int(row["total_cancellations"]),
    )


# ---------------------------------------------------------------------------
# Order summary
# ---------------------------------------------------------------------------

_ORDER_SUMMARY_SQL = text("""
    WITH summary AS (
        SELECT count(*) AS total_orders,
               coalesce(sum(total_amount), 0) AS total_spend
        FROM orders
        WHERE customer_id = :customer_id
    ),
    account AS (
        SELECT extract(epoch FROM now() - created_at) / 86400 AS age_days
        FROM customers
        WHERE id = :customer_id
    ),
    top_cats AS (
        SELECT oi.category
        FROM order_items oi
        JOIN orders o ON o.id = oi.order_id
        WHERE o.customer_id = :customer_id
        GROUP BY oi.category
        ORDER BY count(*) DESC
        LIMIT 5
    )
    SELECT s.total_orders,
           s.total_spend,
           coalesce(a.age_days, 0) AS account_age_days,
           coalesce(array_agg(tc.category), ARRAY[]::text[]) AS top_categories
    FROM summary s
    LEFT JOIN account a ON true
    LEFT JOIN top_cats tc ON true
    GROUP BY s.total_orders, s.total_spend, a.age_days
""")


async def fetch_order_summary(
    session: AsyncSession,
    customer_id: str,
) -> OrderSummary:
    """Fetch high-level order profile for a customer from the replica."""
    result = await session.execute(
        _ORDER_SUMMARY_SQL,
        {"customer_id": customer_id},
        execution_options=_EXEC_OPTIONS,
    )
    row = result.mappings().first()
    if row is None:
        return OrderSummary(
            customer_id=customer_id,
            total_orders=0,
            total_spend=0.0,
            account_age_days=0,
            top_categories=[],
        )
    raw_categories = row["top_categories"] or []
    categories = [c for c in raw_categories if c is not None]
    return OrderSummary(
        customer_id=customer_id,
        total_orders=int(row["total_orders"]),
        total_spend=float(row["total_spend"]),
        account_age_days=int(row["account_age_days"]),
        top_categories=categories,
    )


# ---------------------------------------------------------------------------
# Linked accounts
# ---------------------------------------------------------------------------

_LINKED_ACCOUNTS_SQL = text("""
    WITH target AS (
        SELECT id, shipping_address_hash, payment_token_hash, phone, device_id
        FROM customers
        WHERE id = :customer_id
    ),
    links AS (
        SELECT DISTINCT c.id AS linked_id,
               CASE
                   WHEN c.shipping_address_hash = t.shipping_address_hash
                        AND t.shipping_address_hash IS NOT NULL
                   THEN 'address'
                   WHEN c.payment_token_hash = t.payment_token_hash
                        AND t.payment_token_hash IS NOT NULL
                   THEN 'payment_token'
                   WHEN c.phone = t.phone
                        AND t.phone IS NOT NULL
                   THEN 'phone'
                   WHEN c.device_id = t.device_id
                        AND t.device_id IS NOT NULL
                   THEN 'device_id'
               END AS link_type
        FROM customers c
        CROSS JOIN target t
        WHERE c.id != t.id
          AND (
              (c.shipping_address_hash = t.shipping_address_hash
               AND t.shipping_address_hash IS NOT NULL)
              OR (c.payment_token_hash = t.payment_token_hash
                  AND t.payment_token_hash IS NOT NULL)
              OR (c.phone = t.phone
                  AND t.phone IS NOT NULL)
              OR (c.device_id = t.device_id
                  AND t.device_id IS NOT NULL)
          )
    ),
    abuse_scores AS (
        SELECT l.linked_id,
               CASE WHEN lo.total > 0
                    THEN lr.return_count::float / lo.total
                    ELSE 0
               END AS abuse_rate
        FROM links l
        LEFT JOIN LATERAL (
            SELECT count(*) AS total
            FROM orders
            WHERE customer_id = l.linked_id
        ) lo ON true
        LEFT JOIN LATERAL (
            SELECT count(*) AS return_count
            FROM returns
            WHERE customer_id = l.linked_id
        ) lr ON true
    )
    SELECT count(DISTINCT l.linked_id) AS linked_account_count,
           coalesce(avg(a.abuse_rate), 0) AS linked_abuse_score,
           coalesce(
               array_agg(DISTINCT l.link_type) FILTER (WHERE l.link_type IS NOT NULL),
               ARRAY[]::text[]
           ) AS linked_by
    FROM links l
    LEFT JOIN abuse_scores a ON a.linked_id = l.linked_id
""")


async def fetch_linked_accounts(
    session: AsyncSession,
    customer_id: str,
) -> LinkedAccounts:
    """Fetch linked-account analysis for a customer from the replica."""
    result = await session.execute(
        _LINKED_ACCOUNTS_SQL,
        {"customer_id": customer_id},
        execution_options=_EXEC_OPTIONS,
    )
    row = result.mappings().first()
    if row is None:
        return LinkedAccounts(
            customer_id=customer_id,
            linked_account_count=0,
            linked_abuse_score=0.0,
            linked_by=[],
        )
    raw_linked_by = row["linked_by"] or []
    linked_by = [lb for lb in raw_linked_by if lb is not None]
    return LinkedAccounts(
        customer_id=customer_id,
        linked_account_count=int(row["linked_account_count"]),
        linked_abuse_score=float(row["linked_abuse_score"]),
        linked_by=linked_by,
    )
