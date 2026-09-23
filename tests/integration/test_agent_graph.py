"""Integration tests for the LangGraph fraud investigation pipeline.

Uses FakeListChatModel and mocked data -- no real DB or LLM calls.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from unittest.mock import AsyncMock, patch

import pytest

from fraud_risk_agent.agent.graph import build_graph
from fraud_risk_agent.agent.nodes import _expected_risk_level, validate_assessment
from fraud_risk_agent.agent.state import FraudAgentState
from fraud_risk_agent.data.queries import (
    CancellationHistory,
    LinkedAccounts,
    OrderSummary,
    ReturnHistory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CUSTOMER_ID = "test-cust-001"


def _make_return_history(**overrides: object) -> dict:
    base = asdict(
        ReturnHistory(
            customer_id=_CUSTOMER_ID,
            repeat_return_count_30d=0,
            repeat_return_count_90d=0,
            repeat_return_count_365d=0,
            return_rate=0.0,
            return_value_ratio=0.0,
            category_adjusted_rate=0.0,
            high_risk_reason_count=0,
            late_window_return_count=0,
            total_orders=20,
            total_returns=0,
        )
    )
    base.update(overrides)
    return base


def _make_cancellation_history(**overrides: object) -> dict:
    base = asdict(
        CancellationHistory(
            customer_id=_CUSTOMER_ID,
            cancel_count_30d=0,
            cancel_count_90d=0,
            cancel_rate=0.0,
            cancel_after_ship_count=0,
            promo_cancel_detected=False,
            cancel_reorder_loop_count=0,
            total_orders=20,
            total_cancellations=0,
        )
    )
    base.update(overrides)
    return base


def _make_order_summary(**overrides: object) -> dict:
    base = asdict(
        OrderSummary(
            customer_id=_CUSTOMER_ID,
            total_orders=20,
            total_spend=1000.0,
            account_age_days=365,
            top_categories=["electronics"],
        )
    )
    base.update(overrides)
    return base


def _make_linked_accounts(**overrides: object) -> dict:
    base = asdict(
        LinkedAccounts(
            customer_id=_CUSTOMER_ID,
            linked_account_count=0,
            linked_abuse_score=0.0,
            linked_by=[],
        )
    )
    base.update(overrides)
    return base


def _make_fetch_data_result(**overrides: object) -> dict:
    """Return a mock fetch_data result with defaults."""
    result = {
        "return_history": _make_return_history(),
        "cancellation_history": _make_cancellation_history(),
        "order_summary": _make_order_summary(),
        "linked_accounts": _make_linked_accounts(),
    }
    result.update(overrides)
    return result


def _initial_state(**overrides: object) -> dict:
    """Return a minimal initial state for graph invocation."""
    base: dict = {
        "customer_id": _CUSTOMER_ID,
        "order_id": None,
        "context": None,
        "assessment_id": str(uuid.uuid4()),
        "return_history": None,
        "cancellation_history": None,
        "order_summary": None,
        "linked_accounts": None,
        "features": None,
        "scoring_result": None,
        "messages": [],
        "tool_call_count": 0,
        "assessment": None,
        "error": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Low-risk path
# ---------------------------------------------------------------------------


async def test_low_risk_path_produces_allow_assessment() -> None:
    """A low-risk customer (score < 40) gets 'allow' with no narrative."""
    low_risk_data = _make_fetch_data_result(
        return_history=_make_return_history(
            return_rate=0.05,
            return_value_ratio=0.02,
            high_risk_reason_count=0,
            late_window_return_count=0,
            total_orders=20,
        ),
    )

    async def mock_fetch_data(state: FraudAgentState) -> dict:
        return low_risk_data

    graph = build_graph()

    with patch("fraud_risk_agent.agent.nodes.fetch_data", new=mock_fetch_data):
        # Rebuild the graph so it picks up the patched node
        pass

    # Instead, patch at the data layer
    mock_session = AsyncMock()

    async def mock_replica_session():
        """Yield a mock session."""

        class _Ctx:
            async def __aenter__(self):
                return mock_session

            async def __aexit__(self, *args):
                pass

        return _Ctx()

    with (
        patch(
            "fraud_risk_agent.agent.nodes.get_replica_session",
            return_value=_ctx_manager_returning(low_risk_data),
        ),
        patch(
            "fraud_risk_agent.agent.nodes.fetch_return_history",
            new_callable=AsyncMock,
            return_value=ReturnHistory(
                **_make_return_history(
                    return_rate=0.05,
                    return_value_ratio=0.02,
                    high_risk_reason_count=0,
                    late_window_return_count=0,
                    total_orders=20,
                )
            ),
        ),
        patch(
            "fraud_risk_agent.agent.nodes.fetch_cancellation_history",
            new_callable=AsyncMock,
            return_value=CancellationHistory(**_make_cancellation_history()),
        ),
        patch(
            "fraud_risk_agent.agent.nodes.fetch_order_summary",
            new_callable=AsyncMock,
            return_value=OrderSummary(**_make_order_summary()),
        ),
        patch(
            "fraud_risk_agent.agent.nodes.fetch_linked_accounts",
            new_callable=AsyncMock,
            return_value=LinkedAccounts(**_make_linked_accounts()),
        ),
    ):
        result = await graph.ainvoke(_initial_state())

    assessment = result.get("assessment")
    assert assessment is not None
    assert assessment["recommended_action"] == "allow"
    assert assessment["risk_level"] == "low"
    assert assessment["narrative"] is None
    assert assessment["narrative_available"] is False
    # Score should be low (under 40)
    assert assessment["score"] < 40


def _ctx_manager_returning(data: object):
    """Return a fake async context manager."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _ctx():
        yield AsyncMock()

    return _ctx()


# ---------------------------------------------------------------------------
# Medium-risk path (investigated)
# ---------------------------------------------------------------------------


async def test_medium_risk_path_triggers_investigation() -> None:
    """A medium-risk customer (score 40-69) gets investigated by the LLM."""
    from langchain_core.messages import AIMessage

    # LLM response for investigation (no tool calls -- just a summary)
    investigation_response = AIMessage(content="Investigation complete. Customer shows patterns.")
    # LLM response for draft_assessment
    draft_response = AIMessage(
        content=json.dumps(
            {
                "recommended_action": "hold_refund_for_review",
                "confidence": 0.75,
                "signals": [
                    {
                        "name": "high_risk_reasons",
                        "description": "Multiple high-risk return reasons",
                        "severity": "high",
                        "evidence": "3 fraud-related return reasons detected",
                    }
                ],
                "evidence": ["3 high-risk return reasons in last 90 days"],
                "narrative": (
                    "Customer shows concerning return patterns with multiple fraud reasons."
                ),
            }
        )
    )

    call_count = 0

    async def mock_ainvoke(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return investigation_response
        return draft_response

    with (
        patch(
            "fraud_risk_agent.agent.nodes.get_replica_session",
            return_value=_ctx_manager_returning(None),
        ),
        patch(
            "fraud_risk_agent.agent.nodes.fetch_return_history",
            new_callable=AsyncMock,
            return_value=ReturnHistory(
                **_make_return_history(
                    return_rate=0.5,
                    return_value_ratio=0.6,
                    high_risk_reason_count=5,
                    late_window_return_count=3,
                    total_orders=20,
                )
            ),
        ),
        patch(
            "fraud_risk_agent.agent.nodes.fetch_cancellation_history",
            new_callable=AsyncMock,
            return_value=CancellationHistory(**_make_cancellation_history()),
        ),
        patch(
            "fraud_risk_agent.agent.nodes.fetch_order_summary",
            new_callable=AsyncMock,
            return_value=OrderSummary(**_make_order_summary()),
        ),
        patch(
            "fraud_risk_agent.agent.nodes.fetch_linked_accounts",
            new_callable=AsyncMock,
            return_value=LinkedAccounts(**_make_linked_accounts()),
        ),
        patch(
            "langchain_anthropic.ChatAnthropic.ainvoke",
            side_effect=mock_ainvoke,
        ),
        patch(
            "langchain_anthropic.ChatAnthropic.bind_tools",
            return_value=AsyncMock(ainvoke=mock_ainvoke),
        ),
    ):
        graph = build_graph()
        result = await graph.ainvoke(_initial_state())

    assessment = result.get("assessment")
    assert assessment is not None
    assert assessment["risk_level"] in ("medium", "high")
    assert assessment["score"] >= 40
    assert assessment["recommended_action"] in (
        "hold_refund_for_review",
        "inspect_return",
        "escalate_to_analyst",
    )
    assert assessment["narrative"] is not None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_catches_risk_level_mismatch() -> None:
    """validate_assessment corrects a score/risk_level mismatch."""
    state: dict = _initial_state(
        assessment={
            "assessment_id": str(uuid.uuid4()),
            "customer_id": _CUSTOMER_ID,
            "order_id": None,
            "score": 75,
            "risk_level": "low",  # wrong -- should be "high"
            "recommended_action": "allow",
            "confidence": 0.5,
            "rules_version": "v1.0",
            "model_version": None,
            "narrative": None,
            "narrative_available": False,
            "signals": [],
            "evidence": [],
            "policy_citations": [],
            "similar_cases": [],
            "fired_rules": [],
            "created_at": "2024-01-01T00:00:00",
        },
        scoring_result={"score": 75, "risk_level": "high", "fired_rules": []},
    )

    result = validate_assessment(state)
    assessment = result["assessment"]
    assert assessment["risk_level"] == "high"
    # The action should also have been corrected from "allow"
    assert assessment["recommended_action"] in (
        "hold_refund_for_review",
        "escalate_to_analyst",
    )


def test_validate_passes_correct_assessment() -> None:
    """validate_assessment passes through a correct assessment unchanged."""
    state: dict = _initial_state(
        assessment={
            "assessment_id": str(uuid.uuid4()),
            "customer_id": _CUSTOMER_ID,
            "order_id": None,
            "score": 15,
            "risk_level": "low",
            "recommended_action": "allow",
            "confidence": 0.95,
            "rules_version": "v1.0",
            "model_version": None,
            "narrative": None,
            "narrative_available": False,
            "signals": [],
            "evidence": [],
            "policy_citations": [],
            "similar_cases": [],
            "fired_rules": [],
            "created_at": "2024-01-01T00:00:00",
        },
        scoring_result={"score": 15, "risk_level": "low", "fired_rules": []},
    )

    result = validate_assessment(state)
    assessment = result["assessment"]
    assert assessment["score"] == 15
    assert assessment["risk_level"] == "low"
    assert assessment["recommended_action"] == "allow"


# ---------------------------------------------------------------------------
# Risk level helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "low"),
        (39, "low"),
        (40, "medium"),
        (69, "medium"),
        (70, "high"),
        (100, "high"),
    ],
)
def test_expected_risk_level(score: int, expected: str) -> None:
    """_expected_risk_level maps scores to risk levels correctly."""
    assert _expected_risk_level(score) == expected
