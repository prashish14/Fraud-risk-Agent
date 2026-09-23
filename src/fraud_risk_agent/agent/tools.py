"""LangChain tool wrappers for the investigator agent.

Each data tool opens a read-only replica session, calls the corresponding
fetch function, and returns a sanitised JSON string (no PII).
RAG tools are placeholders until Phase 4 is implemented.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from langchain_core.tools import tool

from fraud_risk_agent.data.connection import get_replica_session
from fraud_risk_agent.data.queries import (
    fetch_cancellation_history,
    fetch_linked_accounts,
    fetch_order_summary,
    fetch_return_history,
)

logger = logging.getLogger(__name__)


def _sanitise_dict(data: dict) -> dict:
    """Strip PII fields from a data dict before returning to the LLM.

    Currently the dataclasses contain no PII, but this is the
    centralised hook where PII filtering would be added.
    """
    return data


@tool
async def query_return_history(customer_id: str) -> str:
    """Query the return history for a customer.

    Returns aggregated return counts, rates, and risk indicators.
    """
    try:
        async with get_replica_session() as session:
            result = await fetch_return_history(session, customer_id)
        data = _sanitise_dict(asdict(result))
        return json.dumps(data, default=str)
    except Exception:
        logger.exception("Failed to fetch return history for %s", customer_id)
        return json.dumps({"error": "Failed to fetch return history", "customer_id": customer_id})


@tool
async def query_cancellation_history(customer_id: str) -> str:
    """Query the cancellation history for a customer.

    Returns aggregated cancellation counts, rates, and pattern indicators.
    """
    try:
        async with get_replica_session() as session:
            result = await fetch_cancellation_history(session, customer_id)
        data = _sanitise_dict(asdict(result))
        return json.dumps(data, default=str)
    except Exception:
        logger.exception("Failed to fetch cancellation history for %s", customer_id)
        return json.dumps(
            {"error": "Failed to fetch cancellation history", "customer_id": customer_id}
        )


@tool
async def query_order_summary(customer_id: str) -> str:
    """Query the order summary for a customer.

    Returns total orders, spend, account age, and top categories.
    """
    try:
        async with get_replica_session() as session:
            result = await fetch_order_summary(session, customer_id)
        data = _sanitise_dict(asdict(result))
        return json.dumps(data, default=str)
    except Exception:
        logger.exception("Failed to fetch order summary for %s", customer_id)
        return json.dumps({"error": "Failed to fetch order summary", "customer_id": customer_id})


@tool
async def query_linked_accounts(customer_id: str) -> str:
    """Find accounts linked to this customer by shared identifiers.

    Returns linked account count, abuse score, and link types.
    """
    try:
        async with get_replica_session() as session:
            result = await fetch_linked_accounts(session, customer_id)
        data = _sanitise_dict(asdict(result))
        return json.dumps(data, default=str)
    except Exception:
        logger.exception("Failed to fetch linked accounts for %s", customer_id)
        return json.dumps({"error": "Failed to fetch linked accounts", "customer_id": customer_id})


@tool
async def search_fraud_policies(query: str) -> str:
    """Search fraud and return policies for relevant guidance.

    Searches the RAG knowledge base for policy passages matching the query.
    """
    # Placeholder until Phase 4 RAG is implemented
    return "Policy search not yet available. Proceed with available evidence."


@tool
async def find_similar_fraud_cases(signals_summary: str, score: int) -> str:
    """Find past fraud cases similar to the current assessment.

    Searches the case database for past assessments with similar signals
    and scores.
    """
    # Placeholder until Phase 4 RAG is implemented
    return "Case search not yet available. Proceed with available evidence."


INVESTIGATION_TOOLS = [
    query_return_history,
    query_cancellation_history,
    query_order_summary,
    query_linked_accounts,
    search_fraud_policies,
    find_similar_fraud_cases,
]
