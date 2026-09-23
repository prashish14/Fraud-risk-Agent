"""LangGraph TypedDict state schema for the fraud investigation graph."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class FraudAgentState(TypedDict):
    """Typed state flowing through every node of the investigation graph."""

    # Input
    customer_id: str
    order_id: str | None
    context: str | None  # "refund_request", "return_request", etc.
    assessment_id: str

    # Data layer results (populated by fetch_data node)
    return_history: dict | None
    cancellation_history: dict | None
    order_summary: dict | None
    linked_accounts: dict | None

    # Scoring (populated by compute_score node)
    features: dict | None
    scoring_result: dict | None

    # Investigation (LLM messages)
    messages: Annotated[list[BaseMessage], add_messages]
    tool_call_count: int

    # Output
    assessment: dict | None
    error: str | None
