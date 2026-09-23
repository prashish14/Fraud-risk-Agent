"""LangGraph StateGraph definition -- the core processing pipeline.

Graph topology::

    fetch_data -> compute_score_node -> gate
      gate --(low_risk)--> draft_assessment
      gate --(investigate)--> investigate -> draft_assessment
    draft_assessment -> validate_assessment -> END
"""

from __future__ import annotations

import uuid

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from fraud_risk_agent.agent.nodes import (
    compute_score_node,
    draft_assessment,
    fetch_data,
    gate,
    investigate,
    validate_assessment,
)
from fraud_risk_agent.agent.state import FraudAgentState


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> StateGraph:
    """Build and compile the fraud investigation state graph.

    Args:
        checkpointer: Optional checkpoint saver (e.g. ``MemorySaver``
            for tests, ``PostgresSaver`` for production). When ``None``,
            the graph runs without checkpointing.

    Returns:
        A compiled ``StateGraph`` ready to invoke.
    """
    graph = StateGraph(FraudAgentState)

    # Add nodes
    graph.add_node("fetch_data", fetch_data)
    graph.add_node("compute_score", compute_score_node)
    graph.add_node("investigate", investigate)
    graph.add_node("draft_assessment", draft_assessment)
    graph.add_node("validate_assessment", validate_assessment)

    # Edges
    graph.set_entry_point("fetch_data")
    graph.add_edge("fetch_data", "compute_score")
    graph.add_conditional_edges(
        "compute_score",
        gate,
        {
            "low_risk": "draft_assessment",
            "investigate": "investigate",
        },
    )
    graph.add_edge("investigate", "draft_assessment")
    graph.add_edge("draft_assessment", "validate_assessment")
    graph.add_edge("validate_assessment", END)

    return graph.compile(checkpointer=checkpointer)


async def run_assessment(
    customer_id: str,
    order_id: str | None = None,
    context: str | None = None,
) -> dict:
    """Convenience function to run a full fraud assessment.

    Args:
        customer_id: The customer to assess.
        order_id: Optional order triggering the assessment.
        context: Optional context string (e.g. ``"refund_request"``).

    Returns:
        The validated ``FraudAssessment`` as a dict.
    """
    graph = build_graph()

    initial_state: FraudAgentState = {
        "customer_id": customer_id,
        "order_id": order_id,
        "context": context,
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

    result = await graph.ainvoke(initial_state)
    return result.get("assessment", {})
