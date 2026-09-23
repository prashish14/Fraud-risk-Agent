"""Node functions for the LangGraph investigation graph.

Each function receives a ``FraudAgentState`` and returns a dict of
state updates. The LLM is called only in ``investigate`` and
``draft_assessment``; all other nodes are deterministic.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from fraud_risk_agent.agent.models import FraudAssessment
from fraud_risk_agent.agent.state import FraudAgentState
from fraud_risk_agent.agent.tools import INVESTIGATION_TOOLS
from fraud_risk_agent.config import get_settings
from fraud_risk_agent.data.connection import get_replica_session
from fraud_risk_agent.data.queries import (
    fetch_cancellation_history,
    fetch_linked_accounts,
    fetch_order_summary,
    fetch_return_history,
)
from fraud_risk_agent.scoring.config import load_scoring_config
from fraud_risk_agent.scoring.engine import compute_score
from fraud_risk_agent.scoring.features import extract_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for the investigation LLM
# ---------------------------------------------------------------------------

_INVESTIGATION_PROMPT = """\
You are a fraud investigation analyst. You have been given a customer
flagged by the deterministic scoring engine. Your job is to investigate
using the available tools and produce a concise fraud assessment.

SCORING CONTEXT:
- Customer ID: {customer_id}
- Risk score: {score}/100 (risk level: {risk_level})
- Fired rules: {fired_rules}

INSTRUCTIONS:
1. Review the scoring signals that fired.
2. Use the available tools to gather additional context about the customer.
3. Look for patterns that confirm or refute the fraud indicators.
4. Consider whether the signals could have benign explanations.
5. Do NOT invent data. Only reference data you received from tools.

When you have enough evidence, stop calling tools and summarise your findings.
"""

_DRAFT_PROMPT = """\
Based on the investigation so far, produce a structured fraud assessment.

SCORING CONTEXT (these values are authoritative — do not change them):
- Score: {score}
- Risk level: {risk_level}
- Fired rules: {fired_rules}

Produce a JSON object with these fields:
- "recommended_action": one of "allow", "inspect_return",
  "hold_refund_for_review", "escalate_to_analyst"
- "confidence": float 0.0-1.0, your confidence in the recommendation
- "signals": list of {{"name": str, "description": str,
  "severity": "low"|"medium"|"high", "evidence": str}}
- "evidence": list of strings — key evidence points
- "narrative": a 2-4 sentence summary of your findings

Choose recommended_action based on the risk level:
- low risk (score 0-39): "allow"
- medium risk (score 40-69): "inspect_return" or "hold_refund_for_review"
- high risk (score 70-100): "hold_refund_for_review" or "escalate_to_analyst"

Return ONLY the JSON object, no markdown fences or extra text.
"""


# ---------------------------------------------------------------------------
# fetch_data
# ---------------------------------------------------------------------------


async def fetch_data(state: FraudAgentState) -> dict:
    """Fetch all data for the customer in parallel.

    On timeout or error for any individual query, that field is set to
    ``None`` and a warning is logged. The pipeline continues with whatever
    data is available.
    """
    customer_id = state["customer_id"]
    settings = get_settings()
    timeout = settings.llm_timeout_seconds

    async def _fetch_returns() -> dict | None:
        try:
            async with get_replica_session() as session:
                result = await asyncio.wait_for(
                    fetch_return_history(session, customer_id),
                    timeout=timeout,
                )
            return asdict(result)
        except Exception:
            logger.warning("Failed to fetch return history for %s", customer_id, exc_info=True)
            return None

    async def _fetch_cancellations() -> dict | None:
        try:
            async with get_replica_session() as session:
                result = await asyncio.wait_for(
                    fetch_cancellation_history(session, customer_id),
                    timeout=timeout,
                )
            return asdict(result)
        except Exception:
            logger.warning(
                "Failed to fetch cancellation history for %s", customer_id, exc_info=True
            )
            return None

    async def _fetch_orders() -> dict | None:
        try:
            async with get_replica_session() as session:
                result = await asyncio.wait_for(
                    fetch_order_summary(session, customer_id),
                    timeout=timeout,
                )
            return asdict(result)
        except Exception:
            logger.warning("Failed to fetch order summary for %s", customer_id, exc_info=True)
            return None

    async def _fetch_linked() -> dict | None:
        try:
            async with get_replica_session() as session:
                result = await asyncio.wait_for(
                    fetch_linked_accounts(session, customer_id),
                    timeout=timeout,
                )
            return asdict(result)
        except Exception:
            logger.warning("Failed to fetch linked accounts for %s", customer_id, exc_info=True)
            return None

    returns, cancellations, orders, linked = await asyncio.gather(
        _fetch_returns(),
        _fetch_cancellations(),
        _fetch_orders(),
        _fetch_linked(),
    )

    return {
        "return_history": returns,
        "cancellation_history": cancellations,
        "order_summary": orders,
        "linked_accounts": linked,
    }


# ---------------------------------------------------------------------------
# compute_score_node
# ---------------------------------------------------------------------------


def _reconstruct_dataclass(cls, data: dict | None):
    """Reconstruct a dataclass from a dict, returning None if data is None."""
    if data is None:
        return None
    return cls(**data)


async def compute_score_node(state: FraudAgentState) -> dict:
    """Extract features and compute the deterministic score.

    Uses the data fetched by ``fetch_data``. If any data source is
    missing, falls back to zero-valued defaults for that source.
    """
    from fraud_risk_agent.data.queries import (
        CancellationHistory,
        LinkedAccounts,
        OrderSummary,
        ReturnHistory,
    )

    customer_id = state["customer_id"]

    returns = _reconstruct_dataclass(ReturnHistory, state.get("return_history"))
    if returns is None:
        returns = ReturnHistory(
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

    cancellations = _reconstruct_dataclass(CancellationHistory, state.get("cancellation_history"))
    if cancellations is None:
        cancellations = CancellationHistory(
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

    orders = _reconstruct_dataclass(OrderSummary, state.get("order_summary"))
    if orders is None:
        orders = OrderSummary(
            customer_id=customer_id,
            total_orders=0,
            total_spend=0.0,
            account_age_days=0,
            top_categories=[],
        )

    linked = _reconstruct_dataclass(LinkedAccounts, state.get("linked_accounts"))
    if linked is None:
        linked = LinkedAccounts(
            customer_id=customer_id,
            linked_account_count=0,
            linked_abuse_score=0.0,
            linked_by=[],
        )

    features = extract_features(returns, cancellations, orders, linked)

    settings = get_settings()
    config = load_scoring_config(settings.scoring_config_path)
    scoring_result = compute_score(features, config)

    return {
        "features": asdict(features),
        "scoring_result": {
            "score": scoring_result.score,
            "risk_level": scoring_result.risk_level,
            "fired_rules": [asdict(r) for r in scoring_result.fired_rules],
        },
    }


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def gate(state: FraudAgentState) -> str:
    """Conditional edge: route to 'low_risk' or 'investigate'.

    Uses the batch_score_threshold from settings to decide whether
    the customer needs LLM investigation.
    """
    scoring_result = state.get("scoring_result")
    if scoring_result is None:
        return "investigate"

    settings = get_settings()
    score = scoring_result.get("score", 0)

    if score < settings.batch_score_threshold:
        return "low_risk"
    return "investigate"


# ---------------------------------------------------------------------------
# investigate
# ---------------------------------------------------------------------------


async def investigate(state: FraudAgentState) -> dict:
    """Run the LLM with tools to investigate the flagged customer.

    The LLM can call the investigation tools to gather more data.
    Stops after ``max_tool_calls`` invocations or if the LLM stops
    calling tools. On LLM timeout, returns current messages unchanged.
    """
    settings = get_settings()
    scoring_result = state.get("scoring_result", {}) or {}

    fired_rules_summary = json.dumps(scoring_result.get("fired_rules", []), indent=2)

    system_msg = SystemMessage(
        content=_INVESTIGATION_PROMPT.format(
            customer_id=state["customer_id"],
            score=scoring_result.get("score", 0),
            risk_level=scoring_result.get("risk_level", "unknown"),
            fired_rules=fired_rules_summary,
        )
    )

    llm = ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key.get_secret_value(),
        timeout=settings.llm_timeout_seconds,
        max_tokens=4096,
    )
    llm_with_tools = llm.bind_tools(INVESTIGATION_TOOLS)

    messages = list(state.get("messages", []))
    tool_call_count = state.get("tool_call_count", 0)
    max_calls = settings.max_tool_calls

    # Seed the conversation if empty
    if not messages:
        messages.append(
            HumanMessage(
                content=(
                    f"Investigate customer {state['customer_id']} "
                    f"who scored {scoring_result.get('score', 0)}/100 "
                    f"for fraud risk."
                )
            )
        )

    try:
        while tool_call_count < max_calls:
            response = await llm_with_tools.ainvoke(
                [system_msg, *messages],
            )
            messages.append(response)

            # If the LLM didn't call any tools, investigation is done
            if not isinstance(response, AIMessage) or not response.tool_calls:
                break

            # Execute tool calls
            for tc in response.tool_calls:
                tool_call_count += 1
                tool_map = {t.name: t for t in INVESTIGATION_TOOLS}
                tool_fn = tool_map.get(tc["name"])
                if tool_fn is None:
                    from langchain_core.messages import ToolMessage

                    messages.append(
                        ToolMessage(
                            content=f"Unknown tool: {tc['name']}",
                            tool_call_id=tc["id"],
                        )
                    )
                    continue

                try:
                    result = await tool_fn.ainvoke(tc["args"])
                except Exception:
                    logger.exception("Tool %s failed", tc["name"])
                    result = json.dumps({"error": f"Tool {tc['name']} failed"})

                from langchain_core.messages import ToolMessage

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"],
                    )
                )

                if tool_call_count >= max_calls:
                    break

    except Exception:
        logger.exception("LLM investigation failed for %s", state["customer_id"])
        # Continue with whatever messages we have; draft_assessment
        # will fall back to score-only assessment.

    return {
        "messages": messages,
        "tool_call_count": tool_call_count,
    }


# ---------------------------------------------------------------------------
# draft_assessment
# ---------------------------------------------------------------------------


def _action_for_risk_level(risk_level: str) -> str:
    """Map risk level to a default recommended action."""
    if risk_level == "low":
        return "allow"
    if risk_level == "medium":
        return "hold_refund_for_review"
    return "escalate_to_analyst"


async def draft_assessment(state: FraudAgentState) -> dict:
    """Build the FraudAssessment from investigation results.

    For low-risk customers: produce a deterministic assessment with
    no narrative and no LLM call.

    For investigated customers: use the LLM with structured output
    to produce signals, evidence, and a narrative. Falls back to a
    score-only assessment if the LLM fails.
    """
    scoring_result = state.get("scoring_result", {}) or {}
    score = scoring_result.get("score", 0)
    risk_level = scoring_result.get("risk_level", "low")
    fired_rules = scoring_result.get("fired_rules", [])

    settings = get_settings()
    config = load_scoring_config(settings.scoring_config_path)

    base_assessment = {
        "assessment_id": state["assessment_id"],
        "customer_id": state["customer_id"],
        "order_id": state.get("order_id"),
        "score": score,
        "risk_level": risk_level,
        "fired_rules": fired_rules,
        "rules_version": config.version,
        "model_version": settings.llm_model,
    }

    # Low-risk: deterministic assessment, no LLM call
    if risk_level == "low":
        assessment = FraudAssessment(
            **base_assessment,
            recommended_action="allow",
            confidence=0.95,
            narrative=None,
            narrative_available=False,
        )
        return {"assessment": assessment.model_dump(mode="json")}

    # Investigated: use LLM to generate structured output
    try:
        llm = ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_tokens=4096,
        )

        messages = list(state.get("messages", []))
        draft_msg = HumanMessage(
            content=_DRAFT_PROMPT.format(
                score=score,
                risk_level=risk_level,
                fired_rules=json.dumps(fired_rules, indent=2),
            )
        )

        response = await llm.ainvoke([*messages, draft_msg])
        content = response.content if isinstance(response.content, str) else str(response.content)

        # Parse the LLM's JSON response
        llm_assessment = json.loads(content)

        assessment = FraudAssessment(
            **base_assessment,
            recommended_action=llm_assessment.get(
                "recommended_action", _action_for_risk_level(risk_level)
            ),
            confidence=max(0.0, min(1.0, float(llm_assessment.get("confidence", 0.5)))),
            signals=[
                {
                    "name": s.get("name", "unknown"),
                    "description": s.get("description", ""),
                    "severity": s.get("severity", "medium"),
                    "evidence": s.get("evidence", ""),
                }
                for s in llm_assessment.get("signals", [])
            ],
            evidence=llm_assessment.get("evidence", []),
            narrative=llm_assessment.get("narrative"),
            narrative_available=True,
        )
        return {"assessment": assessment.model_dump(mode="json")}

    except Exception:
        logger.exception("Failed to draft LLM assessment for %s", state["customer_id"])
        # Fallback: score-only assessment
        assessment = FraudAssessment(
            **base_assessment,
            recommended_action=_action_for_risk_level(risk_level),
            confidence=0.3,
            narrative="Assessment generated from scoring data only; LLM analysis unavailable.",
            narrative_available=False,
        )
        return {"assessment": assessment.model_dump(mode="json")}


# ---------------------------------------------------------------------------
# validate_assessment
# ---------------------------------------------------------------------------

_RISK_LEVEL_FOR_SCORE = {
    range(0, 40): "low",
    range(40, 70): "medium",
    range(70, 101): "high",
}

_VALID_ACTIONS = {"allow", "inspect_return", "hold_refund_for_review", "escalate_to_analyst"}


def _expected_risk_level(score: int) -> str:
    """Return the expected risk level for a given score."""
    if score <= 39:
        return "low"
    if score <= 69:
        return "medium"
    return "high"


def validate_assessment(state: FraudAgentState) -> dict:
    """Validate the assessment for internal consistency.

    Checks:
    - score and risk_level agree
    - recommended_action is valid
    - required fields are present

    On validation failure, attempts one correction. If the corrected
    assessment still fails, produces a minimal fallback.
    """
    assessment_data = state.get("assessment")
    if assessment_data is None:
        return {"error": "No assessment produced"}

    errors: list[str] = []

    score = assessment_data.get("score", 0)
    risk_level = assessment_data.get("risk_level", "low")
    expected_level = _expected_risk_level(score)

    if risk_level != expected_level:
        errors.append(
            f"risk_level '{risk_level}' does not match score {score} (expected '{expected_level}')"
        )
        assessment_data = {**assessment_data, "risk_level": expected_level}

    action = assessment_data.get("recommended_action")
    if action not in _VALID_ACTIONS:
        errors.append(f"Invalid recommended_action: {action}")
        assessment_data = {
            **assessment_data,
            "recommended_action": _action_for_risk_level(expected_level),
        }
    elif risk_level != expected_level:
        # Risk level was corrected, so ensure the action is consistent
        # with the new risk level.
        corrected_action = _action_for_risk_level(expected_level)
        errors.append(
            f"Corrected recommended_action from '{action}' to "
            f"'{corrected_action}' to match risk level '{expected_level}'"
        )
        assessment_data = {**assessment_data, "recommended_action": corrected_action}

    confidence = assessment_data.get("confidence")
    if confidence is not None and not (0.0 <= confidence <= 1.0):
        errors.append(f"confidence {confidence} out of range [0, 1]")
        assessment_data = {**assessment_data, "confidence": max(0.0, min(1.0, confidence))}

    if errors:
        logger.warning(
            "Assessment validation issues for %s: %s",
            assessment_data.get("customer_id", "unknown"),
            "; ".join(errors),
        )

    # Final validation pass through the Pydantic model
    try:
        validated = FraudAssessment(**assessment_data)
        return {"assessment": validated.model_dump(mode="json")}
    except Exception:
        logger.exception("Assessment validation failed, producing fallback")
        scoring_result = state.get("scoring_result", {}) or {}
        fallback = FraudAssessment(
            assessment_id=state.get("assessment_id", ""),
            customer_id=state.get("customer_id", "unknown"),
            score=scoring_result.get("score", 0),
            risk_level=_expected_risk_level(scoring_result.get("score", 0)),
            recommended_action=_action_for_risk_level(
                _expected_risk_level(scoring_result.get("score", 0))
            ),
            confidence=0.1,
            rules_version="unknown",
            narrative="Fallback assessment after validation failure.",
            narrative_available=False,
            fired_rules=scoring_result.get("fired_rules", []),
        )
        return {"assessment": fallback.model_dump(mode="json"), "error": "Validation failed"}
