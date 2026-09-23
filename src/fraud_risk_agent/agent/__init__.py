"""Fraud investigation agent — LangGraph pipeline.

Public API
----------
- :class:`FraudAssessment` — structured output contract
- :func:`build_graph` — returns the compiled ``StateGraph``
- :func:`run_assessment` — convenience coroutine for a single assessment
"""

from fraud_risk_agent.agent.graph import build_graph, run_assessment
from fraud_risk_agent.agent.models import FraudAssessment

__all__ = [
    "FraudAssessment",
    "build_graph",
    "run_assessment",
]
