"""RAG knowledge base — policies and resolved cases."""

from fraud_risk_agent.knowledge.search import (
    CaseResult,
    PolicyResult,
    find_similar_cases,
    search_policies,
)

__all__ = [
    "CaseResult",
    "PolicyResult",
    "find_similar_cases",
    "search_policies",
]
