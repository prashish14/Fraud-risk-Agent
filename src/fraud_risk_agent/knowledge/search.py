"""RAG search tools — search_policies and find_similar_cases.

These are the real implementations that replace the placeholder stubs
in ``agent/tools.py``. They query the pgvector collections and return
structured results.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from langchain_core.embeddings import Embeddings

from fraud_risk_agent.knowledge.store import get_cases_store, get_policies_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyResult:
    """A single policy search result."""

    content: str
    source: str
    section: str
    score: float


@dataclass(frozen=True)
class CaseResult:
    """A single similar-case search result."""

    content: str
    assessment_id: str
    score: int
    risk_level: str
    analyst_verdict: str | None
    similarity: float


def search_policies(
    query: str,
    embeddings: Embeddings,
    top_k: int = 5,
) -> list[PolicyResult]:
    """Search the policies collection for passages matching *query*."""
    store = get_policies_store(embeddings)
    results = store.similarity_search_with_score(query, k=top_k)

    policy_results = []
    for doc, score in results:
        policy_results.append(
            PolicyResult(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                section=doc.metadata.get("h2", doc.metadata.get("h1", "")),
                score=float(score),
            )
        )
    return policy_results


def find_similar_cases(
    query: str,
    embeddings: Embeddings,
    score_range: int = 15,
    current_score: int | None = None,
    top_k: int = 5,
) -> list[CaseResult]:
    """Find resolved cases similar to the given signals description.

    If *current_score* is provided, results are post-filtered to cases
    within ``±score_range`` of the current score.
    """
    store = get_cases_store(embeddings)
    results = store.similarity_search_with_score(query, k=top_k * 2)

    case_results = []
    for doc, similarity in results:
        case_score = doc.metadata.get("score", 0)
        if current_score is not None and abs(case_score - current_score) > score_range:
            continue

        case_results.append(
            CaseResult(
                content=doc.page_content,
                assessment_id=doc.metadata.get("assessment_id", ""),
                score=case_score,
                risk_level=doc.metadata.get("risk_level", "unknown"),
                analyst_verdict=doc.metadata.get("analyst_verdict"),
                similarity=float(similarity),
            )
        )
        if len(case_results) >= top_k:
            break

    return case_results


def search_policies_as_dicts(
    query: str,
    embeddings: Embeddings,
    top_k: int = 5,
) -> list[dict]:
    """Convenience wrapper returning dicts for JSON serialisation."""
    return [asdict(r) for r in search_policies(query, embeddings, top_k)]


def find_similar_cases_as_dicts(
    query: str,
    embeddings: Embeddings,
    current_score: int | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Convenience wrapper returning dicts for JSON serialisation."""
    return [
        asdict(r)
        for r in find_similar_cases(
            query, embeddings, current_score=current_score, top_k=top_k,
        )
    ]
