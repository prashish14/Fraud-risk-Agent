"""pgvector collection management for policies and resolved cases.

Two collections share the same Postgres database (the fraud DB) but use
separate tables managed by ``langchain_postgres.PGVector``.

The embedding model is injectable for testing: production uses an
Anthropic-compatible embedding model, tests pass ``FakeEmbeddings``.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

from fraud_risk_agent.config import get_settings

logger = logging.getLogger(__name__)

_POLICIES_COLLECTION = "policies"
_RESOLVED_CASES_COLLECTION = "resolved_cases"


def _get_connection_string() -> str:
    """Build the sync connection string for PGVector.

    ``langchain_postgres.PGVector`` expects a sync-style ``postgresql://``
    URL even though it can operate with async drivers internally.
    """
    settings = get_settings()
    url = settings.fraud_database_url
    return url.replace("postgresql+asyncpg://", "postgresql://")


def get_policies_store(embeddings: Embeddings) -> PGVector:
    """Return the PGVector store for the *policies* collection."""
    return PGVector(
        collection_name=_POLICIES_COLLECTION,
        connection=_get_connection_string(),
        embeddings=embeddings,
    )


def get_cases_store(embeddings: Embeddings) -> PGVector:
    """Return the PGVector store for the *resolved_cases* collection."""
    return PGVector(
        collection_name=_RESOLVED_CASES_COLLECTION,
        connection=_get_connection_string(),
        embeddings=embeddings,
    )
