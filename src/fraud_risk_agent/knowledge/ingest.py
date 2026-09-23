"""Document chunking and ingestion pipeline for RAG collections.

Two pipelines:
- **Policies**: Markdown documents split by heading, ingested into the
  ``policies`` pgvector collection.
- **Resolved cases**: ``fraud_cases`` rows with an analyst verdict,
  formatted as text documents and ingested into ``resolved_cases``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

from fraud_risk_agent.knowledge.store import get_cases_store, get_policies_store

logger = logging.getLogger(__name__)

_HEADER_SPLITS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def _chunk_markdown(text: str, source: str) -> list[Document]:
    """Split a Markdown document by headers into LangChain Documents."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADER_SPLITS)
    chunks = splitter.split_text(text)
    for chunk in chunks:
        chunk.metadata["source"] = source
    return chunks


def ingest_policies_from_dir(
    directory: str | Path,
    embeddings: Embeddings,
) -> int:
    """Ingest all ``.md`` files in *directory* into the policies collection.

    Returns the number of chunks ingested.
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.warning("Policies directory does not exist: %s", directory)
        return 0

    all_chunks: list[Document] = []
    for md_file in sorted(directory.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        chunks = _chunk_markdown(text, source=md_file.name)
        all_chunks.extend(chunks)
        logger.info("Chunked %s into %d pieces", md_file.name, len(chunks))

    if not all_chunks:
        logger.info("No policy chunks to ingest")
        return 0

    store = get_policies_store(embeddings)
    store.add_documents(all_chunks)
    logger.info("Ingested %d policy chunks into pgvector", len(all_chunks))
    return len(all_chunks)


def _case_to_document(case: dict) -> Document:
    """Convert a fraud_cases row (as dict) into a Document for RAG."""
    signals = case.get("signals", [])
    if isinstance(signals, str):
        signals = json.loads(signals)
    signal_names = [s.get("name", "") for s in signals if isinstance(s, dict)]

    text_parts = [
        f"Score: {case.get('score', 0)}/100 ({case.get('risk_level', 'unknown')})",
        f"Signals: {', '.join(signal_names) or 'none'}",
        f"Action: {case.get('recommended_action', 'unknown')}",
        f"Verdict: {case.get('analyst_verdict', 'pending')}",
    ]
    if case.get("analyst_notes"):
        text_parts.append(f"Notes: {case['analyst_notes']}")

    return Document(
        page_content="\n".join(text_parts),
        metadata={
            "assessment_id": case.get("assessment_id", ""),
            "score": case.get("score", 0),
            "risk_level": case.get("risk_level", "unknown"),
            "analyst_verdict": case.get("analyst_verdict"),
        },
    )


async def ingest_resolved_cases(
    embeddings: Embeddings | None = None,
) -> int:
    """Ingest resolved fraud cases into the resolved_cases collection.

    Queries ``fraud_cases`` for rows with a non-null ``analyst_verdict``
    and inserts them as RAG documents. Returns the count ingested.
    """
    from sqlalchemy import text

    from fraud_risk_agent.data.connection import get_fraud_session

    async with get_fraud_session() as session:
        result = await session.execute(
            text("""
                SELECT assessment_id, score, risk_level, signals,
                       recommended_action, analyst_verdict, analyst_notes
                FROM fraud_cases
                WHERE analyst_verdict IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 1000
            """)
        )
        rows = result.mappings().all()

    if not rows:
        logger.info("No resolved cases to ingest")
        return 0

    docs = [_case_to_document(dict(row)) for row in rows]

    if embeddings is None:
        raise NotImplementedError(
            "No default embedding model configured yet. "
            "Pass an Embeddings instance explicitly."
        )

    store = get_cases_store(embeddings)
    store.add_documents(docs)
    logger.info("Ingested %d resolved cases into pgvector", len(docs))
    return len(docs)
