"""Nightly batch scanner — find risky customers and run assessments.

Finds customers who crossed scoring thresholds in recent activity,
runs the agent graph for each, and writes results to ``fraud_cases``.
The batch is idempotent and resumable via ``batch_run_id``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from fraud_risk_agent.config import get_settings

logger = logging.getLogger(__name__)

_EXEC_OPTIONS: dict[str, int] = {"timeout": 30_000}


@dataclass
class BatchResult:
    """Summary of a batch run."""

    batch_run_id: str
    total_candidates: int
    processed: int
    skipped: int
    errors: int
    error_details: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Candidate selection (pre-filter on the replica)
# ---------------------------------------------------------------------------

_CANDIDATE_SQL = text("""
    SELECT DISTINCT r.customer_id
    FROM returns r
    WHERE r.return_date >= now() - make_interval(days => :lookback_days)
    GROUP BY r.customer_id
    HAVING count(*) >= :min_returns

    UNION

    SELECT DISTINCT c.customer_id
    FROM cancellations c
    WHERE c.cancel_date >= now() - make_interval(days => :lookback_days)
    GROUP BY c.customer_id
    HAVING count(*) >= :min_cancellations
""")


async def find_threshold_customers(
    session: AsyncSession,
    lookback_days: int = 30,
    min_returns: int = 3,
    min_cancellations: int = 3,
) -> list[str]:
    """Find customer IDs with enough recent activity to warrant scoring.

    This is a cheap pre-filter, not full scoring. It uses simple SQL
    aggregates to narrow the candidate set before the agent runs.
    """
    result = await session.execute(
        _CANDIDATE_SQL,
        {
            "lookback_days": lookback_days,
            "min_returns": min_returns,
            "min_cancellations": min_cancellations,
        },
        execution_options=_EXEC_OPTIONS,
    )
    return [row[0] for row in result.fetchall()]


# ---------------------------------------------------------------------------
# Already-processed check (idempotency)
# ---------------------------------------------------------------------------

_ALREADY_PROCESSED_SQL = text("""
    SELECT customer_id
    FROM fraud_cases
    WHERE batch_run_id = :batch_run_id
""")


async def _get_already_processed(
    session: AsyncSession,
    batch_run_id: str,
) -> set[str]:
    """Return customer IDs already processed in this batch run."""
    result = await session.execute(
        _ALREADY_PROCESSED_SQL,
        {"batch_run_id": batch_run_id},
    )
    return {row[0] for row in result.fetchall()}


# ---------------------------------------------------------------------------
# Persist assessment
# ---------------------------------------------------------------------------

_INSERT_CASE_SQL = text("""
    INSERT INTO fraud_cases (
        customer_id, assessment_id, score, risk_level,
        signals, evidence, policy_citations, similar_cases,
        recommended_action, confidence, rules_version, model_version,
        narrative_available, batch_run_id, created_at, updated_at
    ) VALUES (
        :customer_id, :assessment_id, :score, :risk_level,
        :signals::jsonb, :evidence::jsonb, :policy_citations::jsonb, :similar_cases::jsonb,
        :recommended_action, :confidence, :rules_version, :model_version,
        :narrative_available, :batch_run_id, now(), now()
    )
    ON CONFLICT (assessment_id) DO NOTHING
""")


async def _persist_assessment(
    session: AsyncSession,
    assessment: dict,
    batch_run_id: str,
) -> None:
    """Write a FraudAssessment to the fraud_cases table."""
    import json

    await session.execute(
        _INSERT_CASE_SQL,
        {
            "customer_id": assessment["customer_id"],
            "assessment_id": assessment["assessment_id"],
            "score": assessment["score"],
            "risk_level": assessment["risk_level"],
            "signals": json.dumps(assessment.get("signals", [])),
            "evidence": json.dumps(assessment.get("evidence", [])),
            "policy_citations": json.dumps(assessment.get("policy_citations", [])),
            "similar_cases": json.dumps(assessment.get("similar_cases", [])),
            "recommended_action": assessment["recommended_action"],
            "confidence": assessment["confidence"],
            "rules_version": assessment["rules_version"],
            "model_version": assessment.get("model_version"),
            "narrative_available": assessment.get("narrative_available", True),
            "batch_run_id": batch_run_id,
        },
    )
    await session.commit()


# ---------------------------------------------------------------------------
# Single-customer assessment
# ---------------------------------------------------------------------------


async def _assess_customer(
    customer_id: str,
    batch_run_id: str,
) -> dict | None:
    """Run the agent graph for one customer and persist the result.

    Returns the assessment dict on success, None on failure.
    Failures are logged but never raised — they must not abort the batch.
    """
    from fraud_risk_agent.data.connection import get_fraud_session

    try:
        # Lazy import to avoid circular dependency and to allow the batch
        # to run even if the agent module has optional dependencies missing.
        from fraud_risk_agent.agent.graph import run_assessment

        assessment = await run_assessment(customer_id=customer_id)
        if hasattr(assessment, "model_dump"):
            assessment_dict = assessment.model_dump()
        else:
            assessment_dict = assessment

        async with get_fraud_session() as session:
            await _persist_assessment(session, assessment_dict, batch_run_id)

        logger.info(
            "Assessment complete",
            extra={
                "customer_id": customer_id,
                "score": assessment_dict.get("score"),
                "risk_level": assessment_dict.get("risk_level"),
                "batch_run_id": batch_run_id,
            },
        )
        return assessment_dict

    except Exception:
        logger.exception(
            "Assessment failed for customer",
            extra={"customer_id": customer_id, "batch_run_id": batch_run_id},
        )
        return None


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


async def run_batch(
    batch_run_id: str | None = None,
    max_customers: int | None = None,
    dry_run: bool = False,
    concurrency: int = 5,
) -> BatchResult:
    """Run the nightly batch scan.

    Args:
        batch_run_id: UUID for idempotency. If None, a new one is generated.
            Re-running with the same ID skips already-completed assessments.
        max_customers: Cap on how many customers to process (for testing).
        dry_run: If True, find candidates but do not run assessments.
        concurrency: Max parallel assessments via asyncio.Semaphore.

    Returns:
        A :class:`BatchResult` summarising the run.
    """
    from fraud_risk_agent.data.connection import get_fraud_session, get_replica_session

    if batch_run_id is None:
        batch_run_id = str(uuid.uuid4())

    settings = get_settings()
    result = BatchResult(
        batch_run_id=batch_run_id, total_candidates=0, processed=0, skipped=0, errors=0,
    )

    logger.info("Batch run starting", extra={"batch_run_id": batch_run_id, "dry_run": dry_run})

    # Step 1: find candidates on the replica
    async with get_replica_session() as session:
        candidates = await find_threshold_customers(
            session,
            lookback_days=30,
            min_returns=settings.batch_score_threshold // 15,  # rough pre-filter
            min_cancellations=settings.batch_score_threshold // 15,
        )

    result.total_candidates = len(candidates)

    if max_customers is not None:
        candidates = candidates[:max_customers]

    logger.info(
        "Found candidates",
        extra={
            "total": result.total_candidates,
            "processing": len(candidates),
            "batch_run_id": batch_run_id,
        },
    )

    if dry_run:
        result.finished_at = datetime.utcnow()
        result.duration_seconds = (result.finished_at - result.started_at).total_seconds()
        return result

    # Step 2: check which customers were already processed (idempotency)
    async with get_fraud_session() as session:
        already_done = await _get_already_processed(session, batch_run_id)

    to_process = [c for c in candidates if c not in already_done]
    result.skipped = len(candidates) - len(to_process)

    logger.info(
        "After idempotency filter",
        extra={
            "to_process": len(to_process),
            "skipped": result.skipped,
            "batch_run_id": batch_run_id,
        },
    )

    # Step 3: run assessments with bounded concurrency
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded_assess(cid: str) -> dict | None:
        async with semaphore:
            return await _assess_customer(cid, batch_run_id)

    tasks = [_bounded_assess(cid) for cid in to_process]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    for i, outcome in enumerate(outcomes):
        if isinstance(outcome, Exception):
            result.errors += 1
            result.error_details.append(f"{to_process[i]}: {outcome!r}")
            logger.exception(
                "Unexpected batch error",
                extra={"customer_id": to_process[i], "batch_run_id": batch_run_id},
            )
        elif outcome is None:
            result.errors += 1
        else:
            result.processed += 1

    # Step 4: ingest resolved cases into RAG (if available)
    try:
        from fraud_risk_agent.knowledge.ingest import ingest_resolved_cases

        await ingest_resolved_cases()
        logger.info("Updated resolved_cases RAG collection", extra={"batch_run_id": batch_run_id})
    except (ImportError, NotImplementedError):
        logger.debug("RAG ingest not available yet, skipping")

    result.finished_at = datetime.utcnow()
    result.duration_seconds = (result.finished_at - result.started_at).total_seconds()

    logger.info(
        "Batch run complete",
        extra={
            "batch_run_id": batch_run_id,
            "processed": result.processed,
            "skipped": result.skipped,
            "errors": result.errors,
            "duration_seconds": result.duration_seconds,
        },
    )
    return result
