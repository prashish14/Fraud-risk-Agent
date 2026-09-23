"""Seed the local database with sample order, return and cancellation data.

Usage: python scripts/seed_test_data.py
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from sqlalchemy import text

from fraud_risk_agent.data.connection import get_fraud_session

logger = logging.getLogger(__name__)

_SAMPLE_CUSTOMERS = [
    {
        "customer_id": "C-1001",
        "score": 75,
        "risk_level": "high",
        "signals": '[{"name": "return_rate_high", "severity": "high"}]',
        "recommended_action": "escalate_to_analyst",
        "confidence": 0.85,
    },
    {
        "customer_id": "C-1002",
        "score": 45,
        "risk_level": "medium",
        "signals": '[{"name": "cancel_after_ship", "severity": "medium"}]',
        "recommended_action": "hold_refund_for_review",
        "confidence": 0.6,
    },
    {
        "customer_id": "C-1003",
        "score": 15,
        "risk_level": "low",
        "signals": "[]",
        "recommended_action": "allow",
        "confidence": 0.95,
    },
]


async def _seed() -> None:
    async with get_fraud_session() as session:
        for customer in _SAMPLE_CUSTOMERS:
            await session.execute(
                text("""
                    INSERT INTO fraud_cases (
                        customer_id, assessment_id, score, risk_level,
                        signals, recommended_action, confidence,
                        rules_version, narrative_available, created_at, updated_at
                    ) VALUES (
                        :customer_id, :assessment_id, :score, :risk_level,
                        :signals::jsonb, :recommended_action, :confidence,
                        :rules_version, :narrative_available, now(), now()
                    ) ON CONFLICT DO NOTHING
                """),
                {
                    **customer,
                    "assessment_id": str(uuid.uuid4()),
                    "rules_version": "v1",
                    "narrative_available": True,
                },
            )
        await session.commit()
    logger.info("Seeded %d sample fraud cases", len(_SAMPLE_CUSTOMERS))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    asyncio.run(_seed())
    print(f"Seeded {len(_SAMPLE_CUSTOMERS)} sample fraud cases.")


if __name__ == "__main__":
    main()
