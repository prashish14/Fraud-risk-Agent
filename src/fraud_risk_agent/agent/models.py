"""Pydantic models for the fraud assessment output contract."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class Signal(BaseModel):
    """One risk signal detected during investigation."""

    name: str
    description: str
    severity: Literal["low", "medium", "high"]
    evidence: str


class PolicyCitation(BaseModel):
    """Reference to a fraud or return policy passage."""

    policy_name: str
    section: str
    relevant_text: str
    chunk_id: str | None = None


class SimilarCase(BaseModel):
    """A past fraud case similar to the current assessment."""

    case_id: str
    score: int
    verdict: str
    similarity_score: float
    summary: str


class FraudAssessment(BaseModel):
    """The structured output contract consumed by A2A peers and the batch job."""

    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str
    order_id: str | None = None
    score: int = Field(ge=0, le=100)
    risk_level: Literal["low", "medium", "high"]
    signals: list[Signal] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    policy_citations: list[PolicyCitation] = Field(default_factory=list)
    similar_cases: list[SimilarCase] = Field(default_factory=list)
    recommended_action: Literal[
        "allow",
        "inspect_return",
        "hold_refund_for_review",
        "escalate_to_analyst",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    rules_version: str
    model_version: str | None = None
    narrative: str | None = None
    narrative_available: bool = True
    fired_rules: list[dict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
