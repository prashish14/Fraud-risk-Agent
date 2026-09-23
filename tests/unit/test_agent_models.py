"""Tests for the fraud assessment Pydantic models."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from fraud_risk_agent.agent.models import FraudAssessment, Signal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_assessment(**overrides: object) -> dict:
    """Return kwargs for a valid FraudAssessment with optional overrides."""
    base = {
        "customer_id": "cust-001",
        "score": 50,
        "risk_level": "medium",
        "recommended_action": "hold_refund_for_review",
        "confidence": 0.8,
        "rules_version": "v1.0",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------------


def test_valid_assessment_round_trips() -> None:
    """A well-formed FraudAssessment validates and serialises correctly."""
    assessment = FraudAssessment(**_valid_assessment())
    assert assessment.customer_id == "cust-001"
    assert assessment.score == 50
    assert assessment.risk_level == "medium"
    assert assessment.recommended_action == "hold_refund_for_review"


def test_default_assessment_id_is_uuid() -> None:
    """The default assessment_id is a valid UUID4 string."""
    assessment = FraudAssessment(**_valid_assessment())
    parsed = uuid.UUID(assessment.assessment_id)
    assert parsed.version == 4


def test_explicit_assessment_id_preserved() -> None:
    """An explicitly set assessment_id is not overwritten."""
    explicit_id = "my-custom-id-123"
    assessment = FraudAssessment(**_valid_assessment(assessment_id=explicit_id))
    assert assessment.assessment_id == explicit_id


# ---------------------------------------------------------------------------
# Score validation
# ---------------------------------------------------------------------------


def test_score_at_zero_accepted() -> None:
    """Score of 0 is valid."""
    assessment = FraudAssessment(**_valid_assessment(score=0, risk_level="low"))
    assert assessment.score == 0


def test_score_at_100_accepted() -> None:
    """Score of 100 is valid."""
    assessment = FraudAssessment(**_valid_assessment(score=100, risk_level="high"))
    assert assessment.score == 100


def test_score_negative_rejected() -> None:
    """A negative score is rejected."""
    with pytest.raises(ValidationError):
        FraudAssessment(**_valid_assessment(score=-1))


def test_score_above_100_rejected() -> None:
    """A score above 100 is rejected."""
    with pytest.raises(ValidationError):
        FraudAssessment(**_valid_assessment(score=101))


# ---------------------------------------------------------------------------
# Risk level validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_valid_risk_levels(level: str) -> None:
    """All three risk levels are accepted."""
    assessment = FraudAssessment(**_valid_assessment(risk_level=level))
    assert assessment.risk_level == level


def test_invalid_risk_level_rejected() -> None:
    """An unknown risk level is rejected."""
    with pytest.raises(ValidationError):
        FraudAssessment(**_valid_assessment(risk_level="critical"))


# ---------------------------------------------------------------------------
# Recommended action validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    ["allow", "inspect_return", "hold_refund_for_review", "escalate_to_analyst"],
)
def test_valid_recommended_actions(action: str) -> None:
    """All four recommended actions are accepted."""
    assessment = FraudAssessment(**_valid_assessment(recommended_action=action))
    assert assessment.recommended_action == action


def test_invalid_action_rejected() -> None:
    """An unknown recommended action is rejected."""
    with pytest.raises(ValidationError):
        FraudAssessment(**_valid_assessment(recommended_action="ban_customer"))


# ---------------------------------------------------------------------------
# Confidence validation
# ---------------------------------------------------------------------------


def test_confidence_at_zero_accepted() -> None:
    """Confidence of 0.0 is valid."""
    assessment = FraudAssessment(**_valid_assessment(confidence=0.0))
    assert assessment.confidence == 0.0


def test_confidence_at_one_accepted() -> None:
    """Confidence of 1.0 is valid."""
    assessment = FraudAssessment(**_valid_assessment(confidence=1.0))
    assert assessment.confidence == 1.0


def test_confidence_negative_rejected() -> None:
    """Negative confidence is rejected."""
    with pytest.raises(ValidationError):
        FraudAssessment(**_valid_assessment(confidence=-0.1))


def test_confidence_above_one_rejected() -> None:
    """Confidence above 1.0 is rejected."""
    with pytest.raises(ValidationError):
        FraudAssessment(**_valid_assessment(confidence=1.1))


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def test_signals_list_accepted() -> None:
    """Signals can be provided as a list of Signal dicts."""
    signals = [
        {
            "name": "high_return_rate",
            "description": "Return rate is 3x category average",
            "severity": "high",
            "evidence": "Return rate: 45%",
        }
    ]
    assessment = FraudAssessment(**_valid_assessment(signals=signals))
    assert len(assessment.signals) == 1
    assert isinstance(assessment.signals[0], Signal)
    assert assessment.signals[0].severity == "high"


def test_signal_invalid_severity_rejected() -> None:
    """A signal with an invalid severity is rejected."""
    signals = [
        {
            "name": "test",
            "description": "test",
            "severity": "critical",
            "evidence": "test",
        }
    ]
    with pytest.raises(ValidationError):
        FraudAssessment(**_valid_assessment(signals=signals))


# ---------------------------------------------------------------------------
# Optional fields
# ---------------------------------------------------------------------------


def test_narrative_defaults_to_none() -> None:
    """Narrative is None by default."""
    assessment = FraudAssessment(**_valid_assessment())
    assert assessment.narrative is None


def test_order_id_defaults_to_none() -> None:
    """order_id is None by default."""
    assessment = FraudAssessment(**_valid_assessment())
    assert assessment.order_id is None


def test_fired_rules_defaults_to_empty_list() -> None:
    """fired_rules is an empty list by default."""
    assessment = FraudAssessment(**_valid_assessment())
    assert assessment.fired_rules == []


def test_created_at_is_set_automatically() -> None:
    """created_at is populated automatically."""
    assessment = FraudAssessment(**_valid_assessment())
    assert assessment.created_at is not None
