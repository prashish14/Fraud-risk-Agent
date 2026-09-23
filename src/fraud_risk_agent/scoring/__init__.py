"""Scoring engine -- deterministic risk assessment from features and YAML config."""

from fraud_risk_agent.scoring.config import (
    ScoringConfig,
    SignalRule,
    Thresholds,
    load_scoring_config,
)
from fraud_risk_agent.scoring.engine import FiredRule, ScoringResult, compute_score
from fraud_risk_agent.scoring.features import FeatureSet, extract_features

__all__ = [
    "FeatureSet",
    "FiredRule",
    "ScoringConfig",
    "ScoringResult",
    "SignalRule",
    "Thresholds",
    "compute_score",
    "extract_features",
    "load_scoring_config",
]
