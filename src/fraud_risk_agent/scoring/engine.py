"""Deterministic scoring engine -- pure functions, no I/O."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fraud_risk_agent.scoring.config import ScoringConfig, SignalRule
from fraud_risk_agent.scoring.features import FeatureSet

# Operator dispatch -- never use eval().
_OPERATORS: dict[str, Callable[[float | int | bool, float | int | bool], bool]] = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
}


@dataclass(frozen=True)
class FiredRule:
    """A rule that contributed points to the score."""

    signal_name: str
    points: int
    description: str
    feature_value: float | int | bool
    threshold: float | int | bool


@dataclass(frozen=True)
class ScoringResult:
    """Output of :func:`compute_score` -- the deterministic assessment."""

    score: int  # 0-100
    risk_level: str  # "low", "medium", "high"
    fired_rules: list[FiredRule]


def _evaluate_condition(
    value: float | int | bool,
    rule: SignalRule,
) -> bool:
    """Check whether *value* satisfies the rule's operator and threshold."""
    compare = _OPERATORS.get(rule.operator)
    if compare is None:
        msg = f"Unknown operator: {rule.operator!r}"
        raise ValueError(msg)
    return compare(value, rule.threshold)


def _classify_risk(score: int, config: ScoringConfig) -> str:
    """Map a numeric score to a risk-level label."""
    if score <= config.thresholds.low_max:
        return "low"
    if score <= config.thresholds.medium_max:
        return "medium"
    return "high"


def compute_score(features: FeatureSet, config: ScoringConfig) -> ScoringResult:
    """Compute a risk score from features using the given config.

    Pure function -- no I/O, no side effects, fully deterministic.

    Rules:
        1. Iterate over ``config.signals`` in insertion order.
        2. For each signal, look up the matching feature by ``rule.field``.
        3. If ``requires_min_orders`` is ``True`` and
           ``features.total_orders < config.thresholds.min_orders_for_rates``,
           skip the signal.
        4. If the feature value is ``None`` (e.g. ``return_rate`` when there
           are fewer than *min_orders_for_rates* orders), skip the signal.
        5. Evaluate the operator condition.
        6. Sum fired-rule points and cap at 100.
        7. Classify into a risk level via the configured thresholds.
    """
    fired: list[FiredRule] = []

    for signal_name, rule in config.signals.items():
        # Skip rate-based signals when there aren't enough orders.
        if (
            rule.requires_min_orders
            and features.total_orders < config.thresholds.min_orders_for_rates
        ):
            continue

        value = getattr(features, rule.field, None)

        # Skip signals whose feature is absent (None).
        if value is None:
            continue

        if _evaluate_condition(value, rule):
            fired.append(
                FiredRule(
                    signal_name=signal_name,
                    points=rule.points,
                    description=rule.description,
                    feature_value=value,
                    threshold=rule.threshold,
                )
            )

    raw_score = sum(r.points for r in fired)
    capped_score = min(raw_score, 100)
    risk_level = _classify_risk(capped_score, config)

    return ScoringResult(
        score=capped_score,
        risk_level=risk_level,
        fired_rules=fired,
    )
