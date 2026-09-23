"""YAML scoring config loader and validation."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SignalRule(BaseModel, frozen=True):
    """One scoring signal rule from the YAML config."""

    description: str
    field: str
    operator: str  # ">", ">=", "==", "<", "<="
    threshold: float | int | bool
    points: int = Field(ge=1, le=50)
    requires_min_orders: bool = False


class Thresholds(BaseModel, frozen=True):
    """Risk-level band boundaries and order minimum for rate signals."""

    low_max: int = 39
    medium_max: int = 69
    min_orders_for_rates: int = 5


class ScoringConfig(BaseModel, frozen=True):
    """Immutable scoring configuration loaded from YAML."""

    version: str
    thresholds: Thresholds
    signals: dict[str, SignalRule]


def load_scoring_config(path: str | Path) -> ScoringConfig:
    """Load and validate scoring config from a YAML file.

    Raises:
        FileNotFoundError: If *path* does not exist.
        pydantic.ValidationError: If the YAML content fails schema validation.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    return ScoringConfig(**data)
