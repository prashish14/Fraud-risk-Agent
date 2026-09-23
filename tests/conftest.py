"""Shared test fixtures.

Unit tests use no external services.
Integration tests use testcontainers for Postgres + pgvector.
"""

import pytest


@pytest.fixture
def scoring_config_path() -> str:
    """Path to the test scoring rules."""
    return "config/scoring_rules_v1.yaml"
