"""Tests for application configuration."""

from fraud_risk_agent.config import Settings


def test_default_settings_load() -> None:
    """Settings can be instantiated with defaults (no env file needed)."""
    settings = Settings()
    assert settings.llm_model == "claude-sonnet-4-20250514"
    assert settings.max_tool_calls == 8
    assert settings.batch_score_threshold == 40
    assert settings.log_format == "json"
    assert settings.a2a_port == 8000


def test_secrets_not_exposed_in_repr() -> None:
    """SecretStr fields are masked in string representations."""
    settings = Settings(anthropic_api_key="sk-test-secret-key-12345")  # type: ignore[arg-type]
    text = repr(settings)
    assert "sk-test-secret-key-12345" not in text
    assert "SecretStr('**********')" in text or "SecretStr" in text
