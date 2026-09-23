"""Application configuration via environment variables."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the fraud risk agent.

    All values come from environment variables. Secrets are wrapped in
    ``SecretStr`` so they never appear in logs or repr output.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ────────────────────────────────────────────────────
    replica_database_url: str = Field(
        default="postgresql+asyncpg://fraud_agent:localdev@localhost:5432/fraud_agent",
        description="Read-only e-commerce replica connection string",
    )
    fraud_database_url: str = Field(
        default="postgresql+asyncpg://fraud_agent:localdev@localhost:5432/fraud_agent",
        description="Read-write connection for the agent's own tables",
    )

    # ── LLM ─────────────────────────────────────────────────────────
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Anthropic API key for Claude",
    )
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model identifier for the investigator LLM",
    )
    llm_timeout_seconds: int = Field(
        default=60,
        description="Wall-clock budget for the investigation step",
    )
    max_tool_calls: int = Field(
        default=8,
        description="Maximum tool invocations per investigation",
    )

    # ── Observability ───────────────────────────────────────────────
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        description="Optional LangSmith API key for tracing",
    )
    langsmith_project: str = Field(
        default="fraud-risk-agent",
        description="LangSmith project name",
    )
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        description="OpenTelemetry collector OTLP endpoint",
    )
    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="json",
        description="'json' for production, 'console' for development",
    )

    # ── Scoring ─────────────────────────────────────────────────────
    scoring_config_path: str = Field(
        default="config/scoring_rules_v1.yaml",
        description="Path to the YAML scoring rules file",
    )
    batch_score_threshold: int = Field(
        default=40,
        description="Minimum score to trigger LLM investigation",
    )

    # ── A2A ─────────────────────────────────────────────────────────
    a2a_auth_secret: SecretStr = Field(
        default=SecretStr(""),
        description="Shared secret or JWKS URL for A2A auth",
    )
    a2a_allowed_callers: str = Field(
        default="",
        description="Comma-separated list of allowed A2A caller IDs",
    )
    a2a_host: str = Field(default="0.0.0.0")
    a2a_port: int = Field(default=8000)


def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
