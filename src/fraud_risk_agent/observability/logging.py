"""Structured logging setup with structlog.

Configures JSON output in production, coloured console in dev.
Binds common context keys (assessment_id, customer_id hash) and
drops PII fields from log events.
"""

from __future__ import annotations

import hashlib
import logging
import logging.config
from pathlib import Path

import structlog

from fraud_risk_agent.config import get_settings

_PII_FIELDS = frozenset({"email", "name", "address", "phone", "ip_address"})


def _hash_customer_id(customer_id: str) -> str:
    """One-way hash for log correlation without leaking the real ID."""
    return hashlib.sha256(customer_id.encode()).hexdigest()[:12]


def _drop_pii(_logger: logging.Logger, _method: str, event_dict: dict) -> dict:
    """Remove known PII fields from structured log events."""
    for field in _PII_FIELDS:
        event_dict.pop(field, None)
    if "customer_id" in event_dict:
        event_dict["customer_id_hash"] = _hash_customer_id(str(event_dict.pop("customer_id")))
    return event_dict


def configure_logging() -> None:
    """Initialise structlog and stdlib logging.

    Uses the YAML config from ``config/logging.yaml`` as the stdlib base,
    then layers structlog processors on top.
    """
    settings = get_settings()

    yaml_path = Path("config/logging.yaml")
    if yaml_path.exists():
        import yaml

        with yaml_path.open() as f:
            log_cfg = yaml.safe_load(f)
        logging.config.dictConfig(log_cfg)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _drop_pii,
    ]

    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root = logging.getLogger()
    for handler in root.handlers:
        handler.setFormatter(formatter)
