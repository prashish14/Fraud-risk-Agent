"""OpenTelemetry and LangSmith tracing configuration.

Initialises an OTLP span exporter when ``OTEL_EXPORTER_OTLP_ENDPOINT``
is set, and enables LangSmith tracing when ``LANGSMITH_API_KEY`` is set.
"""

from __future__ import annotations

import logging

from fraud_risk_agent.config import get_settings

logger = logging.getLogger(__name__)


def configure_tracing() -> None:
    """Set up OpenTelemetry tracing if an OTLP endpoint is configured."""
    settings = get_settings()
    endpoint = settings.otel_exporter_otlp_endpoint

    if not endpoint:
        logger.debug("OTEL_EXPORTER_OTLP_ENDPOINT not set, tracing disabled")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": "fraud-risk-agent"})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        logger.info("OpenTelemetry tracing configured", extra={"endpoint": endpoint})
    except ImportError:
        logger.warning(
            "opentelemetry packages not installed, tracing disabled. "
            "Install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc"
        )


def configure_langsmith() -> None:
    """Enable LangSmith tracing if the API key is set."""
    settings = get_settings()

    if not settings.langsmith_api_key:
        logger.debug("LANGSMITH_API_KEY not set, LangSmith tracing disabled")
        return

    import os

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault(
        "LANGSMITH_API_KEY",
        settings.langsmith_api_key.get_secret_value(),
    )
    os.environ.setdefault("LANGSMITH_PROJECT", "fraud-risk-agent")
    logger.info("LangSmith tracing enabled")
