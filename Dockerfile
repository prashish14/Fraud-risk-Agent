# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --upgrade pip && pip install .

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r app && useradd -r -g app app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/
COPY config/ ./config/
COPY migrations/ ./migrations/
COPY scripts/healthcheck.py ./scripts/healthcheck.py

USER app

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python scripts/healthcheck.py

EXPOSE 8000

CMD ["uvicorn", "fraud_risk_agent.a2a.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
