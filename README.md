# Fraud Risk Agent

AI-powered fraud detection for e-commerce, focused on return abuse and
cancellation manipulation patterns.

## What it does

- Computes a **deterministic risk score** (0–100) from return, cancellation
  and identity-linking signals
- For flagged customers (score ≥ 40), an LLM agent investigates using
  SQL lookups, RAG over policies and past resolved cases, and peer agent
  queries
- Returns a structured **FraudAssessment** with score, signals, evidence,
  policy citations, similar cases and a recommended action
- Operates in **advisory mode** — it never blocks a customer or denies a
  refund

## Architecture

```
SQL replica (read-only)  →  Scoring (pure Python, YAML rules)
                               ↓
                         Gate: score < 40 → immediate "low risk" (no LLM)
                               ↓ (score ≥ 40)
                         LangGraph investigator (Claude + tools)
                               ↓
                         Structured FraudAssessment
                               ↓
                         A2A server  ←→  peer agents
```

## Quick start

```bash
# 1. Start Postgres + pgvector
docker compose up -d db

# 2. Set up Python
make install

# 3. Run migrations
make migrate

# 4. Run tests
make test
```

## Configuration

Copy `.env.example` to `.env` and fill in the values. See
`src/fraud_risk_agent/config.py` for all available settings.

## Tech stack

- Python 3.12, LangChain / LangGraph, pgvector (RAG)
- PostgreSQL 16, async SQLAlchemy, Alembic
- FastAPI (A2A server), A2A protocol (JSON-RPC 2.0)
- structlog, OpenTelemetry, Prometheus metrics
