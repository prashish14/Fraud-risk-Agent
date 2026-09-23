# Fraud Risk Agent — Coding Conventions

## Project structure

- `src/fraud_risk_agent/` — source package (six units: data, scoring,
  knowledge, agent, a2a, batch, plus observability)
- `config/` — YAML scoring rules and logging config
- `migrations/` — Alembic migrations (async, Postgres)
- `tests/` — unit, integration, a2a, eval layers
- `scripts/` — CLI entry points and seed scripts

## Commands

```bash
make install          # Create venv and install with dev deps
make lint             # ruff check + format check
make typecheck        # mypy strict
make test             # Unit tests only
make test-integration # Integration + A2A tests (needs Docker)
make test-all         # Everything except eval
docker compose up db  # Start Postgres + pgvector
make migrate          # Apply Alembic migrations
```

## Rules

- **Read-only on replica.** The data layer queries the e-commerce replica
  with `default_transaction_read_only: on`. Never add writes there.
- **Scoring is pure.** `scoring/engine.py` has no I/O, no LLM, no side
  effects. Features in, score out.
- **Parameterized SQL only.** Use `text()` with `:param` bindings. No
  f-strings, no string interpolation in queries.
- **PII never reaches the LLM.** Tools strip names, emails, addresses
  before returning to the agent.
- **Structured output.** The `FraudAssessment` Pydantic model is the
  contract. Peer agents and the batch job both depend on it.
- **Tests use fakes.** `FakeListChatModel` for the LLM,
  `FakeEmbeddings` for RAG, testcontainers for Postgres.
- Secrets come from env vars (`pydantic-settings`). Use `SecretStr`
  for anything sensitive.
- Format with `ruff`. Line length 100. Target Python 3.12.
