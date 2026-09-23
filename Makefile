.PHONY: venv install lint typecheck test test-integration test-all up down migrate seed docker-build clean

PYTHON ?= python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
ALEMBIC := $(VENV)/bin/alembic

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e ".[dev]"

lint:
	$(RUFF) check src/ tests/
	$(RUFF) format --check src/ tests/

format:
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

typecheck:
	$(MYPY) src/fraud_risk_agent/

test:
	$(PYTEST) tests/unit/ -v --tb=short

test-integration:
	$(PYTEST) tests/integration/ tests/a2a/ -v --tb=short

test-all:
	$(PYTEST) tests/ -v --tb=short --ignore=tests/eval

test-eval:
	$(PYTEST) tests/eval/ -v --tb=short -m eval --run-eval

up:
	docker compose up -d

down:
	docker compose down

migrate:
	$(ALEMBIC) -c migrations/alembic.ini upgrade head

seed:
	$(VENV)/bin/python scripts/seed_policies.py
	$(VENV)/bin/python scripts/seed_test_data.py

docker-build:
	docker compose build

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
