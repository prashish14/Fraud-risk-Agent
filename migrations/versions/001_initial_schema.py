"""Create fraud_cases, audit_log, and score_configs tables.

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-09-23
"""

from alembic import op

revision = "a1b2c3d4e5f6"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE fraud_cases (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            customer_id         VARCHAR(64) NOT NULL,
            order_id            VARCHAR(64),
            assessment_id       UUID UNIQUE NOT NULL,
            score               SMALLINT NOT NULL CHECK (score BETWEEN 0 AND 100),
            risk_level          VARCHAR(10) NOT NULL
                                CHECK (risk_level IN ('low', 'medium', 'high')),
            signals             JSONB NOT NULL DEFAULT '[]',
            evidence            JSONB NOT NULL DEFAULT '[]',
            policy_citations    JSONB NOT NULL DEFAULT '[]',
            similar_cases       JSONB NOT NULL DEFAULT '[]',
            recommended_action  VARCHAR(50) NOT NULL,
            confidence          REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
            rules_version       VARCHAR(20) NOT NULL,
            model_version       VARCHAR(50),
            narrative_available BOOLEAN NOT NULL DEFAULT TRUE,
            analyst_verdict     VARCHAR(20),
            analyst_notes       TEXT,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            batch_run_id        UUID
        )
    """)
    op.execute("CREATE INDEX idx_fraud_cases_customer ON fraud_cases(customer_id)")
    op.execute("CREATE INDEX idx_fraud_cases_score ON fraud_cases(score)")
    op.execute("CREATE INDEX idx_fraud_cases_created ON fraud_cases(created_at)")
    op.execute(
        "CREATE INDEX idx_fraud_cases_verdict ON fraud_cases(analyst_verdict) "
        "WHERE analyst_verdict IS NOT NULL"
    )

    op.execute("""
        CREATE TABLE audit_log (
            id              BIGSERIAL PRIMARY KEY,
            assessment_id   UUID NOT NULL REFERENCES fraud_cases(assessment_id),
            event_type      VARCHAR(50) NOT NULL,
            event_data      JSONB NOT NULL DEFAULT '{}',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_audit_assessment ON audit_log(assessment_id)")

    op.execute("""
        CREATE TABLE score_configs (
            id              SERIAL PRIMARY KEY,
            version         VARCHAR(20) UNIQUE NOT NULL,
            config_yaml     TEXT NOT NULL,
            activated_at    TIMESTAMPTZ,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS score_configs")
    op.execute("DROP TABLE IF EXISTS audit_log")
    op.execute("DROP TABLE IF EXISTS fraud_cases")
