"""Phase 1: transactions and accounts, with row-level security enabled and forced.

Revision ID: 0001_core_tables
Revises:
Create Date: 2026-08-22

Values are written as literals rather than imported from ``app.data.schema``. A migration
is a snapshot of the schema at a point in history; importing live application constants
would let a later refactor silently rewrite what this revision did.

Two least-privilege roles are created here, both ``NOLOGIN`` and without passwords so that
no credential enters tracked source:

``riskiq_app``
    What the API connects as. Read-only, and subject to a fail-closed account-isolation
    policy.

``riskiq_pipeline``
    What the Phase 1 feature pipeline connects as. Read-write across the corpus, because a
    bulk load is not an account-scoped request.

**Known gap, carried into BUILD_LOG.md.** ``docker-compose.yml`` still connects the backend
as the ``riskiq`` superuser, which owns these tables. Superusers bypass RLS unconditionally,
so the policies below are correctly *defined* but not yet *effective*. Phase 7 grants
``LOGIN`` to these roles and repoints ``DATABASE_URL`` at ``riskiq_app``; until it does,
security-checklist section 3's "application user is not a superuser and does not own the
tables" item is an open FAIL and should be reported as one.

Phase 7 must also add an analyst-scoped read policy. The dashboard's live feed and ring
views are not account-scoped, and the isolation policy below denies them by design rather
than by oversight.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001_core_tables"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Roles are created idempotently: they are cluster-wide objects, so a second database in
# the same cluster may already have created them.
CREATE_ROLES = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'riskiq_app') THEN
        CREATE ROLE riskiq_app NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'riskiq_pipeline') THEN
        CREATE ROLE riskiq_pipeline NOLOGIN;
    END IF;
END
$$;
"""

DROP_ROLES = """
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'riskiq_app') THEN
        EXECUTE 'DROP OWNED BY riskiq_app';
        EXECUTE 'DROP ROLE riskiq_app';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'riskiq_pipeline') THEN
        EXECUTE 'DROP OWNED BY riskiq_pipeline';
        EXECUTE 'DROP ROLE riskiq_pipeline';
    END IF;
END
$$;
"""

# ENABLE alone is not enough for a table owner — FORCE is what subjects the owner to the
# policies too (security-checklist section 3).
GRANTS_AND_POLICIES: tuple[str, ...] = (
    "GRANT USAGE ON SCHEMA public TO riskiq_app, riskiq_pipeline",
    "GRANT SELECT ON transactions, accounts TO riskiq_app",
    "GRANT SELECT, INSERT, UPDATE, DELETE ON transactions, accounts TO riskiq_pipeline",
    "ALTER TABLE transactions ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE transactions FORCE ROW LEVEL SECURITY",
    "ALTER TABLE accounts ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE accounts FORCE ROW LEVEL SECURITY",
    # current_setting(..., true) yields NULL when the session variable is unset, so the
    # comparison is NULL, so no rows are visible. The limiter fails closed.
    """
    CREATE POLICY transactions_app_account_isolation ON transactions
        FOR SELECT TO riskiq_app
        USING (account_id = current_setting('app.current_account_id', true))
    """,
    """
    CREATE POLICY accounts_app_account_isolation ON accounts
        FOR SELECT TO riskiq_app
        USING (account_id = current_setting('app.current_account_id', true))
    """,
    """
    CREATE POLICY transactions_pipeline_full_access ON transactions
        FOR ALL TO riskiq_pipeline
        USING (true) WITH CHECK (true)
    """,
    """
    CREATE POLICY accounts_pipeline_full_access ON accounts
        FOR ALL TO riskiq_pipeline
        USING (true) WITH CHECK (true)
    """,
)


def upgrade() -> None:
    """Create the Phase 1 tables, roles and row-level security policies."""
    op.create_table(
        "transactions",
        sa.Column("source_dataset", sa.String(length=16), nullable=False),
        sa.Column("transaction_id", sa.String(length=128), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("amount", sa.Numeric(precision=20, scale=4), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=False),
        sa.Column("counterparty_id", sa.String(length=128), nullable=True),
        sa.Column("transaction_type", sa.String(length=32), nullable=True),
        sa.Column("is_fraud", sa.Boolean(), nullable=False),
        sa.Column("split", sa.String(length=8), nullable=False),
        sa.Column("feature_version", sa.String(length=64), nullable=False),
        sa.Column("features", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("source_dataset", "transaction_id"),
        sa.CheckConstraint(
            "source_dataset IN ('ieee_cis', 'paysim')",
            name="ck_transactions_source_dataset",
        ),
        sa.CheckConstraint("split IN ('train', 'val', 'test')", name="ck_transactions_split"),
        sa.CheckConstraint("amount >= 0", name="ck_transactions_amount_non_negative"),
        sa.CheckConstraint(
            "source_dataset <> 'ieee_cis' OR counterparty_id IS NULL",
            name="ck_transactions_ieee_cis_has_no_counterparty",
        ),
    )
    op.create_index(
        "ix_transactions_account_time",
        "transactions",
        ["source_dataset", "account_id", "event_time"],
    )
    op.create_index(
        "ix_transactions_source_split_time",
        "transactions",
        ["source_dataset", "split", "event_time"],
    )
    op.create_index(
        "ix_transactions_counterparty",
        "transactions",
        ["source_dataset", "counterparty_id"],
        postgresql_where=sa.text("counterparty_id IS NOT NULL"),
    )

    op.create_table(
        "accounts",
        sa.Column("source_dataset", sa.String(length=16), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=False),
        sa.Column("uid_strategy", sa.String(length=32), nullable=False),
        sa.Column("first_seen", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=False),
        sa.Column("transaction_count", sa.Integer(), nullable=False),
        sa.Column("fraud_count", sa.Integer(), nullable=False),
        sa.Column("first_split", sa.String(length=8), nullable=False),
        sa.Column("straddles_split", sa.Boolean(), nullable=False),
        sa.Column("feature_version", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("source_dataset", "account_id"),
        sa.CheckConstraint(
            "source_dataset IN ('ieee_cis', 'paysim')",
            name="ck_accounts_source_dataset",
        ),
        sa.CheckConstraint(
            "uid_strategy IN ('native', 'card_addr_d1n', 'card_email', 'card_only', 'singleton')",
            name="ck_accounts_uid_strategy",
        ),
        sa.CheckConstraint(
            "first_split IN ('train', 'val', 'test')",
            name="ck_accounts_first_split",
        ),
        sa.CheckConstraint("transaction_count >= 1", name="ck_accounts_transaction_count_positive"),
        sa.CheckConstraint(
            "fraud_count >= 0 AND fraud_count <= transaction_count",
            name="ck_accounts_fraud_count_within_total",
        ),
        sa.CheckConstraint("last_seen >= first_seen", name="ck_accounts_time_ordered"),
        sa.CheckConstraint(
            "uid_strategy <> 'singleton' OR transaction_count = 1",
            name="ck_accounts_singleton_has_one_transaction",
        ),
    )
    op.create_index(
        "ix_accounts_straddles_split",
        "accounts",
        ["source_dataset", "straddles_split"],
    )
    op.create_index("ix_accounts_uid_strategy", "accounts", ["source_dataset", "uid_strategy"])

    op.execute(sa.text(CREATE_ROLES))
    for statement in GRANTS_AND_POLICIES:
        op.execute(sa.text(statement))


def downgrade() -> None:
    """Drop the Phase 1 tables, their policies, and the roles created alongside them.

    Policies and grants disappear with their tables, so only the roles need explicit
    removal. Roles are cluster-wide, hence the guarded ``DROP OWNED BY`` first.
    """
    op.drop_index("ix_accounts_uid_strategy", table_name="accounts")
    op.drop_index("ix_accounts_straddles_split", table_name="accounts")
    op.drop_table("accounts")

    op.drop_index("ix_transactions_counterparty", table_name="transactions")
    op.drop_index("ix_transactions_source_split_time", table_name="transactions")
    op.drop_index("ix_transactions_account_time", table_name="transactions")
    op.drop_table("transactions")

    op.execute(sa.text(DROP_ROLES))
