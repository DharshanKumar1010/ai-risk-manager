"""Phase 7: the append-only audit_log table, the analyst role, and effective RLS.

Revision ID: 0002_audit_log
Revises: 0001_core_tables
Create Date: 2026-08-26

Values are written as literals rather than imported from application constants, for the same
reason revision 0001 gives: a migration is a snapshot of the schema at a point in history, and
importing live constants would let a later refactor silently rewrite what this revision did.

This revision closes three gaps revision 0001 opened and named:

**RLS becomes effective.** 0001 defined correct policies that no one was subject to, because
``docker-compose.yml`` connects as the ``riskiq`` superuser, which owns the tables, and
superusers bypass row-level security unconditionally. ``riskiq_app`` and ``riskiq_pipeline``
were created ``NOLOGIN`` and so could not be connected as. Both are granted ``LOGIN`` here.

**No password is set here, and none may be.** ``ALTER ROLE ... LOGIN`` sets only the login
flag; a role with no password cannot authenticate under ``scram-sha-256`` regardless. Setting
the password is an operator step, done from the environment — see ``backend/.env.example`` and
``infra/postgres-init/10-app-roles.sh``, which creates the roles with passwords from env vars
on first boot. The ``IF NOT EXISTS`` guard in 0001 means that script and this migration compose
in either order.

**The analyst policy arrives.** 0001's isolation policy filters on
``app.current_account_id`` and so denies the dashboard's live feed and ring views by design
rather than by oversight. ``riskiq_analyst`` is a read-only role that sees the whole estate and
can write nothing, which is what a reviewer console actually needs.

**Append-only is enforced by the database, not by convention.** ``audit_log`` grants only
``SELECT, INSERT``, and defines no ``FOR UPDATE`` or ``FOR DELETE`` policy. A rewrite is
refused even if application code one day asks for one. There is deliberately no sequence-level
``UPDATE`` grant beyond ``USAGE``, which is what ``nextval`` needs.

Two nullable columns are added to ``transactions``. ``device_info`` and ``addr1`` are the raw
inputs behind the four familiarity features, and serving cannot recompute those for a new
transaction without the account's prior values. They are nullable and unbackfilled: existing
rows read as the ``__missing__`` sentinel, which is exactly how the training path already
treats a genuinely absent ``DeviceInfo``. Re-running the Phase 1 pipeline backfills them.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002_audit_log"
down_revision: str | None = "0001_core_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Cluster-wide object, so the same idempotent guard 0001 uses.
CREATE_ANALYST_ROLE = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'riskiq_analyst') THEN
        CREATE ROLE riskiq_analyst NOLOGIN;
    END IF;
END
$$;
"""

DROP_ANALYST_ROLE = """
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'riskiq_analyst') THEN
        EXECUTE 'DROP OWNED BY riskiq_analyst';
        EXECUTE 'DROP ROLE riskiq_analyst';
    END IF;
END
$$;
"""

GRANTS_AND_POLICIES: tuple[str, ...] = (
    # --- Make 0001's roles connectable, which is what makes its policies effective ---------
    "ALTER ROLE riskiq_app LOGIN",
    "ALTER ROLE riskiq_pipeline LOGIN",
    "GRANT USAGE ON SCHEMA public TO riskiq_analyst",
    # --- audit_log: append-only, and scoped ------------------------------------------------
    # No UPDATE, no DELETE, to any role. That is the append-only guarantee.
    "GRANT SELECT, INSERT ON audit_log TO riskiq_app",
    "GRANT USAGE ON SEQUENCE audit_log_audit_id_seq TO riskiq_app",
    "GRANT SELECT ON audit_log TO riskiq_analyst",
    "ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE audit_log FORCE ROW LEVEL SECURITY",
    # current_setting(..., true) yields NULL when the session variable is unset, so the
    # comparison is NULL, so no rows are visible and no row may be written. Fails closed in
    # both directions -- an unscoped session cannot read another account's decisions and
    # cannot forge one against an account it does not hold.
    """
    CREATE POLICY audit_log_app_account_isolation ON audit_log
        FOR SELECT TO riskiq_app
        USING (account_id = current_setting('app.current_account_id', true))
    """,
    """
    CREATE POLICY audit_log_app_insert_own_account ON audit_log
        FOR INSERT TO riskiq_app
        WITH CHECK (account_id = current_setting('app.current_account_id', true))
    """,
    """
    CREATE POLICY audit_log_analyst_read_all ON audit_log
        FOR SELECT TO riskiq_analyst
        USING (true)
    """,
    # --- The analyst's estate-wide read on the Phase 1 tables ------------------------------
    # The dashboard's live feed and ring views are not account-scoped. 0001's isolation policy
    # denies them deliberately; this is the role that is allowed to see across accounts, and it
    # is read-only so that widening visibility does not also widen write access.
    "GRANT SELECT ON transactions, accounts TO riskiq_analyst",
    """
    CREATE POLICY transactions_analyst_read_all ON transactions
        FOR SELECT TO riskiq_analyst
        USING (true)
    """,
    """
    CREATE POLICY accounts_analyst_read_all ON accounts
        FOR SELECT TO riskiq_analyst
        USING (true)
    """,
)

REVOKE_AND_DROP_POLICIES: tuple[str, ...] = (
    "DROP POLICY IF EXISTS accounts_analyst_read_all ON accounts",
    "DROP POLICY IF EXISTS transactions_analyst_read_all ON transactions",
    "REVOKE ALL ON transactions, accounts FROM riskiq_analyst",
    "REVOKE ALL ON SCHEMA public FROM riskiq_analyst",
    "ALTER ROLE riskiq_app NOLOGIN",
    "ALTER ROLE riskiq_pipeline NOLOGIN",
)


def upgrade() -> None:
    """Create the audit log, the analyst role, and the grants that make RLS effective."""
    op.create_table(
        "audit_log",
        sa.Column("audit_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("transaction_id", sa.String(length=128), nullable=False),
        sa.Column("account_id", sa.String(length=128), nullable=False),
        sa.Column(
            "decided_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("decision", sa.String(length=8), nullable=False),
        sa.Column("risk_probability", sa.Float(), nullable=False),
        sa.Column("tier1_score", sa.Float(), nullable=True),
        sa.Column("tier2_reconstruction_error", sa.Float(), nullable=True),
        sa.Column("tier3_ring_risk_score", sa.Float(), nullable=True),
        sa.Column("model_versions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("feature_version", sa.String(length=64), nullable=False),
        sa.Column("top_features", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("cost_estimate", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("degraded", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("degraded_reason", sa.String(length=512), nullable=True),
        sa.PrimaryKeyConstraint("audit_id"),
        sa.CheckConstraint(
            "decision IN ('allow', 'review', 'block')", name="ck_audit_log_decision"
        ),
        sa.CheckConstraint(
            "risk_probability >= 0 AND risk_probability <= 1",
            name="ck_audit_log_risk_probability_unit_interval",
        ),
        # A degraded row that does not say why is not reconstructable, which defeats the
        # purpose of the table. The database refuses it.
        sa.CheckConstraint(
            "NOT degraded OR degraded_reason IS NOT NULL",
            name="ck_audit_log_degraded_has_reason",
        ),
    )
    op.create_index("ix_audit_log_transaction", "audit_log", ["transaction_id", "decided_at"])
    op.create_index("ix_audit_log_decided_at", "audit_log", ["decided_at"])
    op.create_index("ix_audit_log_account_time", "audit_log", ["account_id", "decided_at"])

    # The raw inputs behind the familiarity features. Nullable and unbackfilled: an existing
    # row reads as the __missing__ sentinel, which is how the training path already treats an
    # absent DeviceInfo, so serving degrades to a value the model has seen rather than to a
    # fabricated one.
    op.add_column("transactions", sa.Column("device_info", sa.String(length=256), nullable=True))
    op.add_column("transactions", sa.Column("addr1", sa.Float(), nullable=True))

    op.execute(sa.text(CREATE_ANALYST_ROLE))
    for statement in GRANTS_AND_POLICIES:
        op.execute(sa.text(statement))


def downgrade() -> None:
    """Drop the audit log and the analyst role, and return 0001's roles to NOLOGIN.

    Policies and grants on ``audit_log`` disappear with the table; the ones this revision added
    to the Phase 1 tables do not, so they are dropped explicitly before the role is removed.
    """
    for statement in REVOKE_AND_DROP_POLICIES:
        op.execute(sa.text(statement))

    op.drop_column("transactions", "addr1")
    op.drop_column("transactions", "device_info")

    op.drop_index("ix_audit_log_account_time", table_name="audit_log")
    op.drop_index("ix_audit_log_decided_at", table_name="audit_log")
    op.drop_index("ix_audit_log_transaction", table_name="audit_log")
    op.drop_table("audit_log")

    op.execute(sa.text(DROP_ANALYST_ROLE))
