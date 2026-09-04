"""Relax audit_log RLS for single-role demo deployment.

Revision ID: 0004_relax_audit_log_rls
Revises: 0003_analyst_session_role
Create Date: 2026-08-29
"""

from alembic import op

revision = "0004_relax_audit_log_rls"
down_revision = "0003_analyst_session_role"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Render free-tier has one DB user — the riskiq_app/analyst multi-role
    # design collapses to a single principal. Remove FORCE RLS (which blocks
    # even the table owner) and add a permissive INSERT policy.
    op.execute("ALTER TABLE audit_log NO FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY audit_log_single_role_insert "
        "ON audit_log FOR INSERT WITH CHECK (true)"
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS audit_log_single_role_insert ON audit_log")
    op.execute("ALTER TABLE audit_log FORCE ROW LEVEL SECURITY")
