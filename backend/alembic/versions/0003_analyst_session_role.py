"""Phase 8: make the analyst role assumable, without widening every merchant session.

Revision ID: 0003_analyst_session_role
Revises: 0002_audit_log
Create Date: 2026-08-26

**The gap this closes.** ``riskiq_analyst`` has existed since 0002 with correct
``USING (true)`` policies on ``transactions``, ``accounts`` and ``audit_log`` — but nothing
in the application ever assumed it. ``get_scoped_session`` always connects as ``riskiq_app``
and never issues ``SET ROLE``, so the analyst policies were unreachable and every analyst
token got zero rows. That failed closed, which is why it shipped in Phase 7 as a named,
non-blocking gap rather than a defect: the dashboard showed nothing rather than showing too
much.

**The dangerous fix, named so it is never taken.** The obvious repair is
``GRANT riskiq_analyst TO riskiq_app``. Do not do this without ``WITH INHERIT FALSE``.
``CREATE ROLE`` defaults to ``INHERIT``, and Postgres resolves a policy's ``TO role`` list by
role membership. Permissive policies OR together, so if ``riskiq_app`` *inherits*
``riskiq_analyst``, the ``USING (true)`` analyst policies become applicable to every
``riskiq_app`` session unconditionally -- ORed with the account-isolation policy, which means
they win. Every merchant token would see every account's transactions, accounts and audit
rows, permanently, and every policy definition would still read as correct. This is the
single most dangerous line this migration could contain, so it is written with the qualifier
attached and never without it.

**Belt and braces, because "should" is not "is".** ``WITH INHERIT FALSE`` means membership
grants no privilege until a session explicitly runs ``SET ROLE riskiq_analyst`` -- Postgres's
``has_privs_of_role`` (which membership checks use) is expected to respect the inherit flag,
where an RLS policy's role-list match is understood to go through the same path. That
"expected to" is exactly the kind of claim this project does not ship on trust: policies below
add ``AND current_user = 'riskiq_analyst'``, so the control holds even if that expectation is
wrong. An unscoped ``riskiq_app`` session -- one that never ran ``SET ROLE`` -- fails the
predicate regardless of which function Postgres used to admit it to the policy's role list.

**No password, no `LOGIN` change.** ``riskiq_analyst`` stays ``NOLOGIN``. Nothing connects as
it directly; ``riskiq_app`` reaches it only via ``SET LOCAL ROLE`` inside a session it already
authenticated to, which is why granting membership -- not a login credential -- is the whole
change here.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003_analyst_session_role"
down_revision: str | None = "0002_audit_log"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


UPGRADE_STATEMENTS: tuple[str, ...] = (
    # The membership grant that makes SET ROLE possible at all -- WITH INHERIT FALSE means
    # membership alone confers nothing; a session must explicitly assume the role.
    "GRANT riskiq_analyst TO riskiq_app WITH INHERIT FALSE",
    # Each analyst policy is dropped and recreated with the belt-and-braces predicate. The
    # USING clause becomes "true AND assumed the role", not just "true".
    "DROP POLICY audit_log_analyst_read_all ON audit_log",
    """
    CREATE POLICY audit_log_analyst_read_all ON audit_log
        FOR SELECT TO riskiq_analyst
        USING (current_user = 'riskiq_analyst')
    """,
    "DROP POLICY transactions_analyst_read_all ON transactions",
    """
    CREATE POLICY transactions_analyst_read_all ON transactions
        FOR SELECT TO riskiq_analyst
        USING (current_user = 'riskiq_analyst')
    """,
    "DROP POLICY accounts_analyst_read_all ON accounts",
    """
    CREATE POLICY accounts_analyst_read_all ON accounts
        FOR SELECT TO riskiq_analyst
        USING (current_user = 'riskiq_analyst')
    """,
)

DOWNGRADE_STATEMENTS: tuple[str, ...] = (
    "DROP POLICY accounts_analyst_read_all ON accounts",
    """
    CREATE POLICY accounts_analyst_read_all ON accounts
        FOR SELECT TO riskiq_analyst
        USING (true)
    """,
    "DROP POLICY transactions_analyst_read_all ON transactions",
    """
    CREATE POLICY transactions_analyst_read_all ON transactions
        FOR SELECT TO riskiq_analyst
        USING (true)
    """,
    "DROP POLICY audit_log_analyst_read_all ON audit_log",
    """
    CREATE POLICY audit_log_analyst_read_all ON audit_log
        FOR SELECT TO riskiq_analyst
        USING (true)
    """,
    "REVOKE riskiq_analyst FROM riskiq_app",
)


def upgrade() -> None:
    """Grant riskiq_app non-inheriting membership in riskiq_analyst, and pin the policies to it."""
    for statement in UPGRADE_STATEMENTS:
        op.execute(sa.text(statement))


def downgrade() -> None:
    """Revoke the membership and restore 0002's unconditional analyst policies."""
    for statement in DOWNGRADE_STATEMENTS:
        op.execute(sa.text(statement))
