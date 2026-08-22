"""Tests that the ORM, the canonical schema and the migration agree with each other.

Three definitions of the same thing exist by necessity — Python literal types, ORM check
constraints, and a migration that must stay a frozen snapshot. Composing the SQL from the
Python literals would remove the duplication but is barred outright by security-checklist
section 3, and an exception "just for constants" is how that rule stops being enforceable.
So the duplication stands and these tests are what keep it honest.

The row-level-security assertions are section 3 of the checklist expressed as tests: a
table holding transaction data with RLS merely enabled, or enabled with no policy, is a
finding, and it should fail here rather than in review.
"""

import re
from pathlib import Path
from typing import get_args

import pytest
from sqlalchemy import Table

from app.data.raw_spec import SourceDataset
from app.data.schema import (
    AMOUNT_DECIMAL_PLACES,
    AMOUNT_MAX_DIGITS,
    Split,
    UidStrategy,
)
from app.db.base import Base
from app.models.account import (
    FIRST_SPLIT_VALUES_SQL,
    UID_STRATEGY_VALUES_SQL,
    Account,
)
from app.models.transaction import (
    SOURCE_DATASET_VALUES_SQL,
    SPLIT_VALUES_SQL,
    Transaction,
)

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1] / "alembic" / "versions" / "0001_core_tables.py"
)

RLS_TABLES = ("transactions", "accounts")


@pytest.fixture(scope="module")
def migration_sql() -> str:
    """Return the Phase 1 migration's source text."""
    return MIGRATION_PATH.read_text(encoding="utf-8")


def quoted_values(sql: str) -> set[str]:
    """Return every single-quoted literal in a SQL fragment."""
    return set(re.findall(r"'([^']*)'", sql))


def orm_tables() -> tuple[Table, ...]:
    """Return the Phase 1 tables as ``Table`` objects.

    Read off ``Base.metadata`` rather than ``Model.__table__``, which is typed as the
    broader ``FromClause`` and so exposes neither ``constraints`` nor ``indexes``.
    """
    return tuple(Base.metadata.tables[name] for name in RLS_TABLES)


class TestConstraintsMatchPythonLiterals:
    """A value added to a Literal type but not to its constraint would pass silently."""

    def test_source_dataset_constraint_matches_the_literal(self) -> None:
        assert quoted_values(SOURCE_DATASET_VALUES_SQL) == set(get_args(SourceDataset))

    def test_split_constraint_matches_the_literal(self) -> None:
        assert quoted_values(SPLIT_VALUES_SQL) == set(get_args(Split))

    def test_first_split_constraint_matches_the_literal(self) -> None:
        assert quoted_values(FIRST_SPLIT_VALUES_SQL) == set(get_args(Split))

    def test_uid_strategy_constraint_matches_the_literal(self) -> None:
        assert quoted_values(UID_STRATEGY_VALUES_SQL) == set(get_args(UidStrategy))


class TestMigrationMatchesTheOrm:
    """The migration is a snapshot, so drift from the ORM has to be caught explicitly."""

    def test_migration_file_exists(self) -> None:
        assert MIGRATION_PATH.is_file()

    @pytest.mark.parametrize("table", RLS_TABLES)
    def test_migration_creates_each_table(self, migration_sql: str, table: str) -> None:
        assert f'op.create_table(\n        "{table}"' in migration_sql

    def test_every_orm_column_appears_in_the_migration(self, migration_sql: str) -> None:
        for table in orm_tables():
            for column in table.columns:
                assert (
                    f'"{column.name}"' in migration_sql
                ), f"{table.name}.{column.name} is in the ORM but not the migration"

    def test_every_orm_constraint_name_appears_in_the_migration(self, migration_sql: str) -> None:
        for table in orm_tables():
            for constraint in table.constraints:
                if constraint.name and str(constraint.name).startswith("ck_"):
                    assert (
                        f'"{constraint.name}"' in migration_sql
                    ), f"{table.name}.{constraint.name} is in the ORM but not the migration"

    def test_every_orm_index_appears_in_the_migration(self, migration_sql: str) -> None:
        for table in orm_tables():
            for index in table.indexes:
                assert (
                    f'"{index.name}"' in migration_sql
                ), f"{table.name}.{index.name} is in the ORM but not the migration"

    def test_amount_precision_matches_the_schema_contract(self, migration_sql: str) -> None:
        assert f"precision={AMOUNT_MAX_DIGITS}, scale={AMOUNT_DECIMAL_PLACES}" in migration_sql


class TestRowLevelSecurity:
    """security-checklist section 3, as tests rather than as a review checklist item."""

    @pytest.mark.parametrize("table", RLS_TABLES)
    def test_rls_is_enabled(self, migration_sql: str, table: str) -> None:
        assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY" in migration_sql

    @pytest.mark.parametrize("table", RLS_TABLES)
    def test_rls_is_forced(self, migration_sql: str, table: str) -> None:
        """ENABLE alone leaves the table owner unconstrained; FORCE is the real control."""
        assert f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY" in migration_sql

    @pytest.mark.parametrize("table", RLS_TABLES)
    def test_each_table_has_at_least_one_policy(self, migration_sql: str, table: str) -> None:
        """RLS with no policy denies everything and reads as a bug rather than a control."""
        policies = re.findall(rf"CREATE POLICY\s+(\w+)\s+ON\s+{table}\b", migration_sql)
        assert policies, f"{table} has RLS enabled but no policy"

    @pytest.mark.parametrize("table", RLS_TABLES)
    def test_account_isolation_policy_fails_closed(self, migration_sql: str, table: str) -> None:
        """current_setting(..., true) is NULL when unset, so an unset session sees nothing."""
        assert (
            "current_setting('app.current_account_id', true)" in migration_sql
        ), f"{table} isolation policy must key on the session account setting"

    def test_no_credential_is_embedded_in_the_migration(self, migration_sql: str) -> None:
        """Roles are created NOLOGIN; Phase 7 attaches credentials from the environment.

        Matches the SQL form ``PASSWORD '...'`` rather than the bare word, which appears
        legitimately in the module docstring explaining why there is no password here.
        """
        assert not re.search(r"PASSWORD\s+'", migration_sql, re.IGNORECASE)
        assert "NOLOGIN" in migration_sql

    def test_roles_are_created_idempotently(self, migration_sql: str) -> None:
        """Roles are cluster-wide, so a second database may already have created them."""
        guard = "IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'riskiq_app')"
        assert guard in migration_sql

    def test_application_role_is_read_only(self, migration_sql: str) -> None:
        assert "GRANT SELECT ON transactions, accounts TO riskiq_app" in migration_sql


class TestMetadataRegistration:
    """Alembic autogeneration sees only what has registered against Base.metadata."""

    @pytest.mark.parametrize("table", RLS_TABLES)
    def test_table_is_registered(self, table: str) -> None:
        assert table in Base.metadata.tables

    def test_transactions_primary_key_is_composite(self) -> None:
        """IEEE-CIS and PaySim ids overlap, so transaction_id alone is not unique."""
        assert [column.name for column in Transaction.__table__.primary_key] == [
            "source_dataset",
            "transaction_id",
        ]

    def test_accounts_primary_key_is_composite(self) -> None:
        assert [column.name for column in Account.__table__.primary_key] == [
            "source_dataset",
            "account_id",
        ]

    def test_transactions_has_no_foreign_key_to_accounts(self) -> None:
        """accounts is derived from transactions in the same pipeline run, so a FK would
        impose an insert ordering for no integrity gain — and an FK between two RLS-forced
        tables can leak the existence of rows the policy hides."""
        assert Transaction.__table__.foreign_keys == set()
