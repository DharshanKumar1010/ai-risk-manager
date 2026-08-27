"""``get_scoped_session`` and ``get_analyst_session`` — what SQL each principal shape triggers.

These two dependencies are overridden wholesale in ``tests/conftest.py``'s ``app`` fixture, so
every HTTP-level test in the suite exercises the *routes* without ever running either
function's own body. That leaves the one property Phase 8 depends on — that scoring never
assumes the analyst role, and that an analyst read never falls back to account scoping —
unverified anywhere else. This file drives both functions directly, as async generators, over
a stub session that records the exact statement text and parameters each executes.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.core.security import Principal
from app.db import session as session_module
from app.db.session import get_analyst_session, get_scoped_session


class RecordingSession:
    """Records every statement it is asked to execute, and nothing else."""

    def __init__(self) -> None:
        """Start with an empty call log."""
        self.executed: list[tuple[str, dict[str, Any] | None]] = []

    async def execute(self, statement: Any, parameters: dict[str, Any] | None = None) -> None:
        """Record the statement's compiled text and its parameters."""
        self.executed.append((str(statement), parameters))

    async def rollback(self) -> None:
        """No-op; nothing here holds a transaction to discard."""


class RecordingSessionmaker:
    """A callable returning one ``RecordingSession``, and an async context manager over it."""

    def __init__(self) -> None:
        """Create the one session every call to this factory yields."""
        self.session = RecordingSession()

    def __call__(self) -> "RecordingSessionmaker":
        """Sessionmakers are called to produce a session context manager; return self."""
        return self

    async def __aenter__(self) -> RecordingSession:
        """Yield the recording session."""
        return self.session

    async def __aexit__(self, *exc_info: object) -> None:
        """Nothing to close."""


@pytest.fixture
def sessionmaker(monkeypatch: pytest.MonkeyPatch) -> RecordingSessionmaker:
    """Replace ``get_sessionmaker`` so no dependency here opens a real connection."""
    factory = RecordingSessionmaker()
    monkeypatch.setattr(session_module, "get_sessionmaker", lambda: factory)
    return factory


async def _drain(iterator: AsyncIterator[Any]) -> None:
    """Advance an async generator dependency past its single ``yield``."""
    async for _ in iterator:
        return


class TestGetAnalystSession:
    """The dependency every analyst read route uses."""

    async def test_an_analyst_principal_assumes_the_role(
        self, sessionmaker: RecordingSessionmaker
    ) -> None:
        principal = Principal(subject="reviewer-1", account_id=None, scopes=("analyst",))
        await _drain(get_analyst_session(principal))
        assert sessionmaker.session.executed == [("SET LOCAL ROLE riskiq_analyst", None)]

    async def test_a_merchant_principal_scopes_by_account_instead(
        self, sessionmaker: RecordingSessionmaker
    ) -> None:
        principal = Principal(subject="merchant-1", account_id="acct-1", scopes=("audit:read",))
        await _drain(get_analyst_session(principal))
        [(statement, parameters)] = sessionmaker.session.executed
        assert "SET LOCAL ROLE" not in statement
        assert "set_config" in statement
        assert parameters == {"account_id": "acct-1"}

    async def test_a_principal_with_neither_scope_nor_account_issues_no_statement(
        self, sessionmaker: RecordingSessionmaker
    ) -> None:
        """Fails closed: no SET ROLE, no set_config, so every policy denies by default."""
        principal = Principal(subject="nobody", account_id=None, scopes=("audit:read",))
        await _drain(get_analyst_session(principal))
        assert sessionmaker.session.executed == []

    async def test_a_merchant_holding_analyst_scope_still_assumes_the_role(
        self, sessionmaker: RecordingSessionmaker
    ) -> None:
        """Analyst scope wins the branch even alongside an account_id -- see the next test for
        why that combination must never reach `get_scoped_session` on a write.
        """
        principal = Principal(
            subject="dual-1", account_id="acct-1", scopes=("analyst", "audit:read")
        )
        await _drain(get_analyst_session(principal))
        assert sessionmaker.session.executed == [("SET LOCAL ROLE riskiq_analyst", None)]


class TestGetScopedSessionNeverAssumesTheAnalystRole:
    """The write-path dependency. Regression cover for the exact failure the two-function
    split exists to prevent.
    """

    async def test_an_analyst_plus_score_write_principal_issues_no_role_switch(
        self, sessionmaker: RecordingSessionmaker
    ) -> None:
        """This is the token shape that would 500 on POST /score if the dependencies were
        merged: analyst scope, a matching account_id, holding score:write. Ownership passes
        on the merits (`require_account_ownership` has no analyst branch), so this principal
        really does reach the write path -- and here it must get plain account scoping, not
        a role that holds no INSERT grant on audit_log.
        """
        principal = Principal(
            subject="dual-1", account_id="acct-1", scopes=("analyst", "score:write")
        )
        await _drain(get_scoped_session(principal))
        [(statement, parameters)] = sessionmaker.session.executed
        assert "SET LOCAL ROLE" not in statement
        assert "set_config" in statement
        assert parameters == {"account_id": "acct-1"}

    async def test_a_pure_analyst_principal_with_no_account_gets_no_statement(
        self, sessionmaker: RecordingSessionmaker
    ) -> None:
        """The reviewer console's own token never reaches POST /score in practice, but if it
        did, this dependency must not hand it estate-wide write scoping either.
        """
        principal = Principal(subject="reviewer-1", account_id=None, scopes=("analyst",))
        await _drain(get_scoped_session(principal))
        assert sessionmaker.session.executed == []
