"""``webhook_write_session`` / ``webhook_read_session`` -- what SQL they actually trigger.

Both are overridden wholesale by ``tests/conftest.py``'s ``app`` fixture (to a ``FakeSession``,
so every HTTP-level webhook test runs without a real RLS wiring at all), and both drive
``get_scoped_session``/``get_analyst_session`` by calling them directly as plain async
generators rather than through FastAPI's own dependency resolution -- see
``app/api/webhooks.py``'s module docstring for why. That combination means no other test in the
suite exercises what these two functions' bodies actually do. Mirrors ``test_db_session.py``'s
``RecordingSession`` approach for the same reason that file gives: driving the routes through
`TestClient` never reaches this code at all.

Security review of this phase (BUILD_LOG.md's Phase 9 entry) found ``webhook_read_session``
routed through ``get_analyst_session``, which holds a ``SET LOCAL ROLE riskiq_analyst`` branch
unreachable only because :data:`app.api.webhooks.WEBHOOK_PRINCIPAL_SUBJECT`'s principal always
carries ``scopes=()`` -- an invariant nothing enforced. Fixed by routing both webhook session
dependencies through ``get_scoped_session``, which has no such branch to accidentally reach.
This file is what makes "no such branch to reach" a tested property rather than a read of the
source.
"""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from app.api.webhooks import WebhookScoringInputs, webhook_read_session, webhook_write_session
from tests.test_db_session import RecordingSessionmaker, sessionmaker  # noqa: F401


async def _drain(iterator: AsyncIterator[Any]) -> None:
    """Advance an async generator dependency past its single ``yield``."""
    async for _ in iterator:
        return


def _extracted(account_id: str = "acct-1") -> WebhookScoringInputs:
    """Return a minimal WebhookScoringInputs -- only account_id matters to either dependency."""
    return WebhookScoringInputs(
        transaction_id="T-1",
        account_id=account_id,
        event_time=datetime(2018, 5, 5, 14, 30, tzinfo=UTC),
        amount=Decimal("150.00"),
        raw_columns={},
    )


class TestWebhookWriteSessionScopesByAccountOnly:
    async def test_it_issues_set_config_not_set_role(
        self, sessionmaker: RecordingSessionmaker  # noqa: F811
    ) -> None:
        await _drain(webhook_write_session(_extracted("acct-1")))
        [(statement, parameters)] = sessionmaker.session.executed
        assert "SET LOCAL ROLE" not in statement
        assert "set_config" in statement
        assert parameters == {"account_id": "acct-1"}

    async def test_the_scope_matches_the_extracted_account_id_exactly(
        self, sessionmaker: RecordingSessionmaker  # noqa: F811
    ) -> None:
        await _drain(webhook_write_session(_extracted("acct-some-other-one")))
        [(_statement, parameters)] = sessionmaker.session.executed
        assert parameters == {"account_id": "acct-some-other-one"}


class TestWebhookReadSessionNeverAssumesTheAnalystRole:
    """Regression cover for the security-review finding: this must route through
    get_scoped_session (no SET ROLE branch at all), not get_analyst_session (which has one)."""

    async def test_it_issues_set_config_not_set_role(
        self, sessionmaker: RecordingSessionmaker  # noqa: F811
    ) -> None:
        await _drain(webhook_read_session(_extracted("acct-1")))
        [(statement, parameters)] = sessionmaker.session.executed
        assert "SET LOCAL ROLE" not in statement
        assert "set_config" in statement
        assert parameters == {"account_id": "acct-1"}

    async def test_it_reads_the_same_account_it_was_extracted_for(
        self, sessionmaker: RecordingSessionmaker  # noqa: F811
    ) -> None:
        """merchant_context must never accidentally read across accounts -- there is no
        estate-wide branch reachable here at all, unlike get_analyst_session."""
        await _drain(webhook_read_session(_extracted("acct-victim-shaped-name")))
        [(_statement, parameters)] = sessionmaker.session.executed
        assert parameters == {"account_id": "acct-victim-shaped-name"}
