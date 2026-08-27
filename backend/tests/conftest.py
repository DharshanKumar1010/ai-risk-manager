"""Shared test fixtures."""

import hashlib
import hmac
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.core.security import create_access_token
from app.db.session import get_analyst_session, get_scoped_session
from app.main import create_app

#: The signing key every test token is minted with. Not a secret: it signs nothing that exists
#: outside the test process, and `Settings` refuses to boot staging or production with a
#: placeholder key, so it cannot escape into a deployment.
TEST_SIGNING_KEY = "test-only-signing-key-not-a-real-secret"


class PermissiveLimiter:
    """A rate limiter that admits everything.

    Installed by default so that the several dozen tests which are *not* about rate limiting do
    not each need a live Redis. The real limiter's behaviour — the 429 and, more importantly,
    the fail-closed 503 — is tested directly against :class:`app.core.rate_limit.RateLimiter`
    in ``test_rate_limit.py`` rather than through this stub, because a stub that also stood in
    for those tests would be testing itself.
    """

    def __init__(self) -> None:
        """Record calls, so a test can assert the limiter was consulted at all."""
        self.calls: list[str] = []

    async def check(self, identity: str) -> None:
        """Admit the request, remembering who asked."""
        self.calls.append(identity)

    async def close(self) -> None:
        """Nothing to release."""


class FakeResult:
    """The subset of a SQLAlchemy ``Result`` the routes actually call."""

    def __init__(self, rows: list[Any]) -> None:
        """Hold the rows this result will yield."""
        self._rows = rows

    def all(self) -> list[Any]:
        """Return every row."""
        return list(self._rows)

    def scalars(self) -> "FakeResult":
        """Return self; the fake stores already-unwrapped rows."""
        return self


#: Sentinel distinguishing "no get_result configured, fall back to rows[0]" from "get_result was
#: explicitly set to None, meaning .get() should report not-found." A plain `None` default could
#: not carry that distinction.
_GET_RESULT_UNSET = object()


class FakeSession:
    """An ``AsyncSession`` stand-in for tests that are not about the database.

    Deliberately narrow. It answers reads with whatever ``rows`` it was given and assigns a
    surrogate key on flush, which is enough for the routes to run end to end without Postgres.
    It is **not** a substitute for the row-level-security tests: those assert what the database
    does, and a fake that pretended to enforce a policy would be asserting its own behaviour.
    Those live in ``test_orm_constraints.py``, against the migration itself.
    """

    def __init__(self, rows: list[Any] | None = None) -> None:
        """Start with the rows reads should return, and no writes recorded."""
        self.rows = rows if rows is not None else []
        self.added: list[Any] = []
        self.committed = False
        self.scoped_to: str | None = None
        # A separate store from `rows`: app/core/merchant_context.py's account lookup and
        # app/api/webhooks.py's known-account gate both call .get() on the *same* session that
        # read_account_history/compute_merchant_context also .execute() against for a different
        # row shape (transaction history tuples, then decision strings) -- sharing one list
        # between .get() and .execute() would make a test that needs both fail one or the other.
        # Unset by default, so existing tests that mutate `.rows` and rely on the historical
        # `.get() -> rows[0]` fallback (test_api_scoring.py's explain-route tests) keep working
        # unchanged; a test that needs .get() to answer independently of .rows sets this instead.
        self.get_result: Any = _GET_RESULT_UNSET

    async def execute(self, statement: Any, parameters: Any = None) -> FakeResult:
        """Answer a read, or record the row-level-security scoping call."""
        if parameters and "account_id" in parameters:
            self.scoped_to = parameters["account_id"]
            return FakeResult([])
        return FakeResult(self.rows)

    def add(self, row: Any) -> None:
        """Record a pending insert."""
        self.added.append(row)

    async def flush(self) -> None:
        """Assign surrogate keys, as the database would."""
        for position, row in enumerate(self.added, start=1):
            if getattr(row, "audit_id", None) is None:
                row.audit_id = position

    async def commit(self) -> None:
        """Mark the transaction committed."""
        self.committed = True

    async def rollback(self) -> None:
        """Discard pending writes."""
        self.added.clear()

    async def get(self, _model: Any, _key: Any) -> Any:
        """Return ``get_result`` if a test set one explicitly, else fall back to the first held
        row, or None -- see ``get_result``'s own comment for why the two are kept separate."""
        if self.get_result is not _GET_RESULT_UNSET:
            return self.get_result
        return self.rows[0] if self.rows else None


@pytest.fixture
def settings() -> Settings:
    """Return settings pinned to the CI environment with a test-only signing key."""
    return Settings(environment="ci", jwt_secret_key=TEST_SIGNING_KEY)


@pytest.fixture
def session() -> FakeSession:
    """Return the fake session the app fixture installs."""
    return FakeSession()


@pytest.fixture
def app(settings: Settings, session: FakeSession) -> FastAPI:
    """Return an application with the rate limiter and database stubbed out.

    Both substitutions are for reach, not for convenience: without them every authorization
    test would need a live Redis and Postgres, and a test suite that cannot run without
    infrastructure is one that stops being run.
    """
    from app.api.webhooks import webhook_read_session, webhook_write_session

    application = create_app(settings)
    application.state.rate_limiter = PermissiveLimiter()
    # Both session dependencies point at the same fake so a test can inspect one `session`
    # fixture regardless of which route (write path vs read path) it exercises.
    application.dependency_overrides[get_scoped_session] = lambda: session
    application.dependency_overrides[get_analyst_session] = lambda: session
    # The webhook route declares its own session dependencies (webhook_write_session /
    # webhook_read_session) rather than get_scoped_session/get_analyst_session directly --
    # see app/api/webhooks.py's module docstring for why -- so overriding those two above does
    # not reach it. Overridden here at the same fake for the same reason.
    application.dependency_overrides[webhook_write_session] = lambda: session
    application.dependency_overrides[webhook_read_session] = lambda: session
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Return a test client bound to a freshly built application.

    The ``with`` block runs the lifespan hook, so model loading is exercised. Where the
    artefacts are absent the bundle is None and ``/score`` returns 503, which is the documented
    behaviour rather than a broken fixture.
    """
    with TestClient(app) as test_client:
        yield test_client


def auth_header(
    settings: Settings,
    *,
    subject: str = "merchant-1",
    account_id: str | None = "acct-1",
    scopes: tuple[str, ...] = (),
) -> dict[str, str]:
    """Return an Authorization header for a token with the given claims."""
    token = create_access_token(
        subject=subject, account_id=account_id, scopes=scopes, settings=settings
    )
    return {"Authorization": f"Bearer {token}"}


def score_payload(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid ``POST /score`` body, with overrides applied."""
    payload: dict[str, Any] = {
        "transaction_id": "T-1",
        "account_id": "acct-1",
        "event_time": "2018-05-05T14:30:00Z",
        "amount": "150.00",
        "raw_columns": {"ProductCD": "W", "card1": 13926, "card4": "visa"},
    }
    payload.update(overrides)
    return payload


def razorpay_webhook_payload(**entity_overrides: Any) -> dict[str, Any]:
    """Return a minimal valid Razorpay ``payment.captured`` webhook envelope.

    ``entity_overrides`` are applied to ``payload.payment.entity`` -- mirrors ``score_payload``'s
    shape, but nested the way Razorpay's real envelope is. ``notes.riskiq_account_id`` carries
    this project's Phase 9 integration convention (see ``app/api/webhooks.py``'s module
    docstring); override it by passing ``notes={...}`` directly.
    """
    entity: dict[str, Any] = {
        "id": "pay_T1",
        "amount": 15000,  # paise; 150.00 INR, matching score_payload's amount
        "currency": "INR",
        "method": "card",
        "card": {"network": "Visa", "type": "credit"},
        "notes": {"riskiq_account_id": "acct-1"},
        "created_at": int(datetime(2018, 5, 5, 14, 30, tzinfo=UTC).timestamp()),
    }
    entity.update(entity_overrides)
    return {"event": "payment.captured", "payload": {"payment": {"entity": entity}}}


def razorpay_signature(settings: Settings, body: bytes) -> str:
    """Return the ``X-Razorpay-Signature`` value Razorpay would send for ``body``, computed
    with this settings object's webhook secret -- so a test can sign a payload the way
    Razorpay does."""
    return hmac.new(
        settings.razorpay_webhook_secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()


def signed_webhook_body(
    settings: Settings, payload: dict[str, Any]
) -> tuple[bytes, dict[str, str]]:
    """Return the exact bytes to post and the header carrying their signature.

    Serialised once so the signed bytes and the posted bytes are provably the same object --
    re-serialising ``payload`` a second time for the request would risk key-ordering or
    whitespace differences producing a signature that does not match what is actually sent.
    """
    body = json.dumps(payload).encode("utf-8")
    return body, {"X-Razorpay-Signature": razorpay_signature(settings, body)}
