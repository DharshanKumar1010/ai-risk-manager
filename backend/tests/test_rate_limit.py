"""Rate limiting, and specifically the fail-closed requirement.

security-checklist item 5.2 is the one this file exists for: *the limiter fails closed on a
Redis outage for auth-sensitive endpoints — a limiter that silently allows everything when its
backing store is down is not a limiter.* That is the behaviour most likely to be "fixed" by a
later change that finds 503s inconvenient, so it is pinned here explicitly, with the reason.

These tests drive :class:`RateLimiter` directly rather than through the permissive stub the
other suites install, because a stub standing in for these would be testing itself.
"""

from typing import Any

import pytest
from fastapi import HTTPException

from app.config import Settings
from app.core.rate_limit import KEY_PREFIX, RateLimiter, _identity
from tests.conftest import TEST_SIGNING_KEY


class FakePipeline:
    """A Redis pipeline that counts INCRs in a dict, or raises."""

    def __init__(self, store: dict[str, int], failure: Exception | None) -> None:
        """Hold the shared counter store and an optional failure to raise on execute."""
        self._store = store
        self._failure = failure
        self._key: str | None = None

    def incr(self, key: str) -> None:
        """Queue an increment."""
        self._key = key

    def expire(self, key: str, seconds: int, nx: bool = False) -> None:
        """Queue an expiry. The fake has no clock, so this only records intent."""

    async def execute(self) -> list[Any]:
        """Run the queued commands, or fail as configured."""
        if self._failure is not None:
            raise self._failure
        assert self._key is not None
        self._store[self._key] = self._store.get(self._key, 0) + 1
        return [self._store[self._key], True]


class FakeRedis:
    """The two methods :class:`RateLimiter` calls."""

    def __init__(self, failure: Exception | None = None) -> None:
        """Start empty, optionally failing every pipeline execution."""
        self.store: dict[str, int] = {}
        self._failure = failure
        self.closed = False

    def pipeline(self) -> FakePipeline:
        """Return a pipeline over the shared store."""
        return FakePipeline(self.store, self._failure)

    async def aclose(self) -> None:
        """Record that the pool was released."""
        self.closed = True


def _limiter(requests: int = 3, failure: Exception | None = None) -> tuple[RateLimiter, FakeRedis]:
    """Return a limiter wired to a fake Redis, bypassing connection setup."""
    settings = Settings(
        environment="ci",
        jwt_secret_key=TEST_SIGNING_KEY,
        rate_limit_requests=requests,
        rate_limit_window_seconds=60,
    )
    limiter = RateLimiter(settings)
    redis = FakeRedis(failure)
    limiter._client = redis
    return limiter, redis


class TestFailsClosed:
    """The item that decides this module's shape."""

    @pytest.mark.parametrize(
        "failure",
        [
            ConnectionError("redis is down"),
            TimeoutError("redis timed out"),
            OSError("no route to host"),
            RuntimeError("unexpected client state"),
        ],
    )
    async def test_a_redis_outage_refuses_the_request(self, failure: Exception) -> None:
        """503, not "allowed". An outage must not become an open door."""
        limiter, _ = _limiter(failure=failure)
        with pytest.raises(HTTPException) as raised:
            await limiter.check("sub:merchant-1")
        assert raised.value.status_code == 503

    async def test_the_outage_response_does_not_leak_the_cause(self) -> None:
        """The caller learns the limiter is unavailable, not what the backend is."""
        limiter, _ = _limiter(failure=ConnectionError("redis://prod-cache-7:6379 refused"))
        with pytest.raises(HTTPException) as raised:
            await limiter.check("sub:merchant-1")
        assert "redis" not in str(raised.value.detail).lower()
        assert "prod-cache-7" not in str(raised.value.detail)


class TestBudgetEnforcement:
    """The ordinary path: a budget that is consumed and then exhausted."""

    async def test_requests_within_the_budget_are_admitted(self) -> None:
        limiter, _ = _limiter(requests=3)
        for _ in range(3):
            await limiter.check("sub:merchant-1")

    async def test_the_request_after_the_budget_is_refused(self) -> None:
        limiter, _ = _limiter(requests=3)
        for _ in range(3):
            await limiter.check("sub:merchant-1")
        with pytest.raises(HTTPException) as raised:
            await limiter.check("sub:merchant-1")
        assert raised.value.status_code == 429
        assert raised.value.headers is not None
        assert raised.value.headers["Retry-After"] == "60"

    async def test_budgets_are_independent_per_identity(self) -> None:
        """One caller exhausting its budget must not refuse another."""
        limiter, _ = _limiter(requests=1)
        await limiter.check("sub:merchant-1")
        await limiter.check("sub:merchant-2")
        with pytest.raises(HTTPException):
            await limiter.check("sub:merchant-1")

    async def test_keys_are_namespaced(self) -> None:
        """So the limiter's keys are distinguishable from anything else in the database."""
        limiter, redis = _limiter()
        await limiter.check("sub:merchant-1")
        assert all(key.startswith(KEY_PREFIX) for key in redis.store)

    async def test_close_releases_the_pool(self) -> None:
        limiter, redis = _limiter()
        await limiter.check("sub:merchant-1")
        await limiter.close()
        assert redis.closed is True


class TestIdentity:
    """What the budget is keyed on."""

    def test_an_authenticated_caller_is_keyed_on_its_subject(self) -> None:
        """Not on the address: one integration behind a NAT must not exhaust everyone."""
        request = _request_with(principal_subject="merchant-7", host="10.0.0.5")
        assert _identity(request) == "sub:merchant-7"

    def test_an_anonymous_caller_falls_back_to_its_address(self) -> None:
        request = _request_with(principal_subject=None, host="10.0.0.5")
        assert _identity(request) == "ip:10.0.0.5"

    def test_a_forwarded_for_header_is_not_consulted(self) -> None:
        """It is caller-controlled: trusting it would let anyone mint unlimited identities."""
        request = _request_with(principal_subject=None, host="10.0.0.5", forwarded_for="1.2.3.4")
        assert _identity(request) == "ip:10.0.0.5"


def _request_with(
    *, principal_subject: str | None, host: str, forwarded_for: str | None = None
) -> Any:
    """Return the smallest object :func:`_identity` reads."""

    class State:
        """Request state carrying an optional principal."""

    class Principal:
        """Just the attribute ``_identity`` reads."""

        subject = principal_subject

    class Client:
        """Peer address."""

        def __init__(self, address: str) -> None:
            self.host = address

    class Request:
        """A stand-in exposing ``state``, ``client`` and ``headers``."""

        def __init__(self) -> None:
            self.state = State()
            if principal_subject is not None:
                self.state.principal = Principal()  # type: ignore[attr-defined]
            self.client = Client(host)
            self.headers = {"X-Forwarded-For": forwarded_for} if forwarded_for else {}

    return Request()
