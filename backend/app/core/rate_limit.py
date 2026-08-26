"""Redis-backed rate limiting.

Security-checklist section 5 asks for three things, and the third is the one that decides the
shape of this module: the limiter must **fail closed**. A limiter that silently allows every
request when its backing store is down is not a limiter — it is a limiter-shaped hole that
opens exactly when the service is already under stress, which is when an attacker is most
likely to be the reason it is under stress.

So :func:`RateLimiter.check` raises 503 when Redis is unreachable, rather than returning
"allowed". That is a deliberate availability trade: the service refuses traffic it cannot
account for. ``/health`` is exempt, so an orchestrator can still tell the process is alive and
does not restart a container over a Redis outage.

The algorithm is a fixed window, implemented as ``INCR`` plus ``EXPIRE`` on first increment,
pipelined into one round trip. A fixed window admits up to 2x the nominal rate across a window
boundary; a sliding-log window would not, at the cost of storing every request timestamp. At
this service's scale the burst is not the threat the limiter is here for — scripted credential
and probe traffic is — and the simpler structure has fewer ways to be wrong.

Keys are namespaced and derived from the *authenticated principal* where there is one, falling
back to the client address. Keying on the address alone would let one authenticated caller
exhaust the budget for everyone behind a shared NAT.
"""

import logging
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.core.security import CurrentPrincipal

logger = logging.getLogger(__name__)

#: Key prefix, so the limiter's keys are distinguishable from anything else in the database.
KEY_PREFIX = "riskiq:ratelimit"

#: Returned to a caller that has exhausted its budget. The window length is echoed so a
#: well-behaved client can back off correctly; it discloses nothing about other callers.
RATE_LIMIT_DETAIL = "Rate limit exceeded"


class RateLimiter:
    """Fixed-window limiter over Redis.

    Holds the client lazily so that importing this module, or building the app in a test that
    never issues a request, does not open a connection.
    """

    def __init__(self, settings: Settings) -> None:
        """Store configuration. The Redis client is created on first use.

        Args:
            settings: Supplies the Redis DSN, the request budget and the window length.
        """
        self._settings = settings
        self._client: Any | None = None

    async def _redis(self) -> Any:
        """Return the shared Redis client, creating it on first call."""
        if self._client is None:
            import redis.asyncio as redis

            # ``redis.asyncio`` ships no type information for ``from_url``, so the call is
            # untyped under strict mode. Annotating the result rather than widening the mypy
            # override keeps the rest of the client's surface checked.
            client: Any = redis.from_url(  # type: ignore[no-untyped-call]
                self._settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            self._client = client
        return self._client

    async def close(self) -> None:
        """Release the Redis connection pool. Called from the application's shutdown hook."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def check(self, identity: str) -> None:
        """Consume one unit of ``identity``'s budget, or reject the request.

        Args:
            identity: What the budget is keyed on — the authenticated subject where there is
                one, otherwise the client address.

        Raises:
            HTTPException: 429 when the budget for the current window is exhausted; 503 when
                Redis cannot be reached. The 503 is the fail-closed behaviour required by
                security-checklist item 5.2 and is deliberate, not an oversight.
        """
        window = self._settings.rate_limit_window_seconds
        key = f"{KEY_PREFIX}:{window}:{identity}"
        try:
            client = await self._redis()
            pipeline = client.pipeline()
            pipeline.incr(key)
            pipeline.expire(key, window, nx=True)
            used, _ = await pipeline.execute()
        except HTTPException:
            raise
        except Exception as exc:
            # Deliberately not "allow on error". See the module docstring.
            logger.warning("rate limiter unavailable, refusing request: %s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate limiting is unavailable",
            ) from exc

        if int(used) > self._settings.rate_limit_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=RATE_LIMIT_DETAIL,
                headers={"Retry-After": str(window)},
            )


def get_rate_limiter(request: Request) -> RateLimiter:
    """Return the process-wide limiter held on application state.

    Raises:
        HTTPException: 503 if the application was built without a limiter. Refusing is the
            fail-closed reading of a missing limiter, consistent with a broken one.
    """
    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rate limiting is unavailable",
        )
    return limiter


def _identity(request: Request) -> str:
    """Return the budget key for this request.

    Prefers the authenticated subject, which survives a NAT and cannot be spoofed by a header.
    Falls back to the peer address for a request that has not authenticated yet. ``X-Forwarded-
    For`` is deliberately **not** consulted: it is caller-controlled, so trusting it would let
    anyone mint an unlimited number of identities and defeat the limiter entirely.
    """
    principal = getattr(request.state, "principal", None)
    if principal is not None:
        return f"sub:{principal.subject}"
    client = request.client
    return f"ip:{client.host}" if client is not None else "ip:unknown"


async def enforce_rate_limit(
    request: Request,
    principal: CurrentPrincipal,
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> None:
    """FastAPI dependency applying the limiter to one request.

    It takes the principal as a dependency rather than reading it opportunistically, and that
    is load-bearing in two ways. FastAPI resolves a dependency's own dependencies first, so
    authentication runs before the limiter: a request with no token gets 401 rather than a 503
    from a limiter that could not reach Redis, which is what security-checklist item 2.5 asks
    for and what a caller needs in order to know what to fix. And because the principal is
    resolved by the time :func:`_identity` runs, the budget is keyed per caller rather than per
    address — one authenticated integration behind a shared NAT cannot exhaust everyone else's.

    The parameter is unused by name on purpose; ``_identity`` reads it from request state,
    where :func:`app.core.security.get_current_principal` bound it.
    """
    await limiter.check(_identity(request))


def build_rate_limiter(settings: Settings | None = None) -> RateLimiter:
    """Construct a limiter from settings. Used by the application factory."""
    return RateLimiter(settings or get_settings())
