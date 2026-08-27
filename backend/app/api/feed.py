"""``POST /auth/ws-ticket`` and ``GET /ws/feed`` — the live scoring feed.

Two routes, because a bearer token cannot travel in a WebSocket handshake the way it travels
in a normal request. A browser's ``WebSocket`` constructor sets no headers, so the only place
for a credential to go is the URL -- and a full-privilege, hour-long JWT in a URL lands in
uvicorn's access log, any reverse proxy's request log, and the browser's own history. The fix
is not a different place to put the same credential; it is a credential that is safe to log.
``POST /auth/ws-ticket`` mints one: audience-scoped so it cannot authenticate anything but this
socket, and thirty seconds old by the time anyone could read it out of a log line. See
:func:`app.core.security.mint_ws_ticket` for the mechanism.

**The origin check is not optional.** ``CORSMiddleware`` does not run for
``scope["type"] == "websocket"`` -- FastAPI's CORS middleware is HTTP-only, so without an
explicit check here a WebSocket route has no same-origin protection at all. Today the bearer
credential lives in JS memory rather than a cookie, so cross-site WebSocket hijacking is not
directly exploitable -- but the check costs four lines and removing the reasoning about why it
was skipped is more expensive than writing it.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)

from app.api.schemas import WsTicketResponse
from app.core.feed import FeedBroadcaster
from app.core.rate_limit import RateLimiter, enforce_rate_limit
from app.core.security import (
    SCOPE_ANALYST,
    SCOPE_AUDIT_READ,
    SCOPE_EXPLAIN_READ,
    Principal,
    decode_ws_ticket,
    mint_ws_ticket,
    require_scopes,
)

logger = logging.getLogger(__name__)

auth_router = APIRouter(prefix="/auth", tags=["auth"])
feed_router = APIRouter(prefix="/ws", tags=["feed"])

#: How often the server pings an open connection. Render and most proxies close an idle socket
#: around 100s; well inside that keeps the connection from being reaped as idle.
PING_INTERVAL_SECONDS = 25

#: WebSocket close codes used below. 1008 is "policy violation" (auth/origin failure); 1003 is
#: "unsupported data" (this is a broadcast-only socket, so a client frame is a protocol
#: violation, not a message to route anywhere); 1013 is "try again later" (rate limited),
#: distinct from 1008 so a client can tell a bad ticket from a budget it should back off on.
CLOSE_POLICY_VIOLATION = 1008
CLOSE_UNSUPPORTED_DATA = 1003
CLOSE_TRY_AGAIN_LATER = 1013


@auth_router.post(
    "/ws-ticket",
    response_model=WsTicketResponse,
    summary="Mint a short-lived ticket authenticating one live-feed connection",
    dependencies=[Depends(enforce_rate_limit)],
)
async def mint_ticket(
    principal: Annotated[
        Principal,
        Depends(require_scopes(SCOPE_ANALYST, SCOPE_AUDIT_READ, SCOPE_EXPLAIN_READ)),
    ],
    request: Request,
) -> WsTicketResponse:
    """Return a ticket the caller passes as ``?ticket=`` when opening ``GET /ws/feed``.

    Requires the same scopes the feed itself requires (``analyst``, ``audit:read`` and
    ``explain:read``), so a caller who could not open the feed cannot mint a ticket for it
    either -- the ticket route is not a wider door than the socket it opens.

    ``explain:read`` is required here, not just ``analyst`` + ``audit:read``, because
    ``FeedEvent`` carries ``risk_probability`` -- the same figure ``GET /audit/entry/{id}
    /explain`` gates behind ``explain:read`` specifically (``security.py``'s own docstring:
    "useful to a strictly smaller set of callers than the decision record itself"). Gating the
    feed on a narrower pair of scopes than the number it actually ships would make that
    narrower gate decorative -- true today only because every analyst persona this service
    mints happens to also carry ``explain:read``, which is exactly the kind of coincidence that
    stops being true the day a caller's scope set changes for an unrelated reason.

    Settings come from ``request.app.state.settings``, not the process singleton -- the same
    reasoning ``get_current_principal`` and the demo-token route give for the same choice.
    """
    settings = request.app.state.settings
    ticket = mint_ws_ticket(principal, settings)
    return WsTicketResponse(ticket=ticket, expires_in=settings.ws_ticket_expiry_seconds)


def _origin_is_allowed(origin: str | None, allowed: tuple[str, ...]) -> bool:
    """Return whether a WebSocket handshake's Origin header is one this deployment serves."""
    return origin is not None and origin in allowed


#: Generous headroom over a real ticket's actual size (a compact JWS with this principal shape
#: runs well under 1KB) -- bounded at all so a client cannot hand `jwt.decode` a multi-megabyte
#: string before anything rejects it, matching every other string on this API's own bound
#: (schemas.py's MAX_RAW_VALUE_LENGTH and friends).
MAX_TICKET_LENGTH = 2048


@feed_router.websocket("/feed")
async def live_feed(
    websocket: WebSocket, ticket: Annotated[str, Query(max_length=MAX_TICKET_LENGTH)]
) -> None:
    """The live scoring feed. Analyst-only, broadcast-only, ticket-authenticated.

    Every accepted connection: verifies the ticket and the scopes it carries, checks Origin
    (CORSMiddleware does not run for this scope type -- see the module docstring), subscribes
    to the process-wide :class:`FeedBroadcaster`, sends a ``hello`` so the client can
    distinguish "connected and idle" from "still connecting", then relays published events
    until the socket closes. A ping every :data:`PING_INTERVAL_SECONDS` keeps the connection
    from being reaped as idle by a proxy; a concurrent receive task exists only to notice a
    disconnect and to refuse any client-sent frame -- this is a broadcast-only channel, so an
    unsolicited client frame is a bug or a probe, not a message to act on.
    """
    settings = websocket.app.state.settings

    # Rate limited before anything else runs, including ticket verification -- `Depends` does
    # not execute for a websocket route the way it does for an HTTP one (no Authorization
    # header exists yet to resolve a principal from), so this is called directly rather than
    # through `enforce_rate_limit`. Keyed by address, the same fallback `_identity` uses for a
    # request that has not authenticated yet -- there is no verified principal at this point,
    # only a client-supplied ticket string that has not been checked.
    limiter: RateLimiter | None = getattr(websocket.app.state, "rate_limiter", None)
    if limiter is None:
        await websocket.close(code=CLOSE_TRY_AGAIN_LATER)
        return
    client = websocket.client
    try:
        await limiter.check(f"ip:{client.host}" if client is not None else "ip:unknown")
    except HTTPException:
        await websocket.close(code=CLOSE_TRY_AGAIN_LATER)
        return

    try:
        principal = decode_ws_ticket(ticket, settings)
    except Exception:
        await websocket.close(code=CLOSE_POLICY_VIOLATION)
        return

    missing_scopes = {SCOPE_ANALYST, SCOPE_AUDIT_READ, SCOPE_EXPLAIN_READ} - set(principal.scopes)
    if missing_scopes:
        await websocket.close(code=CLOSE_POLICY_VIOLATION)
        return

    if not _origin_is_allowed(websocket.headers.get("origin"), settings.cors_allow_origins):
        await websocket.close(code=CLOSE_POLICY_VIOLATION)
        return

    broadcaster: FeedBroadcaster | None = getattr(websocket.app.state, "feed_broadcaster", None)
    if broadcaster is None or not broadcaster.admit(principal.subject):
        await websocket.close(code=CLOSE_POLICY_VIOLATION)
        return

    await websocket.accept()
    try:
        # A second admit()-then-subscribe() re-check, not a redundant one: two concurrent
        # handshakes for the same principal can both pass the admit() check above before
        # either subscribes, so subscribe()'s own re-check is what actually enforces the
        # limit under a race. Caught here, inside the accepted connection, so a refusal still
        # gets a clean 1008 close rather than an unhandled exception leaving the socket
        # accepted with no relay/heartbeat tasks ever started.
        queue = broadcaster.subscribe(principal.subject)
    except RuntimeError:
        await websocket.close(code=CLOSE_POLICY_VIOLATION)
        return

    async def relay() -> None:
        """Forward published events to the socket until it closes."""
        await websocket.send_json({"type": "hello", "server_time": _now_iso()})
        while True:
            event = await queue.get()
            await websocket.send_json(event)

    async def heartbeat() -> None:
        """Keep the connection alive through idle-socket-reaping proxies."""
        while True:
            await asyncio.sleep(PING_INTERVAL_SECONDS)
            await websocket.send_json({"type": "ping"})

    async def guard_against_client_frames() -> None:
        """This socket is broadcast-only; any client frame ends the connection.

        Also the only path that observes a client-initiated disconnect: ``receive_text``
        raises ``WebSocketDisconnect`` when the client closes, which ends this task and,
        through the ``wait`` below, the whole connection.
        """
        while True:
            await websocket.receive_text()
            await websocket.close(code=CLOSE_UNSUPPORTED_DATA)
            return

    relay_task = asyncio.ensure_future(relay())
    heartbeat_task = asyncio.ensure_future(heartbeat())
    guard_task = asyncio.ensure_future(guard_against_client_frames())
    tasks = (relay_task, heartbeat_task, guard_task)

    try:
        # asyncio.wait does not propagate a task's exception to its caller -- it returns
        # normally with the task in `done`, exception attached. WebSocketDisconnect (a normal
        # client-close, surfaced by guard_task's receive_text) is the expected way this ends;
        # anything else is logged, because a real bug here would otherwise vanish silently.
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exc = task.exception()
            if exc is not None and not isinstance(exc, WebSocketDisconnect):
                logger.warning("live feed connection ended abnormally: %r", exc)
    finally:
        for task in tasks:
            task.cancel()
        broadcaster.unsubscribe(principal.subject, queue)


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string, for the hello/ping payloads."""
    return datetime.now(UTC).isoformat()
