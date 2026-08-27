"""``POST /auth/ws-ticket`` and ``GET /ws/feed`` — scope gating, origin checks, and the live wire.

The unit-level guarantees (ticket audience isolation, ``FeedBroadcaster`` overflow handling)
are covered in ``test_security.py`` and ``test_feed_broadcaster.py`` respectively. This file is
the one place that exercises the actual routes: the scopes ``POST /auth/ws-ticket`` demands, and
a real ``GET /ws/feed`` connection carrying a real published event end to end.
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import Settings
from app.core.feed import FeedBroadcaster
from app.core.security import (
    SCOPE_ANALYST,
    SCOPE_AUDIT_READ,
    SCOPE_EXPLAIN_READ,
    SCOPE_SCORE,
)
from tests.conftest import auth_header

#: The full scope set both `POST /auth/ws-ticket` and `GET /ws/feed` require. `explain:read` is
#: in the set alongside `analyst`/`audit:read` because `FeedEvent` carries `risk_probability` --
#: see feed.py's `mint_ticket` docstring for why the gate must not be narrower than what it ships.
_TICKET_SCOPES = (SCOPE_ANALYST, SCOPE_AUDIT_READ, SCOPE_EXPLAIN_READ)


class TestMintTicketScopes:
    """``POST /auth/ws-ticket`` must demand exactly the scopes the socket itself demands."""

    def test_an_analyst_token_mints_a_ticket(self, client: TestClient, settings: Settings) -> None:
        headers = auth_header(settings, subject="analyst-1", account_id=None, scopes=_TICKET_SCOPES)
        response = client.post("/auth/ws-ticket", headers=headers)
        assert response.status_code == 200
        body = response.json()
        assert body["ticket"]
        assert body["expires_in"] == settings.ws_ticket_expiry_seconds

    def test_a_merchant_token_is_refused(self, client: TestClient, settings: Settings) -> None:
        """Holding score:write is not the same as holding analyst + audit:read + explain:read."""
        headers = auth_header(settings, subject="merchant-1", scopes=(SCOPE_SCORE,))
        response = client.post("/auth/ws-ticket", headers=headers)
        assert response.status_code == 403

    def test_analyst_scope_alone_is_not_enough(
        self, client: TestClient, settings: Settings
    ) -> None:
        """The ticket route must demand audit:read and explain:read too, not just analyst."""
        headers = auth_header(
            settings, subject="analyst-1", account_id=None, scopes=(SCOPE_ANALYST,)
        )
        response = client.post("/auth/ws-ticket", headers=headers)
        assert response.status_code == 403

    def test_analyst_and_audit_read_without_explain_read_is_refused(
        self, client: TestClient, settings: Settings
    ) -> None:
        """Security-review regression cover: the feed ships `risk_probability`, the same figure
        `explain:read` gates elsewhere on this API -- a token missing it must not mint a ticket,
        even though it holds every other scope the route used to check."""
        headers = auth_header(
            settings,
            subject="analyst-1",
            account_id=None,
            scopes=(SCOPE_ANALYST, SCOPE_AUDIT_READ),
        )
        response = client.post("/auth/ws-ticket", headers=headers)
        assert response.status_code == 403

    def test_an_unauthenticated_caller_is_refused(self, client: TestClient) -> None:
        response = client.post("/auth/ws-ticket")
        assert response.status_code == 401


class TestLiveFeedConnection:
    """``GET /ws/feed`` — ticket verification, origin checking, and the actual relay."""

    def _ticket(self, client: TestClient, settings: Settings) -> str:
        headers = auth_header(settings, subject="analyst-1", account_id=None, scopes=_TICKET_SCOPES)
        response = client.post("/auth/ws-ticket", headers=headers)
        ticket: str = response.json()["ticket"]
        return ticket

    def test_a_valid_ticket_connects_and_receives_hello(
        self, client: TestClient, settings: Settings
    ) -> None:
        ticket = self._ticket(client, settings)
        with client.websocket_connect(
            f"/ws/feed?ticket={ticket}",
            headers={"origin": settings.cors_allow_origins[0]},
        ) as websocket:
            hello = websocket.receive_json()
            assert hello["type"] == "hello"
            assert "server_time" in hello

    def test_a_published_event_reaches_a_connected_socket(
        self, app: FastAPI, client: TestClient, settings: Settings
    ) -> None:
        ticket = self._ticket(client, settings)
        broadcaster: FeedBroadcaster = app.state.feed_broadcaster
        with client.websocket_connect(
            f"/ws/feed?ticket={ticket}",
            headers={"origin": settings.cors_allow_origins[0]},
        ) as websocket:
            websocket.receive_json()  # hello
            broadcaster.publish({"type": "decision", "audit_id": 1})
            event = websocket.receive_json()
            assert event == {"type": "decision", "audit_id": 1}

    def test_a_garbage_ticket_is_refused(self, client: TestClient, settings: Settings) -> None:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/ws/feed?ticket=not-a-real-ticket",
                headers={"origin": settings.cors_allow_origins[0]},
            ):
                pass
        assert exc_info.value.code == 1008

    def test_an_ordinary_bearer_token_does_not_open_the_socket(
        self, client: TestClient, settings: Settings
    ) -> None:
        """The audience swap in mint_ws_ticket must actually be enforced at the route too."""
        headers = auth_header(settings, subject="analyst-1", account_id=None, scopes=_TICKET_SCOPES)
        bearer_token = headers["Authorization"].removeprefix("Bearer ")
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                f"/ws/feed?ticket={bearer_token}",
                headers={"origin": settings.cors_allow_origins[0]},
            ):
                pass
        assert exc_info.value.code == 1008

    def test_a_disallowed_origin_is_refused(self, client: TestClient, settings: Settings) -> None:
        ticket = self._ticket(client, settings)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                f"/ws/feed?ticket={ticket}",
                headers={"origin": "https://evil.example"},
            ):
                pass
        assert exc_info.value.code == 1008

    def test_a_missing_origin_is_refused(self, client: TestClient, settings: Settings) -> None:
        ticket = self._ticket(client, settings)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(f"/ws/feed?ticket={ticket}"):
                pass
        assert exc_info.value.code == 1008

    def test_a_client_sent_frame_closes_the_connection(
        self, client: TestClient, settings: Settings
    ) -> None:
        """This is a broadcast-only channel; an unsolicited client frame ends it."""
        ticket = self._ticket(client, settings)
        with client.websocket_connect(
            f"/ws/feed?ticket={ticket}",
            headers={"origin": settings.cors_allow_origins[0]},
        ) as websocket:
            websocket.receive_json()  # hello
            websocket.send_text("hello server")
            frame = websocket.receive()
            assert frame["type"] == "websocket.close"
            assert frame["code"] == 1003


class TestLiveFeedIsRateLimited:
    """Security-review regression cover: ``GET /ws/feed`` had no rate limiting at all --
    every other route on this API carries ``enforce_rate_limit``, but ``Depends`` does not run
    for a websocket route, so the check has to be made explicitly inside the handler."""

    def test_an_exhausted_budget_closes_the_socket_before_the_ticket_is_even_checked(
        self, app: FastAPI, client: TestClient, settings: Settings
    ) -> None:
        class RejectingLimiter:
            async def check(self, identity: str) -> None:
                raise HTTPException(status_code=429, detail="rate limited")

            async def close(self) -> None:
                """Nothing to release. Matches PermissiveLimiter's shape -- app.main's
                lifespan hook calls this unconditionally on shutdown."""

        app.state.rate_limiter = RejectingLimiter()
        with pytest.raises(WebSocketDisconnect) as exc_info:
            # A garbage ticket: if the limiter were not checked first, this would fail with
            # 1008 (bad ticket) instead of 1013, which is exactly the ordering this test pins.
            with client.websocket_connect(
                "/ws/feed?ticket=not-a-real-ticket",
                headers={"origin": settings.cors_allow_origins[0]},
            ):
                pass
        assert exc_info.value.code == 1013

    def test_no_limiter_installed_fails_closed_not_open(
        self, app: FastAPI, client: TestClient, settings: Settings
    ) -> None:
        """Matches the fail-closed rule `RateLimiter`/`get_rate_limiter` already apply to
        every HTTP route -- a missing limiter must refuse, not silently admit."""
        app.state.rate_limiter = None
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/ws/feed?ticket=not-a-real-ticket",
                headers={"origin": settings.cors_allow_origins[0]},
            ):
                pass
        assert exc_info.value.code == 1013

    def test_a_permissive_limiter_still_lets_a_valid_ticket_through(
        self, client: TestClient, settings: Settings
    ) -> None:
        """Guard on the guard: the rate-limit check must not itself block a normal connection."""
        headers = auth_header(settings, subject="analyst-1", account_id=None, scopes=_TICKET_SCOPES)
        ticket = client.post("/auth/ws-ticket", headers=headers).json()["ticket"]
        with client.websocket_connect(
            f"/ws/feed?ticket={ticket}",
            headers={"origin": settings.cors_allow_origins[0]},
        ) as websocket:
            hello = websocket.receive_json()
            assert hello["type"] == "hello"


class TestTicketLengthIsBounded:
    """Security-review regression cover: `ticket` had no length bound, so a client could hand
    `jwt.decode` an arbitrarily large string before anything rejected it."""

    def test_an_oversized_ticket_is_refused(self, client: TestClient, settings: Settings) -> None:
        from app.api.feed import MAX_TICKET_LENGTH

        oversized = "a" * (MAX_TICKET_LENGTH + 1)
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                f"/ws/feed?ticket={oversized}",
                headers={"origin": settings.cors_allow_origins[0]},
            ):
                pass
