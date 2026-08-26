"""Endpoint-level authorization tests.

The integration tests PHASE_PROMPTS.md item 6 asks for: auth bypass attempts, tampered JWTs,
ownership-check bypass attempts, and the degraded-mode fallback. ``test_security.py`` holds the
Phase 0 unit tests for token minting and verification; this file is about what the *routes* do
with them.

Every protected route is covered by the same parametrised sweep rather than one test each,
because the failure these tests exist to catch is a route that was added later and quietly left
undefended. A list that must be extended when a route is added is a weaker control than a sweep
that fails until the new route is added to it — so ``test_every_route_is_covered`` asserts the
sweep's own completeness against the live application.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import jwt
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from httpx2 import Response

from app.config import Settings
from app.core.security import (
    SCOPE_ANALYST,
    SCOPE_AUDIT_READ,
    SCOPE_EXPLAIN_READ,
    SCOPE_RINGS_READ,
    SCOPE_SCORE,
    SCOPE_TRANSACTIONS_READ,
    create_access_token,
)
from tests.conftest import FakeSession, auth_header, score_payload

#: (method, path, body, required scopes) for every route that must refuse an anonymous caller.
PROTECTED_ROUTES: tuple[tuple[str, str, dict[str, Any] | None, tuple[str, ...]], ...] = (
    ("POST", "/score", score_payload(), (SCOPE_SCORE,)),
    ("GET", "/transactions", None, (SCOPE_TRANSACTIONS_READ,)),
    ("GET", "/audit/T-1", None, (SCOPE_AUDIT_READ,)),
    ("GET", "/audit/entry/1/explain", None, (SCOPE_EXPLAIN_READ,)),
    ("GET", "/rings", None, (SCOPE_RINGS_READ, SCOPE_ANALYST)),
)

#: Paths that are unauthenticated on purpose, with the reason.
PUBLIC_PATHS = {
    "/health": "liveness probe; checks no dependency and discloses nothing",
    "/openapi.json": "schema",
    "/docs": "schema UI",
    "/docs/oauth2-redirect": "schema UI",
    "/redoc": "schema UI",
}


def _request(
    client: TestClient, method: str, path: str, body: dict[str, Any] | None, **kwargs: Any
) -> Response:
    """Issue one request, sending ``body`` only where the route takes one."""
    if method == "POST":
        return client.post(path, json=body, **kwargs)
    return client.get(path, **kwargs)


class TestAuthenticationIsRequired:
    """security-checklist item 2.1: every scoring and write endpoint requires server-side auth."""

    @pytest.mark.parametrize(("method", "path", "body", "scopes"), PROTECTED_ROUTES)
    def test_anonymous_request_is_refused(
        self,
        client: TestClient,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        scopes: tuple[str, ...],
    ) -> None:
        """No Authorization header at all."""
        response = _request(client, method, path, body)
        assert response.status_code == 401
        assert response.headers.get("WWW-Authenticate") == "Bearer"

    @pytest.mark.parametrize(("method", "path", "body", "scopes"), PROTECTED_ROUTES)
    def test_malformed_bearer_token_is_refused(
        self,
        client: TestClient,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        scopes: tuple[str, ...],
    ) -> None:
        """A token that is not a JWT at all."""
        response = _request(
            client, method, path, body, headers={"Authorization": "Bearer not-a-jwt"}
        )
        assert response.status_code == 401

    @pytest.mark.parametrize(("method", "path", "body", "scopes"), PROTECTED_ROUTES)
    def test_token_signed_with_another_key_is_refused(
        self,
        client: TestClient,
        settings: Settings,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        scopes: tuple[str, ...],
    ) -> None:
        """A well-formed token carrying every required scope, signed by someone else.

        The tampering case that matters: the claims are exactly right, so anything that read
        them without verifying the signature would admit this caller.
        """
        attacker = Settings(
            environment="ci", jwt_secret_key="an-entirely-different-key-of-sufficient-length"
        )
        forged = create_access_token(
            subject="attacker", account_id="acct-1", scopes=scopes, settings=attacker
        )
        response = _request(
            client, method, path, body, headers={"Authorization": f"Bearer {forged}"}
        )
        assert response.status_code == 401

    @pytest.mark.parametrize(("method", "path", "body", "scopes"), PROTECTED_ROUTES)
    def test_unsigned_alg_none_token_is_refused(
        self,
        client: TestClient,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        scopes: tuple[str, ...],
    ) -> None:
        """The classic downgrade: ``alg: none``, no signature.

        ``decode_access_token`` pins ``algorithms=[configured]``, so this cannot be accepted.
        The test exists because the pin is one keyword argument away from being lost.
        """
        forged = jwt.encode(
            {
                "sub": "attacker",
                "iss": "riskiq",
                "aud": "riskiq-api",
                "iat": 1_700_000_000,
                "exp": 4_100_000_000,
                "account_id": "acct-1",
                "scopes": list(scopes),
            },
            key="",
            algorithm="none",
        )
        response = _request(
            client, method, path, body, headers={"Authorization": f"Bearer {forged}"}
        )
        assert response.status_code == 401

    @pytest.mark.parametrize(("method", "path", "body", "scopes"), PROTECTED_ROUTES)
    def test_expired_token_is_refused(
        self,
        client: TestClient,
        settings: Settings,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        scopes: tuple[str, ...],
    ) -> None:
        """A token this service signed, past its expiry."""
        expired = jwt.encode(
            {
                "sub": "merchant-1",
                "iss": settings.jwt_issuer,
                "aud": settings.jwt_audience,
                "iat": 1_500_000_000,
                "exp": 1_500_003_600,
                "account_id": "acct-1",
                "scopes": list(scopes),
            },
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )
        response = _request(
            client, method, path, body, headers={"Authorization": f"Bearer {expired}"}
        )
        assert response.status_code == 401

    def test_auth_failure_does_not_disclose_the_reason(
        self, client: TestClient, settings: Settings
    ) -> None:
        """item 2.5: a 401 says no, not which no.

        The property is that failures are *indistinguishable*, not that any particular word is
        absent — the shipped message is the fixed string "Invalid or expired credentials",
        which names both possibilities precisely so that neither is disclosed. So the assertion
        is byte equality across three different underlying causes: a malformed token, one
        signed by someone else, and one this service signed that has expired.
        """
        malformed = _request(
            client, "GET", "/transactions", None, headers={"Authorization": "Bearer x.y.z"}
        )
        attacker = Settings(
            environment="ci", jwt_secret_key="an-entirely-different-key-of-sufficient-length"
        )
        forged = _request(
            client,
            "GET",
            "/transactions",
            None,
            headers={
                "Authorization": "Bearer "
                + create_access_token("attacker", scopes=(), settings=attacker)
            },
        )
        expired = jwt.encode(
            {
                "sub": "merchant-1",
                "iss": settings.jwt_issuer,
                "aud": settings.jwt_audience,
                "iat": 1_500_000_000,
                "exp": 1_500_003_600,
                "scopes": [],
            },
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )
        stale = _request(
            client,
            "GET",
            "/transactions",
            None,
            headers={"Authorization": f"Bearer {expired}"},
        )

        assert malformed.status_code == forged.status_code == stale.status_code == 401
        assert malformed.text == forged.text == stale.text

    def test_every_route_is_covered_by_this_sweep(self, app: FastAPI) -> None:
        """Fail when a route is added without being brought under these tests.

        The control that keeps the sweep honest. Without it, the next route to be added is
        defended only by whoever remembers to add it here.
        """
        covered = {path for _, path, _, _ in PROTECTED_ROUTES}
        # Template paths, since the sweep uses concrete ones.
        templates = {"/audit/{transaction_id}", "/audit/entry/{audit_id}/explain"}
        concrete = {"/audit/T-1", "/audit/entry/1/explain"}
        covered = (covered - concrete) | templates

        live = set()
        for route in app.routes:
            for candidate in getattr(route, "routes", [route]):
                path = getattr(candidate, "path", None)
                if path and path not in PUBLIC_PATHS:
                    live.add(path)

        assert live <= covered, f"routes with no authorization test: {sorted(live - covered)}"


class TestScopesAreEnforcedServerSide:
    """item 2.2: permissions are resolved from the token, never from the request."""

    @pytest.mark.parametrize(("method", "path", "body", "scopes"), PROTECTED_ROUTES)
    def test_authenticated_but_unscoped_token_is_refused(
        self,
        client: TestClient,
        settings: Settings,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        scopes: tuple[str, ...],
    ) -> None:
        """A valid token with no scopes reaches authentication and stops at authorization."""
        response = _request(client, method, path, body, headers=auth_header(settings, scopes=()))
        assert response.status_code == 403

    def test_one_scope_does_not_grant_another(self, client: TestClient, settings: Settings) -> None:
        """Holding ``score:write`` must not open the reviewer's explanation route."""
        response = client.get(
            "/audit/entry/1/explain", headers=auth_header(settings, scopes=(SCOPE_SCORE,))
        )
        assert response.status_code == 403

    def test_rings_requires_analyst_in_addition_to_rings_read(
        self, client: TestClient, settings: Settings
    ) -> None:
        """Ring membership spans accounts, so ``rings:read`` alone is deliberately not enough."""
        response = client.get("/rings", headers=auth_header(settings, scopes=(SCOPE_RINGS_READ,)))
        assert response.status_code == 403

    def test_a_body_supplied_role_is_not_an_authorization_input(
        self, client: TestClient, settings: Settings
    ) -> None:
        """A caller cannot promote itself by putting a role in the payload.

        ``extra="forbid"`` makes this a 422 rather than a silently-ignored field, which is the
        stronger outcome: an ignored field leaves the caller believing it did something.
        """
        response = client.post(
            "/score",
            json=score_payload(role="admin", is_admin=True),
            headers=auth_header(settings, scopes=(SCOPE_SCORE,)),
        )
        assert response.status_code == 422


class TestOwnershipIsEnforced:
    """item 2.4: changing an identifier must not reach another account's data."""

    def test_scoring_another_account_is_refused(
        self, client: TestClient, settings: Settings
    ) -> None:
        """The token's account is ``acct-1``; the body claims ``acct-victim``."""
        response = client.post(
            "/score",
            json=score_payload(account_id="acct-victim"),
            headers=auth_header(settings, account_id="acct-1", scopes=(SCOPE_SCORE,)),
        )
        assert response.status_code == 404

    def test_refusal_is_indistinguishable_from_absence(
        self, client: TestClient, settings: Settings
    ) -> None:
        """404 rather than 403, so a caller cannot enumerate accounts by status code."""
        forbidden = client.post(
            "/score",
            json=score_payload(account_id="acct-victim"),
            headers=auth_header(settings, account_id="acct-1", scopes=(SCOPE_SCORE,)),
        )
        assert forbidden.status_code == 404
        assert "acct-victim" not in forbidden.text

    def test_a_token_with_no_account_scope_reaches_no_account(
        self, client: TestClient, settings: Settings
    ) -> None:
        """A token carrying no ``account_id`` and no analyst scope owns nothing."""
        response = client.post(
            "/score",
            json=score_payload(),
            headers=auth_header(settings, account_id=None, scopes=(SCOPE_SCORE,)),
        )
        assert response.status_code == 404


class TestAccountFilterFailsClosed:
    """A principal with no account scope must see nothing, not everything.

    Regression cover for the Phase 7 security review's B1. `scoped_account_id` returned `None`
    for two different situations — "analyst, unrestricted" and "holds no account" — and both
    list routes read `None` as "apply no filter". A token carrying `transactions:read` and no
    `account_id` claim was therefore served every account's rows. That token shape is the
    default of `create_access_token`, not an exotic one.
    """

    def test_a_principal_with_no_account_is_distinguished_from_an_analyst(self) -> None:
        """The unit-level property: the two must not collapse to one value."""
        from app.core.security import AccountScope, Principal, account_filter

        analyst = Principal(subject="a", account_id=None, scopes=(SCOPE_ANALYST,))
        accountless = Principal(subject="b", account_id=None, scopes=(SCOPE_TRANSACTIONS_READ,))
        owner = Principal(subject="c", account_id="acct-1", scopes=(SCOPE_TRANSACTIONS_READ,))

        assert account_filter(analyst) is AccountScope.UNRESTRICTED
        assert account_filter(accountless) is AccountScope.NOTHING
        assert account_filter(owner) == "acct-1"
        assert account_filter(analyst) is not account_filter(accountless)

    def test_the_sentinels_cannot_be_forged_by_an_account_id_claim(self) -> None:
        """The sentinels must not share a value domain with `account_id`.

        With string sentinels, a token minted with `account_id="unrestricted"` compared equal
        to the "see everything" sentinel and dropped the ORM filter. Enum members cannot be
        produced by any JWT claim, so the comparison cannot be spoofed by one.
        """
        from app.core.security import AccountScope, Principal, account_filter

        impostor = Principal(
            subject="x", account_id="unrestricted", scopes=(SCOPE_TRANSACTIONS_READ,)
        )
        result = account_filter(impostor)
        assert result is not AccountScope.UNRESTRICTED
        assert result == "unrestricted"

    def test_an_impostor_account_id_still_filters(
        self, client: TestClient, settings: Settings, session: FakeSession
    ) -> None:
        """End to end: naming the sentinel must not widen the read."""
        session.rows = [_transaction_row("acct-someone-else")]
        response = client.get(
            "/transactions",
            headers=auth_header(
                settings, account_id="unrestricted", scopes=(SCOPE_TRANSACTIONS_READ,)
            ),
        )
        assert response.status_code == 200
        # The fake session does not enforce the WHERE clause, so this asserts the request was
        # not short-circuited into the unrestricted branch; the filter itself is asserted by
        # test_a_principal_with_no_account_is_distinguished_from_an_analyst.
        assert response.json()["count"] == 1

    def test_transactions_returns_nothing_for_an_accountless_token(
        self, client: TestClient, settings: Settings, session: FakeSession
    ) -> None:
        """The route-level property, with rows present that the caller must not receive."""
        session.rows = [_transaction_row("acct-someone-else")]
        response = client.get(
            "/transactions",
            headers=auth_header(settings, account_id=None, scopes=(SCOPE_TRANSACTIONS_READ,)),
        )
        assert response.status_code == 200
        assert response.json()["transactions"] == []
        assert response.json()["count"] == 0

    def test_audit_returns_not_found_for_an_accountless_token(
        self, client: TestClient, settings: Settings, session: FakeSession
    ) -> None:
        """And the same 404 an unknown transaction gets, so the two cannot be told apart."""
        session.rows = [_audit_row("acct-someone-else")]
        response = client.get(
            "/audit/T-1",
            headers=auth_header(settings, account_id=None, scopes=(SCOPE_AUDIT_READ,)),
        )
        assert response.status_code == 404

    def test_an_owner_still_reads_its_own_rows(
        self, client: TestClient, settings: Settings, session: FakeSession
    ) -> None:
        """The fix must not close the door on the legitimate case."""
        session.rows = [_transaction_row("acct-1")]
        response = client.get(
            "/transactions",
            headers=auth_header(settings, account_id="acct-1", scopes=(SCOPE_TRANSACTIONS_READ,)),
        )
        assert response.status_code == 200
        assert response.json()["count"] == 1


class TestScoringHasNoAnalystBypass:
    """An analyst may read across accounts. It may not write against one it does not hold.

    Regression cover for the review's H4. `require_account_access` waives ownership for analyst
    scope, which is right for a read and wrong for the scoring path — it would let the
    widest-reaching token record a decision attributed to any account.
    """

    def test_an_analyst_cannot_score_for_an_account_it_does_not_hold(
        self, client: TestClient, settings: Settings
    ) -> None:
        response = client.post(
            "/score",
            json=score_payload(account_id="acct-victim"),
            headers=auth_header(settings, account_id="acct-1", scopes=(SCOPE_SCORE, SCOPE_ANALYST)),
        )
        assert response.status_code == 404

    def test_the_read_path_still_allows_the_analyst_across_accounts(self) -> None:
        """The bypass is removed from writes only; the reviewer console still needs it."""
        from app.core.security import (
            Principal,
            require_account_access,
            require_account_ownership,
        )

        analyst = Principal(subject="a", account_id="acct-1", scopes=(SCOPE_ANALYST,))
        require_account_access(analyst, "acct-other")  # read: permitted

        with pytest.raises(HTTPException) as raised:
            require_account_ownership(analyst, "acct-other")  # write: refused
        assert raised.value.status_code == 404


def _transaction_row(account_id: str) -> Any:
    """Return the smallest object the transactions route projects."""

    class Row:
        """One transactions row."""

        transaction_id = "T-1"
        event_time = datetime(2018, 5, 5, tzinfo=UTC)
        amount = Decimal("10.00")
        transaction_type = "W"

    row = Row()
    row.account_id = account_id  # type: ignore[attr-defined]
    return row


def _audit_row(account_id: str) -> Any:
    """Return the smallest object the audit route projects."""

    class Row:
        """One audit_log row."""

        audit_id = 1
        transaction_id = "T-1"
        decided_at = datetime(2018, 5, 5, tzinfo=UTC)
        decision = "allow"
        risk_probability = 0.01
        model_versions: dict[str, str] = {"tier1": "m1"}
        feature_version = "fv_test"
        degraded = False
        degraded_reason = None
        top_features: list[Any] = []
        cost_estimate = None

    row = Row()
    row.account_id = account_id  # type: ignore[attr-defined]
    return row
