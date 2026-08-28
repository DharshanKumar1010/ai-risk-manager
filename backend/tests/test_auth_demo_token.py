"""``POST /auth/demo-token`` — mounting, persona-to-scope mapping, and expiry honesty."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.core.security import (
    SCOPE_ANALYST,
    SCOPE_AUDIT_READ,
    SCOPE_EXPLAIN_READ,
    SCOPE_RINGS_READ,
    SCOPE_SCORE,
    SCOPE_TRANSACTIONS_READ,
    decode_access_token,
)
from app.main import create_app
from tests.conftest import TEST_SIGNING_KEY, PermissiveLimiter


class TestRouterMounting:
    """The route must not exist at all outside local/ci."""

    @pytest.mark.parametrize("environment", ["staging", "production"])
    def test_the_route_does_not_exist_outside_local_and_ci(self, environment: str) -> None:
        settings = Settings(
            environment=environment,  # type: ignore[arg-type]
            jwt_secret_key="a-deployed-signing-key-of-sufficient-length-here",
            entity_anonymization_key="a-deployed-anonymization-key-of-sufficient-length",
            razorpay_webhook_secret="a-deployed-webhook-secret-of-sufficient-length-here",
        )
        app = create_app(settings)
        app.state.rate_limiter = PermissiveLimiter()
        with TestClient(app) as client:
            response = client.post("/auth/demo-token", json={"persona": "analyst"})
        assert response.status_code == 404

    @pytest.mark.parametrize("environment", ["staging", "production"])
    def test_the_route_is_absent_from_the_deployed_schema(self, environment: str) -> None:
        """Not just refused -- undiscoverable. The path must not appear in the OpenAPI schema."""
        settings = Settings(
            environment=environment,  # type: ignore[arg-type]
            jwt_secret_key="a-deployed-signing-key-of-sufficient-length-here",
            entity_anonymization_key="a-deployed-anonymization-key-of-sufficient-length",
            razorpay_webhook_secret="a-deployed-webhook-secret-of-sufficient-length-here",
        )
        app = create_app(settings)
        app.state.rate_limiter = PermissiveLimiter()
        with TestClient(app) as client:
            schema = client.get("/openapi.json").json()
        assert "/auth/demo-token" not in schema["paths"]

    @pytest.mark.parametrize("environment", ["local", "ci"])
    def test_the_route_exists_in_local_and_ci(self, environment: str, client: TestClient) -> None:
        response = client.post("/auth/demo-token", json={"persona": "analyst"})
        assert response.status_code != 404


def _deployed_settings(environment: str, **overrides: object) -> Settings:
    """Settings for a staging/production Settings() construction, with test-only secrets."""
    return Settings(
        environment=environment,  # type: ignore[arg-type]
        jwt_secret_key="a-deployed-signing-key-of-sufficient-length-here",
        entity_anonymization_key="a-deployed-anonymization-key-of-sufficient-length",
        razorpay_webhook_secret="a-deployed-webhook-secret-of-sufficient-length-here",
        **overrides,  # type: ignore[arg-type]
    )


class TestDemoModeOverride:
    """`demo_mode` is a second, independent way to mount the router -- see Settings.demo_mode.
    It must not change what the `environment` check gates for anything else, so every test
    here pins environment to staging/production and varies demo_mode alone.

    A security review of the first version of this flag found it granted the full local/ci
    persona set -- including 'analyst', which reaches explain:read/rings:read -- to an
    anonymous internet caller, reassembling the explain-oracle evasion path Phase 7/8 built
    those scope gates to prevent. The fix restricts what demo_mode (not local/ci) may mint;
    these tests pin that restriction, not just that the router mounts.
    """

    @pytest.mark.parametrize("environment", ["staging", "production"])
    def test_demo_mode_mounts_the_route_outside_local_and_ci(self, environment: str) -> None:
        settings = _deployed_settings(environment, demo_mode=True)
        app = create_app(settings)
        app.state.rate_limiter = PermissiveLimiter()
        with TestClient(app) as client:
            response = client.get("/openapi.json")
        assert "/auth/demo-token" in response.json()["paths"]

    @pytest.mark.parametrize("environment", ["staging", "production"])
    def test_demo_mode_defaults_to_off(self, environment: str) -> None:
        """Constructing Settings without demo_mode keeps the Phase 9.5 behaviour unchanged."""
        settings = _deployed_settings(environment)
        assert settings.demo_mode is False
        app = create_app(settings)
        app.state.rate_limiter = PermissiveLimiter()
        with TestClient(app) as client:
            response = client.post("/auth/demo-token", json={"persona": "analyst"})
        assert response.status_code == 404

    @pytest.mark.parametrize("environment", ["staging", "production"])
    def test_demo_mode_refuses_the_analyst_persona(self, environment: str) -> None:
        """The blocking finding: demo_mode must not let an anonymous caller reach explain/rings
        scope, however the account id is set."""
        settings = _deployed_settings(environment, demo_mode=True)
        app = create_app(settings)
        app.state.rate_limiter = PermissiveLimiter()
        with TestClient(app) as client:
            response = client.post("/auth/demo-token", json={"persona": "analyst"})
        assert response.status_code == 403

    @pytest.mark.parametrize("environment", ["staging", "production"])
    def test_demo_mode_refuses_an_account_id_outside_the_allowlist(
        self, environment: str
    ) -> None:
        """demo_account_ids defaults to empty, so demo_mode alone mints nothing until an
        operator explicitly allowlists ids -- fail closed, not fail open."""
        settings = _deployed_settings(environment, demo_mode=True)
        assert settings.demo_account_ids == ()
        app = create_app(settings)
        app.state.rate_limiter = PermissiveLimiter()
        with TestClient(app) as client:
            response = client.post(
                "/auth/demo-token", json={"persona": "merchant", "account_id": "acct-1"}
            )
        assert response.status_code == 403

    @pytest.mark.parametrize("environment", ["staging", "production"])
    def test_demo_mode_mints_a_merchant_token_for_an_allowlisted_account(
        self, environment: str
    ) -> None:
        settings = _deployed_settings(
            environment, demo_mode=True, demo_account_ids=("acct-1",)
        )
        app = create_app(settings)
        app.state.rate_limiter = PermissiveLimiter()
        with TestClient(app) as client:
            response = client.post(
                "/auth/demo-token", json={"persona": "merchant", "account_id": "acct-1"}
            )
        assert response.status_code == 200
        assert set(response.json()["scopes"]) == {
            SCOPE_SCORE,
            SCOPE_TRANSACTIONS_READ,
            SCOPE_AUDIT_READ,
        }

    @pytest.mark.parametrize("environment", ["staging", "production"])
    def test_demo_mode_still_refuses_an_account_id_not_on_the_allowlist(
        self, environment: str
    ) -> None:
        """The allowlist is exact -- being non-empty does not open every account id."""
        settings = _deployed_settings(
            environment, demo_mode=True, demo_account_ids=("acct-1",)
        )
        app = create_app(settings)
        app.state.rate_limiter = PermissiveLimiter()
        with TestClient(app) as client:
            response = client.post(
                "/auth/demo-token", json={"persona": "merchant", "account_id": "acct-2"}
            )
        assert response.status_code == 403

    @pytest.mark.parametrize("environment", ["local", "ci"])
    def test_the_restriction_never_fires_in_local_or_ci_even_with_demo_mode_set(
        self, environment: str
    ) -> None:
        """demo_mode is redundant in local/ci (the router is already mounted) and must not
        additionally narrow what a trusted local/ci caller can do."""
        settings = Settings(
            environment=environment,  # type: ignore[arg-type]
            jwt_secret_key=TEST_SIGNING_KEY,
            entity_anonymization_key="test-only-anonymization-key-not-a-real-secret",
            razorpay_webhook_secret="test-only-webhook-secret-not-a-real-secret",
            demo_mode=True,
        )
        app = create_app(settings)
        app.state.rate_limiter = PermissiveLimiter()
        with TestClient(app) as client:
            response = client.post("/auth/demo-token", json={"persona": "analyst"})
        assert response.status_code == 200

    def test_reject_placeholder_secret_outside_local_is_unaffected_by_demo_mode(self) -> None:
        """Pins the isolation claim in Settings.demo_mode's docstring: the Phase 9.5
        placeholder-secret refusal keys on `environment` alone, never on this flag."""
        with pytest.raises(ValueError):
            Settings(
                environment="production",
                jwt_secret_key="dev-only-placeholder-change-me-before-deploy",
                entity_anonymization_key="a-deployed-anonymization-key-of-sufficient-length",
                razorpay_webhook_secret="a-deployed-webhook-secret-of-sufficient-length-here",
                demo_mode=True,
            )


class TestPersonaScoping:
    """The one property this endpoint exists to guarantee: personas map to fixed scope sets."""

    def test_merchant_persona_cannot_see_why(self, client: TestClient, settings: Settings) -> None:
        response = client.post(
            "/auth/demo-token",
            json={"persona": "merchant", "account_id": "acct-1"},
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body["scopes"]) == {SCOPE_SCORE, SCOPE_TRANSACTIONS_READ, SCOPE_AUDIT_READ}
        assert SCOPE_ANALYST not in body["scopes"]
        assert SCOPE_EXPLAIN_READ not in body["scopes"]

        principal = decode_access_token(body["access_token"], settings)
        assert principal.account_id == "acct-1"
        assert SCOPE_ANALYST not in principal.scopes

    def test_analyst_persona_cannot_score(self, client: TestClient, settings: Settings) -> None:
        response = client.post("/auth/demo-token", json={"persona": "analyst"})
        assert response.status_code == 200
        body = response.json()
        assert set(body["scopes"]) == {
            SCOPE_ANALYST,
            SCOPE_TRANSACTIONS_READ,
            SCOPE_AUDIT_READ,
            SCOPE_EXPLAIN_READ,
            SCOPE_RINGS_READ,
        }
        assert SCOPE_SCORE not in body["scopes"]

        principal = decode_access_token(body["access_token"], settings)
        assert principal.account_id is None

    def test_the_request_cannot_name_scopes_directly(self, client: TestClient) -> None:
        """extra='forbid' refuses a scopes field outright rather than ignoring it."""
        response = client.post(
            "/auth/demo-token",
            json={"persona": "merchant", "account_id": "acct-1", "scopes": ["analyst"]},
        )
        assert response.status_code == 422

    def test_merchant_without_account_id_is_refused(self, client: TestClient) -> None:
        response = client.post("/auth/demo-token", json={"persona": "merchant"})
        assert response.status_code == 422

    def test_analyst_with_account_id_is_refused(self, client: TestClient) -> None:
        """Accepting one would invite a caller to believe analyst scope is account-bound."""
        response = client.post(
            "/auth/demo-token", json={"persona": "analyst", "account_id": "acct-1"}
        )
        assert response.status_code == 422


class TestExpiryIsHonest:
    """expires_in must be the real lifetime of the token, not a stapled-on shorter number."""

    def test_the_token_actually_expires_when_the_response_says_it_does(
        self, client: TestClient, settings: Settings
    ) -> None:
        import jwt

        response = client.post("/auth/demo-token", json={"persona": "analyst"})
        body = response.json()

        claims = jwt.decode(
            body["access_token"],
            TEST_SIGNING_KEY,
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
        )
        lifetime = claims["exp"] - claims["iat"]
        assert lifetime == body["expires_in"]

    def test_the_demo_token_lifetime_is_shorter_than_the_apps_default(
        self, client: TestClient, settings: Settings
    ) -> None:
        """Regression cover: the handler must copy settings, not mutate the shared singleton."""
        response = client.post("/auth/demo-token", json={"persona": "analyst"})
        assert response.json()["expires_in"] < settings.jwt_expiry_seconds

    def test_the_shared_settings_object_is_not_mutated_by_minting(
        self, client: TestClient, settings: Settings
    ) -> None:
        before = settings.jwt_expiry_seconds
        client.post("/auth/demo-token", json={"persona": "analyst"})
        assert settings.jwt_expiry_seconds == before


class TestMintedTokenWorksAgainstRealRoutes:
    """The token this route mints must actually authenticate against the real dependencies."""

    def test_the_analyst_token_authenticates(self, client: TestClient) -> None:
        token = client.post("/auth/demo-token", json={"persona": "analyst"}).json()["access_token"]
        response = client.get("/transactions", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200

    def test_the_merchant_token_cannot_reach_the_explain_route(self, client: TestClient) -> None:
        token = client.post(
            "/auth/demo-token", json={"persona": "merchant", "account_id": "acct-1"}
        ).json()["access_token"]
        response = client.get(
            "/audit/entry/1/explain", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403


class TestRateLimiting:
    """Phase 9.5: this was previously the only route on the service with no rate limit at
    all -- a caller with no bearer token could mint tokens at line rate."""

    def test_the_limiter_is_consulted(self, app: FastAPI, client: TestClient) -> None:
        client.post("/auth/demo-token", json={"persona": "analyst"})
        limiter = app.state.rate_limiter
        assert any(identity.startswith("ip:") for identity in limiter.calls)

    def test_no_limiter_installed_fails_closed(self, app: FastAPI, client: TestClient) -> None:
        app.state.rate_limiter = None
        response = client.post("/auth/demo-token", json={"persona": "analyst"})
        assert response.status_code == 503
