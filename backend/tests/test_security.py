"""Tests for JWT minting and verification.

These are the Phase 0 floor for ``app.core.security``. Phase 7 adds the endpoint-level
auth-bypass and ownership-bypass integration tests.
"""

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.config import MIN_JWT_SECRET_BYTES, PLACEHOLDER_JWT_SECRET, Settings
from app.core.security import (
    Principal,
    create_access_token,
    decode_access_token,
    decode_ws_ticket,
    mint_ws_ticket,
)


def test_token_round_trip(settings: Settings) -> None:
    """A freshly minted token decodes back to the principal it was minted for."""
    token = create_access_token(
        subject="analyst-1",
        account_id="acct-42",
        scopes=("score:read",),
        settings=settings,
    )

    principal = decode_access_token(token, settings=settings)

    assert principal.subject == "analyst-1"
    assert principal.account_id == "acct-42"
    assert principal.scopes == ("score:read",)


def test_tampered_signature_is_rejected(settings: Settings) -> None:
    """A token signed with a different key must not verify."""
    foreign = Settings(environment="ci", jwt_secret_key="a-different-key-long-enough-for-hs256")
    token = create_access_token(subject="attacker", settings=foreign)

    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token, settings=settings)

    assert exc_info.value.status_code == 401


def test_wrong_audience_is_rejected(settings: Settings) -> None:
    """A token minted for another audience must not be accepted by this API."""
    other_audience = settings.model_copy(update={"jwt_audience": "some-other-api"})
    token = create_access_token(subject="analyst-1", settings=other_audience)

    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token, settings=settings)

    assert exc_info.value.status_code == 401


def test_garbage_token_is_rejected(settings: Settings) -> None:
    """A malformed token is rejected without leaking why."""
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token("not-a-jwt", settings=settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid or expired credentials"


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_placeholder_secret_refuses_to_boot_when_deployed(environment: str) -> None:
    """A deployed environment must not start on the well-known placeholder key."""
    with pytest.raises(ValueError, match="placeholder"):
        Settings(environment=environment, jwt_secret_key=PLACEHOLDER_JWT_SECRET)  # type: ignore[arg-type]


def test_short_signing_key_is_rejected() -> None:
    """An HMAC key below 32 bytes is refused outright, not merely warned about.

    RFC 7518 section 3.2 requires an HS256 key at least as long as the hash output.
    PyJWT only warns; for a fraud-detection service a warning is not enough.
    """
    with pytest.raises(ValidationError):
        Settings(environment="ci", jwt_secret_key="a" * (MIN_JWT_SECRET_BYTES - 1))


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_entity_anonymization_placeholder_refuses_to_boot_when_deployed(
    environment: str,
) -> None:
    """Security-review regression cover: a deployed service must not export Tier-3 entity ids
    anonymized with the well-known placeholder key, which anyone who has read this repo could
    use to reverse them by dictionary attack."""
    with pytest.raises(ValueError, match="entity_anonymization_key"):
        Settings(
            environment=environment,  # type: ignore[arg-type]
            jwt_secret_key="a-deployed-signing-key-of-sufficient-length-here",
        )


def test_a_real_entity_anonymization_key_boots_fine_when_deployed() -> None:
    """Guard on the guard: the validator must not refuse every deployed boot outright."""
    Settings(
        environment="production",  # type: ignore[arg-type]
        jwt_secret_key="a-deployed-signing-key-of-sufficient-length-here",
        entity_anonymization_key="a-deployed-anonymization-key-of-sufficient-length",
    )


def test_ws_audience_matching_the_api_audience_is_rejected() -> None:
    """The whole safety argument for a separate ws ticket audience -- see
    mint_ws_ticket's docstring -- depends on the two audiences never being equal."""
    with pytest.raises(ValueError, match="jwt_ws_audience"):
        Settings(environment="ci", jwt_audience="riskiq-api", jwt_ws_audience="riskiq-api")


def test_distinct_ws_and_api_audiences_boot_fine() -> None:
    """Guard on the guard."""
    Settings(environment="ci", jwt_audience="riskiq-api", jwt_ws_audience="riskiq-ws")


class TestWsTicketAudienceIsolation:
    """A ticket must authenticate exactly one thing: the socket it was minted for.

    The whole safety argument in ``mint_ws_ticket``'s docstring rests on the audience swap
    actually being enforced both ways. These tests are that argument, checked.
    """

    def test_a_ticket_carries_the_principals_claims(self, settings: Settings) -> None:
        principal = Principal(
            subject="analyst-1", account_id=None, scopes=("analyst", "audit:read")
        )
        ticket = mint_ws_ticket(principal, settings)

        decoded = decode_ws_ticket(ticket, settings)

        assert decoded.subject == "analyst-1"
        assert decoded.scopes == ("analyst", "audit:read")

    def test_a_ws_ticket_is_rejected_by_ordinary_token_verification(
        self, settings: Settings
    ) -> None:
        """A leaked ticket must not double as a bearer token against a REST route."""
        principal = Principal(subject="analyst-1", scopes=("analyst",))
        ticket = mint_ws_ticket(principal, settings)

        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(ticket, settings)

        assert exc_info.value.status_code == 401

    def test_an_ordinary_bearer_token_is_rejected_as_a_ws_ticket(self, settings: Settings) -> None:
        """The reverse direction: a normal token must not open the socket either."""
        token = create_access_token(subject="analyst-1", scopes=("analyst",), settings=settings)

        with pytest.raises(HTTPException) as exc_info:
            decode_ws_ticket(token, settings)

        assert exc_info.value.status_code == 401

    def test_a_ticket_minted_for_one_deployments_settings_is_rejected_by_another(self) -> None:
        first = Settings(environment="ci", jwt_secret_key="a-signing-key-of-sufficient-length-1")
        second = Settings(environment="ci", jwt_secret_key="a-signing-key-of-sufficient-length-2")
        principal = Principal(subject="analyst-1", scopes=("analyst",))
        ticket = mint_ws_ticket(principal, first)

        with pytest.raises(HTTPException) as exc_info:
            decode_ws_ticket(ticket, second)

        assert exc_info.value.status_code == 401

    def test_the_ticket_expiry_is_the_configured_short_lifetime(self, settings: Settings) -> None:
        """Regression cover: minting must not silently fall back to the full-length expiry."""
        import jwt

        principal = Principal(subject="analyst-1", scopes=("analyst",))
        ticket = mint_ws_ticket(principal, settings)

        claims = jwt.decode(
            ticket,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_ws_audience,
        )
        assert claims["exp"] - claims["iat"] == settings.ws_ticket_expiry_seconds
        assert settings.ws_ticket_expiry_seconds < settings.jwt_expiry_seconds

    def test_minting_does_not_mutate_the_shared_settings_object(self, settings: Settings) -> None:
        before_audience = settings.jwt_audience
        before_expiry = settings.jwt_expiry_seconds
        principal = Principal(subject="analyst-1", scopes=("analyst",))

        mint_ws_ticket(principal, settings)

        assert settings.jwt_audience == before_audience
        assert settings.jwt_expiry_seconds == before_expiry
