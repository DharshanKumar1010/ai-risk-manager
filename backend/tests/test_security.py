"""Tests for JWT minting and verification.

These are the Phase 0 floor for ``app.core.security``. Phase 7 adds the endpoint-level
auth-bypass and ownership-bypass integration tests.
"""

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.config import MIN_JWT_SECRET_BYTES, PLACEHOLDER_JWT_SECRET, Settings
from app.core.security import create_access_token, decode_access_token


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
