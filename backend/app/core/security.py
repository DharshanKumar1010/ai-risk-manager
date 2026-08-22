"""JWT authentication.

Phase 0 scope: token minting and verification, plus the FastAPI dependency that turns a
bearer token into a :class:`Principal`. Phase 7 attaches this dependency to every scoring
and write endpoint and adds the ownership checks.

The rule this module exists to enforce: a caller's identity and permissions come from the
verified token and from nowhere else. A ``role``, ``account_id`` or ``is_admin`` field in
a request body is input to be validated, never an authorization decision.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings

bearer_scheme = HTTPBearer(auto_error=False)


class Principal(BaseModel):
    """An authenticated caller, derived solely from a verified token."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: str = Field(description="Stable identifier of the authenticated caller.")
    account_id: str | None = Field(
        default=None,
        description="Account this principal may access. None means no account scope.",
    )
    scopes: tuple[str, ...] = Field(
        default=(),
        description="Permissions granted by the token issuer, checked server-side.",
    )


def create_access_token(
    subject: str,
    account_id: str | None = None,
    scopes: tuple[str, ...] = (),
    settings: Settings | None = None,
) -> str:
    """Mint a signed access token for ``subject``.

    Args:
        subject: Stable identifier of the caller the token represents.
        account_id: Account scope to embed in the token, if any.
        scopes: Permissions to grant.
        settings: Configuration override, used by tests. Defaults to process settings.

    Returns:
        The encoded JWT as a compact serialised string.
    """
    cfg = settings or get_settings()
    now = datetime.now(UTC)
    payload: dict[str, object] = {
        "sub": subject,
        "iss": cfg.jwt_issuer,
        "aud": cfg.jwt_audience,
        "iat": now,
        "exp": now + timedelta(seconds=cfg.jwt_expiry_seconds),
        "jti": uuid.uuid4().hex,
        "account_id": account_id,
        "scopes": list(scopes),
    }
    return jwt.encode(payload, cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm)


def decode_access_token(token: str, settings: Settings | None = None) -> Principal:
    """Verify ``token`` and return the principal it represents.

    Signature, expiry, issuer and audience are all verified, and the accepted algorithm
    is pinned to the configured one so a token cannot downgrade itself to ``none``.

    Args:
        token: The encoded JWT from the Authorization header.
        settings: Configuration override, used by tests. Defaults to process settings.

    Returns:
        The verified :class:`Principal`.

    Raises:
        HTTPException: 401 if the token is missing, malformed, expired, or fails any
            verification step. The reason is deliberately not echoed to the caller.
    """
    cfg = settings or get_settings()
    try:
        claims = jwt.decode(
            token,
            cfg.jwt_secret_key,
            algorithms=[cfg.jwt_algorithm],
            issuer=cfg.jwt_issuer,
            audience=cfg.jwt_audience,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    raw_scopes = claims.get("scopes") or []
    if not isinstance(raw_scopes, list):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return Principal(
        subject=str(claims["sub"]),
        account_id=claims.get("account_id"),
        scopes=tuple(str(scope) for scope in raw_scopes),
    )


async def get_current_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> Principal:
    """FastAPI dependency resolving the bearer token into a verified principal.

    Args:
        credentials: Parsed Authorization header, or None when the header is absent.

    Returns:
        The verified :class:`Principal`.

    Raises:
        HTTPException: 401 when no bearer token was supplied or verification failed.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_access_token(credentials.credentials)
