"""JWT authentication.

Phase 0 scope: token minting and verification, plus the FastAPI dependency that turns a
bearer token into a :class:`Principal`. Phase 7 attaches this dependency to every scoring
and write endpoint and adds the scope and ownership checks.

The rule this module exists to enforce: a caller's identity and permissions come from the
verified token and from nowhere else. A ``role``, ``account_id`` or ``is_admin`` field in
a request body is input to be validated, never an authorization decision.

Phase 7 adds three things to the Phase 0 floor:

**Scopes are checked server-side**, by :func:`require_scopes`, against the tuple carried in
the verified token. There is no code path in which a scope reaches an authorization decision
from anywhere other than a signature-checked claim.

**Ownership is checked separately from authentication**, by :func:`require_account_access` on
reads and :func:`require_account_ownership` on writes. An authenticated caller is not thereby
entitled to an arbitrary ``account_id`` in a path or query. ``analyst`` scope waives the check
on **reads only**: the write path has its own function with no bypass, because the application
always connects as ``riskiq_app`` and never assumes the ``riskiq_analyst`` role, so "the analyst
role holds no write grant" is not a control that is actually in force. See :func:`account_filter`.

**The principal is bound to the request**, so the rate limiter can key a budget on the caller
rather than on an address that a shared NAT makes meaningless.
"""

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings

bearer_scheme = HTTPBearer(auto_error=False)

#: Scope granting the estate-wide read the reviewer console needs. Held by analysts, never by
#: a merchant integration. It widens *visibility* only, and that is enforced in application code
#: (:func:`require_account_ownership` has no analyst branch) rather than by the database: the
#: ``riskiq_analyst`` role exists and holds no write grant, but nothing issues ``SET ROLE``, so
#: an analyst token currently operates with ``riskiq_app``'s grants. Do not rely on the role.
SCOPE_ANALYST = "analyst"

#: Permission to submit a transaction for scoring.
SCOPE_SCORE = "score:write"

#: Permission to read scored transaction history.
SCOPE_TRANSACTIONS_READ = "transactions:read"

#: Permission to read the audit trail.
SCOPE_AUDIT_READ = "audit:read"

#: Permission to read Tier-3 ring membership.
SCOPE_RINGS_READ = "rings:read"

#: Permission to retrieve the attribution behind a decision. Separated from ``audit:read``
#: because feature attribution is an evasion oracle and is useful to a strictly smaller set of
#: callers than the decision record itself -- see the security checklist's model-exposure item.
SCOPE_EXPLAIN_READ = "explain:read"


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
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> Principal:
    """FastAPI dependency resolving the bearer token into a verified principal.

    Settings are taken from application state rather than from the process singleton, so an
    app built by :func:`app.main.create_app` with an explicit ``Settings`` — which is how every
    test builds one — verifies against the key that app was configured with. Reading the
    ``lru_cache``d singleton here would make a test's signing key silently inert.

    The resolved principal is bound to ``request.state`` so that the rate limiter, which runs
    after this dependency, can key a budget on the caller.

    Args:
        request: The active request, for application state and principal binding.
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
    settings: Settings | None = getattr(request.app.state, "settings", None)
    principal = decode_access_token(credentials.credentials, settings)
    request.state.principal = principal
    return principal


CurrentPrincipal = Annotated[Principal, Depends(get_current_principal)]


def require_scopes(*required: str) -> Callable[[Principal], Awaitable[Principal]]:
    """Return a dependency admitting only principals holding every scope in ``required``.

    The check reads the tuple on the verified :class:`Principal`, which came from a
    signature-checked claim. Nothing in the request body or query string can reach it.

    A missing scope is 403, not 404: the caller is authenticated, and the route exists. Item
    2.5 of the checklist is about not disclosing *whether a resource exists* — which is the
    business of :func:`require_account_access` below, and is handled there.

    Args:
        required: Scope strings that must all be present.

    Returns:
        A FastAPI dependency yielding the principal unchanged when it is authorised.
    """

    async def dependency(principal: CurrentPrincipal) -> Principal:
        """Admit the principal, or refuse with 403."""
        missing = [scope for scope in required if scope not in principal.scopes]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient scope",
            )
        return principal

    return dependency


def require_account_ownership(principal: Principal, account_id: str) -> None:
    """Refuse anyone but the account's own holder. No analyst bypass.

    Used on the write path. :func:`require_account_access` lets an analyst through, which is
    right for a read — the reviewer console has to see across accounts — and wrong for a write:
    it would let the widest-reaching token in the system record a decision attributed to any
    account. The justification for the read bypass is that the analyst *database* role holds no
    write grant, and that justification does not currently hold, because the application always
    connects as ``riskiq_app`` and never assumes the analyst role. So the write path does not
    rely on it.

    Args:
        principal: The verified caller.
        account_id: The account the request is trying to write against.

    Raises:
        HTTPException: 404 when the caller is not the account holder, matching
            :func:`require_account_access` so the two cannot be told apart by status code.
    """
    if principal.account_id is not None and principal.account_id == account_id:
        return
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")


def require_account_access(principal: Principal, account_id: str) -> None:
    """Refuse a principal that does not own ``account_id``.

    The ownership rule the security checklist names in item 2.4: an authenticated user must
    not reach another account's data by changing an identifier in a URL. Two ways to pass —
    the token's own ``account_id`` matches, or the token carries analyst scope, which is what
    the reviewer console legitimately needs and which the analyst database role keeps
    read-only.

    Args:
        principal: The verified caller.
        account_id: The account the request is trying to reach.

    Raises:
        HTTPException: 404 when access is refused. Deliberately *not* 403: a 403 on an account
            that exists and a 404 on one that does not would let a caller enumerate account
            identifiers by watching the status code. Both answer "no such thing here for you".
    """
    if SCOPE_ANALYST in principal.scopes:
        return
    if principal.account_id is not None and principal.account_id == account_id:
        return
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")


class AccountScope(Enum):
    """How wide a list read may reach, when it is not confined to one account.

    An :class:`~enum.Enum` rather than sentinel strings, and that is load-bearing rather than
    tidiness. A string sentinel shares its value domain with ``account_id``, so a token minted
    with ``account_id="unrestricted"`` would compare equal to the sentinel and drop the filter.
    Enum members cannot be produced by any JWT claim, so the comparison cannot be spoofed by
    one. The whole point of this type is that "may see everything" and "may see nothing" are
    unforgeable and distinct; a string would have reintroduced a narrower version of exactly
    the bug it exists to fix.
    """

    #: The reader may see the whole estate. Analysts only.
    UNRESTRICTED = "unrestricted"

    #: The reader may see nothing. Fails closed.
    NOTHING = "nothing"


def account_filter(principal: Principal) -> str | AccountScope:
    """Return the account a list read must be filtered to, or how wide it may reach.

    Used by the list endpoints, where there is no identifier in the URL to check and the
    filter therefore *is* the authorization. Returning the token's account — never a
    client-supplied one — is what makes the filter an authorization decision rather than a
    convenience.

    **Three outcomes, not two, and conflating two of them was a real hole.** An earlier version
    returned ``str | None`` where ``None`` meant "analyst, no filter". But a non-analyst
    principal whose token carries no ``account_id`` claim also has no account — and it returned
    ``None`` for that case too. The callers read ``None`` as "apply no filter", so a token with
    ``transactions:read`` and no ``account_id`` was served every account's rows. That token
    shape is not exotic: ``create_access_token`` defaults ``account_id`` to ``None`` and
    ``decode_access_token`` does not require the claim.

    The fix is to make "may see everything" and "may see nothing" different, unforgeable values,
    so a caller cannot get the first by failing to specify anything — or by naming it.

    Returns:
        :attr:`AccountScope.UNRESTRICTED` for an analyst, :attr:`AccountScope.NOTHING` for a
        principal with no account scope, and otherwise the account id to confine the read to.
    """
    if SCOPE_ANALYST in principal.scopes:
        return AccountScope.UNRESTRICTED
    if principal.account_id is None:
        return AccountScope.NOTHING
    return principal.account_id
