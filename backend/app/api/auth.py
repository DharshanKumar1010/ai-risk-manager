"""``POST /auth/demo-token`` — mint a walkthrough token. Local and CI only.

There is no token-issuing endpoint anywhere else in this service, by design: every other
token this repo has ever minted came from :func:`app.core.security.create_access_token`,
called directly by a script or a test, never over HTTP. This route exists solely so a judge
running the dashboard locally can authenticate as a merchant or a reviewer without a copy of
that recipe — it is a walkthrough convenience, not an identity provider.

**The one rule that keeps it from being a scope-escalation endpoint.** The request names a
*persona*, never a scope list. ``PERSONA_SCOPES`` below is the only place scopes are decided,
and it is a fixed module constant. If this route ever grows a request field that lets a caller
name scopes directly, it has become the exact thing security-checklist item 2.2 forbids: a
caller-supplied privilege claim reaching an authorization decision.

**Why the router is registered conditionally rather than gated inside the handler.** Checking
``settings.environment`` inside the function would still leave the route mounted, reachable,
and discoverable at ``/openapi.json`` in every environment — a dead code path is one commit
away from a live one if someone deletes the check without noticing what it was doing.
:func:`app.main.create_app` includes this router only when ``environment in ("local", "ci")``,
so in staging or production the path does not exist; a request to it 404s the same way any
unrouted path would, indistinguishable from a typo.
"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from app.core.rate_limit import enforce_webhook_rate_limit
from app.core.security import (
    SCOPE_ANALYST,
    SCOPE_AUDIT_READ,
    SCOPE_EXPLAIN_READ,
    SCOPE_RINGS_READ,
    SCOPE_SCORE,
    SCOPE_TRANSACTIONS_READ,
    create_access_token,
)

router = APIRouter(prefix="/auth", tags=["auth (demo only)"])

Persona = Literal["merchant", "analyst"]

#: The one mapping this route is allowed to consult. A merchant integration can score and read
#: its own history; it holds neither ``analyst`` nor ``explain:read`` nor ``rings:read``, so
#: the two-token walkthrough's central point -- a merchant cannot see why, an analyst can --
#: is a property of the scopes minted here, not of anything the caller asked for.
PERSONA_SCOPES: dict[Persona, tuple[str, ...]] = {
    "merchant": (SCOPE_SCORE, SCOPE_TRANSACTIONS_READ, SCOPE_AUDIT_READ),
    "analyst": (
        SCOPE_ANALYST,
        SCOPE_TRANSACTIONS_READ,
        SCOPE_AUDIT_READ,
        SCOPE_EXPLAIN_READ,
        SCOPE_RINGS_READ,
    ),
}

#: A short lifetime, because this token is minted for a browser session watching a demo, not
#: for a long-running integration.
DEMO_TOKEN_EXPIRY_SECONDS = 1800


class DemoTokenRequest(BaseModel):
    """Which persona to mint a walkthrough token for.

    ``account_id`` is required for ``merchant`` and refused for ``analyst``: a merchant's
    every capability is account-scoped, and an analyst's is deliberately not -- accepting one
    from an analyst request would invite a caller to believe analyst scope is narrower than it
    is. It is plain tenant selection, not a privilege the caller is granting itself: the seed
    script (``python -m app.data.seed_demo``) prints the account ids this can name, and any of
    them is equally available to any caller of this endpoint, in this environment only.
    """

    model_config = ConfigDict(extra="forbid")

    persona: Persona
    account_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Required for persona='merchant'; must be omitted for 'analyst'.",
    )


class DemoTokenResponse(BaseModel):
    """A freshly minted walkthrough token."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    persona: Persona
    scopes: tuple[str, ...]
    expires_in: int = Field(description="Seconds until the token expires.")


@router.post(
    "/demo-token",
    response_model=DemoTokenResponse,
    summary="Mint a short-lived walkthrough token (local/CI only)",
    dependencies=[Depends(enforce_webhook_rate_limit)],
)
async def mint_demo_token(payload: DemoTokenRequest, request: Request) -> DemoTokenResponse:
    """Mint a token carrying exactly the scopes ``PERSONA_SCOPES`` names for the persona.

    Settings come from ``request.app.state.settings``, not the process singleton -- the same
    reasoning :func:`app.core.security.get_current_principal` gives for the same choice. A
    token minted against the cached singleton's signing key would fail verification in any
    test built with an explicit ``Settings``, which is how every test in this suite builds one.

    Raises:
        HTTPException: 422 when ``account_id`` is missing for ``merchant`` or supplied for
            ``analyst`` -- the same status FastAPI's own body validation would use, so a
            caller sees one consistent error shape from this route rather than two.
    """
    if payload.persona == "merchant" and payload.account_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="account_id is required for persona='merchant'",
        )
    if payload.persona == "analyst" and payload.account_id is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="account_id must be omitted for persona='analyst'; analyst scope is not "
            "account-bound",
        )

    scopes = PERSONA_SCOPES[payload.persona]
    # create_access_token signs whatever cfg.jwt_expiry_seconds says into `exp` -- it takes no
    # per-call override. A copy with that one field replaced is what keeps the expires_in this
    # route reports true, rather than a shorter number stapled onto a longer-lived token.
    base_settings = request.app.state.settings
    settings = base_settings.model_copy(update={"jwt_expiry_seconds": DEMO_TOKEN_EXPIRY_SECONDS})
    token = create_access_token(
        subject=f"demo-{payload.persona}",
        account_id=payload.account_id,
        scopes=scopes,
        settings=settings,
    )
    return DemoTokenResponse(
        access_token=token,
        persona=payload.persona,
        scopes=scopes,
        expires_in=DEMO_TOKEN_EXPIRY_SECONDS,
    )
