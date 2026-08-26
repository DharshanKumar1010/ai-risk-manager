"""``GET /rings`` — Tier-3 abuse rings.

Responses expose ring membership and the size that drove each flag, but never the decision
threshold or per-feature weights — that would make the endpoint an evasion oracle. The per-
account ring *score* is withheld for the same reason: publishing a continuous score alongside
membership hands a ring operator a gradient to descend, letting them shrink or restructure a
ring until it drops below a threshold they can infer from where membership flips.

Analyst-scoped, and unavoidably so. Rings span accounts by construction, so there is no
account-scoped view of one that would still be a ring. This route reads the loaded snapshot
rather than the database, so it is unaffected by the fact that the ``riskiq_analyst`` role is
not yet assumed by the application — but do not read that role as the control here. The control
is :func:`require_scopes`.

**A stated limitation, because this endpoint would otherwise overclaim.** Registry entry 26
records Tier-3's per-transaction contribution as *below no-skill* on IEEE-CIS, and Phase 5's
ablation measured its effect on the fused ranking at -0.0001 with an interval spanning zero.
These rings are a reviewer's investigative lead, not a decision. Nothing here moves a score.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.api.schemas import RingListResponse, RingMember, RingResponse
from app.core.rate_limit import enforce_rate_limit
from app.core.security import SCOPE_ANALYST, SCOPE_RINGS_READ, Principal, require_scopes
from app.core.serving import ModelBundle

router = APIRouter(prefix="/rings", tags=["rings"])

#: Page size bounds, and a cap on how many members one ring reports. A ring can be large; a
#: response that inlined every member of every ring would be unbounded in two dimensions.
DEFAULT_LIMIT = 20
MAX_LIMIT = 100
MAX_MEMBERS = 200


@router.get(
    "",
    response_model=RingListResponse,
    summary="List flagged abuse rings (analysts only)",
    dependencies=[Depends(enforce_rate_limit)],
)
async def list_rings(
    principal: Annotated[Principal, Depends(require_scopes(SCOPE_RINGS_READ, SCOPE_ANALYST))],
    request: Request,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0, le=10_000)] = 0,
    min_size: Annotated[int, Query(ge=2, le=10_000)] = 2,
) -> RingListResponse:
    """Return flagged rings, largest first.

    Args:
        principal: The verified caller, holding both ``rings:read`` and ``analyst``.
        request: For the loaded model bundle.
        limit: Page size, bounded.
        offset: Rings to skip, bounded.
        min_size: Smallest ring to report. A two-account "ring" is usually a coincidence.

    Returns:
        A page of rings with their membership.

    Raises:
        HTTPException: 503 when no ring snapshot is loaded. Tier-3 is an enrichment and the
            service runs without it; this route is the one place that absence is not
            degradable, because there is nothing else to return.
    """
    bundle: ModelBundle | None = getattr(request.app.state, "model_bundle", None)
    if bundle is None or bundle.tier3 is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ring snapshot is unavailable",
        )
    tier3 = bundle.tier3

    members: dict[str, list[str]] = {}
    for account, ring_id in tier3.ring_of.items():
        # Membership is reported only for rings the layer actually flagged. An account sitting
        # in a component that scored below the operating point is not a finding, and listing it
        # would turn "we grouped these accounts" into "we suspect these accounts".
        if tier3.scores.get(account, 0.0) >= tier3.threshold:
            members.setdefault(ring_id, []).append(account)

    ranked = sorted(
        (
            (ring_id, accounts)
            for ring_id, accounts in members.items()
            if tier3.ring_sizes.get(ring_id, 0) >= min_size
        ),
        key=lambda pair: (-tier3.ring_sizes.get(pair[0], 0), pair[0]),
    )
    page = ranked[offset : offset + limit]

    rings = tuple(
        RingResponse(
            ring_id=ring_id,
            ring_size=int(tier3.ring_sizes.get(ring_id, 0)),
            members=tuple(
                RingMember(account_id=account) for account in sorted(accounts)[:MAX_MEMBERS]
            ),
            snapshot_end=tier3.snapshot_end,
        )
        for ring_id, accounts in page
    )
    return RingListResponse(rings=rings, count=len(rings), model_version=tier3.model_id)
