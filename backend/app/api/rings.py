"""``GET /rings`` — Tier-3 abuse rings.

Responses expose ring membership and the size that drove each flag, but never the decision
threshold or per-feature weights — that would make the endpoint an evasion oracle. The per-
account ring *score* is withheld for the same reason: publishing a continuous score alongside
membership hands a ring operator a gradient to descend, letting them shrink or restructure a
ring until it drops below a threshold they can infer from where membership flips.

Analyst-scoped, and unavoidably so. Rings span accounts by construction, so there is no
account-scoped view of one that would still be a ring. This route reads the loaded snapshot
rather than the database, so it needs no session dependency at all — the control here is
:func:`require_scopes` alone, not row-level security.

**A stated limitation, because this endpoint would otherwise overclaim.** Registry entry 26
records Tier-3's per-transaction contribution as *below no-skill* on IEEE-CIS, and Phase 5's
ablation measured its effect on the fused ranking at -0.0001 with an interval spanning zero.
These rings are a reviewer's investigative lead, not a decision. Nothing here moves a score.

**Phase 8 adds ``nodes``/``edges`` for the network graph view.** An account node is the plain
``account_id``, already exposed on ``members``. An entity node is never the raw shared
fingerprint (a card/device composite) -- it is a truncated hash minted by
:func:`app.models.tier3_graph.export_ring_edges` at training time, so no reviewer response on
this API ever carries that identity signal in the clear.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.api.schemas import (
    RingGraphEdge,
    RingGraphNode,
    RingListResponse,
    RingMember,
    RingResponse,
)
from app.core.rate_limit import enforce_rate_limit
from app.core.security import SCOPE_ANALYST, SCOPE_RINGS_READ, Principal, require_scopes
from app.core.serving import ModelBundle
from app.models.tier3_graph import Tier3Model

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

    rings = tuple(_build_ring_response(tier3, ring_id, accounts) for ring_id, accounts in page)
    return RingListResponse(rings=rings, count=len(rings), model_version=tier3.model_id)


def _build_ring_response(tier3: Tier3Model, ring_id: str, accounts: list[str]) -> RingResponse:
    """Assemble one ring, capping membership, nodes and edges consistently.

    ``nodes``/``edges`` are capped to the *same* account set ``members`` is -- not sliced
    independently at ``MAX_MEMBERS``, which would either disagree with ``members`` about which
    accounts are represented or, on a ring with more than ``MAX_MEMBERS`` accounts, drop every
    entity node outright (`export_ring_edges` lists every account node before any entity node,
    so a blind tuple slice at the cap keeps only accounts). Filtering both to the capped account
    set keeps the three fields describing one consistent, bounded picture of the ring.
    """
    capped_accounts = sorted(accounts)[:MAX_MEMBERS]
    capped_account_set = set(capped_accounts)

    all_nodes = tier3.ring_nodes.get(ring_id, ())
    all_edges = tier3.ring_edges.get(ring_id, ())
    kept_node_ids = capped_account_set | {
        node.node_id
        for node in all_nodes
        if node.kind == "entity"
        and any(
            (edge.source in capped_account_set and edge.target == node.node_id)
            or (edge.target in capped_account_set and edge.source == node.node_id)
            for edge in all_edges
        )
    }

    return RingResponse(
        ring_id=ring_id,
        ring_size=int(tier3.ring_sizes.get(ring_id, 0)),
        members=tuple(RingMember(account_id=account) for account in capped_accounts),
        snapshot_end=tier3.snapshot_end,
        # Empty tuples, not a missing field, when this ring's model predates Phase 8's
        # topology export -- `Tier3Model.ring_nodes`/`ring_edges` default to `{}`.
        nodes=tuple(
            RingGraphNode(node_id=node.node_id, kind=node.kind, entity_type=node.entity_type)
            for node in all_nodes
            if node.node_id in kept_node_ids
        ),
        edges=tuple(
            RingGraphEdge(source=edge.source, target=edge.target)
            for edge in all_edges
            if edge.source in kept_node_ids and edge.target in kept_node_ids
        ),
    )
