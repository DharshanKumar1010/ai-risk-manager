"""``POST /score`` — run a transaction through the scoring stack.

Every route here requires the JWT dependency from ``app.core.security``, writes through
``app.core.audit.write_audit_record``, and applies the Tier-3 timeout with the degraded-mode
fallback.

The ordering inside the handler is the part worth reading. The audit row is written and
committed **before** the response is built, so a failed write fails the request. Returning the
decision and logging the failure would produce exactly the state the audit trail exists to make
impossible: a decision in the world with no record of how it was reached.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import FeedEvent, ScoreRequest, ScoreResponse
from app.core.audit import write_audit_record
from app.core.feed import FeedBroadcaster
from app.core.rate_limit import enforce_rate_limit
from app.core.security import (
    SCOPE_SCORE,
    Principal,
    require_account_ownership,
    require_scopes,
)
from app.core.serving import ModelBundle, score_transaction
from app.db.session import get_scoped_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/score", tags=["scoring"])


def get_bundle(request: Request) -> ModelBundle:
    """Return the model bundle held on application state.

    Raises:
        HTTPException: 503 when the models are not loaded. A scoring service without models is
            unavailable, not broken — the distinction matters to whatever is deciding whether
            to route traffic here.
    """
    bundle: ModelBundle | None = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring models are unavailable",
        )
    return bundle


@router.post(
    "",
    response_model=ScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score one transaction",
    dependencies=[Depends(enforce_rate_limit)],
)
async def score(
    payload: ScoreRequest,
    principal: Annotated[Principal, Depends(require_scopes(SCOPE_SCORE))],
    session: Annotated[AsyncSession, Depends(get_scoped_session)],
    bundle: Annotated[ModelBundle, Depends(get_bundle)],
    request: Request,
) -> ScoreResponse:
    """Assemble features, score, price the decision, audit it, and return the outcome.

    Args:
        payload: The transaction. Raw source columns only — the engineered vector is built
            here, from these fields and the account's own history.
        principal: The verified caller, holding ``score:write``.
        session: Database session, already scoped to the caller's account for row-level
            security.
        bundle: The loaded models.
        request: For application settings.

    Returns:
        The decision and the audit handle, and nothing else quantitative.

    Raises:
        HTTPException: 404 when the caller does not own the account named in the body; 422
            when a raw column is not part of the model's input definition; 503 when the
            decision could not be assembled.
    """
    require_account_ownership(principal, payload.account_id)

    unknown = sorted(set(payload.raw_columns) - bundle.allowed_raw_columns)
    if unknown:
        # Names only, never the values — echoing the input back is checklist item 4.4. The
        # names are the caller's own keys and are what makes the error actionable.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "error": "unknown_raw_columns",
                "columns": unknown[:20],
                "hint": "Engineered features are computed server-side and cannot be supplied.",
            },
        )

    settings = request.app.state.settings
    try:
        outcome, record = await score_transaction(
            session,
            bundle,
            settings,
            transaction_id=payload.transaction_id,
            account_id=payload.account_id,
            event_time=payload.event_time,
            amount=payload.amount,
            raw_columns=dict(payload.raw_columns),
        )
    except ValueError as exc:
        # A feature-version mismatch or a missing feature. Both mean the vector this service
        # assembled is not the one the model was fitted against, and scoring anyway would
        # attach a wrong decision to a correct-looking audit row.
        logger.error("scoring refused for an assembled vector: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring is unavailable",
        ) from exc

    audit_id = await write_audit_record(session, record)
    await session.commit()

    # Published strictly after commit succeeds, never before. The audit trail's whole ordering
    # discipline exists to prevent a decision existing without its record; publishing earlier
    # would show a decision on the live feed that could still roll back. A missing or
    # unavailable broadcaster (most tests, and any deployment that has not wired one) is not a
    # scoring failure -- the feed is an enrichment of the response, not a precondition for it.
    broadcaster: FeedBroadcaster | None = getattr(request.app.state, "feed_broadcaster", None)
    if broadcaster is not None:
        broadcaster.publish(
            FeedEvent(
                audit_id=audit_id,
                transaction_id=payload.transaction_id,
                account_id=payload.account_id,
                decided_at=record.decided_at,
                decision=outcome.decision,
                risk_probability=outcome.probability,
                amount=str(payload.amount),
                degraded=outcome.degraded,
                model_version=outcome.model_versions["tier1"],
            ).model_dump(mode="json")
        )

    return ScoreResponse(
        transaction_id=payload.transaction_id,
        decision=outcome.decision,
        audit_id=audit_id,
        degraded=outcome.degraded,
        decided_at=record.decided_at,
        model_version=outcome.model_versions["tier1"],
    )
