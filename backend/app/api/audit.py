"""``GET /audit/{transaction_id}`` — the audit trail for one decision.

Read-only by construction: no route here may expose an update or delete path. That is enforced
below the application too — the Phase 7 migration grants ``audit_log`` only ``SELECT, INSERT``
and defines no ``FOR UPDATE`` or ``FOR DELETE`` policy, so the database refuses a rewrite even
if a future route asks for one.

Two routes, deliberately split by what they disclose:

``GET /audit/{transaction_id}``
    The decision and its provenance. Available to anyone who may read the account.

``GET /audit/entry/{audit_id}/explain``
    Which features drove the decision. Behind ``explain:read`` **and** ``analyst``, because
    this is the evasion oracle the security checklist names. It is served rather than withheld
    because a reviewer working a queue cannot do the job without it, and Phase 8's drill-down
    is built on it. The cost arms are **not** served here at any scope — ``DecisionCost`` is
    complete as an oracle, and Phase 6 widened the carried gate to cover every field of it.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    AuditEntryResponse,
    AuditListResponse,
    ExplanationResponse,
    FeatureContribution,
)
from app.core.rate_limit import enforce_rate_limit
from app.core.security import (
    SCOPE_ANALYST,
    SCOPE_AUDIT_READ,
    SCOPE_EXPLAIN_READ,
    AccountScope,
    Principal,
    account_filter,
    require_account_access,
    require_scopes,
)
from app.db.session import get_scoped_session
from app.models.audit_log import AuditLog

router = APIRouter(prefix="/audit", tags=["audit"])

#: Most recorded decisions returned for one transaction. A transaction can be scored more than
#: once — a replay, or a redelivered webhook — but not many times.
MAX_ENTRIES = 50


@router.get(
    "/{transaction_id}",
    response_model=AuditListResponse,
    summary="Read the audit trail for one transaction",
    dependencies=[Depends(enforce_rate_limit)],
)
async def read_audit_trail(
    principal: Annotated[Principal, Depends(require_scopes(SCOPE_AUDIT_READ))],
    session: Annotated[AsyncSession, Depends(get_scoped_session)],
    transaction_id: Annotated[str, Path(min_length=1, max_length=128)],
) -> AuditListResponse:
    """Return every recorded decision for ``transaction_id``, newest first.

    Raises:
        HTTPException: 404 when the transaction has no decisions the caller may see. The same
            status is returned whether the transaction does not exist or belongs to someone
            else — distinguishing them would let a caller enumerate identifiers.
    """
    scope = account_filter(principal)
    if scope is AccountScope.NOTHING:
        # Fail closed, and with the same 404 an unknown transaction gets, so a caller cannot
        # tell "you may see nothing" from "no such transaction".
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    statement = (
        select(AuditLog)
        .where(AuditLog.transaction_id == transaction_id)
        .order_by(AuditLog.decided_at.desc())
        .limit(MAX_ENTRIES)
    )
    if scope is not AccountScope.UNRESTRICTED:
        statement = statement.where(AuditLog.account_id == scope)

    rows = (await session.execute(statement)).scalars().all()
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    entries = tuple(_to_entry(row) for row in rows)
    return AuditListResponse(transaction_id=transaction_id, entries=entries)


@router.get(
    "/entry/{audit_id}/explain",
    response_model=ExplanationResponse,
    summary="Read the attribution behind one decision (reviewers only)",
    dependencies=[Depends(enforce_rate_limit)],
)
async def explain_decision(
    principal: Annotated[
        Principal, Depends(require_scopes(SCOPE_EXPLAIN_READ, SCOPE_ANALYST))
    ],
    session: Annotated[AsyncSession, Depends(get_scoped_session)],
    audit_id: Annotated[int, Path(ge=1)],
) -> ExplanationResponse:
    """Return the feature attribution behind one recorded decision.

    The route the security checklist would flag if it were unauthenticated, and the reasons it
    is not. It requires ``analyst`` **in addition to** ``explain:read``, matching ``/rings``.
    Gating on ``explain:read`` alone would have made the boundary depend on an issuance policy
    that does not exist anywhere in this repo — there is no token endpoint and nothing
    constrains which scopes a merchant integration is granted, so "a merchant's token does not
    carry it" would have been an assumption rather than a control. Owner match is deliberately
    not sufficient here: ``require_account_access`` passes for the account's owner, which is
    exactly the party this attribution must never reach.

    The cost arms are not returned at all. See :class:`ExplanationResponse`.

    Raises:
        HTTPException: 404 when no such row is visible to this caller.
    """
    row = await session.get(AuditLog, audit_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    require_account_access(principal, row.account_id)

    contributions = tuple(
        FeatureContribution(feature=str(name), contribution=float(value))
        for name, value in (row.top_features or [])
    )
    return ExplanationResponse(
        audit_id=row.audit_id,
        transaction_id=row.transaction_id,
        decision=row.decision,  # type: ignore[arg-type]
        risk_probability=row.risk_probability,
        top_features=contributions,
        model_versions=dict(row.model_versions),
        feature_version=row.feature_version,
        degraded=row.degraded,
    )


def _to_entry(row: AuditLog) -> AuditEntryResponse:
    """Project one stored row onto what an audit reader may see.

    ``top_features``, ``cost_estimate`` and any banding of the probability are absent by
    construction rather than by omission: this function is the only thing that builds an
    :class:`AuditEntryResponse`, and that schema has no field for any of them. The band was
    removed after review pointed out that the account holder can read its own audit rows, which
    reassembles the probe loop ``POST /score`` was hardened against.
    """
    return AuditEntryResponse(
        audit_id=row.audit_id,
        transaction_id=row.transaction_id,
        account_id=row.account_id,
        decided_at=row.decided_at,
        decision=row.decision,  # type: ignore[arg-type]
        model_versions=dict(row.model_versions),
        feature_version=row.feature_version,
        degraded=row.degraded,
        degraded_reason=row.degraded_reason,
    )
