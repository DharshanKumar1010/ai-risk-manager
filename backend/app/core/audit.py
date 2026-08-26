"""The audit-trail choke point.

Every scoring decision RiskIQ makes is written through :func:`write_audit_record` and
through no other path. This is the buildathon's audit-trail requirement, not an optional
log — a decision that reached a caller without a corresponding audit row is a defect.

Phase 0 defined the record shape and the single entry point. Phase 7 implements persistence
against the append-only ``audit_log`` table and wires it into ``POST /score``.

Two properties this module is responsible for, both of which are enforced rather than assumed:

**A decision is never returned without its audit row.** ``/score`` writes and commits before
it serialises a response, so a failed write fails the request. The alternative — returning the
decision and logging the failure — produces exactly the state the audit trail exists to make
impossible: a decision in the world with no record of how it was reached.

**The record carries strictly more than the response does.** ``top_features`` and
``cost_estimate`` are stored here and never served to the transacting party; both are evasion
oracles, and the constraint travels with the fields rather than living in a reviewer's memory.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog

Decision = Literal["allow", "review", "block"]


class AuditRecord(BaseModel):
    """One immutable record of a scoring decision.

    The fields exist so that a past decision can be reconstructed exactly: which model
    versions ran, which feature definition fed them, what each layer said, what was
    decided, and whether the system was running degraded at the time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: str
    account_id: str = Field(
        min_length=1,
        max_length=128,
        description="Owning account. Required because the audit table's row-level security "
        "policy filters on it — a record without one could not be scoped to a reader, and "
        "the fail-closed policy would make it invisible to everyone.",
    )
    decided_at: datetime
    decision: Decision
    risk_probability: float = Field(ge=0.0, le=1.0)

    tier1_score: float | None = None
    tier2_reconstruction_error: float | None = None
    tier3_ring_risk_score: float | None = None

    model_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Layer name to model_version, for every layer that contributed.",
    )
    feature_version: str = Field(
        description="Hash from the Phase 1 feature store identifying the feature "
        "definition that produced this transaction's inputs.",
    )

    top_features: tuple[tuple[str, float], ...] | None = Field(
        default=None,
        description="Top contributing features by absolute TreeSHAP value, from the Phase 5 "
        "meta-learner. Recorded because the audit trail has to be able to answer *why* a "
        "decision was made, not only what it was. An evasion oracle: this field is for the "
        "audit store and internal reviewers, and must never be echoed to the transacting "
        "party -- see the security checklist's model-exposure section.",
    )
    cost_estimate: dict[str, Any] | None = Field(
        default=None,
        description="``DecisionCost.to_audit_dict()`` for this decision. Strictly worse than "
        "top_features as an oracle -- the sign of its expected saving is the decision "
        "boundary itself -- so no field of it may reach a response body.",
    )

    degraded: bool = Field(
        default=False,
        description="True when a layer was skipped and the decision was made without it.",
    )
    degraded_reason: str | None = Field(
        default=None,
        description="Why degraded mode was entered. Required whenever degraded is True.",
    )

    @model_validator(mode="after")
    def require_reason_when_degraded(self) -> "AuditRecord":
        """Refuse a degraded record that does not say why.

        The table carries the same rule as a check constraint. It is duplicated here so the
        failure surfaces as a validation error at the point the record is built, naming the
        field, rather than as an ``IntegrityError`` from the driver one layer down.
        """
        if self.degraded and not self.degraded_reason:
            raise ValueError(
                "degraded_reason is required when degraded is True. A record saying a layer "
                "was missing without saying which or why cannot reconstruct its decision, "
                "which is the one thing this table exists to guarantee."
            )
        return self


async def write_audit_record(session: AsyncSession, record: AuditRecord) -> int:
    """Append one decision to the immutable audit trail.

    The insert is flushed rather than committed. The caller owns the transaction, which is
    what lets ``/score`` commit the audit row and the response it describes as one unit — a
    committed decision with an uncommitted audit row is the failure this table prevents.

    Args:
        session: Active database session for the current request.
        record: The decision to record.

    Returns:
        The surrogate ``audit_id`` of the row written. ``/score`` returns it to the caller as
        an opaque handle: it identifies the decision for a later authenticated lookup without
        disclosing anything about how the decision was reached.
    """
    row = AuditLog(
        transaction_id=record.transaction_id,
        account_id=record.account_id,
        decided_at=record.decided_at,
        decision=record.decision,
        risk_probability=record.risk_probability,
        tier1_score=record.tier1_score,
        tier2_reconstruction_error=record.tier2_reconstruction_error,
        tier3_ring_risk_score=record.tier3_ring_risk_score,
        model_versions=dict(record.model_versions),
        feature_version=record.feature_version,
        # JSONB has no tuple type; the pairs round-trip as two-element arrays.
        top_features=(
            [[name, value] for name, value in record.top_features]
            if record.top_features is not None
            else None
        ),
        cost_estimate=record.cost_estimate,
        degraded=record.degraded,
        degraded_reason=record.degraded_reason,
    )
    session.add(row)
    await session.flush()
    return int(row.audit_id)
