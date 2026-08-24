"""The audit-trail choke point.

Every scoring decision RiskIQ makes is written through :func:`write_audit_record` and
through no other path. This is the buildathon's audit-trail requirement, not an optional
log — a decision that reached a caller without a corresponding audit row is a defect.

Phase 0 defines the record shape and the single entry point. Phase 7 implements
persistence against the append-only ``audit_log`` table and wires it into ``POST /score``.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

Decision = Literal["allow", "review", "block"]


class AuditRecord(BaseModel):
    """One immutable record of a scoring decision.

    The fields exist so that a past decision can be reconstructed exactly: which model
    versions ran, which feature definition fed them, what each layer said, what was
    decided, and whether the system was running degraded at the time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: str
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

    degraded: bool = Field(
        default=False,
        description="True when a layer was skipped and the decision was made without it.",
    )
    degraded_reason: str | None = Field(
        default=None,
        description="Why degraded mode was entered. Required whenever degraded is True.",
    )


async def write_audit_record(session: AsyncSession, record: AuditRecord) -> None:
    """Append one decision to the immutable audit trail.

    Args:
        session: Active database session for the current request.
        record: The decision to record.

    Raises:
        NotImplementedError: Until Phase 7 implements persistence. Raising rather than
            silently discarding is deliberate — a no-op audit writer would let Phase 7
            ship a scoring endpoint that appears to be audited and is not.
    """
    raise NotImplementedError(
        "Audit persistence is implemented in Phase 7 against the append-only audit_log "
        "table. No scoring endpoint may ship before it."
    )
