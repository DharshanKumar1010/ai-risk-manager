"""ORM model for the append-only ``audit_log`` table.

Every row records one scoring decision: the transaction id, all four layer scores and flags,
the final decision, the ``model_version`` of every layer involved, the ``feature_version``
hash from the Phase 1 feature store, whether degraded mode was used and why, and a timestamp.

This table is append-only by design. No UPDATE or DELETE path may exist in application code —
see ``.claude/skills/security-checklist/SKILL.md`` section 7. "By design" is not left to
convention: the Phase 7 migration grants only ``SELECT, INSERT`` to the application role and
defines no ``FOR UPDATE`` or ``FOR DELETE`` policy, so the database refuses a rewrite even if
application code one day asks for one.

Three shape decisions worth stating:

**The primary key is a surrogate, not the transaction id.** A transaction can be scored more
than once — Phase 7's ``/replay`` enhancement re-scores a past transaction against current
model versions, and Phase 9's webhook can deliver the same payment event twice. Keying on
``transaction_id`` would force those into an UPSERT, which is exactly the UPDATE path this
table must not have. Each scoring event is its own row.

**``model_versions`` is JSONB, not a column per layer.** Which layers contributed varies per
decision: Phase 5 retired Tiers 2 and 3 from the shipped ensemble, degraded mode drops a layer
at runtime, and a future phase may add one. A fixed column set would go stale; the mapping
records exactly the layers that ran.

**``top_features`` is stored but never served to the transacting party.** It is what lets the
audit trail answer *why*, and it is an evasion oracle — the constraint is carried on
``AuditRecord.top_features`` and enforced at the response boundary by ``MetaResult.public()``.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

#: Check-constraint text, kept as literal SQL rather than built from the Python literal, for
#: the same reason ``app/models/transaction.py`` does it: composing SQL by string formatting is
#: barred outright by the security checklist, and an exception "just for constants" is how that
#: rule stops being enforceable. ``tests/test_orm_constraints.py`` asserts it stays in step with
#: ``app.core.audit.Decision``.
DECISION_VALUES_SQL = "decision IN ('allow', 'review', 'block')"

#: A degraded row must say why. The audit trail's whole claim is that a past decision can be
#: reconstructed; "a layer was missing, reason unrecorded" fails that claim, so the database
#: refuses the row rather than storing an unreconstructable one.
DEGRADED_REASON_SQL = "NOT degraded OR degraded_reason IS NOT NULL"


class AuditLog(Base):
    """One immutable record of one scoring decision."""

    __tablename__ = "audit_log"

    audit_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        doc="Surrogate key. A transaction may be scored more than once; each is its own row.",
    )

    transaction_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        doc="The transaction this decision was made about.",
    )
    account_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        doc="Owning account. Present so row-level security can scope an audit read the same "
        "way it scopes a transaction read — without it the isolation policy would have "
        "nothing to filter on and audit reads would have to bypass RLS entirely.",
    )
    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
        doc="When the decision was taken.",
    )

    decision: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        doc="allow, review or block.",
    )
    risk_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="The calibrated fraud probability the decision rested on.",
    )

    tier1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    tier2_reconstruction_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    tier3_ring_risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    model_versions: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        doc="Layer name to registry model_id, for every layer that contributed.",
    )
    feature_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Hash identifying the feature definition behind this decision's inputs.",
    )

    top_features: Mapped[list[Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        doc="Top contributors by absolute TreeSHAP value. Internal reviewers only — never "
        "echoed to the transacting party. See the security checklist's model-exposure item.",
    )
    cost_estimate: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        doc="DecisionCost.to_audit_dict(). Server-side only for any response read by a party "
        "authenticated as the transacting merchant: the sign of its expected saving is the "
        "decision boundary. One exception -- Phase 9's Razorpay webhook, whose caller is "
        "authenticated by a shared HMAC secret, not a merchant JWT. See "
        "app/api/schemas.py's module docstring.",
    )

    degraded: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
        doc="True when a layer was skipped and the decision was made without it.",
    )
    degraded_reason: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        doc="Why degraded mode was entered. Required whenever degraded is true.",
    )

    __table_args__ = (
        CheckConstraint(DECISION_VALUES_SQL, name="ck_audit_log_decision"),
        CheckConstraint(
            "risk_probability >= 0 AND risk_probability <= 1",
            name="ck_audit_log_risk_probability_unit_interval",
        ),
        CheckConstraint(DEGRADED_REASON_SQL, name="ck_audit_log_degraded_has_reason"),
        # The lookup ``GET /audit/{transaction_id}`` performs, newest first.
        Index("ix_audit_log_transaction", "transaction_id", "decided_at"),
        # The dashboard's live feed: most recent decisions across the estate.
        Index("ix_audit_log_decided_at", "decided_at"),
        # The account-scoped read, which is what the RLS policy filters on.
        Index("ix_audit_log_account_time", "account_id", "decided_at"),
    )
