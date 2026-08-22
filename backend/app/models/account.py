"""ORM model for the ``accounts`` table.

A derived summary of each account's activity, written by the Phase 1 pipeline in the same
run that writes ``transactions``. Tier-2 builds its per-account sequences from these rows
and Tier-3 builds graph nodes from them.

It also carries the two measurements that decide how much the account-level features are
worth on IEEE-CIS, which has **no account column** and so requires a constructed UID:

``uid_strategy``
    Which rung of the fallback ladder produced this identifier. A ``singleton`` account
    has exactly one transaction and no history, so every per-account feature computed for
    it — velocity, amount z-score against its own past — is noise rather than signal. The
    share of rows landing here is reported by the Phase 1 data-quality report, and if it is
    high, Tier-2's premise weakens and that must surface in Phase 1 rather than Phase 3.

``straddles_split``
    Whether this account's transactions fall in more than one chronological split. IEEE-CIS
    propagates a reported chargeback across *subsequent* transactions on the same account,
    card, email and billing address within roughly 120 days, so labels cluster tightly by
    account. A straddling account is therefore a measured leak surface across the split
    boundary. The chronological split stands as specified; this column is what lets every
    metrics report state the size of that surface instead of assuming it away.

Row-Level Security is enabled and forced on this table by the Phase 1 migration — see
``.claude/skills/security-checklist/SKILL.md`` section 3.
"""

from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.transaction import SOURCE_DATASET_VALUES_SQL

#: See the note on the equivalent constant in ``app.models.transaction``: literal SQL, kept
#: in step with ``app.data.schema.UidStrategy`` by ``tests/test_orm_constraints.py``.
UID_STRATEGY_VALUES_SQL = (
    "uid_strategy IN ('native', 'card_addr_d1n', 'card_email', 'card_only', 'singleton')"
)
FIRST_SPLIT_VALUES_SQL = "first_split IN ('train', 'val', 'test')"


class Account(Base):
    """One account's derived activity summary within a single corpus."""

    __tablename__ = "accounts"

    source_dataset: Mapped[str] = mapped_column(String(16), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(128), primary_key=True)

    uid_strategy: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="How this identifier was derived. 'native' for PaySim, which has real account "
        "columns; the constructed ladder for IEEE-CIS, which has none.",
    )

    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    transaction_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fraud_count: Mapped[int] = mapped_column(Integer, nullable=False)

    first_split: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        doc="The split this account's earliest transaction fell in.",
    )
    straddles_split: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="True when this account's transactions span more than one split — the "
        "measured leak surface described in this module's docstring.",
    )

    feature_version: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        CheckConstraint(SOURCE_DATASET_VALUES_SQL, name="ck_accounts_source_dataset"),
        CheckConstraint(UID_STRATEGY_VALUES_SQL, name="ck_accounts_uid_strategy"),
        CheckConstraint(FIRST_SPLIT_VALUES_SQL, name="ck_accounts_first_split"),
        CheckConstraint("transaction_count >= 1", name="ck_accounts_transaction_count_positive"),
        CheckConstraint(
            "fraud_count >= 0 AND fraud_count <= transaction_count",
            name="ck_accounts_fraud_count_within_total",
        ),
        CheckConstraint("last_seen >= first_seen", name="ck_accounts_time_ordered"),
        # A singleton account has exactly one transaction, by definition. Enforcing it here
        # means the singleton share reported by the data-quality report cannot drift from
        # what the strategy label claims.
        CheckConstraint(
            "uid_strategy <> 'singleton' OR transaction_count = 1",
            name="ck_accounts_singleton_has_one_transaction",
        ),
        # Both indexes serve the data-quality report, which counts the straddling accounts
        # and the singleton share per corpus on every pipeline run.
        Index("ix_accounts_straddles_split", "source_dataset", "straddles_split"),
        Index("ix_accounts_uid_strategy", "source_dataset", "uid_strategy"),
    )
