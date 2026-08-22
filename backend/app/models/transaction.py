"""ORM model for the ``transactions`` table.

The single source of truth every downstream tier reads from. One row per transaction from
either corpus, carrying the canonical fields, the engineered feature vector, the
chronological ``split`` assignment, and the ``feature_version`` hash that ties the vector
to the definition which produced it.

Two shape decisions worth stating, because neither is the obvious default:

**The primary key is composite.** IEEE-CIS ``TransactionID`` and PaySim's synthesised row
identifier occupy overlapping integer ranges, so ``transaction_id`` alone is not unique
across sources. ``(source_dataset, transaction_id)`` is.

**Engineered features live in JSONB, not in columns.** The two corpora share almost no
raw fields — IEEE-CIS brings ~434, PaySim brings 11 — so a wide table would be ~440
columns, roughly 95% NULL on any given row, and ``feature_version`` would be decorative
because the column set could never change with it. The raw ``V1``-``V339`` block never
enters Postgres at all; model training reads the parquet materialisation in
``data/processed/``, which the pipeline asserts is row-for-row consistent with this table.

Row-Level Security is enabled and forced on this table by the Phase 1 migration — see
``.claude/skills/security-checklist/SKILL.md`` section 3.
"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, CheckConstraint, DateTime, Index, Numeric, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.data.schema import AMOUNT_DECIMAL_PLACES, AMOUNT_MAX_DIGITS, FeatureVector
from app.db.base import Base

#: Check-constraint text, kept as literal SQL rather than built from the Python literals.
#: ``tests/test_orm_constraints.py`` asserts these stay in step with ``app.data.schema``;
#: composing SQL by string formatting is barred outright by the security checklist, and an
#: exception "just for constants" is how that rule stops being enforceable.
SOURCE_DATASET_VALUES_SQL = "source_dataset IN ('ieee_cis', 'paysim')"
SPLIT_VALUES_SQL = "split IN ('train', 'val', 'test')"


class Transaction(Base):
    """One scored-or-scoreable transaction from either corpus."""

    __tablename__ = "transactions"

    source_dataset: Mapped[str] = mapped_column(
        String(16),
        primary_key=True,
        doc="Which corpus this row came from. Never mixed in a training set.",
    )
    transaction_id: Mapped[str] = mapped_column(
        String(128),
        primary_key=True,
        doc="Unique within source_dataset.",
    )

    event_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc="Derived from the source's own clock via its anchor in app.data.schema. "
        "Comparable within a source, never across sources.",
    )
    amount: Mapped[Decimal] = mapped_column(
        Numeric(AMOUNT_MAX_DIGITS, AMOUNT_DECIMAL_PLACES),
        nullable=False,
    )
    account_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        doc="PaySim nameOrig, or the constructed IEEE-CIS UID.",
    )
    counterparty_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        doc="PaySim nameDest. Always NULL for IEEE-CIS, which records no money-flow edge.",
    )
    transaction_type: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        doc="IEEE-CIS ProductCD or PaySim type.",
    )

    is_fraud: Mapped[bool] = mapped_column(Boolean, nullable=False)
    split: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        doc="Assigned chronologically within this row's source. Never by shuffle.",
    )
    feature_version: Mapped[str] = mapped_column(String(64), nullable=False)
    features: Mapped[FeatureVector] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="The engineered vector. Its key set is fixed by feature_version.",
    )

    __table_args__ = (
        CheckConstraint(SOURCE_DATASET_VALUES_SQL, name="ck_transactions_source_dataset"),
        CheckConstraint(SPLIT_VALUES_SQL, name="ck_transactions_split"),
        CheckConstraint("amount >= 0", name="ck_transactions_amount_non_negative"),
        # Structural guard against an adapter inventing a money-flow edge that IEEE-CIS
        # does not record. Without it, a fabricated counterparty would flow straight into
        # the Tier-3 graph as though it had been observed.
        CheckConstraint(
            "source_dataset <> 'ieee_cis' OR counterparty_id IS NULL",
            name="ck_transactions_ieee_cis_has_no_counterparty",
        ),
        # Trailing-window velocity features and Tier-2 sequence assembly both walk one
        # account's history in time order; this is the index that makes that a range scan.
        Index("ix_transactions_account_time", "source_dataset", "account_id", "event_time"),
        # Split-boundary assertions and per-split loads.
        Index("ix_transactions_source_split_time", "source_dataset", "split", "event_time"),
        # Tier-3 edge construction on the PaySim money-flow graph. Partial, because every
        # IEEE-CIS row has a NULL counterparty and would otherwise bloat the index.
        Index(
            "ix_transactions_counterparty",
            "source_dataset",
            "counterparty_id",
            postgresql_where=text("counterparty_id IS NOT NULL"),
        ),
    )
