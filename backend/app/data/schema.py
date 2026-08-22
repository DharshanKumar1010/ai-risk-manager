"""The canonical transaction contract every source maps into and every model reads.

One schema, several adapters. IEEE-CIS, PaySim and — from Phase 9 — Razorpay test-mode
webhooks all translate into :class:`TransactionFeatures`, so a model written against it
does not know or care which corpus a row came from.

What the schema deliberately does **not** do is merge the corpora. ``source_dataset`` is a
first-class discriminator, and three rules follow from it:

1. Splits are computed per source. IEEE-CIS ``TransactionDT`` (seconds, ~182 days) and
   PaySim ``step`` (hours, 30 days) have no common origin, so concatenating them into one
   timeline would fabricate an ordering that does not exist.
2. No model trains across sources. Their base rates differ by roughly 27x (3.50% against
   0.129%), so a pooled metric is uninterpretable.
3. No fitted score crosses between them. The Tier-3 ring detector is one *algorithm* run
   over two graphs; a PaySim ring score is never attached to an IEEE-CIS transaction.

The label lives on :class:`LabelledTransaction`, not on :class:`TransactionFeatures`. A
live scoring call has no label, and an optional ``is_fraud`` on the scoring contract is an
invitation to leak one into a feature vector.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.data.raw_spec import SourceDataset

Split = Literal["train", "val", "test"]

#: Chronological split order. The pipeline assigns these by time, never by shuffle — see
#: ``.claude/skills/ml-evaluation-standards/SKILL.md`` section 1.
SPLIT_ORDER: tuple[Split, ...] = ("train", "val", "test")

#: Fractions of each source's own timeline, applied per source.
SPLIT_FRACTIONS: dict[Split, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

#: How an account identifier was derived. IEEE-CIS has no account column, so a UID is
#: constructed and each row records which rung of the fallback ladder produced it — a row
#: on ``singleton`` has no usable history, and per-account features are noise for it.
UidStrategy = Literal[
    "native",
    "card_addr_d1n",
    "card_email",
    "card_only",
    "singleton",
]

#: Values an engineered feature may take. Constrained to what survives a JSONB round-trip
#: unchanged, so the feature vector read back from Postgres equals the one written.
FeatureValue = float | int | bool | str | None
FeatureVector = dict[str, FeatureValue]

#: Upper bound on any transaction amount, in the source's own currency units. Chosen to
#: clear PaySim's largest transfer (~92.4M) with headroom. An unbounded amount is both a
#: validation hole and a model-poisoning vector — see
#: ``.claude/skills/security-checklist/SKILL.md`` section 4.
MAX_TRANSACTION_AMOUNT = Decimal("100000000")

#: Precision of the ``amount`` column, enforced here so a mismatch fails at the schema
#: boundary rather than as a silent rounding inside Postgres.
AMOUNT_MAX_DIGITS = 20
AMOUNT_DECIMAL_PLACES = 4

# --- Time anchors ------------------------------------------------------------------
# Both source clocks are offsets from an unstated origin, so each gets an anchor. The
# anchors exist to produce a sortable, hour-of-day-bearing timestamp; they are NEVER
# compared across sources, and picking a different one changes no downstream result.

#: IEEE-CIS ``TransactionDT`` is a second offset. This origin is the widely-inferred
#: community value and is an ASSUMPTION, labelled as one wherever a date is displayed.
#: Only ordering and ``TransactionDT % 86400`` (hour-of-day) are load-bearing.
IEEE_CIS_EPOCH = datetime(2017, 12, 1, tzinfo=UTC)

#: PaySim ``step`` is an hour index over a 30-day simulation with no stated start date.
#: This origin is arbitrary by construction and carries no claim about real time.
PAYSIM_EPOCH = datetime(2017, 1, 1, tzinfo=UTC)


class TransactionFeatures(BaseModel):
    """One transaction as every model sees it — the scoring contract.

    This is the input type in every tier's ``score()`` signature and the target of every
    adapter, including Phase 9's Razorpay webhook translation. It carries no label and no
    split: those are training metadata, and a live scoring call has neither.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: str = Field(
        min_length=1,
        max_length=128,
        description="Unique within source_dataset, not across sources.",
    )
    source_dataset: SourceDataset
    event_time: datetime = Field(
        description="Timezone-aware. Derived from the source's own clock via its anchor.",
    )
    amount: Decimal = Field(
        ge=Decimal(0),
        le=MAX_TRANSACTION_AMOUNT,
        max_digits=AMOUNT_MAX_DIGITS,
        decimal_places=AMOUNT_DECIMAL_PLACES,
    )
    account_id: str = Field(
        min_length=1,
        max_length=128,
        description="PaySim nameOrig, or the constructed IEEE-CIS UID.",
    )
    counterparty_id: str | None = Field(
        default=None,
        max_length=128,
        description="PaySim nameDest. Always None for IEEE-CIS, which has no money-flow edge.",
    )
    transaction_type: str | None = Field(
        default=None,
        max_length=32,
        description="IEEE-CIS ProductCD or PaySim type.",
    )
    feature_version: str = Field(
        min_length=1,
        max_length=64,
        description="Hash from the feature store identifying the definition behind "
        "`features`, so any past prediction traces to an exact feature set.",
    )
    features: FeatureVector = Field(
        default_factory=dict,
        description="The engineered vector. Keys are fixed by feature_version.",
    )

    @field_validator("event_time")
    @classmethod
    def require_timezone_aware(cls, value: datetime) -> datetime:
        """Reject naive datetimes.

        Every split boundary and every trailing window in this project is a time
        comparison. A naive datetime compares fine right up until one side of the
        comparison is aware, and then it raises inside a pipeline stage rather than here.
        """
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_time must be timezone-aware")
        return value

    @model_validator(mode="after")
    def reject_counterparty_on_ieee_cis(self) -> "TransactionFeatures":
        """Refuse a counterparty on a corpus that has none.

        IEEE-CIS records no destination account — only shared-entity co-occurrence. A
        populated ``counterparty_id`` here would mean an adapter invented a money-flow edge,
        which would then feed the Tier-3 graph as though it were observed.
        """
        if self.source_dataset == "ieee_cis" and self.counterparty_id is not None:
            raise ValueError(
                "IEEE-CIS has no origin/destination structure; counterparty_id must be None"
            )
        return self


class LabelledTransaction(BaseModel):
    """A transaction plus the training metadata the pipeline assigns to it.

    Kept separate from :class:`TransactionFeatures` so that the label is structurally
    absent from anything a live scoring path touches.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction: TransactionFeatures
    is_fraud: bool
    split: Split


def ieee_cis_event_time(transaction_dt: int) -> datetime:
    """Convert an IEEE-CIS ``TransactionDT`` offset into a timestamp.

    Args:
        transaction_dt: Seconds since the dataset's unstated reference point.

    Returns:
        A timezone-aware timestamp anchored at :data:`IEEE_CIS_EPOCH`. Correct in ordering
        and hour-of-day; the calendar date carries the anchor's assumption.
    """
    return IEEE_CIS_EPOCH + timedelta(seconds=transaction_dt)


def paysim_event_time(step: int) -> datetime:
    """Convert a PaySim ``step`` (1-based hour index) into a timestamp.

    Args:
        step: Hour index over the 30-day simulation, starting at 1.

    Returns:
        A timezone-aware timestamp anchored at :data:`PAYSIM_EPOCH`. The anchor is
        arbitrary; only ordering and the one-hour granularity are meaningful.
    """
    return PAYSIM_EPOCH + timedelta(hours=step - 1)
