"""Request and response schemas for the HTTP surface.

Every model here is ``extra="forbid"``. That is security-checklist item 4.1, and on this
service it does real work rather than being hygiene: a scoring request carries ~90 optional
raw columns, so a typo in a field name would otherwise be silently dropped and the transaction
scored against a vector missing the feature the caller meant to send.

**What a response may carry is decided here, once.** Three things are excluded from every
response body on this service, and each has a specific reason recorded where it is defined:

``top_features``
    Feature attribution says which signals mattered, which is the same as saying which to
    avoid next time. Reviewer-only, behind ``explain:read``.

Any field of ``DecisionCost``
    Strictly worse: the sign of ``expected_saving_from_blocking`` *is* the decision boundary,
    so a handful of probe transactions binary-searches the largest amount that evades review at
    a given risk score. Phase 6 widened the carried gate to cover the whole object.

The operating threshold, the calibrated probability, and any coarse banding of it
    Together with the amount, each reconstructs the same boundary. ``POST /score`` returns the
    decision and nothing quantitative about it.

**A correction worth recording, because the first version of this file got it wrong.** An
earlier ``ScoreResponse`` carried a three-value ``risk_band`` in place of the probability, and
this docstring claimed that recovering a boundary from it "takes O(n) probes per band edge
instead of O(log n) against a continuous score". That is false. ``risk_band`` is *monotone* in
the probability, so binary search over any monotone input — the amount, say — locates an edge
in O(log n) exactly as it would against a continuous score. Coarsening the output reduces the
bits per probe; it does not defeat search. Worse, the band edges were published as tracked
constants, so a located edge was a *calibrated* anchor rather than an unknown one, and two
anchors plus two decision flips over-determine the cost matrix.

The band is therefore gone from the scoring response. What remains is the decision itself,
which is a one-bit oracle that no fraud API can avoid emitting — the caller has to be told
whether the transaction was allowed. The residual exposure is real and is bounded by the rate
limiter and by ``POST /score`` being authenticated per account; closing it properly needs
per-``(account_id, transaction_id)`` idempotency so that probing costs the caller a distinct
transaction, which is recorded as a Phase 9 prerequisite in ``BUILD_LOG.md``.

**And the same correction had to be made twice.** The first fix removed the band from
``POST /score`` and left it on ``GET /audit/{transaction_id}``, reasoning that "that path is
authenticated, account-scoped and not a probe loop". That was false for the same reason the
first claim was: the account-scoped party *is* the probing party. A merchant holding
``score:write`` and ``audit:read`` posts a transaction, reads back its own audit row, and
bisects on the amount — the probe loop reassembled across two calls. So the band is gone from
every response on this service. A reviewer who needs magnitude reads ``risk_probability`` on
the analyst-only explain route.

:func:`risk_band` itself is kept because the dashboard colours a feed by it, but it is computed
from a probability that only an analyst can obtain.
"""

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.audit import Decision
from app.data.schema import (
    AMOUNT_DECIMAL_PLACES,
    AMOUNT_MAX_DIGITS,
    MAX_TRANSACTION_AMOUNT,
)

#: Coarse risk bands. Used on the **audit** path only, never on a scoring response — see the
#: module docstring for why coarsening does not defeat a search over a monotone input.
RiskBand = Literal["low", "elevated", "high"]

#: Where the band edges sit on the probability scale.
RISK_BAND_ELEVATED = 0.05
RISK_BAND_HIGH = 0.25

#: Ceiling on the number of raw columns one request may carry. The model reads 91; anything
#: beyond that is either a mistake or an attempt to make validation expensive.
MAX_RAW_COLUMNS = 128

#: Longest string accepted as a raw categorical value. Comfortably above the longest level in
#: the fitted category sets, and short enough that a payload cannot be used to push memory.
MAX_RAW_VALUE_LENGTH = 256

#: Bound on any numeric raw column. IEEE-CIS counters and deltas live far inside this; the
#: bound exists because an unbounded numeric that reaches a model is both a validation hole and
#: a model-poisoning vector — security-checklist item 4.2.
MAX_RAW_NUMERIC = 1e12

RawValue = float | int | str | bool | None


def risk_band(probability: float) -> RiskBand:
    """Return the coarse band a calibrated probability falls into."""
    if probability >= RISK_BAND_HIGH:
        return "high"
    if probability >= RISK_BAND_ELEVATED:
        return "elevated"
    return "low"


class ScoreRequest(BaseModel):
    """One transaction submitted for scoring.

    The caller supplies canonical fields plus the corpus's raw source columns. It does **not**
    supply the engineered feature vector: that is assembled server-side from these fields and
    the account's own history, because a caller-supplied vector would let it choose its own
    score while leaving a correct-looking audit row.

    ``account_id`` is validated as input and is *not* an authorization decision. The route
    checks it against the verified token before scoring anything.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(
        min_length=1,
        max_length=128,
        description="Caller's identifier for this transaction. Recorded on the audit row.",
    )
    account_id: str = Field(
        min_length=1,
        max_length=128,
        description="The transacting account. Checked against the token, never trusted as an "
        "authorization input on its own.",
    )
    event_time: datetime = Field(
        description="When the transaction occurred. Must be timezone-aware: every trailing "
        "window in the feature set is a time comparison, and a naive datetime fails one of "
        "those deep inside assembly rather than here.",
    )
    amount: Annotated[
        Decimal,
        Field(
            ge=Decimal(0),
            le=MAX_TRANSACTION_AMOUNT,
            max_digits=AMOUNT_MAX_DIGITS,
            decimal_places=AMOUNT_DECIMAL_PLACES,
            description="Bounded on both sides. A negative amount would invert the cost "
            "layer's ranking score and switch blocking off for the row.",
        ),
    ]
    raw_columns: dict[str, RawValue] = Field(
        default_factory=dict,
        description="The corpus's raw source columns. Keys are validated against the model's "
        "own input definition; every derived feature is refused and computed server-side.",
    )

    @field_validator("event_time")
    @classmethod
    def require_timezone_aware(cls, value: datetime) -> datetime:
        """Reject a naive datetime, matching the pipeline's own contract."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_time must be timezone-aware")
        return value

    @field_validator("raw_columns")
    @classmethod
    def bound_raw_values(cls, value: dict[str, RawValue]) -> dict[str, RawValue]:
        """Bound the size and the values of the raw column map.

        The key *set* is checked against the loaded model in the route, which is where the
        model is available. This validator enforces what can be checked without it: how many
        columns, how long a string, how large a number.
        """
        if len(value) > MAX_RAW_COLUMNS:
            raise ValueError(f"at most {MAX_RAW_COLUMNS} raw columns may be supplied")
        for column, entry in value.items():
            if len(column) > MAX_RAW_VALUE_LENGTH:
                raise ValueError("raw column names must be shorter than 256 characters")
            if isinstance(entry, str) and len(entry) > MAX_RAW_VALUE_LENGTH:
                raise ValueError(f"raw column {column!r} exceeds the maximum value length")
            if isinstance(entry, bool):
                continue
            if isinstance(entry, (int, float)) and not -MAX_RAW_NUMERIC <= entry <= MAX_RAW_NUMERIC:
                raise ValueError(f"raw column {column!r} is outside the accepted numeric range")
        return value


class ScoreResponse(BaseModel):
    """What a scoring call returns to the transacting party.

    Built from a strict subset of :class:`app.core.serving.ScoringOutcome`. The fields that are
    absent are absent on purpose — see the module docstring.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: str
    decision: Decision = Field(
        description="allow, review or block. Chosen by the Phase 6 cost policy on the "
        "transaction's own amount, not by a fixed probability cut. This is the only thing "
        "this response says about the risk, and it is the irreducible minimum: a scoring API "
        "has to tell the caller what happened to the transaction."
    )
    audit_id: int = Field(
        description="Opaque handle for this decision's audit row. Identifies the decision for "
        "a later authenticated lookup and discloses nothing about how it was reached.",
    )
    degraded: bool = Field(
        description="True when a layer was unavailable and the decision was made without it."
    )
    decided_at: datetime
    model_version: str = Field(
        description="Registry id of the layer that produced the risk estimate. Names which "
        "model decided, which the audit trail needs; reveals no operating point."
    )


class TransactionSummary(BaseModel):
    """One scored transaction in a history listing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: str
    account_id: str
    event_time: datetime
    amount: Decimal
    transaction_type: str | None


class TransactionListResponse(BaseModel):
    """A page of scored transactions, scoped to what the caller may see."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transactions: tuple[TransactionSummary, ...]
    count: int = Field(ge=0, description="Rows in this page, not the total matching.")


class AuditEntryResponse(BaseModel):
    """One audit row, as returned to an authenticated reader.

    Carries the decision and its provenance. ``top_features`` and the cost breakdown are
    **not** here: they are served only by the reviewer-scoped explanation route, because
    attribution and cost arms are evasion oracles and a great many more callers legitimately
    need to see *that* a decision happened than need to see how close it was to the boundary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    audit_id: int
    transaction_id: str
    account_id: str
    decided_at: datetime
    decision: Decision
    model_versions: dict[str, str]
    feature_version: str
    degraded: bool
    degraded_reason: str | None


class AuditListResponse(BaseModel):
    """Every recorded decision for one transaction, newest first."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: str
    entries: tuple[AuditEntryResponse, ...]


class AuditFeedResponse(BaseModel):
    """A page of recorded decisions across transactions, newest first.

    What ``GET /audit`` returns — the decision table's and the live feed's backlog source.
    Distinct from :class:`AuditListResponse`, which is scoped to one already-known
    ``transaction_id``; this is the unscoped list, so it carries ``count`` the way
    :class:`TransactionListResponse` does rather than a single fixed identifier.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: tuple[AuditEntryResponse, ...]
    count: int = Field(ge=0, description="Rows in this page, not the total matching.")


class FeatureContribution(BaseModel):
    """One feature's signed contribution to a decision, in margin space."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str
    contribution: float


class ExplanationResponse(BaseModel):
    """Why one decision came out the way it did. Analyst-scoped.

    Attribution is an evasion oracle, served deliberately rather than withheld: a reviewer
    working a queue cannot do the job without knowing which signals drove a decision, and
    Phase 8's drill-down is built on it. The route requires ``analyst`` in addition to
    ``explain:read``, so owning the account is not sufficient to open it.

    **``DecisionCost`` is not here, and that is a hard line rather than a trim.**
    ``causal_cost.py`` states it at the method that builds the object: every field of
    ``to_audit_dict()`` is an oracle and *together they are complete*. The sign of
    ``expected_saving_from_blocking`` is the decision boundary itself; ``cost_if_blocked`` is
    ``(1-p) * r``, so a reader holding it and the probability recovers the review cost;
    ``assumptions`` prints the cost matrix in plain English. Phase 6 widened the carried gate
    to cover the whole object, so no field of it reaches any response body on this service.

    What a reviewer actually needs from that object is *which features drove this*, not the
    signed distance to the boundary and the direction to move — so the attribution is served
    and the cost arms are not.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    audit_id: int
    transaction_id: str
    decision: Decision
    risk_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="The calibrated probability the decision rested on. Analyst-only: this is "
        "the number POST /score deliberately withholds from the transacting party.",
    )
    top_features: tuple[FeatureContribution, ...]
    model_versions: dict[str, str]
    feature_version: str
    degraded: bool = Field(
        description="Whether a layer was unavailable. A reviewer reading an attribution needs "
        "to know it may be missing a layer's contribution."
    )


class RingMember(BaseModel):
    """One account in a flagged ring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    account_id: str


class RingGraphNode(BaseModel):
    """One node of a flagged ring's topology, for the Phase 8 network view.

    An account node's ``node_id`` is the plain ``account_id`` -- already exposed on
    ``RingResponse.members``, so hashing it here would just be a second encoding of a value this
    same response already carries in the clear. An entity node's ``node_id`` is a truncated
    SHA-256 of the raw composite fingerprint (a ``card1``/``card4``/... tuple, or a device
    fingerprint) -- never the fingerprint itself, which is exactly the kind of identity signal
    no other response on this API returns. See
    :func:`app.models.tier3_graph.export_ring_edges`, which produces the value stored here; this
    schema only carries it across the wire.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    kind: str = Field(description="'account' or 'entity'.")
    entity_type: str | None = Field(
        default=None,
        description="The fingerprint spec name (e.g. 'device_fp'). None for account nodes.",
    )


class RingGraphEdge(BaseModel):
    """One edge of a flagged ring's topology: an account-account or account-entity incidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(description="A RingGraphNode.node_id.")
    target: str = Field(description="A RingGraphNode.node_id.")


class RingResponse(BaseModel):
    """One abuse ring, as exposed to a reviewer.

    Carries membership and the size that drove the flag. It does **not** carry the operating
    threshold or the per-feature weights behind the ring score — the router's own Phase 0
    docstring names that as the line, because a ring endpoint that reports how far a ring sat
    from the threshold tells a ring operator exactly how much to shrink.

    ``nodes``/``edges`` are the Phase 8 addition: the same ring's topology, for the network
    graph view. Empty on a ring trained before Phase 8 -- ``Tier3Model.load`` tolerates an
    artifact with no stored topology rather than requiring every registered model retrained.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ring_id: str
    ring_size: int = Field(ge=0)
    members: tuple[RingMember, ...]
    snapshot_end: datetime | None
    nodes: tuple[RingGraphNode, ...] = ()
    edges: tuple[RingGraphEdge, ...] = ()


class RingListResponse(BaseModel):
    """A page of flagged rings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rings: tuple[RingResponse, ...]
    count: int = Field(ge=0)
    model_version: str


class FeedEvent(BaseModel):
    """One decision, pushed to every subscribed analyst socket as it is made.

    Analyst-only, matching the WS route's own scope gate: this event carries
    ``risk_probability`` and no coarser banding of it, for the same reason
    :class:`ExplanationResponse` does -- ``risk_band`` is kept in this module precisely so the
    frontend can compute it client-side from a probability only an analyst ever receives. No
    field of ``DecisionCost`` appears here either; that gate is not scope-dependent, it does
    not relax for any caller.

    ``amount`` comes from the scoring request itself, not from any persisted row -- ``POST
    /score`` does not write to ``transactions``, so there is nowhere else this event could
    read it from.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["decision"] = "decision"
    audit_id: int
    transaction_id: str
    account_id: str
    decided_at: datetime
    decision: Decision
    risk_probability: float = Field(ge=0.0, le=1.0)
    amount: str
    degraded: bool
    model_version: str


class WsTicketResponse(BaseModel):
    """A short-lived, audience-scoped credential for one ``GET /ws/feed`` connection.

    See :func:`app.core.security.mint_ws_ticket` for why this exists rather than reusing the
    caller's ordinary bearer token in the WebSocket URL.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticket: str
    expires_in: int = Field(description="Seconds until the ticket expires.")
