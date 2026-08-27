"""Request and response schemas for the HTTP surface.

Every model here is ``extra="forbid"``, with one named exception (``RazorpayPaymentEntity`` /
``RazorpayWebhookEnvelope``, see below). That is security-checklist item 4.1, and on this
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

**One deliberate exception, added in Phase 9.** ``RazorpayWebhookResponse`` carries
``risk_score``, ``cost_estimate`` (every field of ``DecisionCost``) and ``merchant_context``,
which together are exactly the disclosure this docstring spent three paragraphs closing. The
reason it is safe here and nowhere else: every other response on this service is read by the
account it describes, authenticated by that account's own JWT -- the probing party and the
authorized party are the same caller, which is what made the gate necessary in the first place.
``POST /webhooks/razorpay/transaction`` has no JWT at all; its only caller is whoever holds
``settings.razorpay_webhook_secret`` (Razorpay, in a real deployment), verified over the raw
request body before anything is parsed -- see ``app/core/webhook_security.py``. A merchant has
no path to *authenticate as* this route's caller, so a merchant cannot mint its own signed
request and read an arbitrary response. ``top_features`` stays excluded even here: only the
three fields above were reasoned about and authorized, and attribution is a different, worse
oracle (see the ``top_features`` paragraph above).

**This exception's precondition depends on a second fix, found in the same security review.**
The account whose ``merchant_context`` gets disclosed is read from
``notes["riskiq_account_id"]`` -- signed by Razorpay's channel, but not a claim Razorpay itself
verifies, so a caller who can shape a payment's ``notes`` (this project's own checkout
integration, or a customer of it if that integration is not careful) could otherwise name an
*arbitrary* account and have this route disclose that account's data, independent of who holds
the webhook secret. ``app/api/webhooks.py``'s ``_require_known_account`` narrows this to
accounts already present in ``accounts`` -- see that function's docstring for exactly what it
does and does not close. Full rationale recorded in ``BUILD_LOG.md``'s Phase 9 entry.

**A second, request-side exception, also from Phase 9.** ``RazorpayPaymentEntity`` and
``RazorpayWebhookEnvelope`` -- the schemas ``app/api/webhooks.py`` validates the raw webhook
body against -- declare ``extra="ignore"``, not ``"forbid"``, which on every other schema in
this file is security-checklist item 4.1 and is not optional. The reasoning above for item 4.1
does not transfer here: it exists to catch a typo in a field name *this project defines*, so a
caller's mistake fails loudly instead of scoring against a vector silently missing the feature
they meant to send. Razorpay's real payment entity carries on the order of 25 fields --
``status``, ``order_id``, ``fee``, ``tax``, ``acquirer_data``, ``error_code``, and more -- of
which this service reads seven. The shape is Razorpay's, not this project's to constrain, and
``extra="forbid"`` on it would 422 on essentially every real webhook Razorpay sends, which is a
functional failure, not a caught mistake. This was raised directly against security-checklist
item 4.1 during Phase 9's review and kept as a named exception rather than silently deviating
from it -- see ``BUILD_LOG.md``'s Phase 9 entry for the record of that decision.
"""

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

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


# --- Razorpay webhook (Phase 9) -------------------------------------------------------------
#
# See this module's docstring for the disclosure exception these schemas carry.


class RazorpayPaymentEntity(BaseModel):
    """The subset of Razorpay's payment entity this service reads.

    ``extra="ignore"``, not ``"forbid"`` -- unlike every request schema elsewhere in this
    file, this shape is Razorpay's, not ours to constrain. A field this service does not read
    is not a caller-chosen input reaching a model; it is simply not read. Validated only after
    :func:`app.core.webhook_security.verify_razorpay_signature` has already checked the raw
    body, so a malformed payload cannot be crafted to probe parsing behaviour without first
    forging a valid signature.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1, max_length=128)
    amount: int = Field(
        ge=0,
        le=int(MAX_TRANSACTION_AMOUNT) * 100,
        description="Smallest " "currency unit (paise for INR).",
    )
    currency: str = Field(max_length=8)
    method: str | None = Field(default=None, max_length=MAX_RAW_VALUE_LENGTH)
    card: dict[str, str] | None = Field(
        default=None,
        description="Only the two keys _extract_raw_columns reads (network, type) matter; "
        "typed dict[str, str] with a value-length bound (not dict[str, Any]) so a caller cannot "
        "reach the scorer with an arbitrarily large or nested object under this key.",
    )
    notes: dict[str, str] = Field(
        default_factory=dict,
        description="Razorpay's arbitrary merchant-supplied metadata. This project's Phase 9 "
        "integration convention reads the RiskIQ account this transaction's history belongs "
        "to from notes['riskiq_account_id'] -- Razorpay has no field that means this natively, "
        "so a merchant's checkout integration must stamp it here at order/payment creation.",
    )

    @field_validator("card", "notes")
    @classmethod
    def bound_string_map_values(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        """Bound key/value length on the two caller-shaped string maps this schema accepts,
        mirroring ScoreRequest.bound_raw_values -- a webhook payload is caller-influenced data
        reaching this service same as a /score request body is, even though Razorpay's channel
        signs it."""
        if value is None:
            return value
        for key, entry in value.items():
            if len(key) > MAX_RAW_VALUE_LENGTH or len(entry) > MAX_RAW_VALUE_LENGTH:
                raise ValueError("card/notes entries must be shorter than 256 characters")
        return value

    created_at: int = Field(
        ge=0,
        le=4_102_444_800,  # 2100-01-01T00:00:00Z
        description="Unix epoch seconds. Bounded, like `amount` above, so an absurd or "
        "corrupted value fails validation here rather than raising OverflowError out of "
        "datetime.fromtimestamp in app/api/webhooks.py -- that call is not itself wrapped in "
        "the try/except around the two model_validate* calls, so an unbounded value would "
        "reach it as an unhandled 500 instead of the intended 422.",
    )


class RazorpayWebhookEnvelope(BaseModel):
    """The outer shape Razorpay posts. ``payload`` is read narrowly by ``app/api/webhooks.py``,
    not modelled in full -- Razorpay's envelope carries more than this service uses."""

    model_config = ConfigDict(extra="ignore")

    event: Literal["payment.authorized", "payment.captured", "payment.failed"]
    payload: dict[str, Any]


class DecisionCostBlock(BaseModel):
    """``DecisionCost`` mirrored in full for the response body -- the one deliberate exception
    to this file's disclosure gate. See the module docstring and ``BUILD_LOG.md``'s Phase 9
    entry for why it is safe on this one route."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Decision
    expected_cost: float
    cost_if_blocked: float
    cost_if_allowed: float
    expected_saving_from_blocking: float
    fraud_probability: float = Field(ge=0.0, le=1.0)
    amount: float
    assumptions: list[str]


class MerchantRiskContext(BaseModel):
    """How this transaction's decision fits this merchant's own history.

    See ``BUILD_LOG.md``'s Phase 9 entry, "what merchant_context does not measure", for the
    caveats that apply to this block as a whole -- none of these figures are validated against
    a held-out split, and ``fraud_rate_last_100``/``baseline_fraud_rate`` are drawn from
    differently-scoped, non-commensurable populations despite sitting in the same object.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    decisions_considered: int = Field(
        ge=0,
        le=100,
        description="How many of this account's audit_log rows fraud_rate_last_100 was "
        "computed over. Below 100 for a merchant with fewer than 100 recorded decisions. "
        "merchant_context is read after the current decision's audit row commits, so that "
        "decision is included in this count and in fraud_rate_last_100 -- at decisions_"
        "considered == 1, a single blocked transaction reads as fraud_rate_last_100 == 1.0.",
    )
    fraud_rate_last_100: float = Field(
        ge=0.0,
        le=1.0,
        description="NOT a verified fraud rate -- see fraud_rate_basis, carried alongside it in "
        "this same object rather than left to documentation a JSON consumer will not read. "
        "audit_log carries no ground-truth label, only the decision this service made: this is "
        "the fraction of decisions_considered whose decision was 'review' or 'block'. Unstable "
        "at low decisions_considered (see that field), and inflated by an undeduplicated "
        "redelivered webhook event, which writes a second audit_log row for the same payment.",
    )
    fraud_rate_basis: str = Field(
        description="Fixed value 'decision_proxy_no_ground_truth_label', identifying "
        "fraud_rate_last_100 as a decision-proxy in the response body itself -- not only in "
        "this schema's documentation, which a caller reading raw JSON never sees.",
    )
    baseline_fraud_rate: float | None = Field(
        default=None,
        description="accounts.fraud_count / accounts.transaction_count for this account_id -- "
        "a lifetime, offline aggregate over train+val+test combined, built once by the Phase 1 "
        "pipeline. Not point-in-time-safe: for a demo account, this can include transactions "
        "chronologically later than the one currently being scored, so a live decision may be "
        "compared against a baseline that has, in effect, seen its own future. Not a model "
        "input -- this is a display-only figure, read by no training or scoring path -- but the "
        "comparison it invites against fraud_rate_last_100 should be read with that in mind. "
        "None when this account_id was never seen by that pipeline, the expected case for a "
        "real Razorpay merchant. Also scoped to accounts.source_dataset, unlike "
        "fraud_rate_last_100 (audit_log carries no source_dataset column) -- the two figures "
        "are not necessarily drawn from the same population even where both are present.",
    )
    baseline_transaction_count: int | None = Field(
        default=None,
        description="accounts.transaction_count backing baseline_fraud_rate. None with it.",
    )
    amount_zscore_vs_own_history: float | None = Field(
        default=None,
        description="From HistoryAnomalyFeatures. None when the account has fewer than "
        "ZSCORE_MIN_PRIOR prior transactions, or when Tier-1's fitted spec did not select this "
        "column as a model feature for the serving corpus. Unlike velocity_zscore_1h below, "
        "this one *is* a Tier-1 model input -- it contributed to risk_score.",
    )
    amount_anomaly: bool = Field(
        description="abs(amount_zscore_vs_own_history) > 2. A fixed, untuned threshold -- not "
        "evaluated for precision or recall against any held-out split.",
    )
    velocity_zscore_1h: float | None = Field(
        default=None,
        description="NOT a Tier-1 model input, unlike amount_zscore_vs_own_history above -- a "
        "diagnostic z-score, computed only for this response, of this transaction's trailing-1h "
        "count against the same count measured at each of the account's own prior transactions. "
        "Did not contribute to risk_score. None below ZSCORE_MIN_PRIOR prior transactions.",
    )
    velocity_anomaly: bool = Field(
        description="abs(velocity_zscore_1h) > 2. Same untuned-threshold caveat as "
        "amount_anomaly; unlike amount_anomaly, also not a signal the model saw.",
    )
    decision_rationale: str = Field(
        description="A short, deterministic sentence built from decision plus which flags "
        "fired -- not free text from a model, and not a SHAP-style attribution (that is "
        "top_features, withheld from every response including this one). Names which flags "
        "fired alongside the decision; does not assert that either one caused it, and "
        "velocity_anomaly in particular was not a signal risk_score was computed from.",
    )


class RazorpayWebhookResponse(BaseModel):
    """The response to a verified Razorpay webhook call.

    See this module's docstring for why ``risk_score``, ``cost_estimate`` and
    ``merchant_context`` are authorized here and nowhere else on this service. There is no
    ``confidence`` field: nothing this system produces measures calibration or ensemble
    agreement distinctly from ``risk_score`` itself, and BUILD_LOG.md records that one was
    considered and dropped rather than invented.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: str
    account_id: str
    decision: Decision
    audit_id: int
    degraded: bool
    decided_at: datetime
    model_version: str
    risk_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Calibrated fraud probability -- deliberately returned here, unlike "
        "ScoreResponse. See this module's docstring.",
    )
    cost_estimate: DecisionCostBlock
    merchant_context: MerchantRiskContext
    timestamp: datetime
