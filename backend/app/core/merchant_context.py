"""How one decision fits this merchant's own history -- Phase 9's ``merchant_context``.

Two, and only two, data sources feed this, and neither adds a query beyond what
``score_transaction`` and ``POST /webhooks/razorpay/transaction`` already run:

``audit_log``
    This account's most recent recorded decisions -- the only ground this service has to stand
    on for "how has this merchant been trending", since ``audit_log`` carries no verified fraud
    label, only the ``allow``/``review``/``block`` decision that was made, and (unlike
    ``accounts``) no ``source_dataset`` column to scope the read by corpus. See
    :class:`app.api.schemas.MerchantRiskContext` for why ``fraud_rate_last_100`` is named for
    what it is not, not just what it is -- and for ``fraud_rate_basis``, which carries that same
    caveat in the response payload itself, not only in this schema's documentation.

``accounts``
    A lifetime, offline aggregate built once by the Phase 1 pipeline, over train+val+test
    combined. Not point-in-time-safe and not live-updating -- it is the only "baseline" this
    project has, and it is ``None`` for any account the pipeline never saw, which is the
    expected case for a real Razorpay merchant.

The amount/velocity anomaly flags are **not** a third query: they are read off
:class:`app.core.serving.HistoryAnomalyFeatures`, which ``score_transaction`` already computed
from the same account-history read it performs for every scoring call. Recomputing them here
against ``audit_log`` (which carries no amount or velocity columns at all) would be both a
second implementation of the same measurement and, for the reasons in
``BUILD_LOG.md``'s Phase 9 entry, impossible: live-webhook-scored transactions are not persisted
to ``transactions``, so ``audit_log`` is not a source these two features could be recomputed
from even if it carried the columns.
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.serving import HistoryAnomalyFeatures
from app.data.raw_spec import SourceDataset
from app.models.account import Account
from app.models.audit_log import AuditLog

#: How many of the account's most recent decisions fraud_rate_last_100 is computed over.
#: Matches the field name and BUILD_LOG.md's Phase 9 entry; not configurable, since a caller
#: choosing this width could otherwise smooth or sharpen the rate to taste.
MAX_DECISION_HISTORY = 100

#: |z| beyond this is flagged. Matches app.data.features.ZSCORE_MIN_PRIOR's "at least 2 prior
#: observations" reasoning in spirit: a two-sigma threshold is a common, unremarkable default,
#: not a value tuned against this project's data. Not evaluated for precision/recall against
#: any held-out split -- see BUILD_LOG.md's Phase 9 entry, "what merchant_context does not
#: measure".
ANOMALY_ZSCORE_THRESHOLD = 2.0

#: What fraud_rate_last_100 actually is, carried in the response payload itself -- not only in
#: the OpenAPI schema description, which a caller reading raw JSON never sees. ml-evaluator
#: review of this phase found that a caveat living solely in documentation, next to a real
#: label-derived figure (baseline_fraud_rate) with a plausible-looking value in the same
#: numeric neighbourhood, is not a caveat a JSON consumer will encounter. This constant is
#: that caveat, on the wire.
FRAUD_RATE_BASIS = "decision_proxy_no_ground_truth_label"


@dataclass(frozen=True)
class MerchantContext:
    """Mirrors :class:`app.api.schemas.MerchantRiskContext` field for field. See that class's
    field docstrings for what each figure does and does not measure."""

    decisions_considered: int
    fraud_rate_last_100: float
    fraud_rate_basis: str
    baseline_fraud_rate: float | None
    baseline_transaction_count: int | None
    amount_zscore_vs_own_history: float | None
    amount_anomaly: bool
    velocity_zscore_1h: float | None
    velocity_anomaly: bool
    decision_rationale: str


def _velocity_zscore(history: HistoryAnomalyFeatures) -> float | None:
    """Return the incoming transaction's velocity_count_1h as a z-score against the account's
    own prior distribution, or None when either side of that comparison is unavailable."""
    if (
        history.velocity_count_1h is None
        or history.prior_velocity_count_1h_mean is None
        or history.prior_velocity_count_1h_std is None
        or history.prior_velocity_count_1h_std == 0
    ):
        return None
    return (
        history.velocity_count_1h - history.prior_velocity_count_1h_mean
    ) / history.prior_velocity_count_1h_std


def _rationale(decision: str, amount_anomaly: bool, velocity_anomaly: bool) -> str:
    """Build the one-sentence, deterministic explanation the response carries.

    Deliberately not free text from a model: every sentence this can produce is one of a fixed,
    enumerable set, chosen from the same booleans the response already discloses -- it adds
    nothing an attacker could not already read off amount_anomaly/velocity_anomaly directly.
    """
    if amount_anomaly and velocity_anomaly:
        reason = "Amount and transaction velocity both unusual for this merchant's history."
    elif amount_anomaly:
        reason = "Amount unusual for this merchant's history."
    elif velocity_anomaly:
        reason = "Transaction velocity unusual for this merchant's history."
    else:
        reason = "Transaction within merchant's normal envelope."
    if decision == "block":
        return f"{reason} Blocked on the combined risk signal."
    if decision == "review":
        return f"{reason} Flagged for review on the combined risk signal."
    return f"{reason} Low risk."


async def compute_merchant_context(
    session: AsyncSession,
    source_dataset: SourceDataset,
    account_id: str,
    history: HistoryAnomalyFeatures,
    decision: str,
) -> MerchantContext:
    """Build the merchant_context block for one webhook response.

    Args:
        session: Read session, scoped for row-level security -- see
            ``app/api/webhooks.py``'s ``webhook_read_session``.
        source_dataset: Which corpus's ``accounts`` row to read the baseline from.
        account_id: The merchant (== RiskIQ account) this transaction belongs to.
        history: The current transaction's own history-derived features, already computed by
            :func:`app.core.serving.score_transaction` -- not re-derived here.
        decision: The decision just made, for the rationale sentence.

    Returns:
        The context block. Every field degrades to ``None``/``0.0`` rather than raising when
        its data is unavailable -- a new merchant with no recorded history is the expected
        case, not an error.
    """
    rows = (
        (
            await session.execute(
                select(AuditLog.decision)
                .where(AuditLog.account_id == account_id)
                .order_by(AuditLog.decided_at.desc())
                .limit(MAX_DECISION_HISTORY)
            )
        )
        .scalars()
        .all()
    )
    decisions_considered = len(rows)
    flagged = sum(1 for row_decision in rows if row_decision in ("review", "block"))
    fraud_rate_last_100 = flagged / decisions_considered if decisions_considered else 0.0

    account = await session.get(Account, (source_dataset, account_id))
    baseline_fraud_rate = (
        account.fraud_count / account.transaction_count
        if account is not None and account.transaction_count
        else None
    )
    baseline_transaction_count = account.transaction_count if account is not None else None

    amount_z = history.amount_zscore_vs_own_history
    amount_anomaly = amount_z is not None and abs(amount_z) > ANOMALY_ZSCORE_THRESHOLD
    velocity_z = _velocity_zscore(history)
    velocity_anomaly = velocity_z is not None and abs(velocity_z) > ANOMALY_ZSCORE_THRESHOLD

    return MerchantContext(
        decisions_considered=decisions_considered,
        fraud_rate_last_100=fraud_rate_last_100,
        fraud_rate_basis=FRAUD_RATE_BASIS,
        baseline_fraud_rate=baseline_fraud_rate,
        baseline_transaction_count=baseline_transaction_count,
        amount_zscore_vs_own_history=amount_z,
        amount_anomaly=amount_anomaly,
        velocity_zscore_1h=velocity_z,
        velocity_anomaly=velocity_anomaly,
        decision_rationale=_rationale(decision, amount_anomaly, velocity_anomaly),
    )
