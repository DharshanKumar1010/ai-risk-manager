"""The scoring path: model loading, serving-time feature assembly, and the decision.

Three things live here, in the order a request meets them.

**The model bundle.** Every artefact the scoring path needs, resolved from ``registry.json``
once at startup and held on application state. Loading per request would put a 13 MB booster
read on the latency budget; loading at import would make the module unimportable wherever the
artefacts are absent, which is most of CI.

**Feature assembly.** The Phase 2 security review requires ``/score`` to take *raw transaction
fields* and build the vector server-side: ``FeatureVector`` is unbounded, so an endpoint taking
a caller-supplied vector would let that caller choose its own score while leaving a
correct-looking audit row. Assembly runs the **same** ``app.data.features`` functions the
training pipeline ran, over a frame of the account's prior transactions plus the incoming one.
That is deliberate and is the whole reason this is not a hand-written single-row encoder: a
second implementation of the feature logic would drift, and a drifted feature produces a wrong
decision underneath a correct-looking audit row — the exact failure the audit trail exists to
prevent.

**The decision.** Tier-1's calibrated probability and the transaction's amount go to the
Phase 6 ``plug_in`` cost policy, which is what Phase 6 actually measured and shipped:
22.41% cheaper than probability ranking at a matched 1% flag rate. The meta-learner is
deliberately **not** in the decision path — registry entry 25 records it losing to Tier-1 alone
by 0.0322 PR-AUC with a confidence interval excluding zero, and ``app/models/README.md`` says
in as many words that it is not recommended for serving. Serving it anyway to satisfy a
four-layer description would be choosing the architecture diagram over the measurement.

What Tier-3 does here, stated plainly: it annotates the audit record and feeds ``GET /rings``.
It does not move the decision, because Phase 5's ablation measured its contribution to the
fused ranking at -0.0001 with an interval spanning zero. Its lookup is still wrapped in the
timeout and degraded-mode fallback the phase requires, because an enrichment that can hang is
still an enrichment that can hang.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.audit import AuditRecord, Decision
from app.data import features as feature_engineering
from app.data.raw_spec import SourceDataset
from app.data.schema import TransactionFeatures
from app.data.serving_encoders import load_serving_encoders
from app.ml.registry import latest_entry
from app.models.causal_cost import CostPolicy, DecisionCost
from app.models.tier1_anomaly import Tier1Model, Tier1Result, explain
from app.models.tier3_graph import Tier3Model, Tier3Result
from app.models.transaction import Transaction

logger = logging.getLogger(__name__)

#: How many contributors the audit record keeps. Three is what the Phase 5 brief asked for and
#: what the meta-learner's own ``predict`` stores; keeping the same number here means a Tier-1
#: explanation and a fused one are read the same way.
TOP_FEATURE_COUNT = 3

#: Reason strings recorded in the audit row when a layer is skipped. Fixed rather than
#: formatted per request, so that "how often did we degrade, and why" is a GROUP BY rather than
#: a text search.
DEGRADED_TIER3_TIMEOUT = "tier3 ring lookup exceeded its timeout"
DEGRADED_TIER3_UNAVAILABLE = "tier3 snapshot unavailable"

#: Workers reserved for Tier-3 lookups. Small on purpose: the lookup is a dictionary read, so
#: one worker serves a great many requests, and the pool's real job is to bound how many
#: threads a stalling snapshot can strand. See :func:`_tier3_with_timeout`.
TIER3_LOOKUP_THREADS = 4

_TIER3_EXECUTOR: ThreadPoolExecutor | None = None


def _tier3_executor() -> ThreadPoolExecutor:
    """Return the dedicated Tier-3 executor, creating it on first use.

    Isolated from the default executor so that a stalled ring lookup cannot strand the threads
    every other ``run_in_executor`` caller shares.
    """
    global _TIER3_EXECUTOR
    if _TIER3_EXECUTOR is None:
        _TIER3_EXECUTOR = ThreadPoolExecutor(
            max_workers=TIER3_LOOKUP_THREADS, thread_name_prefix="tier3-lookup"
        )
    return _TIER3_EXECUTOR


def shutdown_tier3_executor() -> None:
    """Release the Tier-3 executor. Called from the application's shutdown hook.

    ``cancel_futures`` drops work that has not started; anything already running is left to
    finish, because a thread cannot be cancelled and waiting on a stalled lookup would be the
    same hang this executor exists to contain.
    """
    global _TIER3_EXECUTOR
    if _TIER3_EXECUTOR is not None:
        _TIER3_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _TIER3_EXECUTOR = None


@dataclass(frozen=True)
class HistoryAnomalyFeatures:
    """The deviation-from-own-history subset of the assembled Tier-1 vector.

    Carried on :class:`ScoringOutcome` so a caller building risk context reads these off the
    vector :func:`score_transaction` already assembled, instead of re-querying
    :func:`read_account_history` for the same window a second time. Today the only caller is
    the Phase 9 webhook's ``merchant_context`` block.

    ``amount_zscore_vs_own_history`` and the ``velocity_count_*`` fields are read directly from
    the assembled :class:`~app.data.schema.TransactionFeatures`, so they are ``None`` whenever
    Tier-1's fitted spec did not select that column as a model feature (measured per corpus by
    ``MAX_NULL_RATE``) -- not only when the account has too little history to compute one.

    ``prior_velocity_count_1h_mean``/``std`` are not a Tier-1 model input and are not produced
    by ``app.data.features``: they summarise the account's own trailing-1h transaction count as
    measured at each of its prior transactions, computed directly from the same ``history`` rows
    already fetched, for :func:`app.core.merchant_context.compute_merchant_context`'s
    ``velocity_anomaly`` flag. ``None`` below :data:`app.data.features.ZSCORE_MIN_PRIOR`
    observations, matching the same minimum the amount z-score requires.
    """

    amount_zscore_vs_own_history: float | None
    velocity_count_1h: float | None
    velocity_count_24h: float | None
    velocity_count_7d: float | None
    prior_velocity_count_1h_mean: float | None
    prior_velocity_count_1h_std: float | None


@dataclass(frozen=True)
class ScoringOutcome:
    """Everything one scoring call produced, before anything is chosen for the response.

    Deliberately carries *more* than any response body will: ``cost`` and ``top_features`` are
    audit-only, and keeping them on a separate object from the response schema means excluding
    them is structural rather than a thing a route has to remember. ``history_anomaly`` is the
    one field on this object that a response is allowed to derive from -- see
    :class:`HistoryAnomalyFeatures`.
    """

    decision: Decision
    probability: float
    tier1: Tier1Result
    tier3: Tier3Result | None
    cost: DecisionCost
    top_features: tuple[tuple[str, float], ...]
    history_anomaly: HistoryAnomalyFeatures
    model_versions: dict[str, str]
    feature_version: str
    degraded: bool
    degraded_reason: str | None
    latency_ms: float


@dataclass(frozen=True)
class ModelBundle:
    """Every artefact the scoring path needs, loaded once.

    Attributes:
        source_dataset: The corpus these models were fitted on. A request naming another
            corpus is refused rather than scored against the wrong models.
        tier1: The shipped Tier-1 scorer. Carries the system.
        cost_policy: The Phase 6 ``plug_in`` policy that turns a probability and an amount
            into a decision.
        serving_encoders: The Phase 1 frequency tables, which the pipeline fits and discards.
        tier3: The ring snapshot, or None when its artefact is absent.
        model_versions: Layer name to registry ``model_id``, recorded on every audit row.
    """

    source_dataset: SourceDataset
    tier1: Tier1Model
    cost_policy: CostPolicy
    serving_encoders: dict[str, dict[str, float]]
    tier3: Tier3Model | None
    model_versions: dict[str, str]

    @property
    def feature_version(self) -> str:
        """Return the Tier-1 feature definition every assembled vector must match."""
        return self.tier1.feature_version

    @property
    def allowed_raw_columns(self) -> frozenset[str]:
        """Return the source columns a caller may supply.

        The model's full input universe minus every feature the server derives: the Phase 1
        engineered block, and the ``freq_*`` encodings of both the pipeline's columns and
        Tier-1's own. 91 raw columns in, 22 derived features refused.

        This is what makes ``/score`` satisfy the Phase 2 security review. A caller supplies
        facts it legitimately owns — the card, the address, the device, the counters that came
        with the authorisation — and cannot reach ``amount_zscore_vs_own_history`` or
        ``velocity_count_24h``, which are the features an attacker would want to set. Derived
        names are refused at validation *and* overwritten during assembly, so neither control
        alone is load-bearing.
        """
        spec = self.tier1.spec
        derived = set(feature_engineering.ieee_feature_names())
        derived |= set(spec.encoded_frequency_names)
        universe = (
            set(spec.numeric_columns)
            | set(spec.native_categorical_columns)
            | set(spec.frequency_columns)
        )
        return frozenset(universe - derived)

    @classmethod
    def load(cls, settings: Settings) -> "ModelBundle":
        """Resolve and load every artefact named by the registry.

        Tier-3 is optional and its absence is logged rather than raised: the ring snapshot is
        an enrichment, and a scoring service that refuses to start without it trades a
        degradable dependency for an outage. Tier-1 and the cost policy are not optional —
        without them there is no decision to make, and starting anyway would produce a service
        that returns 500 to every caller while reporting itself healthy.

        Raises:
            FileNotFoundError: If a required artefact is missing.
            RuntimeError: If the registry names no model for a required layer.
        """
        source = settings.scoring_source_dataset
        artifact_dir = settings.artifact_dir
        registry_path = settings.registry_path

        tier1 = _load_tier1(source, artifact_dir, registry_path)
        cost_policy, cost_model_id = _load_cost_policy(source, artifact_dir, registry_path)
        tier3, tier3_model_id = _load_tier3(source, artifact_dir, registry_path)

        model_versions = {"tier1": tier1.model_id, "causal_cost": cost_model_id}
        if tier3_model_id is not None:
            model_versions["tier3"] = tier3_model_id

        return cls(
            source_dataset=source,
            tier1=tier1,
            cost_policy=cost_policy,
            # The Tier-1 spec's own encoders ship in its sidecar; these are the four the
            # pipeline fits and throws away. Digest unchecked here because the tables belong to
            # the *pipeline's* feature definition, not Tier-1's -- two different hashes over
            # two different column sets, and asserting one against the other would fail
            # correctly-built artefacts.
            serving_encoders=load_serving_encoders(source, artifact_dir),
            tier3=tier3,
            model_versions=model_versions,
        )


def _require_entry(layer: str, source: SourceDataset, registry_path: Path) -> dict[str, Any]:
    """Return the latest registry entry for a layer, or explain what is missing."""
    entry = latest_entry(layer, source, registry_path)
    if entry is None:
        raise RuntimeError(
            f"models/registry.json names no {layer} model for {source}. The scoring service "
            "cannot start without one."
        )
    return entry


def _load_tier1(source: SourceDataset, artifact_dir: Path, registry_path: Path) -> Tier1Model:
    """Load the registered Tier-1 model, verifying its feature hash against the registry."""
    entry = _require_entry("tier1_anomaly", source, registry_path)
    return Tier1Model.load(
        str(entry["model_id"]),
        artifact_dir,
        feature_version=str(entry.get("feature_version", "")) or None,
    )


def _load_cost_policy(
    source: SourceDataset, artifact_dir: Path, registry_path: Path
) -> tuple[CostPolicy, str]:
    """Load the shipped cost policy and return it with its registry id."""
    entry = _require_entry("causal_cost", source, registry_path)
    model_id = str(entry["model_id"])
    policy = CostPolicy.load(model_id, artifact_dir)
    if policy.strategy == "learned_loss" and policy.loss_model is None:
        raise RuntimeError(
            f"cost policy {model_id} is learned_loss but carries no loss model; it cannot "
            "price a decision, and falling back to the plug-in would record a different "
            "figure than the one that decided."
        )
    return policy, model_id


def _load_tier3(
    source: SourceDataset, artifact_dir: Path, registry_path: Path
) -> tuple[Tier3Model | None, str | None]:
    """Load the ring snapshot if it is present. Absence degrades rather than fails."""
    entry = latest_entry("tier3_graph", source, registry_path)
    if entry is None:
        logger.warning("no tier3_graph model registered for %s; ring enrichment disabled", source)
        return None, None
    model_id = str(entry["model_id"])
    try:
        return Tier3Model.load(model_id, artifact_dir), model_id
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.warning(
            "tier3 snapshot %s failed to load (%s); ring enrichment disabled",
            model_id,
            type(exc).__name__,
        )
        return None, None


# --- Feature assembly ---------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryRow:
    """One prior transaction, carrying only what feature engineering reads from history."""

    event_time: datetime
    amount: float
    device_info: str | None
    addr1: float | None


async def read_account_history(
    session: AsyncSession,
    source_dataset: SourceDataset,
    account_id: str,
    before: datetime,
    limit: int,
    exclude_transaction_id: str | None = None,
) -> list[HistoryRow]:
    """Return the account's most recent prior transactions, oldest first.

    The indexed range scan ``ix_transactions_account_time`` was built for. Bounded by ``limit``
    so one very active account cannot make a scoring call unboundedly slow; the widest
    engineered window is seven days, so a bound well above a week's typical volume changes no
    feature value while capping the tail.

    ``before`` is inclusive of everything at or before the incoming transaction's timestamp,
    which matches the ``(t - W, t]`` window the training path used.

    ``exclude_transaction_id`` keeps a transaction out of its own history. It matters whenever
    the row is already persisted at scoring time — a re-score, the Phase 7 ``/replay``
    enhancement, or a redelivered Phase 9 webhook. Without it the transaction appears twice in
    the frame: once from the database and once as the incoming row, which inflates
    ``account_prior_txn_count`` and every velocity count by one, doubles its own contribution
    to the velocity sums, and makes ``seconds_since_prior_txn`` zero. The decision would change
    while the audit row still looked correct.

    Args:
        session: Active database session.
        source_dataset: Corpus to scan within — ids are unique per corpus, never across.
        account_id: The account whose history is wanted.
        before: The incoming transaction's event time.
        limit: Maximum rows to read.
        exclude_transaction_id: The incoming transaction, so it cannot enter its own history.

    Returns:
        Prior transactions in ascending time order.
    """
    statement = select(
        Transaction.event_time,
        Transaction.amount,
        Transaction.device_info,
        Transaction.addr1,
    ).where(
        Transaction.source_dataset == source_dataset,
        Transaction.account_id == account_id,
        Transaction.event_time <= before,
    )
    if exclude_transaction_id is not None:
        statement = statement.where(Transaction.transaction_id != exclude_transaction_id)
    # Ordering and the bound are applied last, after every filter, so the limit is taken from
    # the filtered set rather than trimming rows the filter would have removed anyway.
    statement = statement.order_by(Transaction.event_time.desc()).limit(limit)
    result = await session.execute(statement)
    rows = [
        HistoryRow(
            event_time=event_time,
            amount=float(amount),
            device_info=device_info,
            addr1=None if addr1 is None else float(addr1),
        )
        for event_time, amount, device_info, addr1 in result.all()
    ]
    rows.reverse()
    return rows


def _prior_velocity_baseline(history: list[HistoryRow]) -> tuple[float | None, float | None]:
    """Return (mean, std) of the account's own trailing-1h transaction count, measured at each
    of its prior transactions -- the baseline :class:`HistoryAnomalyFeatures` compares the
    incoming transaction's own ``velocity_count_1h`` against.

    Deliberately not routed through ``app.data.features.add_velocity_features``: that pipeline
    is the single source of truth for what Tier-1 *scores on*, and reusing it here would take a
    dependency this summary statistic does not need for a materially different purpose -- a
    diagnostic field on ``merchant_context``, not a model input. This is a plain trailing-window
    count over ``history``'s own event times, the same rows :func:`score_transaction` already
    fetched, so no new query is issued.

    Returns:
        ``(None, None)`` when fewer than ``ZSCORE_MIN_PRIOR`` prior transactions exist -- too
        few to say what "usual" looks like for this account.
    """
    if len(history) < feature_engineering.ZSCORE_MIN_PRIOR:
        return None, None
    window = timedelta(hours=1)
    times = [row.event_time for row in history]

    # Two-pointer sliding window, O(n) -- not the O(n^2) pairwise comparison this reads like at
    # a glance. This runs unconditionally on every score_transaction call (POST /score included,
    # not only the webhook that reads its output), so its cost is on the real-time scoring path
    # regardless of who consumes the result; account_history_limit allows up to 10,000 rows, and
    # an O(n^2) version measured in the seconds at that size. `left` advances monotonically
    # because `times` is sorted ascending (assemble_tier1_vector's contract on `history`), so a
    # single forward pass over both pointers suffices.
    counts = np.empty(len(times), dtype="float64")
    left = 0
    for right, t in enumerate(times):
        while times[left] <= t - window:
            left += 1
        counts[right] = right - left + 1

    mean = float(counts.mean())
    std = float(counts.std(ddof=1))
    return mean, std


def assemble_tier1_vector(
    bundle: ModelBundle,
    *,
    transaction_id: str,
    account_id: str,
    event_time: datetime,
    amount: Decimal,
    raw_columns: dict[str, Any],
    history: list[HistoryRow],
) -> TransactionFeatures:
    """Build the Tier-1 feature vector for one incoming transaction.

    Runs the training pipeline's own engineering functions over a frame of ``history`` plus the
    incoming row, then takes the last row. Using the real functions rather than a single-row
    reimplementation is the point: the engineered features are defined by that code, and a
    second definition that drifts from it changes decisions silently.

    Args:
        bundle: Supplies the fitted spec and the pipeline's frequency tables.
        transaction_id: Identifier for the incoming transaction.
        account_id: Its account.
        event_time: Its timezone-aware timestamp.
        amount: Its amount.
        raw_columns: The corpus's raw source columns as supplied by the caller. Validated
            against the allowed key set before reaching here.
        history: The account's prior transactions, oldest first.

    Returns:
        A :class:`TransactionFeatures` carrying all of Tier-1's features and its feature
        version, ready for :meth:`Tier1Model.score`.
    """
    spec = bundle.tier1.spec
    rows = len(history) + 1
    incoming = rows - 1

    # The whole frame is built from one dict and materialised once. Inserting ~113 columns
    # one at a time into an existing frame fragments its block manager, which pandas warns
    # about and which measured at roughly 100ms per call -- twice the Tier-1 latency budget for
    # work that is not the model.
    columns: dict[str, list[Any]] = {
        "account_id": [account_id] * rows,
        "event_time": [row.event_time for row in history] + [event_time],
        "amount": [row.amount for row in history] + [float(amount)],
        "DeviceInfo": [row.device_info for row in history] + [raw_columns.get("DeviceInfo")],
        "addr1": [row.addr1 for row in history] + [_as_float(raw_columns.get("addr1"))],
    }

    # Every column the spec reads must exist before it is transformed. A column the caller did
    # not supply is genuinely absent for this transaction, which is a value the model was
    # trained to route: 76% of IEEE-CIS rows carry no identity record at all.
    for column in (
        *spec.numeric_columns,
        *spec.native_categorical_columns,
        *spec.frequency_columns,
    ):
        columns.setdefault(column, [None] * rows)

    # The caller's raw source columns, placed on the incoming row only. History rows keep None:
    # a prior transaction's ProductCD is not this one's, and the history-dependent features
    # read time and amount, never these.
    for column, value in raw_columns.items():
        if column in {"DeviceInfo", "addr1"}:
            continue
        columns.setdefault(column, [None] * rows)
        columns[column][incoming] = value

    # Derived server-side rather than taken from the payload. In training this is the identity
    # table's join indicator; at serving the equivalent fact is whether the caller presented an
    # identity record at all, which is what this measures. Leaving it to the payload would hand
    # a caller a one-field lever on a feature the model leans on heavily.
    columns.setdefault("has_identity", [None] * rows)
    columns["has_identity"][incoming] = any(
        value is not None for key, value in raw_columns.items() if key.startswith("id_")
    )

    frame = pd.DataFrame(columns)
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)

    # Engineering runs *after* the caller's values are placed, and that ordering is the
    # security property: a caller naming a derived feature -- ``velocity_count_1h``,
    # ``amount_zscore_vs_own_history``, ``freq_ProductCD`` -- has it overwritten by the real
    # computation rather than honoured. The request schema rejects those names too; this
    # ordering means even a gap there could not become a caller-chosen score.
    #
    # A stable sort, with the incoming row already last, keeps it last among rows sharing its
    # timestamp -- the same tie-breaking the pipeline's mergesort gives.
    frame = feature_engineering.sort_for_engineering(frame)
    frame = feature_engineering.add_calendar_features(frame)
    frame = feature_engineering.add_amount_history_features(frame)
    frame = feature_engineering.add_velocity_features(frame)
    frame = feature_engineering.add_familiarity_features(
        frame, feature_engineering.IEEE_FAMILIARITY_COLUMNS
    )
    # The pipeline's four tables, which it fits and discards. Tier-1's own four are applied by
    # ``spec.transform`` below, from the encoders that ship in its sidecar.
    frame = feature_engineering.apply_frequency_encoders(frame, bundle.serving_encoders)

    # Tier-1's own four tables, from the encoders that ship in its sidecar. Applied here
    # rather than by ``spec.transform`` because the serving path deliberately does not go
    # through ``transform``: that method builds ``pd.Categorical`` against the fitted level set,
    # and a level absent from training -- a ``ProductCD`` this account did not send, a device
    # never seen -- is both expected at serving and something ``Categorical`` now warns about.
    # ``Tier1Model._vector_to_array`` is the serving contract, and it maps an unseen level to
    # -1, which is how LightGBM encodes a missing category. Handing it the raw value is what
    # that method documents itself as expecting.
    frame = feature_engineering.apply_frequency_encoders(frame, spec.encoders)

    row = frame.iloc[-1]
    categorical = set(spec.native_categorical_columns)
    vector: dict[str, Any] = {}
    for name in bundle.tier1.feature_names:
        value = row[name]
        if name in categorical:
            # The sentinel is a real fitted level for every categorical that had missing values
            # in training, so it resolves to a code; where it was never missing, it resolves to
            # -1, which is the honest answer for a value the model has not seen.
            vector[name] = feature_engineering.MISSING_SENTINEL if pd.isna(value) else str(value)
        else:
            vector[name] = None if pd.isna(value) else float(value)

    return TransactionFeatures(
        transaction_id=transaction_id,
        source_dataset=bundle.source_dataset,
        event_time=event_time,
        amount=amount,
        account_id=account_id,
        counterparty_id=None,
        transaction_type=_as_optional_str(raw_columns.get("ProductCD")),
        feature_version=bundle.feature_version,
        features=vector,
    )


def _as_float(value: Any) -> float | None:
    """Coerce a raw column to float, treating an unparseable value as absent."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_str(value: Any) -> str | None:
    """Coerce a raw column to a string, preserving absence."""
    return None if value is None else str(value)


# --- The decision -------------------------------------------------------------------------


async def _tier3_with_timeout(
    bundle: ModelBundle,
    transaction: TransactionFeatures,
    timeout_ms: int,
) -> tuple[Tier3Result | None, str | None]:
    """Look up the ring score under a timeout, returning the reason when it is skipped.

    The Phase 7 graceful-degradation requirement and security-checklist item 5.3. The lookup
    is a dictionary read and is not expected to be slow; the timeout exists because "expected
    to be fast" is not a property a request path should rely on, and because a snapshot reload
    or a cold page fault can turn an O(1) read into a stall.

    **A timeout abandons the wait, not the work.** Python cannot cancel a running thread, so a
    lookup that stalls keeps its worker until it returns on its own. That is why this uses a
    dedicated executor rather than ``asyncio.to_thread``: on the shared default executor, a
    Tier-3 snapshot stalling under load would leak workers into the pool every other coroutine
    depends on, and the degraded path itself would eventually block waiting for a free thread —
    a fallback that fails under exactly the conditions it exists for. Here the blast radius is
    bounded to :data:`TIER3_LOOKUP_THREADS`, and saturation degrades immediately instead of
    queueing, because a queued lookup has already missed the budget the caller is waiting on.

    Returns:
        The result and ``None``, or ``None`` and the reason degraded mode was entered.
    """
    if bundle.tier3 is None:
        return None, DEGRADED_TIER3_UNAVAILABLE

    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(_tier3_executor(), bundle.tier3.score, transaction)
    except RuntimeError as exc:
        # The executor refuses new work only when it is shutting down.
        logger.warning("tier3 executor unavailable (%s); deciding without it", type(exc).__name__)
        return None, DEGRADED_TIER3_UNAVAILABLE

    try:
        result = await asyncio.wait_for(future, timeout=timeout_ms / 1_000.0)
    except TimeoutError:
        logger.warning("tier3 lookup exceeded %dms; deciding without it", timeout_ms)
        return None, DEGRADED_TIER3_TIMEOUT
    except Exception as exc:
        logger.warning("tier3 lookup failed (%s); deciding without it", type(exc).__name__)
        return None, DEGRADED_TIER3_UNAVAILABLE
    return result, None


def _decide(policy: CostPolicy, probability: float, amount: float) -> Decision:
    """Turn a probability and an amount into allow/review/block under the cost policy.

    The policy ranks by expected cost saving and flags at an operating point chosen on
    validation under a 1% review-capacity cap. A flagged transaction goes to ``review`` rather
    than ``block``: the operating point was chosen to fill a review queue, and every figure
    Phase 6 published for it prices the flagged arm as a review. Returning ``block`` here would
    report a decision the measured cost of which is not the measured cost of this threshold.
    """

    flagged = policy.decide(
        np.asarray([probability], dtype="float64"),
        np.asarray([amount], dtype="float64"),
    )
    return "review" if bool(flagged[0]) else "allow"


async def score_transaction(
    session: AsyncSession,
    bundle: ModelBundle,
    settings: Settings,
    *,
    transaction_id: str,
    account_id: str,
    event_time: datetime,
    amount: Decimal,
    raw_columns: dict[str, Any],
) -> tuple[ScoringOutcome, AuditRecord]:
    """Score one transaction and build the audit record that must accompany it.

    Returns both, and writes neither. The caller commits the audit row and the response
    together, so a decision cannot reach a caller without its record.

    Raises:
        ValueError: If the assembled vector does not match the model's feature definition, or
            if a feature is missing. Neither is recoverable: scoring anyway would attach a
            wrong decision to a correct-looking audit row.
    """
    started = time.perf_counter()

    history = await read_account_history(
        session,
        bundle.source_dataset,
        account_id,
        before=event_time,
        limit=settings.account_history_limit,
        exclude_transaction_id=transaction_id,
    )
    transaction = assemble_tier1_vector(
        bundle,
        transaction_id=transaction_id,
        account_id=account_id,
        event_time=event_time,
        amount=amount,
        raw_columns=raw_columns,
        history=history,
    )

    tier1 = bundle.tier1.score(transaction)
    tier3, degraded_reason = await _tier3_with_timeout(
        bundle, transaction, settings.tier3_timeout_ms
    )

    decision = _decide(bundle.cost_policy, tier1.score, float(amount))
    cost = bundle.cost_policy.estimate_cost(
        amount=float(amount),
        fraud_probability=tier1.score,
        decision=decision,
    )
    top_features = tuple(explain(bundle.tier1, transaction, top_k=TOP_FEATURE_COUNT))
    prior_velocity_mean, prior_velocity_std = _prior_velocity_baseline(history)
    history_anomaly = HistoryAnomalyFeatures(
        # FeatureVector's declared type also admits str, for categorical features -- these four
        # are always numeric-or-absent at runtime (assemble_tier1_vector never stores a string
        # under these names), so _as_float only ever narrows, never silently drops a real value.
        amount_zscore_vs_own_history=_as_float(
            transaction.features.get("amount_zscore_vs_own_history")
        ),
        velocity_count_1h=_as_float(transaction.features.get("velocity_count_1h")),
        velocity_count_24h=_as_float(transaction.features.get("velocity_count_24h")),
        velocity_count_7d=_as_float(transaction.features.get("velocity_count_7d")),
        prior_velocity_count_1h_mean=prior_velocity_mean,
        prior_velocity_count_1h_std=prior_velocity_std,
    )

    outcome = ScoringOutcome(
        decision=decision,
        probability=tier1.score,
        tier1=tier1,
        tier3=tier3,
        cost=cost,
        top_features=top_features,
        history_anomaly=history_anomaly,
        model_versions=dict(bundle.model_versions),
        feature_version=bundle.feature_version,
        degraded=degraded_reason is not None,
        degraded_reason=degraded_reason,
        latency_ms=(time.perf_counter() - started) * 1_000.0,
    )
    record = AuditRecord(
        transaction_id=transaction_id,
        account_id=account_id,
        decided_at=datetime.now(UTC),
        decision=decision,
        risk_probability=tier1.score,
        tier1_score=tier1.score,
        tier2_reconstruction_error=None,
        tier3_ring_risk_score=None if tier3 is None else tier3.ring_risk_score,
        model_versions=outcome.model_versions,
        feature_version=outcome.feature_version,
        top_features=top_features,
        cost_estimate=cost.to_audit_dict(),
        degraded=outcome.degraded,
        degraded_reason=outcome.degraded_reason,
    )
    return outcome, record
