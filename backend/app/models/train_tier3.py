"""Phase 4 driver: build both Tier-3 graphs, select on validation, measure once on test.

Run as ``python -m app.models.train_tier3``. Produces ``notebooks/tier3_report.md``, the two
ring visualisations the phase gates on, and one ``models/registry.json`` entry per corpus.

--------------------------------------------------------------------------------------

**What this phase has to prove, and the trap it has to avoid.** Tier-3's premise is that
collusion is visible in topology and invisible to per-row and per-account models. On PaySim
that premise runs into a measured problem: the transfer-to-cash-out pairing rule the graph
needs in order to have any multi-hop structure at all is, by itself, a 99.50%-against-0.23%
fraud classifier. It is the simulator's generative rule read back out, the same species as
Tier-1's PaySim PR-AUC of 0.9998 on ``amount == oldbalanceOrg``. Laying a ring metric over it
and reporting the result would measure the edge rule and call it a graph.

So this driver measures four things rather than one, and the first is the sharpest:

1. **Conditional separation.** Among transfers where the pairing rule *already fired*, can
   topology separate a real chain from an amount coincidence? Both classes are present there
   by construction, so a leak-suspicious number is not reachable, and it is the question a
   production system actually faces.
2. **Increment over the baseline.** The pairing rule is registered as an explicit baseline
   with its own precision and recall, and the graph's claim is the paired-bootstrap delta on
   top of it. A baseline that wins is the finding, not a disappointment to be buried.
3. **Ring-level classification.** Each detected candidate ring is fraud-bearing or clean, so
   TN/FP/FN/TP are all defined and the ml-evaluation-standards reporting block applies
   unchanged. Reported with its own ring-level base rate. Two robustness views accompany it:
   recovery against a surrogate ground-truth partition, and enrichment against a random-ring
   null that depends on no surrogate at all.
4. **IEEE-CIS incremental lift.** Held-out PR-AUC of Tier-1 alone against Tier-1 plus
   ``ring_risk_score``. This is the number Phase 5 consumes and the pitch quotes, because it
   is the only one of the four measured on a corpus with no simulator artefact in it.

**Ring-level ground truth does not exist and is not pretended into existence.** PaySim ships
``isFraud`` per transaction and no ring or agent identifier. The surrogate partition used in
(3) is built from the labels — connected components of the fraud-only induced graph — and is
labelled a surrogate everywhere it appears. Where the enrichment view and the surrogate view
disagree, the surrogate is driving the result and the headline is withdrawn.

**Time discipline.** Snapshots advance on a fixed cadence over a trailing window. A
transaction is scored against the most recent snapshot ending at or before its own
``event_time``; the snapshot's own edges never include it. Everything fitted -- the scorer,
the operating threshold, the step window, the entity cap -- is fitted on train snapshots and
selected on validation. Phase 2 recorded what happens when a phase brief is read literally
enough to select on test; this driver selects on validation and touches test once.
"""

import argparse
import json
import logging
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import pandas as pd  # noqa: E402

from app.config import Settings  # noqa: E402
from app.data.raw_spec import SourceDataset  # noqa: E402
from app.data.schema import SPLIT_ORDER, Split  # noqa: E402
from app.ml.cost import (  # noqa: E402
    CostEstimate,
    CostModel,
    cost_at_threshold,
    render_sensitivity,
    review_cost_sweep,
    sensitivity_sweep,
    threshold_for_flag_rate,
)
from app.ml.evaluation import (  # noqa: E402
    LEAK_SUSPICION_PR_AUC,
    ConfusionMatrix,
    EvaluationResult,
    bootstrap_pr_auc,
    bootstrap_pr_auc_delta,
    confusion_at_threshold,
    evaluate,
    pr_auc,
)
from app.ml.registry import (  # noqa: E402
    RegistryEntry,
    append_entry,
    artifact_path,
    build_model_id,
    read_registry,
)
from app.models.tier3_edges import (  # noqa: E402
    CASH_OUT,
    IEEE_FINGERPRINTS,
    NON_CIRCULAR_FINGERPRINTS,
    TRANSFER,
    FingerprintSpec,
    build_chain_edges,
)
from app.models.tier3_graph import (  # noqa: E402
    MIN_RING_SIZE,
    RANDOM_SEED,
    EntityGraph,
    GraphSnapshot,
    IEEECISSharedEntityGraph,
    PaySimMoneyFlowGraph,
    RingFlag,
    RingScorer,
    Tier3Model,
    benchmark_latency,
    build_score_table,
    detect_communities,
    export_ring_edges,
    feature_names_for,
    fit_ring_scorer,
    ring_feature_frame,
)

logger = logging.getLogger(__name__)

#: Ranks below every real score, so an abstaining account never displaces a scored one. The
#: same sentinel Tier-2 uses for windows too short to score.
ABSTAINED_RANK_SENTINEL = -1.0

#: PaySim advances one snapshot per simulated day over a trailing week. The step clock is
#: hourly and the corpus is 31 days, so this yields ~30 snapshots per configuration.
PAYSIM_CADENCE = pd.Timedelta(days=1)
PAYSIM_WINDOW = pd.Timedelta(days=7)

#: IEEE-CIS spans 182 days with no money-flow edge, so its entity graph needs a longer window
#: to accumulate shared-entity co-occurrence and can afford a slower cadence.
IEEE_CADENCE = pd.Timedelta(days=7)
IEEE_WINDOW = pd.Timedelta(days=30)

#: Step windows swept for the PaySim chain edge. Selected on validation.
PAYSIM_STEP_WINDOW_GRID: tuple[int, ...] = (0, 1, 3)

#: Amount tolerances swept. Measured collapse: at +/-0.1% the legitimate-transfer match rate
#: rises from 0.23% to 64.2% and candidate pairs from 2,681 to 2.9M, because PaySim's amounts
#: are dense enough that a relative window admits everything. The signal is exact equality,
#: which is itself the artefact tell -- a real money-flow graph would need tolerance for fees
#: and partial cash-outs, and this one does not because the simulator copies the amount.
PAYSIM_TOLERANCE_GRID: tuple[float, ...] = (0.0,)

#: Entity degree caps swept for IEEE-CIS. Above the cap an entity is a bucket, not an
#: identifier; the measured collapse without a cap is 98,466 accounts on one ``card4`` value.
IEEE_ENTITY_CAP_GRID: tuple[int, ...] = (50, 200)

#: An analyst team can review about this share of traffic. The operating threshold is chosen
#: on validation as the cost-minimising point subject to this cap, matching Tier-1 and Tier-2.
MAX_REVIEW_FLAG_RATE = 0.01

#: Reviewing a ring is not reviewing a transaction: an analyst has to walk several accounts
#: and their links before deciding. Stated as an assumption in the report, and swept.
RING_REVIEW_COST_MULTIPLE = 5.0

#: Above this **validation-split** abstention rate a per-transaction score does not
#: meaningfully exist on the corpus, and the ring becomes the unit of analysis. PaySim measures
#: ~100% here because its origins are near-unique; IEEE-CIS accounts recur and measure far
#: below. Read on validation, never on test: which unit a corpus is evaluated on is a
#: selection like any other.
TRANSACTION_COVERAGE_FLOOR = 0.99

#: Overlap at which a ring counts as a repeat of an earlier one for de-duplication. A stated
#: leakage-control choice, deliberately not fitted: selecting it would mean choosing how much
#: leakage to permit by looking at the result. At 0.5 a ring repeats when a majority of the
#: smaller ring is accounts already seen together.
RING_SEEN_OVERLAP = 0.5

#: Overlap coefficient at which a detected community counts as having recovered a surrogate
#: ground-truth ring. Selected on validation from this grid.
OVERLAP_GRID: tuple[float, ...] = (0.3, 0.5, 0.7)

#: PaySim's amounts carry no currency, so its cost model is scaled to the corpus median and
#: reported in its own units rather than in any real one.
PAYSIM_COST_UNITS = "PaySim amount units (synthetic, no currency)"
IEEE_COST_UNITS = "IEEE-CIS amount units (consistent with USD)"


def load_splits(
    processed_dir: Path, source: SourceDataset, sample: int | None = None
) -> pd.DataFrame:
    """Read one corpus's three splits back as a single time-ordered frame.

    The splits are concatenated deliberately: a rolling window in validation must be allowed
    to see training history, because history is history. What must never happen is a fitted
    quantity crossing forward, and that is enforced by which rows each stage *fits* on, not by
    withholding the past.
    """
    frames: list[pd.DataFrame] = []
    for split in SPLIT_ORDER:
        path = processed_dir / f"{source}_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run `python -m app.data.pipeline` before training Tier-3."
            )
        frame = pd.read_parquet(path)
        frame["split"] = split
        if sample is not None:
            # A chronological head **per split**, never a random sample and never a head of
            # the corpus. The splits are chronological, so a head of the whole frame is
            # entirely train and leaves nothing to select or test on; a shuffled subset would
            # have no rolling window to build at all. Taking a contiguous block from each
            # split keeps all three present and keeps transfer/cash-out pairs intact inside
            # each one, at the cost of a time gap between them -- which is why a sampled run
            # is labelled unreportable rather than merely approximate.
            frame = frame.head(max(1, sample // len(SPLIT_ORDER)))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).sort_values("event_time")
    return combined.reset_index(drop=True)


@dataclass(frozen=True)
class SplitBoundaries:
    """Where one corpus's chronological splits begin and end."""

    train_end: pd.Timestamp
    val_end: pd.Timestamp

    def of(self, moment: pd.Timestamp) -> Split:
        """Return which split a moment falls in."""
        if moment <= self.train_end:
            return "train"
        if moment <= self.val_end:
            return "val"
        return "test"


def split_boundaries(frame: pd.DataFrame) -> SplitBoundaries:
    """Read the split boundaries back off the frame the pipeline already partitioned."""
    return SplitBoundaries(
        train_end=cast(pd.Timestamp, frame.loc[frame["split"] == "train", "event_time"].max()),
        val_end=cast(pd.Timestamp, frame.loc[frame["split"] == "val", "event_time"].max()),
    )


GraphFactory = Callable[[], EntityGraph]


def paysim_factory(step_window: int, amount_tolerance: float) -> GraphFactory:
    """Return a factory building PaySim graphs with these parameters bound.

    A named closure rather than a lambda with default arguments: the grid loop below would
    otherwise capture the loop variable by reference, which is the classic late-binding bug
    that would silently evaluate every configuration with the last one's parameters.
    """

    def build() -> EntityGraph:
        return PaySimMoneyFlowGraph(step_window=step_window, amount_tolerance=amount_tolerance)

    return build


def ieee_factory(
    max_entity_degree: int, specs: Sequence[FingerprintSpec] = IEEE_FINGERPRINTS
) -> GraphFactory:
    """Return a factory building IEEE-CIS entity graphs with these parameters bound."""

    def build() -> EntityGraph:
        return IEEECISSharedEntityGraph(specs=specs, max_entity_degree=max_entity_degree)

    return build


@dataclass
class SnapshotOutcome:
    """Everything one snapshot produced: its rings, and the transactions it went on to score."""

    snapshot_end: pd.Timestamp
    split: Split
    rings: pd.DataFrame
    scored: pd.DataFrame
    seen_accounts: frozenset[str] = frozenset()
    notes: dict[str, Any] = field(default_factory=dict)
    snapshot: GraphSnapshot | None = None


def ring_exposure(window: pd.DataFrame, communities: Sequence[Sequence[str]]) -> dict[str, float]:
    """Return the total amount each ring is exposed to, counting every transaction once.

    An earlier version summed a per-account exposure over both sides of every edge and then
    added it up across ring members, which priced a transfer between two members of the same
    ring at twice its amount -- and every chain-linked PaySim ring is exactly that shape by
    construction. Since the false-negative side of the cost model is the transaction amount,
    that doubled the cost and biased the recommended threshold toward flagging.

    A transaction is attributed to the ring of its origin when the origin is a member, and
    otherwise to the ring of its counterparty, so it is counted once and only once.
    """
    member_ring: dict[str, str] = {}
    for index, members in enumerate(communities):
        for account in members:
            member_ring.setdefault(str(account), f"r{index}")
    if not member_ring or window.empty:
        return {}

    origin = window["account_id"].astype(str).map(member_ring)
    if "counterparty_id" in window.columns:
        counterparty = window["counterparty_id"].astype(str).map(member_ring)
        attributed = origin.fillna(counterparty)
    else:
        attributed = origin
    totals = window.loc[attributed.notna()].groupby(attributed.dropna())["amount"].sum()
    return {str(key): float(value) for key, value in totals.items()}


def fraud_accounts(window: pd.DataFrame) -> set[str]:
    """Return accounts touching a fraudulent transaction inside one window.

    Both sides of the edge count. A mule receiving a fraudulent transfer is part of the ring
    even though that row's ``account_id`` is the victim's.
    """
    fraud = window.loc[window["is_fraud"]]
    accounts = {str(value) for value in fraud["account_id"].dropna()}
    if "counterparty_id" in fraud.columns:
        accounts |= {str(value) for value in fraud["counterparty_id"].dropna()}
    return accounts


def surrogate_rings(window: pd.DataFrame, factory: GraphFactory) -> list[set[str]]:
    """Build the surrogate ground-truth partition from labels alone.

    **This is a surrogate and is labelled as one everywhere it is reported.** PaySim ships no
    ring or agent identifier, only ``isFraud`` per transaction, so a "true ring" has to be
    constructed: the connected components, of at least :data:`MIN_RING_SIZE` accounts, of the
    graph induced by fraudulent transactions alone. It is a defensible construction and it is
    still a construction, which is why the enrichment view exists to check it.
    """
    fraud = window.loc[window["is_fraud"]]
    if fraud.empty:
        return []
    graph = factory()
    graph.insert(fraud)
    end = cast(pd.Timestamp, fraud["event_time"].max()) + pd.Timedelta(seconds=1)
    snapshot = graph.snapshot(end.to_pydatetime())
    return [set(members) for members in detect_communities(snapshot)]


def snapshot_ends(
    frame: pd.DataFrame, cadence: pd.Timedelta, window: pd.Timedelta
) -> list[pd.Timestamp]:
    """Return the chronological snapshot boundaries for one corpus.

    The first boundary is one cadence after the corpus starts, so every snapshot has some
    history behind it; the last is one cadence past the final transaction, so no rows go
    unscored.
    """
    start = cast(pd.Timestamp, frame["event_time"].min())
    stop = cast(pd.Timestamp, frame["event_time"].max())
    ends: list[pd.Timestamp] = []
    moment = start + cadence
    while moment <= stop + cadence:
        ends.append(moment)
        moment = moment + cadence
    return ends


def run_snapshots(
    frame: pd.DataFrame,
    factory: GraphFactory,
    *,
    cadence: pd.Timedelta,
    window: pd.Timedelta,
    boundaries: SplitBoundaries,
    keep_snapshot_for: pd.Timestamp | None = None,
) -> list[SnapshotOutcome]:
    """Walk the corpus in chronological snapshots, scoring forward only.

    For each boundary ``t`` the graph is built from ``[t - window, t)`` and used to score the
    transactions in ``[t, t + cadence)``. A transaction therefore never contributes an edge to
    the snapshot that scores it, which is the invariant the whole layer's honesty rests on and
    which :meth:`EntityGraph.snapshot` re-checks rather than trusting this loop.

    **Structural only, deliberately.** No scorer is applied here. The expensive half of this
    layer is graph construction and community detection, and neither depends on the scorer, so
    one pass produces the features for every split at once and :func:`attach_scores` fits and
    applies afterwards. Doing it the other way round would rebuild every graph a second time
    purely to multiply a matrix.

    Args:
        frame: The full time-ordered corpus.
        factory: Builds a fresh graph per snapshot. Phase 4 rebuilds; the incremental
            enhancement replaces this with insert-and-evict against one long-lived graph.
        cadence: How often the graph refreshes. Also the maximum score staleness.
        window: Trailing width of each snapshot.
        boundaries: Used to label each snapshot with the split its scoring period falls in.
        keep_snapshot_for: Retain the built graph for this one boundary, for the visualisations.

    Returns:
        One :class:`SnapshotOutcome` per boundary that had any rings or any scored rows.
    """
    times = frame["event_time"]
    outcomes: list[SnapshotOutcome] = []

    for end in snapshot_ends(frame, cadence, window):
        window_rows = frame.loc[(times >= end - window) & (times < end)]
        scoring_rows = frame.loc[(times >= end) & (times < end + cadence)]
        if window_rows.empty and scoring_rows.empty:
            continue

        graph = factory()
        graph.insert(window_rows)
        snapshot = graph.snapshot(end.to_pydatetime())
        communities = detect_communities(snapshot)
        features = ring_feature_frame(snapshot, communities)

        rings = features
        if not rings.empty:
            infected = fraud_accounts(window_rows)
            exposure = ring_exposure(window_rows, communities)
            rings = rings.assign(
                account_is_fraudulent=rings["account_id"].isin(infected),
                ring_amount=rings["ring_id"].map(exposure).fillna(0.0),
                snapshot_end=end,
            )

        columns = [
            "transaction_id",
            "account_id",
            "is_fraud",
            "split",
            "transaction_type",
            "amount",
            "step",
        ]
        scored = scoring_rows.loc[:, [c for c in columns if c in scoring_rows.columns]].copy()
        if not scored.empty:
            scored["snapshot_end"] = end
            scored["staleness_seconds"] = (
                (scoring_rows["event_time"] - end).dt.total_seconds().to_numpy()
            )

        outcomes.append(
            SnapshotOutcome(
                snapshot_end=end,
                split=boundaries.of(end),
                rings=rings,
                scored=scored,
                seen_accounts=frozenset(snapshot.account_nodes),
                notes=dict(snapshot.notes),
                snapshot=snapshot if end == keep_snapshot_for else None,
            )
        )
    return outcomes


def stack(outcomes: Sequence[SnapshotOutcome], attribute: str) -> pd.DataFrame:
    """Concatenate one field across snapshots, dropping the empties."""
    frames = [
        getattr(outcome, attribute) for outcome in outcomes if not getattr(outcome, attribute).empty
    ]
    if not frames:
        return pd.DataFrame()
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))


def ring_labels(rings: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-account ring rows to one row per ring, with its label and score.

    A ring is positive when any member account touched fraud inside the snapshot window, and
    its score is the maximum over its members -- the same rule
    :func:`app.models.tier3_graph.build_score_table` uses, so the ring-level and
    transaction-level views cannot disagree about what a ring's score is.
    """
    if rings.empty:
        return pd.DataFrame()
    grouped = rings.groupby(["snapshot_end", "ring_id"], sort=True)
    # The two corpora carry different ring columns by design -- PaySim has chain edges and no
    # entities, IEEE-CIS the reverse -- so aggregate whichever are present rather than a union
    # neither produces.
    aggregations: dict[str, tuple[str, str]] = {
        "size": ("account_id", "size"),
        "is_fraud_ring": ("account_is_fraudulent", "any"),
    }
    for name in ("ring_chain_edge_count", "ring_density", "ring_entity_count"):
        if name in rings.columns:
            aggregations[name] = (name, "first")
    if "ring_amount" in rings.columns:
        # "first", not "sum": ring_amount is already a per-ring total broadcast onto each
        # member row, so summing it would multiply the ring's exposure by its member count.
        aggregations["ring_amount"] = ("ring_amount", "first")
    frame = grouped.agg(**aggregations).reset_index()
    if "ring_risk_score" in rings.columns:
        frame["ring_risk_score"] = grouped["ring_risk_score"].max().to_numpy()
    return frame


def transaction_scores(
    scored: pd.DataFrame,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Return labels and ranking scores, with abstentions ranked last rather than at zero.

    An account Tier-3 has never seen linked is not a clean account. Ranking it at 0.0 would put
    it above nothing and below everything scored, which happens to be right, but writing 0.0
    into the score would also tell Phase 5's meta-learner "maximally clean" about an account
    this layer has no opinion on. The sentinel keeps the ranking honest and keeps the
    abstention visible.
    """
    labels = scored["is_fraud"].to_numpy(dtype=bool)
    values = scored["ring_risk_score"].to_numpy(dtype="float64")
    ranked = np.where(np.isnan(values), ABSTAINED_RANK_SENTINEL, values)
    return labels, ranked


def pairing_baseline(frame: pd.DataFrame, *, step_window: int, tolerance: float) -> pd.DataFrame:
    """Score every PaySim transaction by the pairing rule alone -- the baseline to beat.

    **The comparison that decides whether Phase 4 earned anything.** The chain rule selects
    99.50% of fraudulent transfers against 0.23% of legitimate ones before any graph algorithm
    runs. If the graph cannot beat it, that is the finding, and it is reported as one rather
    than hidden behind a ring metric that inherits the rule's separation.

    Returns:
        The frame with a boolean ``pairing_flag`` column: True where the transaction is either
        a transfer with a matched cash-out or a cash-out matched by one.
    """
    chain = build_chain_edges(frame, step_window=step_window, amount_tolerance=tolerance)
    matched_transfers = set(chain.frame["transfer_txn"].dropna().astype(str))
    matched_cashouts = set(chain.frame["cashout_txn"].dropna().astype(str))
    identifiers = frame["transaction_id"].astype(str)
    types = frame["transaction_type"]
    flag = (types == TRANSFER) & identifiers.isin(matched_transfers)
    flag |= (types == CASH_OUT) & identifiers.isin(matched_cashouts)
    return frame.assign(pairing_flag=flag.to_numpy())


def overlap_coefficient(left: set[str], right: set[str]) -> float:
    """Return ``|A and B| / min(|A|, |B|)``."""
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


@dataclass(frozen=True)
class RecoveryResult:
    """How well detected communities recovered the surrogate ground-truth rings."""

    threshold: float
    detected: int
    truth: int
    matched_detected: int
    matched_truth: int

    @property
    def precision(self) -> float:
        """Return the share of detected rings that matched a surrogate ring."""
        return self.matched_detected / self.detected if self.detected else 0.0

    @property
    def recall(self) -> float:
        """Return the share of surrogate rings a detected ring recovered."""
        return self.matched_truth / self.truth if self.truth else 0.0

    @property
    def f1(self) -> float:
        """Return the harmonic mean of recovery precision and recall."""
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


def measure_recovery(
    detected: Sequence[set[str]], truth: Sequence[set[str]], threshold: float
) -> RecoveryResult:
    """Match detected communities against surrogate rings by overlap coefficient."""
    matched_detected = 0
    matched_truth: set[int] = set()
    for community in detected:
        best_index, best_score = -1, 0.0
        for index, true_ring in enumerate(truth):
            score = overlap_coefficient(community, true_ring)
            if score > best_score:
                best_index, best_score = index, score
        if best_score >= threshold:
            matched_detected += 1
            matched_truth.add(best_index)
    return RecoveryResult(
        threshold=threshold,
        detected=len(detected),
        truth=len(truth),
        matched_detected=matched_detected,
        matched_truth=len(matched_truth),
    )


@dataclass(frozen=True)
class EnrichmentResult:
    """Fraud concentration in top-ranked rings against a random-ring null.

    Depends on no surrogate partition, which is the point: if it and the surrogate-based
    recovery disagree, the surrogate definition is driving the headline and the headline goes.
    """

    k: int
    precision_at_k: float
    base_rate: float
    lift: float

    def describe(self) -> str:
        """Return a one-line summary."""
        return (
            f"top-{self.k} rings: {self.precision_at_k:.3f} fraud-bearing against a "
            f"{self.base_rate:.3f} ring base rate ({self.lift:.1f}x)"
        )


def measure_enrichment(rings: pd.DataFrame, k: int) -> EnrichmentResult:
    """Return precision@k over rings ranked by risk score, against the ring base rate."""
    if rings.empty or "ring_risk_score" not in rings.columns:
        return EnrichmentResult(k=k, precision_at_k=0.0, base_rate=0.0, lift=0.0)
    ordered = rings.sort_values("ring_risk_score", ascending=False)
    top = ordered.head(k)
    base = float(rings["is_fraud_ring"].mean())
    precision = float(top["is_fraud_ring"].mean()) if len(top) else 0.0
    return EnrichmentResult(
        k=int(len(top)),
        precision_at_k=precision,
        base_rate=base,
        lift=precision / base if base else 0.0,
    )


def graph_feature_version(source: SourceDataset, parameters: dict[str, Any]) -> str:
    """Return a hash identifying the exact structural input definition behind a score.

    Deliberately a different namespace from Phase 1's ``fv_`` feature version. Tier-1 and
    Tier-2 read an engineered per-row vector; Tier-3 reads graph topology, and the things that
    change its inputs are the edge rule, the entity cap and the window, none of which the
    Phase 1 feature store knows about. Sharing the prefix would imply the two are
    interchangeable in an audit row when they identify different spaces.
    """
    import hashlib

    payload = json.dumps(
        {
            "source": source,
            "features": list(feature_names_for(source)),
            "parameters": parameters,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "gv_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def attach_scores(
    rings: pd.DataFrame, scored: pd.DataFrame, scorer: RingScorer
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a fitted scorer to cached structural features and propagate to transactions.

    Separated from :func:`run_snapshots` so a configuration's graphs are built once and every
    scorer variant reuses them. The per-account table is rebuilt per snapshot with the same
    max-over-rings rule :func:`app.models.tier3_graph.build_score_table` applies at serving
    time, so the offline metric and the served score cannot drift apart.
    """
    if rings.empty:
        return rings, scored.assign(ring_risk_score=np.nan) if not scored.empty else scored

    scored_rings = rings.assign(ring_risk_score=scorer.score_frame(rings))
    best = (
        scored_rings.groupby(["snapshot_end", "account_id"])["ring_risk_score"].max().reset_index()
    )
    if scored.empty:
        return scored_rings, scored
    merged = scored.merge(
        best,
        how="left",
        left_on=["snapshot_end", "account_id"],
        right_on=["snapshot_end", "account_id"],
    )
    return scored_rings, merged


@dataclass
class Candidate:
    """One configuration's validation result, kept whether it won or lost."""

    label: str
    parameters: dict[str, Any]
    val_ring_pr_auc: float
    val_rings: int
    val_ring_base_rate: float
    rings: pd.DataFrame
    account_rings: pd.DataFrame
    #: The de-duplicated account-level rings the ring metric is computed over. Kept apart from
    #: `account_rings`, which is the full set the score table is built from: recovery is a
    #: ring-level metric and must read the same population as the ring PR-AUC beside it, or the
    #: entry reports 18,839 detected rings next to a ring test of 2,508.
    evaluation_account_rings: pd.DataFrame
    scored: pd.DataFrame
    scorer: RingScorer
    outcomes: list[SnapshotOutcome]
    #: Every account present in each snapshot's graph, scoreable or not.
    seen_accounts: dict[pd.Timestamp, frozenset[str]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return the registry-shaped record of this candidate."""
        return {
            "configuration": self.label,
            "parameters": dict(self.parameters),
            "validation_ring_pr_auc": round(self.val_ring_pr_auc, 6),
            "validation_rings": self.val_rings,
            "validation_ring_base_rate": round(self.val_ring_base_rate, 6),
        }


def describe_operating_point(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    threshold: float,
    model: CostModel,
    unit: str,
) -> dict[str, Any]:
    """Return what the recommended threshold actually costs and how much it flags.

    ``models/README.md`` names this a required part of a complete entry, and the reason is in
    ml-evaluation-standards section 2: precision at a 40% flag rate and precision at a 0.5%
    flag rate describe two different products, so the flag rate travels with the precision.
    """
    estimate = cost_at_threshold(labels, scores, amounts, threshold, model)
    matrix = confusion_at_threshold(labels, scores, threshold)
    return {
        "unit": unit,
        "threshold": threshold,
        "flag_rate": round(estimate.flag_rate, 6),
        "precision": round(matrix.precision, 6),
        "recall": round(matrix.recall, 6),
        "f1": round(matrix.f1, 6),
        "confusion_matrix": matrix.to_dict(),
        "review_capacity_cap": MAX_REVIEW_FLAG_RATE,
        "cost": estimate.to_dict(),
    }


def ring_amounts_of(rings: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Return per-ring exposure, or zeros where the corpus did not supply amounts."""
    if "ring_amount" not in rings.columns:
        return np.zeros(len(rings), dtype="float64")
    return rings["ring_amount"].fillna(0.0).to_numpy(dtype="float64")


def evaluate_rings(
    rings: pd.DataFrame,
    split: Split,
    threshold: float,
    criterion: str,
    name: str,
    cost: CostModel | None = None,
) -> EvaluationResult | None:
    """Measure ring-level classification on one split.

    The unit of analysis is a **candidate ring**, not a transaction, and it is declared here
    rather than implied — the same move Phase 3 made when it declared Tier-2 evaluated per
    account. Because every detected ring is either fraud-bearing or clean, TN/FP/FN/TP are all
    defined and the ml-evaluation-standards reporting block applies with no reinterpretation.
    """
    subset = rings.loc[rings["split"] == split] if "split" in rings.columns else rings
    if subset.empty or subset["is_fraud_ring"].nunique() < 2:
        return None
    labels = subset["is_fraud_ring"].to_numpy(dtype=bool)
    scores = subset["ring_risk_score"].to_numpy(dtype="float64")
    amounts = ring_amounts_of(subset)
    return evaluate(
        name,
        split,
        labels,
        scores,
        threshold=threshold,
        threshold_criterion=criterion,
        interval=bootstrap_pr_auc(labels, scores, seed=RANDOM_SEED),
        cost=(
            cost_at_threshold(labels, scores, amounts, threshold, cost)
            if cost is not None
            else None
        ),
    )


def ring_key(members: Sequence[str]) -> str:
    """Return a stable identity for a ring, independent of which snapshot produced it."""
    return "|".join(sorted(str(member) for member in members))


def assign_rings_to_first_split(
    rings: pd.DataFrame, overlap: float = RING_SEEN_OVERLAP
) -> pd.DataFrame:
    """Keep each ring once, dropping any that substantially repeats an earlier one.

    **The correction that makes the ring metrics mean anything, at the second attempt.**
    Snapshot windows are much wider than the cadence -- seven days against one on PaySim,
    thirty against seven on IEEE-CIS -- so consecutive snapshots share most of their rows and
    the same ring reappears in roughly seven (respectively four) of them. Left alone, the
    scorer is selected and scored on rings it was fitted on, and the bootstrap treats
    near-identical copies as independent draws.

    The first attempt keyed a ring on its **exact** member set, which is not enough and was
    measured not to be: of the rings surviving exact de-duplication in the first PaySim test
    snapshot, **58% still overlapped a training-split ring by at least half**, and on IEEE-CIS
    it was 82%. One member changing makes a new key while leaving the ring the same ring. The
    apparent 2% removal rate on IEEE-CIS was evidence the key was wrong, not evidence its
    rings were independent.

    So a ring is now dropped when it overlaps any already-kept earlier ring by at least
    ``overlap``, measured by overlap coefficient -- intersection over the smaller set, so a
    ring nested inside a larger earlier one counts as a repeat of it. Rings are considered in
    chronological order, so the earliest sighting is the one kept.

    ``overlap`` is a stated leakage-control choice rather than a fitted parameter: at 0.5 a
    ring is a repeat when a majority of the smaller ring is accounts already seen together.
    It is not selected on any split, because selecting it would mean choosing how much
    leakage to permit by looking at the result.
    """
    if rings.empty:
        return rings

    members_by_ring = (
        rings.groupby(["snapshot_end", "ring_id"])["account_id"]
        .apply(lambda values: frozenset(str(value) for value in values))
        .sort_index(level="snapshot_end")
    )

    kept: list[frozenset[str]] = []
    # account -> indices into `kept`, so a candidate is only compared against rings it could
    # actually overlap. Comparing every candidate against every kept ring is quadratic in the
    # tens of thousands and was measured to dominate the phase.
    index: dict[str, set[int]] = {}
    keep_keys: set[tuple[Any, Any]] = set()

    for key, accounts in members_by_ring.items():
        moment, ring_id = cast(tuple[Any, Any], key)
        candidates: set[int] = set()
        for account in accounts:
            candidates |= index.get(account, set())
        if any(
            len(accounts & kept[position]) / min(len(accounts), len(kept[position])) >= overlap
            for position in candidates
        ):
            continue
        keep_keys.add((moment, ring_id))
        position = len(kept)
        kept.append(accounts)
        for account in accounts:
            index.setdefault(account, set()).add(position)

    mask = [
        (moment, ring) in keep_keys
        for moment, ring in zip(rings["snapshot_end"], rings["ring_id"], strict=True)
    ]
    return rings.loc[pd.Series(mask, index=rings.index)].copy()


def label_rings_with_split(rings: pd.DataFrame, boundaries: SplitBoundaries) -> pd.DataFrame:
    """Tag each ring with the split its snapshot's scoring period belongs to."""
    if rings.empty:
        return rings
    return rings.assign(
        split=[boundaries.of(cast(pd.Timestamp, moment)) for moment in rings["snapshot_end"]]
    )


def plot_ring(
    snapshot: GraphSnapshot,
    members: Sequence[str],
    fraud: set[str],
    title: str,
    path: Path,
) -> None:
    """Draw one community, colouring members by label, and save it.

    Two of these are the phase's visual gate: a detected ring and a clean cluster have to be
    visibly different, not merely differently labelled.
    """
    nodes = set(members)
    for account in members:
        for neighbour in snapshot.graph.neighbors(account):
            if snapshot.graph.nodes[neighbour].get("kind", "account") == "entity":
                nodes.add(str(neighbour))
    subgraph = nx.Graph(snapshot.graph.subgraph(nodes))

    figure, axis = plt.subplots(figsize=(8, 6))
    layout = nx.spring_layout(subgraph, seed=RANDOM_SEED, k=0.5)
    accounts = [n for n in subgraph if subgraph.nodes[n].get("kind", "account") == "account"]
    entities = [n for n in subgraph if subgraph.nodes[n].get("kind", "account") == "entity"]
    # Read the `chain` flag, not a `relation` label. The label went away when an edge became
    # able to carry both relations at once, and this filter then matched nothing -- so every
    # inferred chain link was drawn as an ordinary grey edge while the legend underneath still
    # promised a thick red one.
    chain = [(u, v) for u, v, d in subgraph.edges(data=True) if d.get("chain")]
    other = [(u, v) for u, v, d in subgraph.edges(data=True) if not d.get("chain")]

    nx.draw_networkx_edges(subgraph, layout, edgelist=other, ax=axis, alpha=0.35, width=1.0)
    nx.draw_networkx_edges(
        subgraph, layout, edgelist=chain, ax=axis, edge_color="#c0392b", width=2.2
    )
    nx.draw_networkx_nodes(
        subgraph,
        layout,
        nodelist=accounts,
        ax=axis,
        node_size=140,
        node_color=["#c0392b" if n in fraud else "#2c7fb8" for n in accounts],
    )
    if entities:
        nx.draw_networkx_nodes(
            subgraph,
            layout,
            nodelist=entities,
            ax=axis,
            node_size=90,
            node_shape="s",
            node_color="#7f8c8d",
        )
    # Wrap rather than trust the figure width: these captions carry the window, the member
    # count and the fraud-touching count, and the first render clipped both margins of the
    # subtitle. This image is the phase's sanity-check screenshot, so a clipped caption is a
    # defect in the deliverable rather than a cosmetic detail.
    wrapped = "\n".join(textwrap.fill(line, width=72) for line in title.splitlines())
    axis.set_title(wrapped, fontsize=10)
    axis.axis("off")
    axis.text(
        0.01,
        0.01,
        "red = account touching labelled fraud, blue = clean account, "
        "grey square = shared entity, thick red edge = inferred chain link",
        transform=axis.transAxes,
        fontsize=7,
        color="#555555",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


@dataclass
class CorpusReport:
    """Everything one corpus produced, kept together so no headline travels without context."""

    source: SourceDataset
    winner: Candidate
    candidates: list[Candidate]
    threshold: float
    cadence: pd.Timedelta
    window: pd.Timedelta
    ring_test: EvaluationResult | None
    transaction_test: EvaluationResult | None
    abstention_rate: float
    validation_abstention: float = 1.0
    training_window: dict[str, str] = field(default_factory=dict)
    unit: str = "transaction"
    ring_threshold: float = 0.0
    recovery: RecoveryResult | None = None
    enrichment: EnrichmentResult | None = None
    baseline: ConfusionMatrix | None = None
    baseline_pr_auc: float | None = None
    conditional: EvaluationResult | None = None
    increment: tuple[float, float, float] | None = None
    tier1_pr_auc: float | None = None
    fused_pr_auc: float | None = None
    fused_delta: tuple[float, float, float] | None = None
    latency: dict[str, float] = field(default_factory=dict)
    cost_sweep: list[Any] = field(default_factory=list)
    scale_sweep: list[Any] = field(default_factory=list)
    operating_point: dict[str, Any] = field(default_factory=dict)
    model: Tier3Model | None = None
    false_negatives: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def ring_cost_model(base: CostModel) -> CostModel:
    """Return the ring-level cost model: reviewing a ring costs more than reviewing a row.

    An analyst clearing a flagged ring has to walk several accounts and the links between them
    before deciding, so a ring review is priced at :data:`RING_REVIEW_COST_MULTIPLE` times a
    transaction review. The multiple is an assumption, stated as one in the report and swept
    alongside the other cost assumptions.
    """
    return CostModel(
        review_cost=base.review_cost * RING_REVIEW_COST_MULTIPLE,
        chargeback_fee=base.chargeback_fee,
        units=base.units,
        unit_noun="ring",
    )


def choose_threshold(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    amounts: npt.NDArray[np.float64],
    model: CostModel,
    unit: str,
) -> tuple[float, str, CostEstimate | None]:
    """Pick the operating threshold on validation, by cost and under the review-capacity cap.

    Two constraints, in this order: the cost-minimising point is found first, then raised if
    it would flag more than :data:`MAX_REVIEW_FLAG_RATE` of traffic. A threshold justified by
    F1 alone is not a recommendation an analyst team can staff, which is why
    ml-evaluation-standards section 3 asks for the cost justification rather than the metric.

    Args:
        labels: Validation labels, at whatever unit this corpus is evaluated on.
        scores: Validation scores, aligned to ``labels``.
        amounts: Exposure per unit, for pricing a missed positive.
        model: The cost model, already scaled to this corpus and unit.
        unit: Plain-language name of the unit, recorded in the criterion string so a reader
            can never mistake a ring threshold for a transaction one.
    """
    if labels.size == 0:
        return 0.5, f"default; validation produced no scoreable {unit}s", None
    capacity = threshold_for_flag_rate(scores, MAX_REVIEW_FLAG_RATE)
    if not np.isfinite(capacity):
        # Every score is the abstention sentinel, so no finite threshold flags anything. This
        # is a real state on PaySim, where near-unique origins mean an account seen in one
        # window essentially never returns in the next; it is reported, not papered over.
        return (
            0.5,
            f"no finite threshold exists: every validation {unit} abstained",
            None,
        )
    if labels.sum() == 0:
        return (
            float(capacity),
            f"validation flag rate capped at {MAX_REVIEW_FLAG_RATE:.1%}; no validation "
            f"positives among {unit}s to cost",
            None,
        )
    from app.ml.cost import choose_threshold_by_cost

    cost_threshold, _ = choose_threshold_by_cost(labels, scores, amounts, model)
    threshold = float(max(cost_threshold, capacity))
    criterion = (
        f"minimum estimated cost on validation {unit}s, raised to respect a "
        f"{MAX_REVIEW_FLAG_RATE:.1%} review-capacity cap"
        if capacity > cost_threshold
        else f"minimum estimated cost on validation {unit}s"
    )
    return threshold, criterion, cost_at_threshold(labels, scores, amounts, threshold, model)


def run_corpus(
    source: SourceDataset,
    frame: pd.DataFrame,
    configurations: Sequence[tuple[str, dict[str, Any], GraphFactory]],
    *,
    cadence: pd.Timedelta,
    window: pd.Timedelta,
    cost_model: CostModel,
) -> CorpusReport:
    """Sweep configurations on validation, then measure the winner once on test.

    The selection discipline is the one Phase 2 had to learn the hard way: every configuration
    is compared on **validation** ring PR-AUC, the winner is fixed, and only then is the test
    split touched. Losing configurations are kept on the report rather than discarded.
    """
    boundaries = split_boundaries(frame)
    candidates: list[Candidate] = []

    for label, parameters, factory in configurations:
        logger.info("[%s] building snapshots for %s", source, label)
        outcomes = run_snapshots(
            frame, factory, cadence=cadence, window=window, boundaries=boundaries
        )
        rings = label_rings_with_split(stack(outcomes, "rings"), boundaries)
        if rings.empty:
            logger.warning("[%s] %s produced no rings at all; skipped", source, label)
            continue
        # De-duplication is a **metric** device, and its scope is exactly the ring-level
        # evaluation population. It must not reach the scoring path: in production every ring
        # present in the current snapshot is scored, repeats included, and an account whose
        # only ring happened to resemble an earlier one is still an account sitting in a ring.
        # Applying it to both inflated IEEE-CIS's transaction abstention rate from 65.7% to
        # 96.7% -- a number about the evaluation's bookkeeping being reported as a property of
        # the layer.
        evaluation_rings = assign_rings_to_first_split(rings)
        logger.info(
            "[%s] %s: %d of %d account-ring rows survive de-duplication for the ring-level "
            "metric; all %d are kept for scoring",
            source,
            label,
            len(evaluation_rings),
            len(rings),
            len(rings),
        )
        scored = stack(outcomes, "scored")

        # Fitted on de-duplicated training rings, so one recurring ring does not dominate the
        # fit by appearing seven times.
        train_rings = evaluation_rings.loc[evaluation_rings["split"] == "train"]
        if train_rings.empty or train_rings["account_is_fraudulent"].nunique() < 2:
            logger.warning("[%s] %s produced no two-class training rings; skipped", source, label)
            continue
        scorer = fit_ring_scorer(
            train_rings,
            train_rings["account_is_fraudulent"].to_numpy(dtype=bool),
            source,
            seed=RANDOM_SEED,
        )
        # Scored over *all* rings — this feeds the served table and the per-transaction metric.
        scored_rings, scored_txns = attach_scores(rings, scored, scorer)
        # The ring-level metric reads the de-duplicated subset of the same scored rows, so the
        # two views can never disagree about what a ring scored.
        evaluation_scored = scored_rings.merge(
            evaluation_rings[["snapshot_end", "ring_id", "account_id"]],
            on=["snapshot_end", "ring_id", "account_id"],
            how="inner",
        )
        ring_view = ring_labels(evaluation_scored)
        ring_view = label_rings_with_split(ring_view, boundaries)
        validation = ring_view.loc[ring_view["split"] == "val"]
        if validation.empty or validation["is_fraud_ring"].nunique() < 2:
            logger.warning("[%s] %s has no two-class validation rings; skipped", source, label)
            continue

        candidates.append(
            Candidate(
                label=label,
                parameters=parameters,
                val_ring_pr_auc=pr_auc(
                    validation["is_fraud_ring"].to_numpy(dtype=bool),
                    validation["ring_risk_score"].to_numpy(dtype="float64"),
                ),
                val_rings=int(len(validation)),
                val_ring_base_rate=float(validation["is_fraud_ring"].mean()),
                rings=ring_view,
                account_rings=scored_rings,
                evaluation_account_rings=evaluation_scored,
                scored=scored_txns,
                scorer=scorer,
                outcomes=outcomes,
                seen_accounts={outcome.snapshot_end: outcome.seen_accounts for outcome in outcomes},
            )
        )

    if not candidates:
        raise RuntimeError(
            f"no {source} configuration produced a scoreable graph; Tier-3 cannot be "
            "evaluated on this corpus"
        )

    winner = max(candidates, key=lambda item: item.val_ring_pr_auc)
    logger.info("[%s] selected %s on validation ring PR-AUC", source, winner.label)

    test_txns = winner.scored.loc[winner.scored["split"] == "test"]

    def abstention_rate(subset: pd.DataFrame) -> float:
        """Return the share of rows with no ring score at all."""
        if subset.empty or "ring_risk_score" not in subset.columns:
            return 1.0
        return float(np.isnan(subset["ring_risk_score"].to_numpy("float64")).mean())

    # Which unit this corpus is evaluated on is a **selection**, so it is made on validation.
    # It used to be made on the test split's abstention rate, which is a decision taken by
    # looking at test -- ml-evaluation-standards section 1 is categorical about that, however
    # unlikely the decision was to differ. The test rate is still measured and reported,
    # because it is what the reader needs to interpret the abstention line.
    validation_abstention = abstention_rate(winner.scored.loc[winner.scored["split"] == "val"])
    abstention = abstention_rate(test_txns)

    # The unit of analysis differs by corpus, and it is chosen on measurement rather than
    # taste. PaySim origins are near-unique -- 99.95% have degree 1 across the whole corpus --
    # so an account appearing in one window essentially never reappears in the next, and a
    # per-transaction ring score there abstains on almost everything. The ring is the unit that
    # exists on that corpus. IEEE-CIS accounts recur, so the transaction is.
    ring_unit = validation_abstention >= TRANSACTION_COVERAGE_FLOOR
    validation_rings = winner.rings.loc[winner.rings["split"] == "val"]
    notes: list[str] = []

    # The ring result is always operated at a threshold chosen on validation **rings** under
    # the ring cost model, on every corpus. Reusing a transaction-selected threshold to cut a
    # ring confusion matrix -- as an earlier version did on IEEE-CIS -- reports precision and
    # recall at an operating point that was never chosen for that unit, and the criterion
    # string then says "on validation transactions" underneath a ring metric.
    ring_model = ring_cost_model(cost_model)
    ring_threshold, ring_criterion, _ = choose_threshold(
        validation_rings["is_fraud_ring"].to_numpy(dtype=bool),
        validation_rings["ring_risk_score"].to_numpy(dtype="float64"),
        ring_amounts_of(validation_rings),
        ring_model,
        "ring",
    )

    if ring_unit:
        model_for_unit = ring_model
        threshold, criterion = ring_threshold, ring_criterion
        reason = (
            "PaySim origins are near-unique -- 99.95% have degree 1 -- so an account seen in "
            "one snapshot window essentially never returns in the next"
            if source == "paysim"
            else "almost no account recurs across consecutive snapshot windows on this corpus"
        )
        notes.append(
            f"**The unit of analysis on this corpus is the ring, not the transaction.** "
            f"{abstention:.1%} of test transactions abstained, because {reason}. A "
            "per-transaction ring score is therefore structurally unavailable here, which is "
            "why the ring-level result is the one reported and no transaction-level headline "
            "is published from a column that is almost entirely abstentions."
        )
    else:
        model_for_unit = cost_model
        validation_txns = winner.scored.loc[winner.scored["split"] == "val"]
        val_labels, val_scores = transaction_scores(validation_txns)
        threshold, criterion, _ = choose_threshold(
            val_labels,
            val_scores,
            validation_txns["amount"].to_numpy(dtype="float64"),
            model_for_unit,
            "transaction",
        )

    # The ring result is always priced with the ring cost model, including on a corpus whose
    # headline unit is the transaction: an analyst clearing a flagged ring walks several
    # accounts either way, and pricing that at one transaction review would understate it.
    ring_test = evaluate_rings(
        winner.rings,
        "test",
        ring_threshold,
        ring_criterion,
        f"tier3-{source}-ring",
        ring_model,
    )

    transaction_test: EvaluationResult | None = None
    cost_sweep: list[Any] = []
    scale_sweep: list[Any] = []
    operating_point: dict[str, Any] = {}
    if not ring_unit and not test_txns.empty:
        labels, scores = transaction_scores(test_txns)
        amounts = test_txns["amount"].to_numpy(dtype="float64")
        if labels.sum() > 0:
            transaction_test = evaluate(
                f"tier3-{source}-transaction",
                "test",
                labels,
                scores,
                threshold=threshold,
                threshold_criterion=criterion,
                interval=bootstrap_pr_auc(labels, scores, seed=RANDOM_SEED),
                cost=cost_at_threshold(labels, scores, amounts, threshold, cost_model),
                notes=[
                    f"{abstention:.1%} of test transactions abstained (account absent from "
                    "the snapshot or in a component below the minimum ring size). Abstentions "
                    "rank last and never flag; they are not scored as clean."
                ],
            )
            # Swept on **validation**. These sweeps re-choose the threshold at each
            # setting, which is a selection; running them on test would be selecting on test
            # under the name of a sensitivity analysis.
            sweep_labels, sweep_scores = transaction_scores(validation_txns)
            sweep_amounts = validation_txns["amount"].to_numpy(dtype="float64")
            cost_sweep = review_cost_sweep(sweep_labels, sweep_scores, sweep_amounts, cost_model)
            scale_sweep = sensitivity_sweep(sweep_labels, sweep_scores, sweep_amounts, cost_model)
            operating_point = describe_operating_point(
                labels, scores, amounts, threshold, cost_model, "transaction"
            )
    elif ring_unit:
        test_rings_for_cost = winner.rings.loc[winner.rings["split"] == "test"]
        ring_label_array = test_rings_for_cost["is_fraud_ring"].to_numpy(dtype=bool)
        if ring_label_array.sum() > 0:
            ring_score_array = test_rings_for_cost["ring_risk_score"].to_numpy(dtype="float64")
            ring_amount_array = ring_amounts_of(test_rings_for_cost)
            # Swept on validation rings, for the reason above.
            sweep_labels = validation_rings["is_fraud_ring"].to_numpy(dtype=bool)
            sweep_scores = validation_rings["ring_risk_score"].to_numpy(dtype="float64")
            sweep_amounts = ring_amounts_of(validation_rings)
            cost_sweep = review_cost_sweep(
                sweep_labels, sweep_scores, sweep_amounts, model_for_unit
            )
            scale_sweep = sensitivity_sweep(
                sweep_labels, sweep_scores, sweep_amounts, model_for_unit
            )
            operating_point = describe_operating_point(
                ring_label_array,
                ring_score_array,
                ring_amount_array,
                ring_threshold,
                ring_model,
                "ring",
            )

    test_rings = winner.rings.loc[winner.rings["split"] == "test"]
    enrichment = (
        measure_enrichment(test_rings, k=max(1, int(0.1 * len(test_rings))))
        if not test_rings.empty
        else None
    )

    return CorpusReport(
        source=source,
        winner=winner,
        candidates=candidates,
        threshold=threshold,
        cadence=cadence,
        window=window,
        ring_test=ring_test,
        transaction_test=transaction_test,
        abstention_rate=abstention,
        validation_abstention=validation_abstention,
        enrichment=enrichment,
        cost_sweep=cost_sweep,
        ring_threshold=ring_threshold,
        training_window={
            "start": str(frame.loc[frame["split"] == "train", "event_time"].min()),
            "end": str(frame.loc[frame["split"] == "train", "event_time"].max()),
        },
        scale_sweep=scale_sweep,
        operating_point=operating_point,
        unit=("ring" if ring_unit else "transaction"),
        notes=notes,
    )


def add_paysim_measurements(report: CorpusReport, frame: pd.DataFrame) -> None:
    """Measure the pairing baseline and what the graph adds on top of it.

    **This is the part of Phase 4 that decides whether the graph earned anything on PaySim.**
    The chain rule alone selects 99.50% of fraudulent transfers against 0.23% of legitimate
    ones, so anything laid on top of it inherits that separation for free.

    Two things pull them apart, and the first falls out of the ring metric rather than needing
    machinery of its own:

    * **Conditional separation is what the ring-level result already measures.** Every
      candidate ring exists *because* a chain edge created it -- the star filter removes
      everything else -- so the detected-ring population is exactly the pairing rule's output.
      The ring base rate is therefore the share of the rule's own hits that are real, and the
      graph's job inside that population is to rank. A ring PR-AUC equal to the ring base rate
      means the graph adds nothing the rule had not already found.
    * **The baseline's own confusion matrix**, registered as a documented comparison rather
      than discarded, so a reader can see the rule's standalone precision and recall next to
      what the graph did with it.

    The transaction-level conditional test the plan called for is not computable on this
    corpus and the reason is itself a finding: PaySim origins are near-unique, so almost every
    scored transaction abstains. It is recorded as a note rather than quietly omitted.
    """
    parameters = report.winner.parameters
    flagged = pairing_baseline(
        frame,
        step_window=int(parameters["step_window"]),
        tolerance=float(parameters["amount_tolerance"]),
    )
    test_rows = flagged.loc[flagged["split"] == "test"]
    if test_rows.empty:
        report.notes.append("PaySim test split produced no rows for the pairing baseline.")
        return

    labels = test_rows["is_fraud"].to_numpy(dtype=bool)
    baseline_scores = test_rows["pairing_flag"].to_numpy(dtype="float64")
    report.baseline = confusion_at_threshold(labels, baseline_scores, 0.5)
    report.baseline_pr_auc = pr_auc(labels, baseline_scores)

    # The increment: how far the ring score's PR-AUC sits above the no-skill floor for the
    # candidate population, which is the population the pairing rule handed over. The floor is
    # the ring base rate, so this is literally "what did ranking add to the rule's own hits".
    if report.ring_test is not None:
        # The pairing rule flags every candidate ring and ranks none of them, so its
        # ring-level behaviour *is* the no-skill ranker and its PR-AUC on this population is
        # the base rate. The increment is measured against that, paired on the same bootstrap
        # resamples via a constant-score ranker -- shifting the PR-AUC interval by a fixed
        # floor, as an earlier version did, drops the base rate's own resampling variance and
        # reports an interval narrower than the evidence supports.
        #
        # It is emphatically *not* a delta against the rule's transaction-level PR-AUC of
        # 0.9891: that number describes a different population (every transaction, not the
        # candidate rings) and subtracting the two would be comparing unlike things.
        test_rings = report.winner.rings.loc[report.winner.rings["split"] == "test"]
        labels = test_rings["is_fraud_ring"].to_numpy(dtype=bool)
        scores = test_rings["ring_risk_score"].to_numpy(dtype="float64")
        report.increment = (
            report.ring_test.pr_auc - report.ring_test.base_rate,
            *bootstrap_pr_auc_delta(labels, scores, np.zeros_like(scores), seed=RANDOM_SEED),
        )
    else:
        report.notes.append(
            "The ring-level result was not computable, so the graph's increment over the "
            "pairing baseline could not be measured."
        )


def add_surrogate_recovery(
    report: CorpusReport, frame: pd.DataFrame, factory: GraphFactory
) -> None:
    """Match detected communities against the surrogate ground-truth partition.

    Reported next to the enrichment view on purpose. The surrogate is built from labels, so it
    could in principle be defining its own success; enrichment depends on no partition at all.
    Agreement between them is what makes the ring numbers reportable.

    Three things this deliberately does **not** do. It does not match across snapshots:
    consecutive windows overlap heavily, so a pooled match lets a ring from one window
    "recover" a surrogate ring from another simply because the same accounts recur. It does not
    read the raw snapshot output: it reads the same de-duplicated ring population the ring
    metric is computed over, so the two describe one thing. And it does not count a surrogate
    ring once per window it survives into -- a ring that persists across four snapshots is one
    ring, and counting it four times on both sides of the ratio drives recall to 1.0 without
    detecting anything.
    """
    times = frame["event_time"]
    rings = report.winner.evaluation_account_rings
    test_rings = rings.loc[rings["split"] == "test"] if "split" in rings.columns else rings
    if test_rings.empty:
        report.notes.append("Surrogate ring recovery was not measurable: no test rings.")
        return

    matched_detected = 0
    detected_total = 0
    matched_truth_keys: set[str] = set()
    truth_keys: set[str] = set()
    best_overlap = OVERLAP_GRID[0]
    best_f1 = -1.0

    for overlap in OVERLAP_GRID:
        local_matched_detected = 0
        local_detected = 0
        local_matched_truth: set[str] = set()
        local_truth: set[str] = set()

        for moment, snapshot_rings in test_rings.groupby("snapshot_end", sort=True):
            window_rows = frame.loc[
                (times >= cast(pd.Timestamp, moment) - report.window)
                & (times < cast(pd.Timestamp, moment))
            ]
            detected = [
                set(group["account_id"].astype(str))
                for _, group in snapshot_rings.groupby("ring_id", sort=True)
            ]
            truth = surrogate_rings(window_rows, factory)
            if not detected or not truth:
                continue

            local_detected += len(detected)
            keys = [ring_key(sorted(members)) for members in truth]
            local_truth.update(keys)
            result = measure_recovery(detected, truth, overlap)
            local_matched_detected += result.matched_detected
            for index, members in enumerate(truth):
                if any(overlap_coefficient(c, members) >= overlap for c in detected):
                    local_matched_truth.add(keys[index])

        candidate = RecoveryResult(
            threshold=overlap,
            detected=local_detected,
            truth=len(local_truth),
            matched_detected=local_matched_detected,
            matched_truth=len(local_matched_truth),
        )
        if candidate.f1 > best_f1:
            best_f1 = candidate.f1
            best_overlap = overlap
            matched_detected = local_matched_detected
            detected_total = local_detected
            matched_truth_keys = local_matched_truth
            truth_keys = local_truth

    if not detected_total or not truth_keys:
        report.notes.append(
            "Surrogate ring recovery was not measurable: no test snapshot produced both "
            "detected rings and fraud-only components of at least the minimum ring size."
        )
        return

    report.recovery = RecoveryResult(
        threshold=best_overlap,
        detected=detected_total,
        truth=len(truth_keys),
        matched_detected=matched_detected,
        matched_truth=len(matched_truth_keys),
    )


def load_tier1_scores(
    frame: pd.DataFrame, artifact_dir: Path, registry_path: Path
) -> npt.NDArray[np.float64] | None:
    """Score the registered IEEE-CIS Tier-1 model, for the incremental-lift comparison.

    Returns None when the artefact is absent, which is logged and carried into the report as a
    stated gap rather than silently skipped.
    """
    import lightgbm as lgb

    from app.models.tier1_anomaly import ScoreNormaliser, Tier1Model
    from app.models.tier1_features import Tier1InputSpec

    if not registry_path.exists():
        return None
    entries = [
        entry
        for entry in json.loads(registry_path.read_text(encoding="utf-8"))
        if entry.get("layer") == "tier1_anomaly" and entry.get("source_dataset") == "ieee_cis"
    ]
    if not entries:
        return None
    model_id = str(entries[-1]["model_id"])
    # Through the shared guard, not by concatenation. `model_id` here is *file content* --
    # it comes out of registry.json, which append_entry does not re-validate -- so it is not a
    # trusted identifier at the point it becomes a path. Building it by hand is the exact hole
    # artifact_path was added to close, and leaving one call site out is how that class of bug
    # survives a fix.
    sidecar_path = artifact_path(model_id, artifact_dir, ".meta.json")
    booster_path = artifact_path(model_id, artifact_dir, ".txt")
    if not sidecar_path.exists() or not booster_path.exists():
        logger.warning("Tier-1 artefact %s is missing; the lift comparison is omitted", model_id)
        return None

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload = sidecar["spec"]
    spec = Tier1InputSpec(
        source_dataset=payload["source_dataset"],
        numeric_columns=tuple(payload["numeric_columns"]),
        native_categorical_columns=tuple(payload["native_categorical_columns"]),
        frequency_columns=tuple(payload["frequency_columns"]),
        categories={k: tuple(v) for k, v in payload["categories"].items()},
        encoders={k: dict(v) for k, v in payload["encoders"].items()},
        post_settlement_columns=tuple(payload["post_settlement_columns"]),
        dropped=(),
    )
    normaliser = sidecar["normaliser"]
    tier1 = Tier1Model(
        model_id=model_id,
        algorithm=sidecar["algorithm"],
        spec=spec,
        threshold=float(sidecar["threshold"]),
        normaliser=ScoreNormaliser(
            kind=normaliser["kind"], low=normaliser["low"], high=normaliser["high"]
        ),
        estimator=lgb.Booster(model_file=str(booster_path)),
        numeric_only=bool(sidecar["numeric_only"]),
    )
    return np.asarray(tier1.score_frame(frame), dtype="float64")


def add_ieee_lift(
    report: CorpusReport, frame: pd.DataFrame, artifact_dir: Path, registry_path: Path
) -> None:
    """Measure whether ``ring_risk_score`` adds anything to Tier-1 on held-out test.

    **The number Phase 5 consumes and the pitch quotes**, because IEEE-CIS carries no
    simulator artefact. The fusion is a logistic combiner fitted on **validation** -- fitting
    it on test would make the comparison meaningless, and fitting it on train would score
    Tier-1 in-sample on its own training window.
    """
    tier1_all = load_tier1_scores(frame, artifact_dir, registry_path)
    if tier1_all is None:
        report.notes.append(
            "The Tier-1 IEEE-CIS artefact was not on disk, so the incremental-lift comparison "
            "could not be run. This is a gap in the phase, not a null result."
        )
        return

    lookup = dict(zip(frame["transaction_id"].astype(str), tier1_all.tolist(), strict=True))
    scored = report.winner.scored.copy()
    scored["tier1"] = scored["transaction_id"].astype(str).map(lookup).astype("float64")
    scored = scored.loc[scored["tier1"].notna()]

    validation = scored.loc[scored["split"] == "val"]
    test = scored.loc[scored["split"] == "test"]
    if validation.empty or test.empty or test["is_fraud"].nunique() < 2:
        report.notes.append("Incremental lift was not measurable on the available splits.")
        return

    from sklearn.linear_model import LogisticRegression

    def matrix(subset: pd.DataFrame) -> npt.NDArray[np.float64]:
        ring = np.nan_to_num(
            subset["ring_risk_score"].to_numpy(dtype="float64"), nan=ABSTAINED_RANK_SENTINEL
        )
        scoreable = (~subset["ring_risk_score"].isna()).to_numpy(dtype="float64")
        return np.column_stack([subset["tier1"].to_numpy(dtype="float64"), ring, scoreable])

    combiner = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    combiner.fit(matrix(validation), validation["is_fraud"].to_numpy(dtype=bool))

    labels = test["is_fraud"].to_numpy(dtype=bool)
    tier1_only = test["tier1"].to_numpy(dtype="float64")
    fused = np.asarray(combiner.predict_proba(matrix(test))[:, 1], dtype="float64")
    report.tier1_pr_auc = pr_auc(labels, tier1_only)
    report.fused_pr_auc = pr_auc(labels, fused)
    report.fused_delta = (
        report.fused_pr_auc - report.tier1_pr_auc,
        *bootstrap_pr_auc_delta(labels, fused, tier1_only, seed=RANDOM_SEED),
    )


def profile_false_negatives(report: CorpusReport) -> list[str]:
    """Describe what the layer missed, from the false negatives actually observed.

    ml-evaluation-standards section 4 asks for this to be written from real misses rather than
    from imagination, so every line here is a measured share of the test split's own misses.
    """
    if report.unit == "ring":
        rings = report.winner.rings
        test_rings = rings.loc[rings["split"] == "test"] if "split" in rings.columns else rings
        if test_rings.empty:
            return ["No test rings were detected, so no false negatives could be profiled."]
        positives = test_rings.loc[test_rings["is_fraud_ring"]]
        if positives.empty:
            return ["The test split carried no fraud-bearing rings."]
        missed = positives.loc[positives["ring_risk_score"] < report.threshold]
        lines: list[str] = [
            f"{len(missed):,} of {len(positives):,} fraud-bearing test rings "
            f"({len(missed) / len(positives):.1%}) scored below the "
            f"{report.threshold:.4f} operating threshold.",
            f"Every transaction whose account is not in a detected ring is abstained on: "
            f"{report.abstention_rate:.1%} of test transactions. On PaySim that is almost all "
            "of them, because origins are near-unique and a first-time account has no links "
            "to reason about. Tier-3 cannot catch a one-shot fraud by a previously unseen "
            "account at any threshold -- Tier-1 is the only layer covering that case.",
        ]
        if not missed.empty and "size" in missed.columns:
            lines.append(
                f"Missed fraud-bearing rings hold a median of {missed['size'].median():.0f} "
                f"accounts against {positives['size'].median():.0f} for the population."
            )
        return lines

    scored = report.winner.scored
    test = scored.loc[scored["split"] == "test"] if "split" in scored.columns else scored
    if test.empty:
        return ["No test transactions were scored, so no false negatives could be profiled."]

    fraud = test.loc[test["is_fraud"]]
    if fraud.empty:
        return ["The test split carried no positives on this corpus."]

    lines = []
    abstained = fraud["ring_risk_score"].isna()
    lines.append(
        f"{abstained.mean():.1%} of test fraud ({int(abstained.sum()):,} of {len(fraud):,}) "
        "was abstained on outright -- the account appeared in no ring at all, so Tier-3 could "
        "not have caught it at any threshold."
    )
    scoreable = fraud.loc[~abstained]
    if not scoreable.empty:
        below = scoreable["ring_risk_score"] < report.threshold
        lines.append(
            f"{below.mean():.1%} of scoreable test fraud sat in a ring but scored below the "
            f"{report.threshold:.4f} operating threshold."
        )
        if "ring_size" in scoreable.columns and not scoreable.empty:
            missed = scoreable.loc[below]
            if not missed.empty and "transaction_type" in missed.columns:
                counts = missed["transaction_type"].value_counts(normalize=True)
                if not counts.empty:
                    top = ", ".join(f"{name} {share:.0%}" for name, share in counts.head(3).items())
                    lines.append(f"Missed-but-scoreable fraud is composed of: {top}.")
    return lines


def ring_flags_from_frame(rings: pd.DataFrame, snapshot_end: pd.Timestamp) -> list[RingFlag]:
    """Rebuild :class:`RingFlag` objects from one snapshot's scored account-ring rows.

    Exists so the served score table is produced by
    :func:`app.models.tier3_graph.build_score_table` -- the same function the layer's own API
    uses -- rather than by a third hand-written copy of the max-over-rings rule. An earlier
    version reimplemented that rule in three places and the module's only guarantee that they
    agreed was a comment; :func:`test_offline_and_served_score_tables_agree` now pins it.
    """
    flags: list[RingFlag] = []
    for ring_id, group in rings.groupby("ring_id", sort=True):
        flags.append(
            RingFlag(
                ring_id=str(ring_id),
                member_account_ids=tuple(str(value) for value in group["account_id"]),
                member_scores=tuple(float(value) for value in group["ring_risk_score"]),
                size=int(len(group)),
                ring_risk_score=float(group["ring_risk_score"].max()),
                density=float(group.get("ring_density", pd.Series([0.0])).iloc[0]),
                max_degree_centrality=float(group["account_degree_centrality"].max()),
                max_betweenness=float(group["account_betweenness"].max()),
                betweenness_exact=bool(group["betweenness_exact"].all()),
                chain_edge_count=int(group.get("ring_chain_edge_count", pd.Series([0.0])).iloc[0]),
                entity_count=int(group.get("ring_entity_count", pd.Series([0.0])).iloc[0]),
                snapshot_end=snapshot_end.to_pydatetime(),
            )
        )
    return flags


def build_served_model(
    report: CorpusReport,
    model_id: str,
    frame: pd.DataFrame,
    factory: GraphFactory,
    *,
    entity_anonymization_key: bytes,
) -> Tier3Model:
    """Freeze the most recent snapshot's rings into the score table serving reads.

    Serving reads a table, not a graph. The most recent snapshot is the one a request at
    "now" would be scored against, so it is the one that gets frozen; every earlier snapshot
    existed only to produce the evaluation.

    ``frame`` and ``factory`` (the winning configuration's, not any factory) exist for exactly
    one reason: :func:`run_snapshots` never kept the graph itself, only the feature rows it
    produced, so the topology :func:`export_ring_edges` needs for the Phase 8 network view has
    to be rebuilt. Rebuilding costs one extra snapshot -- the same ``window_rows`` slice and
    ``factory().insert(...).snapshot(...)`` call :func:`run_snapshots` already made for this one
    boundary -- not the whole sweep, and reproduces the exact same graph deterministically.

    ``entity_anonymization_key`` is ``Settings.entity_anonymization_key``, UTF-8 encoded by the
    caller -- see :func:`app.models.tier3_graph._anonymized_entity_id` for why an unkeyed hash
    would not actually anonymize an IEEE-CIS shared-entity fingerprint. Required, not defaulted:
    a training run that forgot to pass it should fail loudly rather than silently fall back to
    an empty or fixed key.
    """
    account_rings = report.winner.account_rings
    latest = cast(pd.Timestamp, account_rings["snapshot_end"].max())
    current = account_rings.loc[account_rings["snapshot_end"] == latest]

    flags = ring_flags_from_frame(current, latest)
    scores, ring_of, ring_sizes = build_score_table(flags)

    # Every account the snapshot held, not only the scoreable ones. Without it the two
    # abstention reasons collapse into one and an account Tier-3 *did* see, sitting in a
    # component below the minimum ring size, is recorded in the audit trail as one it has
    # never seen any links for.
    seen = frozenset(
        str(account) for account in report.winner.seen_accounts.get(latest, ())
    ) | frozenset(scores)

    times = frame["event_time"]
    window_rows = frame.loc[(times >= latest - report.window) & (times < latest)]
    graph = factory()
    graph.insert(window_rows)
    snapshot = graph.snapshot(latest.to_pydatetime())
    communities = detect_communities(snapshot)
    topology = export_ring_edges(snapshot, communities, key=entity_anonymization_key)
    # Restricted to scoreable rings (those `build_score_table` actually assigned accounts to) --
    # the same population `ring_sizes`/`ring_of` cover, not every community this snapshot
    # detected structurally.
    scoreable_ring_ids = set(ring_of.values())
    ring_nodes = {
        ring_id: nodes
        for ring_id, (nodes, _edges) in topology.items()
        if ring_id in scoreable_ring_ids
    }
    ring_edges = {
        ring_id: edges
        for ring_id, (_nodes, edges) in topology.items()
        if ring_id in scoreable_ring_ids
    }

    return Tier3Model(
        model_id=model_id,
        source_dataset=report.source,
        threshold=report.threshold,
        scores=scores,
        ring_of=ring_of,
        ring_sizes=ring_sizes,
        snapshot_end=latest.to_pydatetime(),
        seen_accounts=seen,
        scorer=report.winner.scorer,
        parameters=dict(report.winner.parameters),
        ring_nodes=ring_nodes,
        ring_edges=ring_edges,
    )


def register(
    report: CorpusReport, model: Tier3Model, registry_path: Path, artifact_dir: Path
) -> RegistryEntry:
    """Append this corpus's permanent record. Append, never overwrite."""
    artifact = model.save(artifact_dir)
    # The structural input definition -- everything that changes what the graph looks like.
    # The operating threshold is deliberately NOT in here: it is an output of fitting, and
    # models/README.md requires feature_version to reconstruct the inputs the model saw. With
    # it included, two models over identical graphs got different `gv_` hashes.
    structural = dict(report.winner.parameters)
    structural.update(
        {
            "cadence_hours": report.cadence.total_seconds() / 3600.0,
            "window_hours": report.window.total_seconds() / 3600.0,
            "min_ring_size": MIN_RING_SIZE,
        }
    )
    parameters = {**structural, "operating_threshold": report.threshold}

    heldout: dict[str, Any] = {}
    if report.transaction_test is not None:
        heldout = report.transaction_test.to_dict()
    if report.ring_test is not None:
        heldout["ring_level"] = report.ring_test.to_dict()
    heldout["abstention_rate"] = round(report.abstention_rate, 6)
    heldout["validation_abstention_rate"] = round(report.validation_abstention, 6)
    heldout["unit_of_analysis"] = report.unit
    if report.operating_point:
        heldout["capacity_constrained_operating_point"] = report.operating_point
    heldout["latency"] = report.latency
    if report.recovery is not None:
        heldout["surrogate_ring_recovery"] = {
            "overlap_threshold": report.recovery.threshold,
            "detected_rings": report.recovery.detected,
            "surrogate_rings": report.recovery.truth,
            "precision": round(report.recovery.precision, 6),
            "recall": round(report.recovery.recall, 6),
            "f1": round(report.recovery.f1, 6),
            "caveat": "Ground truth is a surrogate derived from transaction labels. PaySim "
            "ships no ring or agent identifier.",
        }
    if report.enrichment is not None:
        heldout["enrichment"] = {
            "k": report.enrichment.k,
            "precision_at_k": round(report.enrichment.precision_at_k, 6),
            "ring_base_rate": round(report.enrichment.base_rate, 6),
            "lift": round(report.enrichment.lift, 3),
        }
    if report.conditional is not None:
        heldout["conditional_separation"] = report.conditional.to_dict()
    if report.increment is not None:
        heldout["ring_lift_over_pairing_rule"] = {
            "delta_pr_auc": round(report.increment[0], 6),
            "ci95": [round(report.increment[1], 6), round(report.increment[2], 6)],
            "measured_against": "the pairing rule's ring-level behaviour, which is to flag "
            "every candidate ring and rank none -- so its PR-AUC on this population equals "
            "the ring base rate. NOT a delta against the rule's transaction-level PR-AUC in "
            "baseline_comparison, which describes a different population.",
        }
    if report.fused_delta is not None:
        heldout["incremental_lift_over_tier1"] = {
            "tier1_pr_auc": round(float(report.tier1_pr_auc or 0.0), 6),
            "fused_pr_auc": round(float(report.fused_pr_auc or 0.0), 6),
            "delta_pr_auc": round(report.fused_delta[0], 6),
            "ci95": [round(report.fused_delta[1], 6), round(report.fused_delta[2], 6)],
        }

    # A leak-suspicious number reaches the permanent record with its caveat attached, not
    # only the rendered report. Prepended so it is the first thing a reader of registry.json
    # sees -- Phase 2 added this to Tiers 1 and 2 after a PaySim PR-AUC of 0.9999 was recorded
    # with no warning on it at all, and the same wire has to hold here.
    caveats: list[str] = []
    for label, result in (
        ("ring-level", report.ring_test),
        ("per-transaction", report.transaction_test),
    ):
        if result is not None and result.lift_over_no_skill < 1.0:
            # Not a leak, the opposite: a headline that loses to a coin toss. It reached the
            # rendered report but not the registry, and registry.json is the audit artefact.
            caveats.append(
                f"BELOW NO-SKILL. The {label} test PR-AUC {result.pr_auc:.4f} is under its "
                f"own no-skill floor of {result.base_rate:.4f} "
                f"({result.lift_over_no_skill:.3f}x) -- ranking by this score is worse than "
                "random on this split. It is recorded because dropping the numbers that did "
                "not work is what ml-evaluation-standards section 4 forbids, not because it "
                "is usable."
            )
        if result is not None and result.is_leak_suspicious:
            diagnosis = (
                " On this corpus the diagnosis is already known: the candidate rings are the "
                "output of an amount-and-step pairing rule that is itself a "
                "99.50%-against-0.23% classifier, so the ring population starts high by "
                "construction."
                if report.source == "paysim"
                else ""
            )
            caveats.append(
                f"DO NOT QUOTE AS A HEADLINE. The {label} test PR-AUC {result.pr_auc:.4f} "
                f"exceeds {LEAK_SUSPICION_PR_AUC} on fraud data, which "
                "ml-evaluation-standards section 4 treats as a leak signal until disproven."
                f"{diagnosis} The interpretable figure is the "
                f"{result.lift_over_no_skill:.3f}x lift over a {result.base_rate:.3f} base "
                "rate, not the headline. See BUILD_LOG.md Phase 4."
            )

    if report.fused_delta is not None and report.fused_delta[2] < 0.0:
        caveats.append(
            f"NEGATIVE FUSION DELTA. Adding this layer's score to Tier-1 moved held-out PR-AUC "
            f"by {report.fused_delta[0]:+.4f} with a 95% CI of "
            f"[{report.fused_delta[1]:+.4f}, {report.fused_delta[2]:+.4f}], an interval that "
            "excludes zero on the negative side. Phase 5 must not treat this score as a "
            "proven input; see BUILD_LOG.md Phase 4, finding 4."
        )

    baselines: list[dict[str, Any]] = [
        candidate.summary() for candidate in report.candidates if candidate is not report.winner
    ]
    if report.baseline is not None:
        baselines.append(
            {
                "baseline": "amount-and-step pairing rule alone, no graph",
                "pr_auc": round(float(report.baseline_pr_auc or 0.0), 6),
                "precision": round(report.baseline.precision, 6),
                "recall": round(report.baseline.recall, 6),
                "f1": round(report.baseline.f1, 6),
                "confusion_matrix": report.baseline.to_dict(),
                "note": "This rule is what makes PaySim's graph multi-hop at all, and it "
                "separates fraud on its own. Tier-3's PaySim claim is the increment over it, "
                "not the absolute number.",
            }
        )

    # Which earlier entries this one replaces. The registry is append-only, so a phase that
    # was re-run leaves its superseded attempts in the file; without this a reader has to go to
    # BUILD_LOG prose to learn which of ten Tier-3 entries is the live one.
    superseded = [
        str(previous["model_id"])
        for previous in read_registry(registry_path)
        if previous.get("layer") == "tier3_graph"
        and previous.get("source_dataset") == report.source
    ]

    entry = RegistryEntry(
        model_id=model.model_id,
        layer="tier3_graph",
        algorithm="louvain_centrality_logistic",
        source_dataset=report.source,
        feature_version=graph_feature_version(report.source, structural),
        training_window=report.training_window,
        hyperparameters=parameters,
        random_seed=RANDOM_SEED,
        heldout_test=heldout,
        baseline_comparison=baselines,
        artifact=f"artifacts/{artifact.name}",
        notes=[
            *caveats,
            *(
                [
                    f"SUPERSEDES {len(superseded)} earlier {report.source} Tier-3 "
                    f"entr{'y' if len(superseded) == 1 else 'ies'} "
                    f"({', '.join(superseded)}). The registry is append-only; those were "
                    "produced by runs whose defects are recorded in BUILD_LOG.md Phase 4."
                ]
                if superseded
                else []
            ),
            *report.notes,
            *report.false_negatives,
        ],
    )
    append_entry(entry, registry_path)
    return entry


def interval_verdict(
    delta: float,
    low: float,
    high: float,
    *,
    gain: str,
    loss: str,
    tie: str,
) -> str:
    """Return the right verdict for a bootstrap interval, including when it is negative.

    Written as one helper because the obvious two-branch version -- "straddles zero" against
    "excludes zero, therefore a win" -- calls a significant *degradation* a win, which is
    exactly the kind of sentence this project exists not to print.
    """
    if low <= 0.0 <= high:
        return tie
    return gain if delta > 0.0 else loss


def render_report(reports: Sequence[CorpusReport], sample: int | None) -> str:
    """Render the phase report in the shape ml-evaluation-standards prescribes."""
    lines: list[str] = [
        "# Tier-3 — network graph abuse-ring detection (Phase 4)",
        "",
        "Generated by `python -m app.models.train_tier3`. Every headline below is measured on "
        "the **held-out test split only**; validation numbers appear solely where a selection "
        "was made and are labelled as such.",
        "",
    ]
    if sample is not None:
        lines += [
            f"> **Sampled run — not reportable.** Only the earliest {sample:,} rows per corpus "
            "were used. Re-run without `--sample` before quoting anything here.",
            "",
        ]

    lines += [
        "## What the graph actually is, measured before it was built",
        "",
        "PaySim's observed money-flow graph is a **star forest**: 99.95% of origins have "
        "degree 1, the maximum is 2, and only 341 of 2,291,054 nodes are both an origin and a "
        "destination. Louvain over that returns `groupby(nameDest)` and betweenness restates "
        "degree. The inferred transfer-to-cash-out chain edge is what creates multi-hop "
        "structure, and Phase 1 measured that it cannot be built on account names: 0.00% of "
        "fraudulent transfers have a `nameDest` reappearing as a fraudulent cash-out's "
        "`nameOrig`.",
        "",
        "**That chain rule is also a simulator artefact.** Exact-amount same-step matching "
        "selects 99.50% of fraudulent transfers against 0.23% of legitimate ones, median one "
        "candidate partner. It is the generative rule read back out. Every PaySim number "
        "below is therefore reported against that rule as an explicit baseline, and the "
        "reportable claim is the increment over it — never the absolute figure.",
        "",
        "IEEE-CIS has no money-flow edge at all, so its rings come from shared-entity "
        "co-occurrence. Its single columns are buckets rather than identifiers — `card4` and "
        "`card6` hold four values each, with 98,466 accounts on one — so entities are "
        "composite fingerprints under a degree cap.",
        "",
    ]

    for report in reports:
        lines += [f"## {report.source}", ""]
        lines += [
            f"- Configuration selected on **validation**: `{report.winner.label}` "
            f"(validation ring PR-AUC {report.winner.val_ring_pr_auc:.4f} over "
            f"{report.winner.val_rings:,} rings at a {report.winner.val_ring_base_rate:.3f} "
            "ring base rate)",
            f"- Snapshot cadence {report.cadence}, trailing window {report.window}. Maximum "
            f"score staleness is one cadence, {report.cadence}.",
            f"- **Unit of analysis: the {report.unit}.** Chosen on measured coverage, not "
            "declared in advance — see the note below where it is not the transaction.",
            f"- Abstention rate on test: **{report.abstention_rate:.1%}** of transactions had "
            "no ring and were abstained on, not scored as clean.",
            "",
        ]
        if report.source == "ieee_cis":
            lines += [
                "> **Most of this corpus's ring signal is circular with the account id.** "
                "`account_id` is constructed as `c{card1}_a{addr1}_d{d1n}`, and the winning "
                "fingerprint is `(card1, card2, card5, addr1)` — so two accounts sharing it "
                "differ *only* in `d1n`, the inferred account-start day. That is either one "
                "card fragmented into several inferred identities or one card genuinely "
                "presenting as several clients, and the graph cannot tell them apart. The "
                "device-only control, which shares no column with the account id, was run as "
                "a third configuration and scores far lower (see the table above). Read the "
                "headline with that gap in mind.",
                "",
            ]

        if len(report.candidates) > 1:
            lines += [
                "### Configurations compared (validation)",
                "",
                "| configuration | val ring PR-AUC | rings | base rate |",
                "|---|---|---|---|",
            ]
            for candidate in report.candidates:
                marker = " **(selected)**" if candidate is report.winner else ""
                lines.append(
                    f"| `{candidate.label}`{marker} | {candidate.val_ring_pr_auc:.4f} | "
                    f"{candidate.val_rings:,} | {candidate.val_ring_base_rate:.3f} |"
                )
            lines.append("")

        if report.ring_test is not None:
            lines += [
                "### Ring-level classification — held-out test",
                "",
                "Unit of analysis is a **candidate ring**, declared rather than implied, the "
                "same move Phase 3 made in evaluating Tier-2 per account. Every detected ring "
                "is fraud-bearing or clean, so the confusion matrix is fully defined.",
                "",
                *(
                    [
                        "**This is also the conditional-separation measurement.** Every "
                        "candidate ring exists because a chain edge created it — the star "
                        "filter removes everything else — so the detected-ring population *is* "
                        "the pairing rule's output. The ring base rate below is the share of "
                        "that rule's own hits which are real, and the graph's job inside the "
                        "population is to rank. A PR-AUC at the base rate would mean the graph "
                        "found nothing the rule had not already found.",
                        "",
                    ]
                    if report.source == "paysim"
                    else [
                        "This corpus has **no chain edge and no pairing rule** — it carries no "
                        "money-flow edge at all, so its rings come from shared-entity "
                        "co-occurrence and the star filter is deliberately skipped. There is "
                        "therefore no conditional-separation reading here: the base rate below "
                        "is simply the share of detected communities that are fraud-bearing.",
                        "",
                    ]
                ),
                "```",
                report.ring_test.render(),
                "```",
                "",
            ]

        if report.transaction_test is not None:
            lines += [
                "### Per-transaction ring risk — held-out test",
                "",
                "```",
                report.transaction_test.render(),
                "```",
                "",
            ]
            if report.transaction_test.lift_over_no_skill < 1.0:
                lines += [
                    "> **This result does not clear its own no-skill floor.** Ranking "
                    "transactions by ring risk alone is worse than random on this split. The "
                    "layer's demonstrated value is at ring level, not per transaction, and "
                    "this line is kept rather than dropped because a metrics section that "
                    "shows only the numbers that worked is not an honest one.",
                    "",
                ]

        if report.baseline is not None:
            lines += [
                "### Baseline: the pairing rule alone, no graph",
                "",
                f"PR-AUC {float(report.baseline_pr_auc or 0.0):.4f} · "
                f"precision {report.baseline.precision:.4f} · "
                f"recall {report.baseline.recall:.4f} · F1 {report.baseline.f1:.4f}",
                "",
                "```",
                "Confusion matrix        Predicted",
                "                     neg        pos",
                f"Actual  neg  {report.baseline.true_negatives:>9,}  "
                f"{report.baseline.false_positives:>9,}",
                f"        pos  {report.baseline.false_negatives:>9,}  "
                f"{report.baseline.true_positives:>9,}",
                "```",
                "",
            ]

        if report.increment is not None:
            delta, low, high = report.increment
            verdict = interval_verdict(
                delta,
                low,
                high,
                gain="The interval stays above the floor, so the ranking the graph adds "
                "inside the pairing rule's own hits is real at this sample size.",
                loss="The interval sits **below** the floor: ranking the pairing rule's hits "
                "by ring risk is worse than not ranking them at all. That is a negative "
                "result and it is reported as one.",
                tie="The interval reaches the no-skill floor: on this corpus the graph adds "
                "**nothing measurable** to what the pairing rule had already selected, and "
                "that is the finding rather than a disappointment to be buried.",
            )
            lines += [
                "### What the graph adds over the pairing rule",
                "",
                "The candidate rings are the pairing rule's output, so the rule's own "
                "performance on this population is the no-skill floor — it flags all of them "
                "and ranks none. The graph's contribution is how far ring PR-AUC sits above "
                "that floor.",
                "",
                f"Ring PR-AUC − ring base rate = **{delta:+.4f}** "
                f"(95% CI {low:+.4f} to {high:+.4f}, bootstrapped over rings)",
                "",
                verdict,
                "",
            ]

        if report.fused_delta is not None:
            delta, low, high = report.fused_delta
            verdict = interval_verdict(
                delta,
                low,
                high,
                gain="The interval excludes zero on the positive side: the ring score carries "
                "ranking power Tier-1 does not have, and Phase 5 should read it as a real "
                "input.",
                loss="The interval excludes zero on the **negative** side: adding "
                "`ring_risk_score` to Tier-1 through a linear combiner makes the ranking "
                "*worse*, not better. Phase 5 must not treat this as a proven input. What the "
                "graph demonstrably does carry is ring-level structure (see the ring result "
                "above); what it does not carry is a per-transaction signal that improves "
                "Tier-1 under this fusion.",
                tie="The interval straddles zero: `ring_risk_score` does not measurably "
                "improve on Tier-1 alone. Phase 5 should treat it as an unproven input.",
            )
            lines += [
                "### Incremental lift over Tier-1 — the number Phase 5 consumes",
                "",
                f"Tier-1 alone PR-AUC {float(report.tier1_pr_auc or 0.0):.4f} → "
                f"Tier-1 + ring score {float(report.fused_pr_auc or 0.0):.4f} "
                f"(**{delta:+.4f}**, 95% CI {low:+.4f} to {high:+.4f})",
                "",
                "The combiner is a logistic regression fitted on **validation**. Fitting it on "
                "test would void the comparison; fitting it on train would score Tier-1 "
                "in-sample on its own training window.",
                "",
                verdict,
                "",
            ]

        if report.recovery is not None:
            recovery = report.recovery
            lines += [
                "### Surrogate ring recovery",
                "",
                "> **This ground truth is a surrogate.** PaySim ships `isFraud` per "
                "transaction and no ring or agent identifier, so a true ring is constructed "
                "as a connected component of the fraud-only induced graph. It is a defensible "
                "construction and it is still a construction.",
                "",
                f"At overlap coefficient ≥ {recovery.threshold:.1f} (selected on validation): "
                f"{recovery.matched_detected:,} of {recovery.detected:,} detected rings "
                f"matched a surrogate ring (precision {recovery.precision:.3f}); "
                f"{recovery.matched_truth:,} of {recovery.truth:,} surrogate rings were "
                f"recovered (recall {recovery.recall:.3f}); F1 {recovery.f1:.3f}.",
                "",
            ]

        if report.enrichment is not None:
            lines += [
                "### Enrichment — the check that needs no surrogate",
                "",
                report.enrichment.describe(),
                "",
                *(
                    [
                        "This view depends on no ground-truth partition. Where it and the "
                        "surrogate recovery disagree, the surrogate definition is driving the "
                        "result and the ring headline is withdrawn.",
                        "",
                    ]
                    if report.recovery is not None
                    else [
                        "This view depends on no ground-truth partition, which is the whole of "
                        "the check available here: **no surrogate recovery was computed for "
                        "this corpus**, because the surrogate partition is built from PaySim's "
                        "money-flow edges and this corpus has none. The corroboration is "
                        "therefore weaker than on PaySim, where the two views agree.",
                        "",
                    ]
                ),
            ]

        if report.scale_sweep or report.cost_sweep:
            lines += [
                "### Cost sensitivity",
                "",
                "ml-evaluation-standards section 3: a recommended threshold is justified by "
                "cost, and the recommendation has to survive the cost assumptions being wrong. "
                f"Both sweeps re-choose the threshold at each setting, on the **validation** "
                f"split, and are measured per {report.unit}.",
                "",
                "> **The two sweeps are not equally informative, and the first cannot be.** "
                "Scaling both cost parameters by the same factor multiplies total cost by that "
                "factor and therefore cannot move the argmin — the threshold is identical at "
                "0.5x, 1.0x and 1.5x by construction, not as evidence of robustness. It is "
                "reported because section 3 names a +/-50% analysis, and it is labelled here "
                "so that its flatness is not read as a result. The **review-cost sweep below "
                "is the informative one**: it varies the false-positive-to-false-negative "
                "*ratio*, which is what the threshold actually depends on.",
                "",
            ]
            if report.scale_sweep:
                lines += [
                    "```",
                    render_sensitivity(report.scale_sweep, "Both costs scaled (+/-50%)"),
                    "```",
                    "",
                ]
            if report.cost_sweep:
                lines += [
                    "```",
                    render_sensitivity(
                        report.cost_sweep, "Review cost raised against a fixed fraud cost"
                    ),
                    "```",
                    "",
                ]

        if report.operating_point:
            point = report.operating_point
            lines += [
                "### Recommended operating point",
                "",
                f"Threshold {point['threshold']:.6f} per {point['unit']} — flags "
                f"**{point['flag_rate']:.2%}** of the population against a "
                f"{point['review_capacity_cap']:.1%} review-capacity cap, at precision "
                f"{point['precision']:.4f} / recall {point['recall']:.4f} / F1 "
                f"{point['f1']:.4f}.",
                "",
            ]

        if report.latency:
            lines += [
                "### Latency",
                "",
                f"p50 {report.latency.get('p50_ms', 0):.3f}ms · "
                f"p95 {report.latency.get('p95_ms', 0):.3f}ms · "
                f"budget {report.latency.get('budget_p95_ms', 0):.0f}ms",
                "",
                "Serving is a dictionary lookup into a precomputed score table. No graph "
                "algorithm runs in the request path, which is what makes Phase 7's Tier-3 "
                "timeout and degraded-mode fallback implementable.",
                "",
            ]

        if report.false_negatives:
            lines += ["### What this does NOT catch", ""]
            lines += [f"- {line}" for line in report.false_negatives]
            lines.append("")

        if report.notes:
            lines += ["### Notes and gaps", ""]
            lines += [f"- {line}" for line in report.notes]
            lines.append("")

    lines += [
        "## Visualisations",
        "",
        "Both drawn from the PaySim money-flow graph. Red nodes are accounts touching labelled "
        "fraud, blue are clean, grey squares are shared entities, and a thick red edge is an "
        "inferred transfer-to-cash-out chain link.",
        "",
        "The two may come from **different snapshot windows**, and each caption names its own. "
        "The detected ring is the riskiest in the most recent window; the clean cluster is the "
        "lowest-scoring ring in the most recent window that holds one with no fraud-touching "
        "member at all. That search is deliberate -- PaySim's ring base rate is well above a "
        "half, and "
        "the final window was measured to hold no clean ring at all, so taking the "
        "lowest-scoring ring in it would have put the word *clean* over a mostly-red graph.",
        "",
        "![detected ring](tier3_ring_detected.png)",
        "",
        "![clean cluster](tier3_cluster_clean.png)",
        "",
    ]
    return "\n".join(lines)


def run(
    processed_dir: Path,
    notebook_dir: Path,
    registry_path: Path,
    artifact_dir: Path,
    *,
    entity_anonymization_key: bytes,
    sample: int | None = None,
    corpora: Sequence[SourceDataset] = ("paysim", "ieee_cis"),
) -> list[CorpusReport]:
    """Run Phase 4 end to end over both corpora."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Tier-3 training run, random seed %d", RANDOM_SEED)
    notebook_dir.mkdir(parents=True, exist_ok=True)
    reports: list[CorpusReport] = []

    for source in corpora:
        frame = load_splits(processed_dir, source, sample)
        logger.info("[%s] %d rows, %d positives", source, len(frame), int(frame.is_fraud.sum()))
        median_amount = float(frame["amount"].median())

        configurations: list[tuple[str, dict[str, Any], GraphFactory]]
        if source == "paysim":
            cost_model = CostModel.scaled_to(median_amount, PAYSIM_COST_UNITS)
            configurations = [
                (
                    f"step_window={window},tolerance={tolerance}",
                    {"step_window": window, "amount_tolerance": tolerance},
                    paysim_factory(window, tolerance),
                )
                for window in PAYSIM_STEP_WINDOW_GRID
                for tolerance in PAYSIM_TOLERANCE_GRID
            ]
            report = run_corpus(
                source,
                frame,
                configurations,
                cadence=PAYSIM_CADENCE,
                window=PAYSIM_WINDOW,
                cost_model=cost_model,
            )
            add_paysim_measurements(report, frame)
            add_surrogate_recovery(
                report,
                frame,
                paysim_factory(
                    int(report.winner.parameters["step_window"]),
                    float(report.winner.parameters["amount_tolerance"]),
                ),
            )
        else:
            cost_model = CostModel.scaled_to(median_amount, IEEE_COST_UNITS)
            configurations = [
                (
                    f"entity_cap={cap},all_fingerprints",
                    {"max_entity_degree": cap, "fingerprints": "all"},
                    ieee_factory(cap),
                )
                for cap in IEEE_ENTITY_CAP_GRID
            ] + [
                (
                    "entity_cap=50,non_circular_only",
                    {"max_entity_degree": 50, "fingerprints": "non_circular"},
                    ieee_factory(50, NON_CIRCULAR_FINGERPRINTS),
                )
            ]
            report = run_corpus(
                source,
                frame,
                configurations,
                cadence=IEEE_CADENCE,
                window=IEEE_WINDOW,
                cost_model=cost_model,
            )
            add_ieee_lift(report, frame, artifact_dir, registry_path)

        winner_factory = next(
            candidate_factory
            for label, _parameters, candidate_factory in configurations
            if label == report.winner.label
        )
        model_id = build_model_id("tier3-graph", "louvain", source)
        model = build_served_model(
            report,
            model_id,
            frame,
            winner_factory,
            entity_anonymization_key=entity_anonymization_key,
        )
        report.model = model
        report.false_negatives = profile_false_negatives(report)

        sample_rows = frame.tail(200)
        transactions = _sample_transactions(sample_rows, source)
        if transactions:
            report.latency = benchmark_latency(model, transactions)

        if sample is None:
            register(report, model, registry_path, artifact_dir)
        else:
            report.notes.append("Sampled run: nothing was written to models/registry.json.")
        reports.append(report)

    _draw_visualisations(reports, notebook_dir, processed_dir, sample)
    (notebook_dir / "tier3_report.md").write_text(render_report(reports, sample), encoding="utf-8")
    logger.info("wrote %s", notebook_dir / "tier3_report.md")
    return reports


def _sample_transactions(frame: pd.DataFrame, source: SourceDataset) -> list[Any]:
    """Build a handful of :class:`TransactionFeatures` for the latency benchmark."""
    from decimal import Decimal

    from app.data.schema import TransactionFeatures

    transactions: list[Any] = []
    if frame.empty:
        return transactions

    identifiers = frame["transaction_id"].astype(str).tolist()
    moments = pd.to_datetime(frame["event_time"]).dt.to_pydatetime().tolist()
    amounts = frame["amount"].to_numpy(dtype="float64").tolist()
    accounts = frame["account_id"].astype(str).tolist()
    versions = frame["feature_version"].astype(str).tolist()
    counterparties = (
        frame["counterparty_id"].tolist()
        if source == "paysim" and "counterparty_id" in frame.columns
        else [None] * len(frame)
    )
    types = (
        frame["transaction_type"].tolist()
        if "transaction_type" in frame.columns
        else [None] * len(frame)
    )

    for index in range(len(frame)):
        try:
            transactions.append(
                TransactionFeatures(
                    transaction_id=identifiers[index],
                    source_dataset=source,
                    event_time=moments[index],
                    amount=Decimal(str(round(amounts[index], 4))),
                    account_id=accounts[index],
                    counterparty_id=(
                        str(counterparties[index]) if pd.notna(counterparties[index]) else None
                    ),
                    transaction_type=(str(types[index]) if pd.notna(types[index]) else None),
                    feature_version=versions[index],
                )
            )
        except Exception:  # noqa: BLE001 - a malformed sample row must not fail the phase
            continue
    return transactions


def _draw_visualisations(
    reports: Sequence[CorpusReport],
    notebook_dir: Path,
    processed_dir: Path,
    sample: int | None,
) -> None:
    """Save one detected ring and one genuinely clean cluster — the phase's visual gate.

    The two images have to be *visibly* different, which means the clean one has to actually be
    clean. Selecting it as "the lowest-scoring ring in the most recent snapshot" does not
    achieve that: PaySim's ring base rate is above 0.75, and the final snapshot was measured to
    hold 885 rings with **none** free of labelled fraud. So the clean side searches snapshots
    backwards until it finds a ring whose every member is untouched by fraud, and the caption
    names the window it came from. If no snapshot has one, the caption says so outright rather
    than calling a mostly-fraudulent ring clean.
    """
    paysim = next((report for report in reports if report.source == "paysim"), None)
    if paysim is None:
        return
    frame = load_splits(processed_dir, "paysim", sample)
    account_rings = paysim.winner.account_rings
    if account_rings.empty:
        return

    per_ring = account_rings.groupby(["snapshot_end", "ring_id"]).agg(
        score=("ring_risk_score", "max"),
        members=("account_id", "size"),
        tainted=("account_is_fraudulent", "sum"),
    )
    latest = cast(pd.Timestamp, account_rings["snapshot_end"].max())

    # Detected: the riskiest ring in the most recent snapshot, which is the one a request at
    # "now" would be scored against.
    # Many rings tie at the top score, so "the highest-scoring ring" alone picks one of them
    # arbitrarily and can land on a four-node path that illustrates nothing. Ties are broken by
    # ring size, a structural property, purely so the picture is legible. This chooses which
    # ring is *drawn*; it selects no metric and changes no reported number.
    current = cast(pd.DataFrame, per_ring.loc[latest]).sort_values(
        ["score", "members"], ascending=[False, False]
    )
    detected = (latest, str(current.index[0]), float(current["score"].iloc[0]))

    # Clean: the most recent snapshot holding a ring with no fraud-touching member at all.
    clean: tuple[pd.Timestamp, str, float] | None = None
    for moment in sorted(per_ring.index.get_level_values(0).unique(), reverse=True):
        spotless = per_ring.loc[moment]
        spotless = spotless.loc[spotless["tainted"] == 0]
        if not spotless.empty:
            ordered = spotless.sort_values("score")
            clean = (moment, str(ordered.index[0]), float(ordered["score"].iloc[0]))
            break

    plans: list[tuple[pd.Timestamp, str, float, str, str]] = [
        (*detected, "Detected ring — highest Tier-3 risk score", "tier3_ring_detected.png")
    ]
    if clean is not None:
        plans.append(
            (
                *clean,
                "Clean cluster — no member touches labelled fraud",
                "tier3_cluster_clean.png",
            )
        )
    else:
        lowest = current.sort_values("score")
        plans.append(
            (
                latest,
                str(lowest.index[0]),
                float(lowest["score"].iloc[0]),
                "Lowest-risk ring — NOT clean: no snapshot holds a ring free of labelled fraud",
                "tier3_cluster_clean.png",
            )
        )

    for moment, ring_id, score, title, filename in plans:
        window_rows = frame.loc[
            (frame["event_time"] >= moment - paysim.window) & (frame["event_time"] < moment)
        ]
        graph = paysim_factory(
            int(paysim.winner.parameters["step_window"]),
            float(paysim.winner.parameters["amount_tolerance"]),
        )()
        graph.insert(window_rows)
        snapshot = graph.snapshot(moment.to_pydatetime())
        infected = fraud_accounts(window_rows)

        rows = account_rings.loc[
            (account_rings["snapshot_end"] == moment) & (account_rings["ring_id"] == ring_id)
        ]
        members = rows["account_id"].astype(str).tolist()
        tainted = int(rows["account_is_fraudulent"].sum())
        plot_ring(
            snapshot,
            members,
            infected,
            f"{title}\nring {ring_id} in the window ending {moment.date()}"
            f", {len(members)} accounts ({tainted} touching labelled fraud), "
            f"ring_risk_score {score:.3f}",
            notebook_dir / filename,
        )
        logger.info(
            "wrote %s (%d accounts, %d fraud-touching, window ending %s)",
            notebook_dir / filename,
            len(members),
            tainted,
            moment.date(),
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train and evaluate the Tier-3 graph layer.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Use only the earliest N rows per corpus. For iteration; results are not "
        "reportable and nothing is written to the registry.",
    )
    parser.add_argument(
        "--corpus",
        choices=["paysim", "ieee_cis", "both"],
        default="both",
        help="Which corpus to run.",
    )
    arguments = parser.parse_args(argv)

    settings = Settings()
    root = Path(__file__).resolve().parents[3]
    corpora: tuple[SourceDataset, ...] = (
        ("paysim", "ieee_cis")
        if arguments.corpus == "both"
        else (cast(SourceDataset, arguments.corpus),)
    )
    run(
        processed_dir=settings.processed_data_dir,
        notebook_dir=root / "notebooks",
        registry_path=root / "models" / "registry.json",
        artifact_dir=root / "models" / "artifacts",
        entity_anonymization_key=settings.entity_anonymization_key.encode("utf-8"),
        sample=arguments.sample,
        corpora=corpora,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
