"""Feature assembly for the Phase 5 meta-learner, and the leakage discipline that governs it.

This module exists because fusing three trained layers is the point in the project where a
contaminated result is easiest to produce and hardest to see. Tier-1, Tier-2 and Tier-3 were
all fitted on the train split. Their scores *on that split* are therefore in-sample, and a
meta-learner fitted on them learns how much to trust each layer from numbers that layer will
never reproduce at serving time. The three tiers are contaminated in three different ways and
each needs a different remedy:

**Tier-1 — in-sample by construction.** Remedied with out-of-fold scoring: forward-chaining
blocks over the train split, where fold *k* is scored by a model fitted only on blocks before
it (:func:`time_blocks`, :func:`build_oof_tier1`). Random K-fold is not available here — it
would let a fold's model train on rows chronologically after the rows it scores, which is the
leak ``ml-evaluation-standards`` section 1 forbids and which no metric would reveal.

**Tier-2 — contaminated *by label*.** The autoencoder fits on fraud-free train windows,
excluded by *account* rather than by row (``tier2_sequences.eligible_training_rows``). So on
train, every fraud row belongs to an account withheld from the fit entirely, while most clean
rows belong to accounts inside it. That manufactures a fraud/clean error gap which will not
reproduce on test, and left alone it makes the meta-learner overweight ``tier2_error``.
:func:`tier2_memorisation_diagnostic` measures the artefact against a control group that holds
the label constant, and the keep/drop verdict is arbitrated on validation, where Tier-2's
scores carry no eligibility filter at all.

**Tier-3 — a frozen snapshot that ends after the test period.** The registered artifact's score
table is one snapshot ending 2018-06-02; joining it to train or validation rows reads the
future. :func:`build_tier3_features` rebuilds from ``train_tier3.run_snapshots`` instead, which
scores ``[t, t + cadence)`` from a graph built on ``[t - window, t)`` and so cannot see the rows
it scores.

**The abstention convention, inherited from Phases 3 and 4.** A missing tier signal is written
as :data:`ABSTENTION_SENTINEL` *and* flagged by a separate ``is_scoreable`` indicator. Never as
``0.0``: a zero reads to a tree as "this layer looked and found nothing wrong", which is a
fabrication about a layer that did not look. Phases 3 and 4 both state this in their own
docstrings and both defer the consequence to here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from app.data.features import feature_names_for
from app.data.raw_spec import SourceDataset
from app.models.tier1_features import DENIED_COLUMNS

if TYPE_CHECKING:  # pragma: no cover - kept out of the runtime import graph
    from app.models.tier1_anomaly import Tier1Model

logger = logging.getLogger("riskiq.meta")

#: Set and logged by the driver, per ml-evaluation-standards section 5.
RANDOM_SEED = 42

#: Written into a tier column when that layer has no opinion on the row, paired always with the
#: block's ``is_scoreable`` indicator. Same value and same reasoning as
#: ``train_tier2.ABSTAINED_RANK_SENTINEL`` and ``train_tier3.ABSTAINED_RANK_SENTINEL``: it ranks
#: below every real score, so an abstaining layer never flags, and it is distinguishable from a
#: genuine low score, so the meta-learner can condition on "did not look" separately from
#: "looked and found nothing".
ABSTENTION_SENTINEL = -1.0

#: Forward-chaining blocks over the train split for out-of-fold Tier-1 scoring. Block 1 is
#: scored by nobody -- it has no predecessor -- and is dropped from the meta-fit set rather than
#: sentinel-filled, because a sentinel there would teach the meta-learner a serving state that
#: never occurs (Tier-1 is never absent in production).
OOF_BLOCKS = 5

#: Rounds for each fold's Tier-1 booster. Fixed rather than early-stopped, and this is a
#: measurement decision, not a shortcut. Early-stopping on the held-out block would select the
#: round count by reading the very rows whose out-of-fold scores are being produced; early
#: stopping on the global validation split would leak validation into a fit whose output is then
#: used to fit the meta-learner. The value is the registered Tier-1 model's ``best_iteration``,
#: read from models/registry.json at run time and asserted against this default.
OOF_TIER1_ROUNDS = 1917


# --- Feature blocks ---------------------------------------------------------------------
# The unit of ablation. Each block is retained or retired as a whole, because "does Tier-2 pay
# for itself" is the question Phase 3 deferred to here, and it is a question about the layer,
# not about one of its columns.

#: The Phase 1 engineered vector, passed through to the meta-learner alongside the tier scores.
#: This is stacking *with passthrough*: the meta-learner sees both Tier-1's output and Tier-1's
#: inputs. Expect its marginal contribution to be small -- Tier-1's LightGBM was fitted on a
#: superset of exactly these columns -- and do not read that smallness as the features being
#: uninformative.
ENGINEERED_BLOCK: tuple[str, ...] = feature_names_for("ieee_cis")

TIER1_BLOCK: tuple[str, ...] = ("tier1_score",)

#: ``tier2_is_anomaly`` is deliberately absent. The registered model's operating threshold is
#: -1.0, so that flag is constant-true on every scoreable window and carries no information; a
#: constant column would still consume a SHAP slot and appear in the importance table.
TIER2_BLOCK: tuple[str, ...] = (
    "tier2_error",
    "tier2_is_scoreable",
    "tier2_sequence_length",
)

#: Tier-3 signals obtainable at serving time from a ``Tier3Result`` as it exists today.
TIER3_SERVED_BLOCK: tuple[str, ...] = (
    "tier3_ring_risk_score",
    "tier3_ring_size",
    "tier3_is_ring_member",
    "tier3_is_scoreable",
    "tier3_seen_not_ringed",
    "tier3_staleness_hours",
)

#: Ring topology. Separated from :data:`TIER3_SERVED_BLOCK` because ``Tier3Result`` does **not**
#: expose these today -- serving them would require extending Tier-3's persisted score table to
#: carry a per-account feature vector. Kept as its own ablatable block so that if it turns out to
#: add nothing the question closes for free, and if it does add something the cost lands in the
#: registry as a costed Phase 7 requirement rather than being discovered during integration.
TIER3_TOPOLOGY_BLOCK: tuple[str, ...] = (
    "tier3_ring_density",
    "tier3_ring_mean_degree",
    "tier3_ring_max_degree",
    "tier3_ring_mean_idf",
    "tier3_ring_spec_count",
    "tier3_ring_entity_count",
    "tier3_account_degree",
    "tier3_account_degree_centrality",
    "tier3_account_betweenness",
    "tier3_account_entity_count",
    "tier3_betweenness_exact",
)

FEATURE_BLOCKS: dict[str, tuple[str, ...]] = {
    "engineered": ENGINEERED_BLOCK,
    "tier1": TIER1_BLOCK,
    "tier2": TIER2_BLOCK,
    "tier3_served": TIER3_SERVED_BLOCK,
    "tier3_topology": TIER3_TOPOLOGY_BLOCK,
}

#: Blocks that cannot be retired by the ablation. Not a statement that they are useful -- their
#: deltas are still measured and reported -- but that retiring them would mean something other
#: than "this layer does not pay for itself". A negative ``tier1`` delta would mean Phase 2's
#: headline is unreproducible, which is a bug signal to investigate, not a finding to act on.
EXEMPT_BLOCKS: frozenset[str] = frozenset({"engineered", "tier1"})

#: Prefix applied to the non-circular Tier-3 control's columns. The control re-runs graph
#: construction with fingerprints that share no column with the constructed ``account_id``.
NON_CIRCULAR_PREFIX = "tier3nc_"


# --- The deny list ----------------------------------------------------------------------

#: Columns that must never reach the meta-learner's matrix.
#:
#: A superset of Tier-1's :data:`DENIED_COLUMNS`, extended with the Tier-3 snapshot outputs that
#: read a label or an amount.
#:
#: ``account_is_fraudulent`` is the dangerous one and the reason this list exists separately.
#: ``train_tier3.run_snapshots`` computes it as ``fraud_accounts(window_rows)`` -- a direct read
#: of the label over the snapshot window -- to supply the ring scorer's training target. It
#: travels in the same frame as the topology features and is one careless ``select_dtypes`` away
#: from the model matrix. A meta-learner given it would report a near-perfect PR-AUC that means
#: nothing whatsoever.
#:
#: ``ring_amount`` is denied because Tier-3's contract is topology-only
#: (``tier3_graph.FORBIDDEN_FEATURE_SOURCES`` and the test that pins it); transaction value
#: already reaches the model as ``amount_log``.
META_DENIED_COLUMNS: frozenset[str] = DENIED_COLUMNS | frozenset(
    {
        "account_is_fraudulent",
        "is_fraud_ring",
        "ring_amount",
        "ring_id",
        "snapshot_end",
        "staleness_seconds",
    }
)


#: Prefixes a Tier-3 column may carry into the matrix. Stripped before the deny-list check.
TIER3_PREFIXES: tuple[str, ...] = ("tier3_", NON_CIRCULAR_PREFIX)


def meta_denied_columns_present(columns: Sequence[str]) -> list[str]:
    """Return any denied column names found in ``columns``, sorted.

    The companion guard to ``tier1_features.denied_columns_present``, widened to the Tier-3
    snapshot outputs. Empty return means clean.

    Tier-3 prefixes are stripped before the comparison, and that is the whole point of this
    function existing separately. The deny list holds bare names like ``account_is_fraudulent``,
    but Tier-3 columns reach the matrix prefixed, so a naive set intersection would wave
    ``tier3_account_is_fraudulent`` straight through -- the single most dangerous column in the
    phase, a direct label read, matched against a guard that cannot see it.
    """
    found: set[str] = set()
    for column in columns:
        bare = column
        for prefix in TIER3_PREFIXES:
            if bare.startswith(prefix):
                bare = bare[len(prefix) :]
                break
        if bare in META_DENIED_COLUMNS or column in META_DENIED_COLUMNS:
            found.add(column)
    return sorted(found)


def require_clean_feature_names(columns: Sequence[str]) -> None:
    """Raise if any denied column reached a feature name list.

    Raises:
        ValueError: Naming every offending column. Raised rather than logged: a denied column
            in the matrix invalidates every number the run produces, so there is no useful
            degraded mode to continue into.
    """
    denied = meta_denied_columns_present(columns)
    if denied:
        raise ValueError(
            "denied columns reached the meta-learner feature set: "
            f"{denied}. These read a label, an identifier or a duplicate of amount_log; "
            "see META_DENIED_COLUMNS for why each is barred."
        )


def block_feature_names(blocks: Sequence[str]) -> tuple[str, ...]:
    """Return the concatenated, deny-list-checked feature names for ``blocks``.

    Order is the order of :data:`FEATURE_BLOCKS`, not the caller's argument order, so that two
    runs requesting the same blocks in different orders produce the same matrix and therefore
    the same feature-version hash.

    Raises:
        KeyError: If a block name is not defined.
        ValueError: If the result contains a denied column.
    """
    unknown = sorted(set(blocks) - set(FEATURE_BLOCKS))
    if unknown:
        raise KeyError(f"unknown feature block(s): {unknown}")
    selected = tuple(
        name for block, names in FEATURE_BLOCKS.items() if block in set(blocks) for name in names
    )
    require_clean_feature_names(selected)
    return selected


# --- Out-of-fold fold assignment --------------------------------------------------------


@dataclass(frozen=True)
class TimeBlocks:
    """Forward-chaining block assignment over one time-ordered split.

    Attributes:
        index: Block number per row, 1-based, aligned to the frame that produced it. Every row
            gets one; block 1 is the earliest.
        boundaries: The ``block_count - 1`` cut timestamps. Row *i* is in block *b* when its
            event time is below ``boundaries[b - 1]`` and not below ``boundaries[b - 2]``.
        block_count: How many blocks were cut.
    """

    index: npt.NDArray[np.int64]
    boundaries: tuple[pd.Timestamp, ...]
    block_count: int

    def rows_in(self, block: int) -> npt.NDArray[np.bool_]:
        """Return the mask selecting one block."""
        return np.asarray(self.index == block, dtype=np.bool_)

    def rows_before(self, block: int) -> npt.NDArray[np.bool_]:
        """Return the mask selecting every block strictly earlier than ``block``.

        This is the fitting set for the fold that scores ``block`` -- the definition that makes
        the scheme forward-chaining rather than a shuffle.
        """
        return np.asarray(self.index < block, dtype=np.bool_)


def time_blocks(event_time: pd.Series, block_count: int = OOF_BLOCKS) -> TimeBlocks:
    """Cut a time-ordered split into equal-row-count blocks on the time axis.

    The cut is a *timestamp*, not a row index, and the comparison is strict ``<`` with ties
    falling on the later side -- the same rule ``app.data.splitting.assign_splits`` uses for the
    train/val/test boundaries. Rows sharing a boundary timestamp therefore all land in the later
    block, so no boundary instant is split across a fitting set and the set it scores.

    Args:
        event_time: Timezone-aware event times. Need not be sorted.
        block_count: How many blocks. Must be at least 2.

    Returns:
        The block assignment, aligned positionally to ``event_time``.

    Raises:
        ValueError: If ``block_count`` is below 2, if ``event_time`` is empty or contains nulls,
            or if the requested cut would leave a block empty. An empty block means a fold with
            nothing to score or nothing to fit on, and silently continuing would quietly shrink
            the meta-fit set without saying so.
    """
    if block_count < 2:
        raise ValueError(f"block_count must be at least 2, got {block_count}")
    if event_time.empty:
        raise ValueError("cannot cut blocks from an empty split")
    if event_time.isna().any():
        raise ValueError("event_time contains nulls; the block cut would be undefined")

    values = event_time.to_numpy()
    ordered = np.sort(values)
    row_count = len(ordered)

    boundaries: list[pd.Timestamp] = []
    for step in range(1, block_count):
        cut = min(int(row_count * step / block_count), row_count - 1)
        boundaries.append(pd.Timestamp(ordered[cut]))

    index = np.full(row_count, block_count, dtype="int64")
    # Walk the boundaries in reverse so the earliest matching cut wins, giving strict `<`.
    for position in range(block_count - 1, 0, -1):
        index[values < boundaries[position - 1]] = position

    counts = np.bincount(index, minlength=block_count + 1)[1:]
    empty = [block for block, count in enumerate(counts, start=1) if count == 0]
    if empty:
        raise ValueError(
            f"block(s) {empty} are empty at block_count={block_count}: the split has too few "
            "distinct timestamps to cut this finely. Use fewer blocks or more rows."
        )
    return TimeBlocks(index=index, boundaries=tuple(boundaries), block_count=block_count)


def assert_forward_chaining(blocks: TimeBlocks, event_time: pd.Series, scored_block: int) -> None:
    """Assert that a fold's fitting rows all precede every row it scores.

    The guard on the out-of-fold scheme, called per fold rather than trusted. A shuffled or
    random-K-fold assignment fails here, which is the point: the contamination it would create
    is invisible in every metric the run reports afterwards.

    Raises:
        ValueError: If any fitting row is not strictly earlier than the earliest scored row, or
            if the fold has no fitting rows at all.
    """
    fit_mask = blocks.rows_before(scored_block)
    score_mask = blocks.rows_in(scored_block)
    if not fit_mask.any():
        raise ValueError(
            f"block {scored_block} has no earlier block to fit on; it cannot be scored "
            "out-of-fold and must be excluded from the meta-fit set"
        )
    if not score_mask.any():
        raise ValueError(f"block {scored_block} is empty")

    values = event_time.to_numpy()
    latest_fit = values[fit_mask].max()
    earliest_scored = values[score_mask].min()
    if not latest_fit < earliest_scored:
        raise ValueError(
            f"fold {scored_block} is not forward-chaining: its fitting set reaches "
            f"{latest_fit}, which is not strictly before the earliest row it scores "
            f"({earliest_scored}). This is the leak out-of-fold scoring exists to prevent."
        )


def rank_normalise(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Map scores to their within-array percentile rank in ``[0, 1]``.

    Applied per fold. The four fold models are fitted on one, two, three and four blocks
    respectively, so they differ in both level and sharpness; the meta-learner would otherwise
    read that drift as signal, since the blocks are consecutive in time. Ranking is invariant to
    any monotone recalibration, so it removes the drift while leaving each fold's ordering --
    the only thing a fold model actually asserts -- untouched.

    Ties take their average rank, so equal scores stay equal.
    """
    if scores.size == 0:
        return scores.astype("float64")
    order = scores.argsort(kind="stable")
    ranks = np.empty(scores.size, dtype="float64")
    ranks[order] = np.arange(scores.size, dtype="float64")

    # Average the ranks within each run of equal values, so ties do not become an arbitrary
    # ordering that the meta-learner could latch onto.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if unique.size != scores.size:
        summed = np.zeros(unique.size, dtype="float64")
        np.add.at(summed, inverse, ranks)
        ranks = (summed / counts)[inverse]

    denominator = max(scores.size - 1, 1)
    return ranks / denominator


@dataclass(frozen=True)
class EmpiricalCdf:
    """A monotone score-to-percentile map fitted on one reference sample.

    Fitted on the full-train Tier-1 model's scores over the train split and applied at
    validation and test, so that the column the meta-learner reads at serving time occupies the
    same ``[0, 1]`` scale as the rank-normalised out-of-fold column it was fitted on.

    Stored as a quantile grid rather than the whole sample, and applied by **linear
    interpolation** between the knots rather than by bucketing into them.

    The distinction is not cosmetic and was measured. Bucketing with ``searchsorted`` collapses
    the mapped column onto exactly ``points`` distinct values: at 1,024 points it took Tier-1's
    88,069 distinct test scores down to 1,024 and cost **0.0073 PR-AUC** (0.5276 to 0.5202)
    purely in ties. Since ``tier1_score`` is the meta-learner's strongest input, that quantisation
    was damaging the fusion layer's best feature and simultaneously depressing the baseline it
    was being compared against. Interpolation keeps the map continuous and strictly monotone
    wherever the reference distribution is, so the ranking survives intact.
    """

    grid: tuple[float, ...]

    @classmethod
    def fit(cls, scores: npt.NDArray[np.float64], points: int = 4_096) -> "EmpiricalCdf":
        """Fit the quantile grid on a reference sample."""
        if scores.size == 0:
            raise ValueError("cannot fit an empirical CDF on an empty sample")
        quantiles = np.linspace(0.0, 1.0, points)
        return cls(grid=tuple(np.quantile(scores.astype("float64"), quantiles).tolist()))

    def apply(self, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map scores onto ``[0, 1]`` by their position in the reference distribution."""
        knots = np.asarray(self.grid, dtype="float64")
        levels = np.linspace(0.0, 1.0, knots.size)
        return np.asarray(np.interp(scores.astype("float64"), knots, levels), dtype="float64")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form for the model sidecar."""
        return {"kind": "empirical_cdf", "grid": list(self.grid)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmpiricalCdf":
        """Rebuild from :meth:`to_dict` output."""
        return cls(grid=tuple(float(value) for value in payload["grid"]))


def require_ieee_cis(source: SourceDataset) -> None:
    """Refuse to fit the meta-learner on any corpus but IEEE-CIS.

    PaySim is barred by measurement, not by preference. Phase 2 found ``amount ==
    oldbalanceOrg`` on 97.49% of PaySim fraud and on 0 of 412,277 legitimate rows, which drives
    Tier-1's PaySim PR-AUC to 0.9998 (the registered matched-inputs model; a losing full-inputs
    candidate reached 0.9999 and was never selected). A meta-learner fitted there would learn
    "Tier-1 is always right" from a simulator's pairing rule and carry that weighting into a
    fusion that is supposed to be measuring exactly the opposite question. Tier-2 has no PaySim
    model at all,
    and Tier-3 abstains on 100% of PaySim test transactions.

    Raises:
        ValueError: For any source other than ``ieee_cis``.
    """
    if source != "ieee_cis":
        raise ValueError(
            f"the meta-learner is fitted on ieee_cis only, got {source!r}. PaySim's Tier-1 "
            "result is a simulator artefact (amount == oldbalanceOrg on 97.49% of fraud), it "
            "has no Tier-2 model, and Tier-3 abstains on 100% of its test transactions. "
            "See BUILD_LOG.md Phase 2 finding 3."
        )


# --- Out-of-fold Tier-1 scoring ---------------------------------------------------------


@dataclass(frozen=True)
class RegisteredTier1:
    """The Tier-1 model as it is actually deployed, rebuilt from its registry entry.

    Attributes:
        model: The reconstructed :class:`~app.models.tier1_anomaly.Tier1Model`.
        model_id: Its registry id, recorded in the meta-learner's provenance.
        best_iteration: The round count early stopping chose on validation. Reused as the fixed
            round count for the out-of-fold folds, which have no honest way to choose their own.
    """

    model: "Tier1Model"
    model_id: str
    best_iteration: int


def load_registered_tier1(artifact_dir: Path, registry_path: Path) -> RegisteredTier1 | None:
    """Rebuild the registered IEEE-CIS Tier-1 model from its artefact and sidecar.

    The reconstruction itself now lives on :meth:`app.models.tier1_anomaly.Tier1Model.load`,
    where Phase 7's serving path also reads it. This function remains the *registry-resolving*
    wrapper: it picks the entry, carries the ``best_iteration`` the meta-learner's folds reuse,
    and keeps the "return None rather than raise" contract that Phase 5's report depends on.

    The "return None rather than raise" contract covers *absence* only: no registry, no entry,
    no artefact on disk. A reconstruction that finds the artefact and cannot trust it -- a
    feature-version hash that does not match the registry, or a sidecar naming an algorithm this
    path refuses to load -- still raises, deliberately. Absence is a stated gap a report can
    carry; drift between the saved spec and what ``fit_tier1_input_spec`` produces means every
    provenance claim downstream is wrong, and degrading quietly would launder that into a
    missing-model note.

    Returns:
        The model, or ``None`` when the registry or artefact is absent -- a stated gap the
        caller carries into the report, never a silent skip.

    Raises:
        RuntimeError: If the rebuilt spec does not hash to the registered ``feature_version``.
        ValueError: If the sidecar names an algorithm that cannot be loaded here.
    """
    from app.models.tier1_anomaly import Tier1Model

    if not registry_path.exists():
        return None
    entries = [
        entry
        for entry in json.loads(registry_path.read_text(encoding="utf-8"))
        if entry.get("layer") == "tier1_anomaly" and entry.get("source_dataset") == "ieee_cis"
    ]
    if not entries:
        return None
    entry = entries[-1]
    model_id = str(entry["model_id"])
    try:
        model = Tier1Model.load(
            model_id,
            artifact_dir,
            feature_version=str(entry.get("feature_version", "")) or None,
        )
    except FileNotFoundError:
        logger.warning("Tier-1 artefact %s is missing", model_id)
        return None

    best_iteration = int(entry.get("hyperparameters", {}).get("best_iteration", OOF_TIER1_ROUNDS))
    return RegisteredTier1(model=model, model_id=model_id, best_iteration=best_iteration)


def _parse_dropped(entries: Any) -> list[Any]:
    """Rebuild ``DroppedColumn`` records from their ``"column (reason)"`` serialisation.

    Delegates to :func:`app.models.tier1_features.parse_dropped`, which is where the parser
    now lives -- beside the formatter it must stay in step with, since the string form is
    hashed into the feature version.
    """
    from app.models.tier1_features import parse_dropped

    return list(parse_dropped(entries))


@dataclass(frozen=True)
class FoldHandicap:
    """One fold model's standing against the full-train model, on a common yardstick.

    The out-of-fold column is produced by models fitted on a fraction of the train split, while
    the column the meta-learner meets at serving time comes from a model fitted on all of it.
    That gap is a real limitation of the scheme, so it is measured on the untouched validation
    split and reported rather than asserted away.
    """

    fold: int
    train_rows: int
    train_positives: int
    validation_pr_auc: float


@dataclass(frozen=True)
class OofTier1:
    """Out-of-fold Tier-1 scores over the train split, plus the provenance to justify them.

    Attributes:
        scores: Rank-normalised out-of-fold score per train row, ``NaN`` in block 1.
        scoreable: Which rows have an out-of-fold score at all -- everything outside block 1.
        blocks: The fold assignment, retained so the driver can report the cut.
        handicap: Per-fold standing against the full-train model.
        feature_versions: One ``fv_`` hash per fold. They differ, because each fold refits its
            own input spec; recording all of them is part of the honest record, since the
            out-of-fold column is not the product of a single feature definition.
        reference_cdf: The full-train model's train-score distribution, used to map validation
            and test scores onto the same ``[0, 1]`` scale the fitted column occupies.
        full_train_validation_pr_auc: The yardstick the handicap table is read against.
        rounds: The fixed round count every fold used.
    """

    scores: npt.NDArray[np.float64]
    scoreable: npt.NDArray[np.bool_]
    blocks: TimeBlocks
    handicap: tuple[FoldHandicap, ...]
    feature_versions: tuple[str, ...]
    reference_cdf: EmpiricalCdf
    full_train_validation_pr_auc: float
    rounds: int


def _fit_fold_booster(spec: Any, frame: pd.DataFrame, rounds: int) -> Any:
    """Fit one fold's Tier-1 booster at a fixed round count, with no early stopping.

    Deliberately not ``train_tier1.fit_lightgbm``: that function early-stops on a validation
    frame, and there is no frame a fold could honestly early-stop on. The held-out block is the
    one being scored, and the global validation split is downstream of this fit.
    """
    import lightgbm as lgb

    from app.models.tier1_features import labels_of
    from app.models.train_tier1 import LIGHTGBM_PARAMS

    matrix = spec.transform(frame)
    dataset = lgb.Dataset(matrix, label=labels_of(frame).astype("int8"))
    return lgb.train(LIGHTGBM_PARAMS, dataset, num_boost_round=rounds)


def build_oof_tier1(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    source: SourceDataset,
    registered: RegisteredTier1,
    block_count: int = OOF_BLOCKS,
) -> OofTier1:
    """Produce out-of-fold Tier-1 scores over the train split.

    Fold *k* is fitted on blocks ``1..k-1`` and scores block *k*, so no row contributes to the
    model that scores it and every fitting row precedes every scored row. Block 1 has no
    predecessor and comes back ``NaN``; the caller drops those rows rather than filling them.

    Each fold refits its own ``fit_tier1_input_spec``. Sharing the full-train spec would be
    cheaper and would leak: the frequency encoders and categorical level sets are fitted on the
    frame they are given, so a shared spec carries the held-out block's own category frequencies
    into its own features.

    Args:
        train_frame: The train split, time-ordered.
        validation_frame: Used only for the fold-handicap table -- never fitted on here.
        source: Must be ``ieee_cis``.
        registered: The deployed Tier-1 model, supplying the fixed round count and the
            reference score distribution.
        block_count: How many forward-chaining blocks to cut.

    Returns:
        The out-of-fold column and its provenance.
    """
    from app.ml.evaluation import pr_auc
    from app.models.tier1_features import fit_tier1_input_spec, labels_of

    require_ieee_cis(source)
    blocks = time_blocks(train_frame["event_time"], block_count=block_count)
    rounds = registered.best_iteration
    logger.info(
        "out-of-fold Tier-1: %d forward-chaining blocks, %d rounds per fold (from %s), "
        "block 1 dropped",
        block_count,
        rounds,
        registered.model_id,
    )

    scores = np.full(len(train_frame), np.nan, dtype="float64")
    validation_labels = labels_of(validation_frame)
    handicap: list[FoldHandicap] = []
    feature_versions: list[str] = []

    for fold in range(2, block_count + 1):
        assert_forward_chaining(blocks, train_frame["event_time"], fold)
        fit_mask = blocks.rows_before(fold)
        score_mask = blocks.rows_in(fold)
        fit_frame = train_frame.loc[fit_mask]
        score_frame = train_frame.loc[score_mask]

        spec = fit_tier1_input_spec(fit_frame, source)
        feature_versions.append(spec.to_feature_definition().feature_version)
        booster = _fit_fold_booster(spec, fit_frame, rounds)

        raw = np.asarray(booster.predict(spec.transform(score_frame)), dtype="float64")
        # Rank within the fold. The folds are consecutive in time and fitted on growing amounts
        # of data, so their raw scores drift in level and sharpness; unranked, the meta-learner
        # would read that drift as signal.
        scores[score_mask] = rank_normalise(raw)

        fold_validation = np.asarray(
            booster.predict(spec.transform(validation_frame)), dtype="float64"
        )
        handicap.append(
            FoldHandicap(
                fold=fold,
                train_rows=int(fit_mask.sum()),
                train_positives=int(labels_of(fit_frame).sum()),
                validation_pr_auc=pr_auc(validation_labels, fold_validation),
            )
        )
        logger.info(
            "fold %d: fitted on %d rows (%d positives), scored %d, val PR-AUC %.4f",
            fold,
            handicap[-1].train_rows,
            handicap[-1].train_positives,
            int(score_mask.sum()),
            handicap[-1].validation_pr_auc,
        )

    full_train_scores = np.asarray(registered.model.score_frame(train_frame), dtype="float64")
    reference_cdf = EmpiricalCdf.fit(full_train_scores)
    full_validation = np.asarray(registered.model.score_frame(validation_frame), dtype="float64")
    full_train_validation_pr_auc = pr_auc(validation_labels, full_validation)
    logger.info(
        "full-train Tier-1 (%s) val PR-AUC %.4f -- the yardstick for the handicap table",
        registered.model_id,
        full_train_validation_pr_auc,
    )

    return OofTier1(
        scores=scores,
        scoreable=np.asarray(blocks.index > 1, dtype=np.bool_),
        blocks=blocks,
        handicap=tuple(handicap),
        feature_versions=tuple(feature_versions),
        reference_cdf=reference_cdf,
        full_train_validation_pr_auc=full_train_validation_pr_auc,
        rounds=rounds,
    )


# --- Tier-2 per-transaction reconstruction error -----------------------------------------


@dataclass(frozen=True)
class Tier2MemorisationDiagnostic:
    """How much of Tier-2's train-split fraud/clean gap is memorisation rather than signal.

    Tier-2 fits on fraud-free train windows, excluded by *account*
    (``tier2_sequences.eligible_training_rows``). So on the train split every fraud row belongs
    to an account withheld from the fit entirely, while most clean rows belong to accounts
    inside it. Some of the resulting error gap is therefore the autoencoder recognising rows it
    was fitted on, not fraud detection -- and that component will not reproduce on test, where
    an account fraudulent-in-test may well have been clean-in-train and inside the fit.

    The control group is what makes this measurable rather than arguable: train rows that are
    **clean but belong to fraud-touching accounts** were excluded from the fit for a reason that
    has nothing to do with their own label. Comparing them against clean rows from included
    accounts holds the label constant and varies only fit membership.

    Attributes:
        included_clean_rows: Clean train rows whose account was inside the fit.
        excluded_clean_rows: Clean train rows whose account was withheld -- the control group.
        fraud_rows: Fraud train rows, all withheld by construction.
        memorisation_auc: P(an excluded clean row scores above an included clean row). 0.5 means
            fit membership alone moves nothing; above 0.5 means part of the observed gap is
            memorisation.
        fraud_auc: P(a fraud row scores above an included clean row) -- the naive gap, which
            mixes signal and memorisation together.
        residual_auc: P(a fraud row scores above an *excluded* clean row). Both sides were
            withheld from the fit, so this is the gap with memorisation held constant, and it is
            the honest estimate of what Tier-2 contributes on the train split.
    """

    included_clean_rows: int
    excluded_clean_rows: int
    fraud_rows: int
    memorisation_auc: float
    fraud_auc: float
    residual_auc: float

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form for the registry note."""
        return {
            "included_clean_rows": self.included_clean_rows,
            "excluded_clean_rows": self.excluded_clean_rows,
            "fraud_rows": self.fraud_rows,
            "memorisation_auc": round(self.memorisation_auc, 6),
            "fraud_auc": round(self.fraud_auc, 6),
            "residual_auc": round(self.residual_auc, 6),
        }

    def describe(self) -> list[str]:
        """Return plain-language lines for the report."""
        inflation = self.fraud_auc - self.residual_auc
        return [
            f"Fit membership alone moves Tier-2's error by AUC {self.memorisation_auc:.4f} "
            f"(0.5 = no effect), measured on {self.excluded_clean_rows:,} clean train rows "
            f"withheld from the fit against {self.included_clean_rows:,} clean rows inside it. "
            "Both groups carry the same label, so this is memorisation, not detection.",
            f"The naive train fraud/clean gap is AUC {self.fraud_auc:.4f}. Holding fit "
            f"membership constant it falls to {self.residual_auc:.4f}, so roughly "
            f"{inflation:+.4f} of the apparent gap is an artefact of the eligibility filter "
            "and will not reproduce on test.",
        ]


def _rank_auc(positive: npt.NDArray[np.float64], negative: npt.NDArray[np.float64]) -> float:
    """Return P(a random positive outranks a random negative), ties counted as half.

    The Mann-Whitney U statistic normalised to ``[0, 1]``. Computed by ranking rather than by
    the O(n*m) pairwise comparison, which at 360k against 34k rows would not finish.
    """
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    order = combined.argsort(kind="stable")
    ranks = np.empty(combined.size, dtype="float64")
    ranks[order] = np.arange(1, combined.size + 1, dtype="float64")
    # Average tied ranks so that equal errors contribute exactly one half.
    unique, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    if unique.size != combined.size:
        summed = np.zeros(unique.size, dtype="float64")
        np.add.at(summed, inverse, ranks)
        ranks = (summed / counts)[inverse]
    positive_rank_sum = ranks[: positive.size].sum()
    u = positive_rank_sum - positive.size * (positive.size + 1) / 2.0
    return float(u / (positive.size * negative.size))


def tier2_memorisation_diagnostic(
    train_frame: pd.DataFrame, errors: pd.Series
) -> Tier2MemorisationDiagnostic:
    """Measure how much of Tier-2's train-split gap is fit memorisation.

    Args:
        train_frame: The train split, carrying ``account_id`` and ``is_fraud``.
        errors: Reconstruction error indexed by ``transaction_id``, as
            :func:`build_tier2_errors` returns it. Rows Tier-2 abstained on are skipped.

    Returns:
        The three AUCs and their group sizes.
    """
    from app.models.tier2_sequences import eligible_training_rows

    eligible = eligible_training_rows(train_frame)
    labels = train_frame["is_fraud"].to_numpy(dtype=bool)
    aligned = train_frame["transaction_id"].astype(str).map(errors)
    measured = aligned.notna().to_numpy()

    included_clean = aligned[measured & ~labels & eligible].to_numpy(dtype="float64")
    excluded_clean = aligned[measured & ~labels & ~eligible].to_numpy(dtype="float64")
    fraud = aligned[measured & labels].to_numpy(dtype="float64")

    return Tier2MemorisationDiagnostic(
        included_clean_rows=int(included_clean.size),
        excluded_clean_rows=int(excluded_clean.size),
        fraud_rows=int(fraud.size),
        memorisation_auc=_rank_auc(excluded_clean, included_clean),
        fraud_auc=_rank_auc(fraud, included_clean),
        residual_auc=_rank_auc(fraud, excluded_clean),
    )


@dataclass(frozen=True)
class Tier2Features:
    """Per-transaction Tier-2 signals, keyed by ``transaction_id``.

    Attributes:
        frame: Columns ``transaction_id``, ``tier2_error``, ``tier2_is_scoreable``,
            ``tier2_sequence_length``. One row per scored transaction; transactions with no
            window at all are simply absent and pick up the sentinel at join time.
        model_id: The registered Tier-2 model these came from.
        feature_version: Its ``fv_`` hash.
        min_length: The scoring floor below which Tier-2 abstains.
        window: The ``W`` the windows were built at.
        coverage: Share of transactions Tier-2 has an opinion on.
    """

    frame: pd.DataFrame
    model_id: str
    feature_version: str
    min_length: int
    window: int
    coverage: float


def build_tier2_errors(
    frames: dict[str, pd.DataFrame],
    *,
    artifact_dir: Path,
    model_id: str,
) -> Tier2Features | None:
    """Score every transaction with the registered Tier-2 autoencoder.

    Tier-2's unit is the *window*, anchored on the transaction being scored, so this returns a
    per-transaction column directly -- Phase 3 aggregated to accounts for reporting, but the
    underlying errors were always per transaction. The anchor is mapped back to its
    ``transaction_id`` through the account-then-time reordering ``assemble_windows`` applies.

    The model's own fitted spec is reused rather than refitted. Refitting would produce a
    different scaler on the same data and silently break the correspondence between these
    errors and the registered model's reported metrics.

    Args:
        frames: The three splits. All are needed together: a test window whose account first
            appeared during the train period genuinely has that history at serving time.
        artifact_dir: ``models/artifacts``.
        model_id: The registered Tier-2 model to load.

    Returns:
        The per-transaction features, or ``None`` when the artefact is absent -- a stated gap,
        never a silent skip.
    """
    from app.ml.registry import artifact_path
    from app.models.tier2_behavioral import Tier2Model
    from app.models.tier2_sequences import assemble_windows, order_full_history

    if not artifact_path(model_id, artifact_dir, ".meta.json").exists():
        logger.warning("Tier-2 artefact %s is missing; its block will be unavailable", model_id)
        return None

    model = Tier2Model.load(model_id, artifact_dir)
    history = order_full_history(frames)
    matrix = model.spec.transform(history)
    windows, order = assemble_windows(matrix, history["account_id"], model.spec.window)

    errors = model.score_windows(windows)
    lengths = windows.lengths
    scoreable = lengths >= model.spec.min_length

    transaction_id = history["transaction_id"].astype(str).to_numpy()[order][windows.anchor_row]
    frame = pd.DataFrame(
        {
            "transaction_id": transaction_id,
            # The abstention convention: the error itself is sentinel-filled and the indicator
            # carries the fact of abstention separately. A short window is scored by the network
            # regardless, but that number is not a measurement the deployed system would act on.
            "tier2_error": np.where(scoreable, errors, ABSTENTION_SENTINEL).astype("float64"),
            "tier2_is_scoreable": scoreable.astype("float64"),
            "tier2_sequence_length": lengths.astype("float64"),
        }
    )
    coverage = float(scoreable.mean()) if scoreable.size else 0.0
    logger.info(
        "tier2: %d windows scored, %.1f%% at or above the min_length=%d floor",
        len(frame),
        100.0 * coverage,
        model.spec.min_length,
    )
    return Tier2Features(
        frame=frame,
        model_id=model_id,
        feature_version=model.spec.to_feature_definition().feature_version,
        min_length=int(model.spec.min_length),
        window=int(model.spec.window),
        coverage=coverage,
    )


# --- Tier-3 ring features ----------------------------------------------------------------

#: Ring columns carried into the meta matrix, before prefixing. The eleven structural features
#: the Phase 4 scorer reads, plus the exactness flag on the betweenness approximation.
#:
#: ``account_is_fraudulent`` is emphatically **not** here. ``run_snapshots`` computes it as
#: ``fraud_accounts(window_rows)`` -- a direct read of the label over the snapshot window -- to
#: supply the ring scorer's training target, and it travels in the same frame as these columns.
TIER3_CARRIED_COLUMNS: tuple[str, ...] = (
    "ring_size",
    "ring_density",
    "ring_mean_degree",
    "ring_max_degree",
    "ring_mean_idf",
    "ring_spec_count",
    "ring_entity_count",
    "account_degree",
    "account_degree_centrality",
    "account_betweenness",
    "account_entity_count",
    "betweenness_exact",
)


@dataclass(frozen=True)
class Tier3Features:
    """Per-transaction Tier-3 signals, rebuilt leak-free from rolling snapshots.

    Attributes:
        frame: ``transaction_id`` plus the prefixed ring columns and indicators.
        prefix: Column prefix, distinguishing the main graph from the non-circular control.
        snapshots: How many snapshot boundaries were walked.
        abstention_rate: Share of corpus transactions with no ring score.
        unscored_rows: Transactions in no snapshot's scoring period at all -- the first cadence
            block, which precedes the first boundary and so is scored by nobody.
        parameters: The graph parameters, for the registry.
    """

    frame: pd.DataFrame
    prefix: str
    snapshots: int
    abstention_rate: float
    unscored_rows: int
    parameters: dict[str, Any]


def build_tier3_features(
    frame: pd.DataFrame,
    *,
    factory: Any,
    boundaries: Any,
    cadence: pd.Timedelta,
    window: pd.Timedelta,
    prefix: str = "tier3_",
    seed: int = RANDOM_SEED,
) -> Tier3Features:
    """Rebuild Tier-3's per-transaction signals from rolling snapshots.

    Not from ``models/artifacts/tier3-*.json``. That artefact holds a single frozen snapshot
    whose ``snapshot_end`` is 2018-06-02 -- *after* the test period -- so joining its score
    table to train or validation rows would score them with a graph built from their own
    future. ``run_snapshots`` instead builds each graph from ``[t - window, t)`` and scores only
    ``[t, t + cadence)``, an invariant ``EntityGraph.snapshot`` re-checks rather than trusting.

    The ring scorer is refitted here because ``Tier3Model.save`` does not serialise it
    (``load`` returns ``scorer=None``). The refit is deterministic given the same snapshots,
    parameters and seed, but **that determinism is not currently covered by a test** -- there is
    no assertion anywhere that it reproduces the registered Phase 4 score table. Treat the
    equivalence as unverified until one exists.

    **A contamination this function does not remove.** The scorer is fitted on train rings and
    then applied to those same train rings, so ``ring_risk_score`` is in-sample on the train
    split in exactly the way Tier-1's raw score is, and with no out-of-fold remedy. That is the
    likely mechanism behind the Tier-3 block scoring far better in the train-fitted ablation
    column than on the validation arbiter. It does not reach the shipped model -- Tier-3 was
    retired on the validation column, which is uncontaminated -- but any future run that retains
    Tier-3 must fix this first.

    De-duplication is applied **only** to the scorer's fitting set. Phase 4 is explicit that it
    is a metric device: applying it to the scoring path would raise abstention from roughly 66%
    to 97% by discarding rings for resembling an earlier one, which is a statement about
    measurement independence, not about whether an account is currently in a ring.

    Args:
        frame: The full corpus, time-ordered, carrying a ``split`` column.
        factory: Builds a fresh graph per snapshot.
        boundaries: ``train_tier3.SplitBoundaries`` -- note this is *not*
            ``app.data.splitting.SplitBoundaries``; the two use opposite comparison conventions.
        cadence: How often the graph refreshes.
        window: Trailing width of each snapshot.
        prefix: Column prefix.
        seed: Passed to the scorer fit and logged.

    Returns:
        The per-transaction features and their provenance.
    """
    from app.models.tier3_graph import fit_ring_scorer
    from app.models.train_tier3 import (
        assign_rings_to_first_split,
        label_rings_with_split,
        run_snapshots,
        stack,
    )

    outcomes = run_snapshots(frame, factory, cadence=cadence, window=window, boundaries=boundaries)
    rings = stack(outcomes, "rings")
    scored = stack(outcomes, "scored")
    if rings.empty or scored.empty:
        raise RuntimeError(
            "the snapshot pass produced no rings or no scored rows; Tier-3 features cannot be "
            "built. Check the cadence and window against the corpus span."
        )

    # Fit on de-duplicated *train* rings only. Selecting the fitting set on anything later would
    # put validation or test structure into a column the meta-learner then reads. The split of a
    # ring is the split its *snapshot boundary* falls in, which is what decides whether the rows
    # it went on to score are training rows.
    rings = label_rings_with_split(rings, boundaries)
    deduplicated = assign_rings_to_first_split(rings)
    train_rings = deduplicated[deduplicated["split"] == "train"]
    if train_rings.empty:
        raise RuntimeError("no training-split rings survived de-duplication; cannot fit a scorer")
    scorer = fit_ring_scorer(
        train_rings,
        train_rings["account_is_fraudulent"].to_numpy(dtype=bool),
        "ieee_cis",
        seed=seed,
    )
    logger.info(
        "tier3[%s]: %d snapshots, %d ring rows (%d after de-duplication), scorer fitted on "
        "%d train rings",
        prefix.rstrip("_"),
        len(outcomes),
        len(rings),
        len(deduplicated),
        len(train_rings),
    )

    rings = rings.assign(ring_risk_score=scorer.score_frame(rings))

    # One row per (snapshot_end, account_id) is expected -- communities are disjoint, so an
    # account lands in exactly one -- but Phase 4's own join guards it with a max(), so guard
    # here too rather than trusting a property of somebody else's algorithm.
    keys = ["snapshot_end", "account_id"]
    duplicates = int(rings.duplicated(subset=keys).sum())
    if duplicates:
        logger.warning(
            "tier3[%s]: %d duplicate (snapshot_end, account_id) ring rows; keeping the largest "
            "ring per key",
            prefix.rstrip("_"),
            duplicates,
        )
        rings = (
            rings.sort_values([*keys, "ring_size", "ring_id"])
            .drop_duplicates(subset=keys, keep="last")
            .reset_index(drop=True)
        )

    carried = [column for column in TIER3_CARRIED_COLUMNS if column in rings.columns]
    merged = scored.merge(
        rings[[*keys, *carried, "ring_risk_score"]], on=keys, how="left", validate="many_to_one"
    )

    # Snapshot boundaries start one cadence after the corpus begins, so the earliest cadence
    # block is in no scoring period at all. Those rows are abstentions like any other, and they
    # are re-attached explicitly rather than silently dropped from the meta matrix.
    all_ids = frame["transaction_id"].astype(str)
    merged["transaction_id"] = merged["transaction_id"].astype(str)
    complete = pd.DataFrame({"transaction_id": all_ids}).merge(
        merged, on="transaction_id", how="left", validate="one_to_one"
    )
    unscored = int(
        complete["ring_risk_score"].isna().sum() - merged["ring_risk_score"].isna().sum()
    )

    scoreable = complete["ring_risk_score"].notna().to_numpy()
    # "Seen but not ringed" separates Tier-3's two abstention reasons, exactly as
    # ``Tier3Model.score`` does: an account the snapshot saw but placed in no qualifying ring is
    # a different state from an account the snapshot never saw at all.
    seen_not_ringed = (
        complete["snapshot_end"].notna().to_numpy() & ~scoreable
        if "snapshot_end" in complete.columns
        else np.zeros(len(complete), dtype=bool)
    )
    staleness = (
        complete["staleness_seconds"].to_numpy(dtype="float64") / 3600.0
        if "staleness_seconds" in complete.columns
        else np.full(len(complete), np.nan)
    )

    output: dict[str, Any] = {"transaction_id": complete["transaction_id"].to_numpy()}
    output[f"{prefix}ring_risk_score"] = np.nan_to_num(
        complete["ring_risk_score"].to_numpy(dtype="float64"), nan=ABSTENTION_SENTINEL
    )
    output[f"{prefix}is_scoreable"] = scoreable.astype("float64")
    output[f"{prefix}seen_not_ringed"] = seen_not_ringed.astype("float64")
    output[f"{prefix}staleness_hours"] = np.nan_to_num(staleness, nan=ABSTENTION_SENTINEL)
    output[f"{prefix}is_ring_member"] = (
        scoreable & (complete["ring_risk_score"].to_numpy(dtype="float64") >= 0.5)
    ).astype("float64")
    for column in carried:
        output[f"{prefix}{column}"] = np.nan_to_num(
            complete[column].to_numpy(dtype="float64"), nan=ABSTENTION_SENTINEL
        )

    result = pd.DataFrame(output)
    require_clean_feature_names([c for c in result.columns if c != "transaction_id"])
    abstention_rate = float((~scoreable).mean()) if len(complete) else 0.0
    logger.info(
        "tier3[%s]: %.1f%% of transactions abstained (%d in no scoring period at all)",
        prefix.rstrip("_"),
        100.0 * abstention_rate,
        unscored,
    )
    return Tier3Features(
        frame=result,
        prefix=prefix,
        snapshots=len(outcomes),
        abstention_rate=abstention_rate,
        unscored_rows=unscored,
        parameters={
            "cadence_hours": cadence.total_seconds() / 3600.0,
            "window_hours": window.total_seconds() / 3600.0,
            "seed": seed,
        },
    )


# --- Assembly ----------------------------------------------------------------------------

#: Carried through the join for the driver's use -- splitting, evaluation, cost -- and stripped
#: before the matrix is built. Every one of them is in :data:`META_DENIED_COLUMNS`.
CARRIER_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "event_time",
    "split",
    "is_fraud",
    "amount",
    "account_id",
)

#: Marks rows with no honest out-of-fold Tier-1 score -- block 1 of the train split. The driver
#: drops them from the fitting set. Not a feature: it exists only on train, so a model given it
#: would be reading which fold a row came from.
OOF_SCOREABLE_COLUMN = "meta_oof_scoreable"


def assemble_meta_frame(
    frames: dict[str, pd.DataFrame],
    *,
    oof: OofTier1,
    registered: RegisteredTier1,
    tier2: Tier2Features | None,
    tier3: Sequence[Tier3Features] = (),
) -> pd.DataFrame:
    """Join the engineered features and all three tier signals into one per-transaction frame.

    The Tier-1 column is assembled from two different sources on purpose, and this is the
    subtlest thing in the module:

    * **On train** it is the rank-normalised out-of-fold score, so no row's feature was produced
      by a model that had seen it.
    * **On validation and test** it is the *registered full-train* model's score, mapped through
      that model's own train-score CDF onto the same ``[0, 1]`` scale. Those splits were never
      in Tier-1's fit, so no out-of-fold machinery is needed there -- and using it would be
      wrong, since the deployed system scores them with the full-train model.

    The CDF is what makes the two halves commensurable. Without it the meta-learner would be
    fitted on percentile ranks and applied to raw probabilities, and every threshold it learned
    would land in the wrong place.

    Args:
        frames: The three splits, keyed ``train``/``val``/``test``.
        oof: Out-of-fold Tier-1 scores over the train split.
        registered: The deployed Tier-1 model, for the validation and test columns.
        tier2: Per-transaction Tier-2 signals, or ``None`` if the artefact was absent.
        tier3: Zero or more Tier-3 feature sets, distinguished by their prefixes.

    Returns:
        One row per transaction across all three splits, carrying the engineered block, the
        available tier blocks, the carrier columns and :data:`OOF_SCOREABLE_COLUMN`.
    """
    parts: list[pd.DataFrame] = []
    for split, frame in frames.items():
        carried = [column for column in CARRIER_COLUMNS if column in frame.columns]
        part = frame.loc[:, [*carried, *ENGINEERED_BLOCK]].copy()
        part["split"] = split

        if split == "train":
            part["tier1_score"] = oof.scores
            part[OOF_SCOREABLE_COLUMN] = oof.scoreable
        else:
            raw = np.asarray(registered.model.score_frame(frame), dtype="float64")
            part["tier1_score"] = oof.reference_cdf.apply(raw)
            part[OOF_SCOREABLE_COLUMN] = True
        parts.append(part)

    combined = pd.concat(parts, ignore_index=True)
    combined["transaction_id"] = combined["transaction_id"].astype(str)

    if tier2 is not None:
        combined = combined.merge(
            tier2.frame, on="transaction_id", how="left", validate="one_to_one"
        )
        # A transaction with no assembled window at all -- Tier-2 never saw it, as opposed to
        # saw it and found the history too short. Both are abstentions and both take the
        # sentinel, but neither takes a zero.
        combined["tier2_error"] = combined["tier2_error"].fillna(ABSTENTION_SENTINEL)
        combined["tier2_is_scoreable"] = combined["tier2_is_scoreable"].fillna(0.0)
        combined["tier2_sequence_length"] = combined["tier2_sequence_length"].fillna(0.0)

    for features in tier3:
        combined = combined.merge(
            features.frame, on="transaction_id", how="left", validate="one_to_one"
        )
        for column in features.frame.columns:
            if column == "transaction_id":
                continue
            fill = (
                0.0 if column.endswith(("is_scoreable", "seen_not_ringed")) else ABSTENTION_SENTINEL
            )
            combined[column] = combined[column].fillna(fill)

    logger.info(
        "meta frame: %d rows, %d columns, %d with an out-of-fold Tier-1 score",
        len(combined),
        combined.shape[1],
        int(combined[OOF_SCOREABLE_COLUMN].sum()),
    )
    return combined


def available_blocks(frame: pd.DataFrame, prefix: str = "tier3_") -> tuple[str, ...]:
    """Return the feature blocks fully present in ``frame``, in canonical order.

    A block is available only when *every* one of its columns is present. A partially present
    block would silently change what "retiring tier2" means between runs.
    """
    present = set(frame.columns)
    available = tuple(
        block
        for block, names in FEATURE_BLOCKS.items()
        if set(_prefixed(names, prefix)).issubset(present)
    )
    missing = tuple(block for block in FEATURE_BLOCKS if block not in available)
    if missing:
        logger.warning("feature block(s) unavailable and excluded from the ablation: %s", missing)
    return available


def _prefixed(names: Sequence[str], prefix: str) -> tuple[str, ...]:
    """Re-point ``tier3_*`` names at an alternative prefix, leaving other blocks alone."""
    if prefix == "tier3_":
        return tuple(names)
    return tuple(
        f"{prefix}{name[len('tier3_'):]}" if name.startswith("tier3_") else name for name in names
    )


def matrix_for(
    frame: pd.DataFrame, blocks: Sequence[str], prefix: str = "tier3_"
) -> tuple[npt.NDArray[np.float64], tuple[str, ...]]:
    """Return the model matrix for ``blocks`` and the feature names behind its columns.

    Raises:
        ValueError: If a denied column reached the selection, or a named feature is absent.
    """
    names = _prefixed(block_feature_names(blocks), prefix)
    require_clean_feature_names(names)
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise ValueError(f"feature(s) absent from the assembled frame: {absent}")
    return frame.loc[:, list(names)].to_numpy(dtype="float64"), names
