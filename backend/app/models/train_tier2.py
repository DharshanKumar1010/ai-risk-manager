"""Train, compare and register the Tier-2 behavioural layer.

Usage::

    python -m app.models.train_tier2
    python -m app.models.train_tier2 --sample 50000 --epochs 3 --skip-registry  # iteration

**IEEE-CIS only.** Phase 1 measured PaySim at 2,768,630 accounts of which 99.9% hold a single
transaction, a maximum of 3, and ``seconds_since_prior_txn`` 99.94% null. A sequence model
over a corpus with no sequences still emits a number, and the number would mean nothing —
the same trap Phase 2 fell into and documented with PaySim's Tier-1 result. The exclusion is
a measured finding and is reported as one, not a silent omission.

**What is being controlled.** An autoencoder that reconstructs everything well separates
nothing, and one that reconstructs short windows well is a sequence-length sensor. So the
comparison carries four references besides the model itself:

===============================  ==========================================================
candidate                        isolates
===============================  ==========================================================
no-skill floor                   what PR-AUC means at the account level
Tier-1, aggregated to account    **whether Tier-2 earns anything at all** over the layer
                                 that already ships
max amount_log per account       whether the autoencoder is just finding big-ticket accounts
dense autoencoder                whether *sequence order* does the work, or the aggregate
                                 alone — the analogue of Tier-1's matched-inputs LightGBM
===============================  ==========================================================

**The split discipline.** The autoencoder fits on fraud-free train windows. Early stopping
reads a clean-validation reconstruction loss — the model's own objective. The latent size,
the window, the abstention threshold N, the operating threshold and the choice of which model
ships all read **validation**. Test is scored exactly once, after every one of those
decisions is already made. Phase 2's recorded lesson was that the phase brief followed
literally produces a contaminated selection; the same applies here and is guarded by
``tests/test_tier2.py::test_model_selection_reads_validation_not_test``.

**Two decisions that both read validation, for different things.** Early stopping reads
reconstruction loss; candidate selection reads PR-AUC. Stopping on PR-AUC would tune the
model against the very metric it is judged by, which is a subtler version of the same
contamination.
"""

import argparse
import copy
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn

from app.config import Settings, get_settings
from app.data.raw_spec import SourceDataset
from app.data.schema import SPLIT_ORDER, TransactionFeatures
from app.data.splitting import find_boundary_overlaps
from app.ml.cost import (
    DEFAULT_CHARGEBACK_FEE,
    DEFAULT_REVIEW_COST,
    CostEstimate,
    CostModel,
    choose_threshold_by_cost,
    cost_at_threshold,
    render_sensitivity,
    review_cost_sweep,
    sensitivity_sweep,
    threshold_for_flag_rate,
)
from app.ml.evaluation import (
    LEAK_SUSPICION_PR_AUC,
    EvaluationResult,
    bootstrap_pr_auc,
    bootstrap_pr_auc_delta,
    evaluate,
    pr_auc,
)
from app.ml.registry import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_REGISTRY_PATH,
    RegistryEntry,
    append_entry,
    build_model_id,
)
from app.models.tier2_behavioral import (
    LATENCY_BENCHMARK_CALLS,
    LATENCY_BUDGET_P95_MS,
    Tier2Model,
    benchmark_latency,
    build_network,
    masked_reconstruction_error,
)
from app.models.tier2_sequences import (
    DEFAULT_WINDOW,
    MIN_SEQUENCE_LENGTH,
    SequenceWindows,
    Tier2SequenceSpec,
    aggregate_to_accounts,
    assemble_windows,
    coverage,
    derive_timestep_frame,
    eligible_training_rows,
    find_future_reads,
    fit_sequence_spec,
    order_full_history,
)

logger = logging.getLogger("riskiq.tier2")

#: Set and logged, per ml-evaluation-standards section 5.
RANDOM_SEED = 42

#: The only corpus Tier-2 trains on. See the module docstring.
SOURCE: SourceDataset = "ieee_cis"

IEEE_COST_UNITS = "IEEE-CIS amount units (consistent with USD)"

#: Ceiling on the share of accounts the capacity-constrained operating point may flag,
#: matching Tier-1's transaction-level ceiling so the two are read on the same scale.
MAX_REVIEW_FLAG_RATE = 0.01

#: LSTM hidden units, encoder and decoder alike. The phase brief's figure — kept, because the
#: capacity that actually decides whether this layer works is the *latent* size below, and
#: holding the hidden size fixed is what makes the latent sweep a controlled comparison.
HIDDEN_SIZE = 128

#: Bottleneck sizes swept at :data:`DEFAULT_WINDOW`, selected on validation PR-AUC.
#:
#: This sweep is the phase's main risk control. At 21 features over 10 timesteps the input is
#: 210 dimensions; an autoencoder with a latent anywhere near that can learn the identity map,
#: reconstruct fraud exactly as faithfully as normal behaviour, and produce two error
#: distributions sitting on top of each other. The brief's ``hidden_size=128`` comes from a
#: prior project that does not exist in this repo, and its capacity ratio does not transfer.
LATENT_GRID: tuple[int, ...] = (8, 16, 32)

#: Window lengths swept at the winning latent, alongside :data:`DEFAULT_WINDOW`.
#:
#: Staged rather than a full grid: 3 latents at W=10, then 3 further windows at the winning
#: latent, is 6 fits instead of 12. The report says it is staged rather than implying the whole
#: product space was searched — a coordinate descent can miss an interaction between window and
#: latent, and claiming otherwise would be a false claim about the search.
#:
#: W=20 is retained beyond the {5, 10, 15} brief because it won the first full run outright;
#: dropping the empirically best window to match a grid would be discarding a measured result.
WINDOW_GRID: tuple[int, ...] = (5, 15, 20)

#: Abstention thresholds swept on validation. N is a **scoring-time** policy, not a training
#: one: the autoencoder fits on every eligible window of at least
#: :data:`MIN_SEQUENCE_LENGTH`, and N only decides which windows get an opinion. So the sweep
#: costs no extra training and the model is bit-identical across it.
MIN_LENGTH_GRID: tuple[int, ...] = (3, 5, 10)

TRAINING_EPOCHS = 30
TRAINING_BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 4

#: Deciles printed as a text table beside every distribution plot. ``notebooks/README.md`` is
#: explicit that a number existing only inside an image is not a result.
DECILES: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


# ==========================================================================================
# Assembled data
# ==========================================================================================


@dataclass(frozen=True)
class SplitWindows:
    """One split's anchored windows plus everything evaluation needs about them."""

    split: str
    windows: SequenceWindows
    labels: npt.NDArray[np.bool_]
    amounts: npt.NDArray[np.float64]
    account_id: npt.NDArray[np.object_]
    lengths: npt.NDArray[np.int32]
    #: Windows whose oldest timestep belongs to an earlier split. Legitimate — a real scorer
    #: has the account's history — but counted and reported rather than left for a reviewer
    #: to discover.
    reaches_earlier_split: int

    def scoreable(self, min_length: int) -> npt.NDArray[np.bool_]:
        """Return which windows are long enough for the model to have an opinion on."""
        return self.lengths >= min_length


@dataclass(frozen=True)
class AssembledCorpus:
    """The full history, windowed once, sliced per split."""

    spec: Tier2SequenceSpec
    history: pd.DataFrame
    order: npt.NDArray[np.intp]
    all_windows: SequenceWindows
    per_split: dict[str, SplitWindows]
    train_eligible: npt.NDArray[np.bool_]
    clean_val: npt.NDArray[np.bool_]
    training_window: dict[str, str]
    rows: dict[str, int]


def load_splits(
    processed_dir: Path, source: SourceDataset, sample: int | None
) -> dict[str, pd.DataFrame]:
    """Load the three chronological splits Phase 1 wrote.

    Identical in contract to Tier-1's loader, including the re-assertion that the files on
    disk still honour the Phase 1 boundary guarantee. Every metric below is meaningless if
    they do not, so it is checked rather than assumed.

    Raises:
        FileNotFoundError: If a split parquet is missing.
        RuntimeError: If the loaded splits are not strictly time-ordered.
    """
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLIT_ORDER:
        path = processed_dir / f"{source}_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m app.data.pipeline` to build the feature store."
            )
        frame = pd.read_parquet(path)
        frames[split] = frame.head(sample) if sample is not None else frame
        logger.info("%s/%s: loaded %d rows from %s", source, split, len(frames[split]), path.name)

    combined = pd.concat(
        [frame.assign(split=split) for split, frame in frames.items()], ignore_index=True
    )
    overlaps = find_boundary_overlaps(combined)
    if overlaps:
        raise RuntimeError(f"{source}: split boundaries overlap on disk — {overlaps}")
    return frames


def assemble(frames: dict[str, pd.DataFrame], window: int) -> AssembledCorpus:
    """Window the whole history once and slice it per split.

    Windows are built over the account's complete history and assigned to the **anchor's**
    split, so a test window may legitimately reach back into the train period — that is what
    a real scorer has in front of it. The reach is one-directional by construction and
    :func:`find_future_reads` re-checks the result against the timestamps rather than
    trusting the construction, which is the form ml-evaluation-standards section 1 asks a
    leakage check to take.
    """
    history = order_full_history(frames)
    split_column = history["split"].to_numpy()

    train_mask = split_column == "train"
    eligible_within_train = eligible_training_rows(history.loc[train_mask])
    train_eligible_rows = np.zeros(len(history), dtype=bool)
    train_eligible_rows[np.flatnonzero(train_mask)] = eligible_within_train

    spec = fit_sequence_spec(
        history.loc[train_mask],
        eligible_within_train,
        SOURCE,
        window=window,
        min_length=MIN_SEQUENCE_LENGTH,
    )
    logger.info(
        "tier2 feature_version=%s, %d per-timestep features, W=%d",
        spec.to_feature_definition().feature_version,
        spec.n_features,
        window,
    )

    matrix = spec.transform(history)
    all_windows, order = assemble_windows(matrix, history["account_id"], window)

    event_time_ns = np.asarray(
        history["event_time"].astype("int64").to_numpy()[order], dtype=np.int64
    )
    future_reads = find_future_reads(all_windows, event_time_ns)
    if future_reads:
        raise RuntimeError(
            f"{future_reads} assembled window(s) contain a timestep later than their own "
            "anchor. Every held-out number below would be fabricated; refusing to continue."
        )
    logger.info("windows: %d assembled, 0 read forward of their anchor", len(all_windows))

    # Everything below is indexed like the base matrix (account-then-time), so per-row
    # metadata is reordered once here rather than at each use.
    ordered_split = split_column[order]
    ordered_labels = history["is_fraud"].to_numpy(dtype=bool)[order]
    ordered_amounts = history["amount"].to_numpy(dtype="float64")[order]
    ordered_accounts = history["account_id"].to_numpy()[order]
    ordered_eligible = train_eligible_rows[order]

    anchor = all_windows.anchor_row
    oldest = all_windows.gather[:, 0]
    split_rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    anchor_rank = np.array([split_rank[value] for value in ordered_split[anchor]])
    oldest_rank = np.array([split_rank[value] for value in ordered_split[oldest]])

    per_split: dict[str, SplitWindows] = {}
    for split in SPLIT_ORDER:
        keep = ordered_split[anchor] == split
        per_split[split] = SplitWindows(
            split=split,
            windows=all_windows.select(keep),
            labels=ordered_labels[anchor][keep],
            amounts=ordered_amounts[anchor][keep],
            account_id=ordered_accounts[anchor][keep].astype(object),
            lengths=all_windows.lengths[keep],
            reaches_earlier_split=int(np.sum((oldest_rank < anchor_rank)[keep])),
        )
        logger.info(
            "%s: %d windows, %d positive, %d reach into an earlier split",
            split,
            len(per_split[split].windows),
            int(per_split[split].labels.sum()),
            per_split[split].reaches_earlier_split,
        )

    # The autoencoder's training set: train-anchored, account clean across the whole train
    # split, and long enough to carry a pattern at all.
    train_eligible = (
        (ordered_split[anchor] == "train")
        & ordered_eligible[anchor]
        & (all_windows.lengths >= MIN_SEQUENCE_LENGTH)
    )

    # Early stopping reads reconstruction loss on *normal* validation sequences. Fraud-bearing
    # accounts are excluded for the same reason fraud-bearing train accounts are: on this
    # corpus a chargeback propagates forward, so the account rather than the row is the unit
    # that is compromised.
    #
    # Only train and validation labels define "fraud-bearing" here. Reading the test label —
    # even only to decide which sequences the early-stopping loss averages over — would let
    # test information choose when training stops, which is the contamination
    # ml-evaluation-standards section 1 forbids. `isin` goes through pandas rather than
    # `np.isin` because these are object-dtype account ids: hashing is O(n), numpy's sort-based
    # path is not, and at 590k rows the difference is minutes.
    anchor_accounts = pd.Series(ordered_accounts[anchor])
    labelled_before_test = np.isin(ordered_split[anchor], ("train", "val"))
    known_fraud_accounts = set(anchor_accounts[ordered_labels[anchor] & labelled_before_test])
    clean_val = (
        (ordered_split[anchor] == "val")
        & (all_windows.lengths >= MIN_SEQUENCE_LENGTH)
        & ~anchor_accounts.isin(known_fraud_accounts).to_numpy()
    )
    logger.info(
        "autoencoder fits on %d fraud-free train windows; early stopping on %d clean "
        "validation windows",
        int(train_eligible.sum()),
        int(clean_val.sum()),
    )

    train_frame = frames["train"]
    return AssembledCorpus(
        spec=spec,
        history=history,
        order=order,
        all_windows=all_windows,
        per_split=per_split,
        train_eligible=train_eligible,
        clean_val=clean_val,
        training_window={
            "start": str(train_frame["event_time"].min()),
            "end": str(train_frame["event_time"].max()),
        },
        rows={split: len(frame) for split, frame in frames.items()},
    )


# ==========================================================================================
# Training
# ==========================================================================================


@dataclass(frozen=True)
class TrainingEpoch:
    """One epoch's losses, for the convergence check the phase gates on."""

    epoch: int
    train_loss: float
    validation_loss: float


def _mean_masked_error(
    network: nn.Module,
    windows: SequenceWindows,
    index: npt.NDArray[np.intp],
    batch_size: int,
) -> float:
    """Return the mean masked reconstruction error over ``index``, without gradients."""
    total, count = 0.0, 0
    network.eval()
    with torch.no_grad():
        for start in range(0, index.size, batch_size):
            chunk = index[start : start + batch_size]
            values, mask = windows.batch(chunk)
            value_tensor = torch.from_numpy(values)
            mask_tensor = torch.from_numpy(mask)
            lengths = mask_tensor.sum(dim=1).to(torch.int64).clamp(min=1)
            errors = masked_reconstruction_error(
                value_tensor, network(value_tensor, lengths), mask_tensor
            )
            total += float(errors.sum())
            count += chunk.size
    return total / count if count else 0.0


def fit_autoencoder(
    algorithm: str,
    spec: Tier2SequenceSpec,
    windows: SequenceWindows,
    train_index: npt.NDArray[np.intp],
    validation_index: npt.NDArray[np.intp],
    *,
    model_id: str,
    latent_size: int,
    hidden_size: int = HIDDEN_SIZE,
    epochs: int = TRAINING_EPOCHS,
    batch_size: int = TRAINING_BATCH_SIZE,
) -> tuple[Tier2Model, list[TrainingEpoch]]:
    """Fit one autoencoder on fraud-free windows, early-stopping on clean validation loss.

    The weights restored at the end are the best-validation-loss weights, not the last epoch's
    — otherwise "early stopping" would detect overfitting and then ship the overfitted model
    anyway.

    Returns:
        The model with a placeholder threshold, and the per-epoch loss curve.
    """
    torch.manual_seed(RANDOM_SEED)
    network = build_network(
        algorithm=algorithm, spec=spec, hidden_size=hidden_size, latent_size=latent_size
    )
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(RANDOM_SEED)

    curve: list[TrainingEpoch] = []
    best_loss = float("inf")
    best_state = copy.deepcopy(network.state_dict())
    since_improvement = 0

    for epoch in range(1, epochs + 1):
        network.train()
        shuffled = rng.permutation(train_index)
        running, seen = 0.0, 0
        for start in range(0, shuffled.size, batch_size):
            chunk = shuffled[start : start + batch_size]
            values, mask = windows.batch(chunk)
            value_tensor = torch.from_numpy(values)
            mask_tensor = torch.from_numpy(mask)
            lengths = mask_tensor.sum(dim=1).to(torch.int64).clamp(min=1)

            optimiser.zero_grad()
            loss = masked_reconstruction_error(
                value_tensor, network(value_tensor, lengths), mask_tensor
            ).mean()
            # torch types Tensor.backward as untyped; it is the standard training call and
            # there is no annotated alternative.
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
            # `detach` before the scalar conversion: reading a grad-tracking tensor as a
            # float warns, and this project's pytest config turns warnings into failures.
            running += float(loss.detach()) * chunk.size
            seen += chunk.size

        train_loss = running / seen if seen else 0.0
        validation_loss = _mean_masked_error(network, windows, validation_index, batch_size)
        curve.append(TrainingEpoch(epoch, train_loss, validation_loss))
        logger.info(
            "%s: epoch %2d  train %.6f  clean-val %.6f",
            model_id,
            epoch,
            train_loss,
            validation_loss,
        )

        if validation_loss < best_loss:
            best_loss, since_improvement = validation_loss, 0
            best_state = copy.deepcopy(network.state_dict())
        else:
            since_improvement += 1
            if since_improvement >= EARLY_STOPPING_PATIENCE:
                logger.info(
                    "%s: early stop at epoch %d (best clean-val %.6f)", model_id, epoch, best_loss
                )
                break

    network.load_state_dict(best_state)
    model = Tier2Model(
        model_id=model_id,
        algorithm=algorithm,
        spec=spec,
        threshold=float("inf"),  # replaced once chosen on validation
        network=network,
        hyperparameters={
            "hidden_size": hidden_size,
            "latent_size": latent_size,
            "window": spec.window,
            "epochs_run": len(curve),
            "epochs_max": epochs,
            "batch_size": batch_size,
            "learning_rate": LEARNING_RATE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "optimiser": "adam",
            "loss": "masked mean squared error over real timesteps",
            "best_clean_validation_loss": best_loss,
        },
    )
    return model, curve


# ==========================================================================================
# Account-level scoring
# ==========================================================================================


#: Score given to an account Tier-2 abstained on, in the system-level evaluation.
#:
#: It encodes "Tier-2 never flags this account", which is the truth of the deployed system —
#: **not** "Tier-2 judged this account normal", which would be a fabrication. The two readings
#: differ and the distinction is why Phase 5's meta-learner receives ``is_scoreable`` rather
#: than this sentinel.
ABSTAINED_RANK_SENTINEL = -1.0

#: Minimum length that abstains on nothing — every window is at least one transaction long.
#: Used for the references, which have no abstention rule of their own.
NO_ABSTENTION = 1


@dataclass(frozen=True)
class AccountScores:
    """One split collapsed to one row per account."""

    scores: npt.NDArray[np.float64]
    labels: npt.NDArray[np.bool_]
    amounts: npt.NDArray[np.float64]
    n_accounts: int
    n_abstained: int

    @property
    def abstention_rate(self) -> float:
        """Return the share of accounts Tier-2 had no opinion on."""
        return self.n_abstained / self.n_accounts if self.n_accounts else 0.0


def to_accounts(
    split: SplitWindows,
    window_scores: npt.NDArray[np.float64],
    min_length: int,
    *,
    scoreable_only: bool,
) -> AccountScores:
    """Collapse per-window scores to per-account, within this split.

    Args:
        split: The split's windows and anchor metadata.
        window_scores: Per-window reconstruction error.
        min_length: Windows shorter than this abstain and contribute no score.
        scoreable_only: When True, accounts with no scoreable window are dropped entirely —
            the model's own performance. When False they are kept at
            :data:`ABSTAINED_RANK_SENTINEL` — the deployed system's performance, which is the
            honest basis for any system-level claim and the one candidate selection uses,
            because it holds the account set fixed across candidates so a paired bootstrap
            is valid.
    """
    masked = np.where(split.scoreable(min_length), window_scores, np.nan)
    scores, labels, amounts, n_accounts = aggregate_to_accounts(
        split.account_id, masked, split.labels, split.amounts, how="max"
    )
    abstained = np.isnan(scores)
    if scoreable_only:
        keep = ~abstained
        return AccountScores(
            scores=scores[keep],
            labels=labels[keep],
            amounts=amounts[keep],
            n_accounts=int(keep.sum()),
            n_abstained=0,
        )
    return AccountScores(
        scores=np.where(abstained, ABSTAINED_RANK_SENTINEL, scores),
        labels=labels,
        amounts=amounts,
        n_accounts=n_accounts,
        n_abstained=int(abstained.sum()),
    )


# ==========================================================================================
# Candidates
# ==========================================================================================


@dataclass
class Candidate:
    """One scorer and its measured results, at the account level."""

    name: str
    model: Tier2Model | None
    min_length: int
    #: PR-AUC on validation accounts. **This is the selection criterion.** Selecting on the
    #: test result would make test a validation set and invalidate the headline
    #: (ml-evaluation-standards section 1).
    validation_pr_auc: float
    validation_threshold: float
    validation_cost: CostEstimate | None
    test_accounts: AccountScores
    result: EvaluationResult
    #: The windowing this candidate was trained and scored against. Candidates from the
    #: window sweep are assembled at different W, so re-measuring the winner after selection
    #: has to go back to *its* corpus — scoring a W=20 model against W=10 windows is a shape
    #: mismatch, and was a real bug caught by ``test_model_selection_reads_validation_not_test``.
    corpus: "AssembledCorpus | None" = None
    curve: list[TrainingEpoch] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def choose_and_evaluate(
    name: str,
    model: Tier2Model | None,
    validation: AccountScores,
    test: AccountScores,
    cost_model: CostModel,
    min_length: int,
    *,
    notes: Sequence[str] = (),
    curve: Sequence[TrainingEpoch] = (),
    corpus: "AssembledCorpus | None" = None,
) -> Candidate:
    """Pick the operating threshold on validation accounts, then measure once on test."""
    threshold, validation_cost = choose_threshold_by_cost(
        validation.labels, validation.scores, validation.amounts, cost_model
    )
    result = evaluate(
        name,
        "test",
        test.labels,
        test.scores,
        threshold=threshold,
        threshold_criterion="minimising estimated cost on validation accounts",
        interval=bootstrap_pr_auc(test.labels, test.scores, seed=RANDOM_SEED),
        cost=cost_at_threshold(test.labels, test.scores, test.amounts, threshold, cost_model),
        notes=notes,
    )
    if model is not None:
        model.threshold = threshold
    return Candidate(
        name=name,
        model=model,
        min_length=min_length,
        validation_pr_auc=pr_auc(validation.labels, validation.scores),
        validation_threshold=threshold,
        validation_cost=validation_cost,
        test_accounts=test,
        result=result,
        corpus=corpus,
        curve=list(curve),
        notes=list(notes),
    )


def load_tier1_account_scores(
    corpus: AssembledCorpus,
    artifact_dir: Path,
    registry_path: Path,
) -> dict[str, npt.NDArray[np.float64]] | None:
    """Score the registered Tier-1 model, for the baseline Tier-2 has to beat.

    **The decisive comparison in this phase.** Tier-2 costs a second model, a second feature
    definition and a serving path that needs an account's history; if aggregating Tier-1's
    existing per-transaction scores to the account level does as well, none of that is earned,
    and that is the finding rather than a disappointment to be buried.

    The model is rebuilt from its registry artefact rather than retrained. Only the scores are
    needed, so ``dropped`` — which feeds ``feature_version`` and nothing else — is not
    reconstructed; this object is never saved, registered or served.

    Returns:
        Per-window Tier-1 scores per split, aligned to ``corpus.per_split``, or None when the
        artefact is absent. Absence is logged and carried into the report as a stated gap,
        never silently skipped.
    """
    import lightgbm as lgb

    from app.models.tier1_anomaly import ScoreNormaliser, Tier1Model
    from app.models.tier1_features import Tier1InputSpec

    if not registry_path.exists():
        logger.warning("%s does not exist; the Tier-1 baseline will be omitted", registry_path)
        return None
    entries = [
        entry
        for entry in json.loads(registry_path.read_text(encoding="utf-8"))
        if entry.get("layer") == "tier1_anomaly" and entry.get("source_dataset") == SOURCE
    ]
    if not entries:
        logger.warning("no Tier-1 %s entry in the registry; the baseline will be omitted", SOURCE)
        return None

    model_id = str(entries[-1]["model_id"])
    sidecar_path = artifact_dir / f"{model_id}.meta.json"
    booster_path = artifact_dir / f"{model_id}.txt"
    if not sidecar_path.exists() or not booster_path.exists():
        logger.warning(
            "Tier-1 artefact %s is not on disk; the baseline will be omitted and the report "
            "will say so",
            model_id,
        )
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
    logger.info("Tier-1 baseline loaded from %s", model_id)

    # Tier-1 scores a transaction; Tier-2 anchors a window at one. Scoring the whole history
    # once and indexing by anchor keeps the two aligned to the same row.
    history_scores = np.full(len(corpus.history), np.nan, dtype="float64")
    for split in SPLIT_ORDER:
        rows = corpus.history["split"].to_numpy() == split
        history_scores[rows] = tier1.score_frame(corpus.history.loc[rows])

    ordered = history_scores[corpus.order]
    return {
        split: ordered[corpus.all_windows.anchor_row][
            corpus.history["split"].to_numpy()[corpus.order][corpus.all_windows.anchor_row] == split
        ]
        for split in SPLIT_ORDER
    }


def amount_max_scores(split: SplitWindows) -> npt.NDArray[np.float64]:
    """Return a per-window ranking by the anchor's own amount.

    The reference every model must beat to have earned its complexity — the account-level
    analogue of Tier-1's ``amount-only ranking``. It is not a strawman: aggregated with
    ``max``, "the account's largest transaction" is a genuinely competitive fraud ranking.
    """
    return np.log1p(split.amounts)


# ==========================================================================================
# Diagnostics
# ==========================================================================================


def spearman(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> float:
    """Return Spearman rank correlation, via Pearson on ranks.

    Computed here rather than pulled from scipy: it is four lines, and scipy is not otherwise
    a dependency of this project.
    """
    if left.size < 2:
        return 0.0
    left_rank = pd.Series(left).rank().to_numpy()
    right_rank = pd.Series(right).rank().to_numpy()
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


#: Window-length bands the stratified diagnostic reports PR-AUC within.
LENGTH_BANDS: tuple[tuple[str, int, int], ...] = (
    ("3-4 transactions", 3, 4),
    ("5-9 transactions", 5, 9),
    ("10+ transactions", 10, 1_000_000),
)


def length_stratified_pr_auc(
    split: SplitWindows,
    window_scores: npt.NDArray[np.float64],
    min_length: int,
) -> list[tuple[str, int, int, float, float]]:
    """Return account-level PR-AUC within bands of window length.

    The companion to the error-vs-length correlation, and the more decisive of the two. If
    reconstruction error rises with sequence length — it does, because regenerating more
    timesteps from one fixed latent is genuinely harder — and fraudulent accounts also hold
    more transactions, then part of the headline is the model ranking accounts by how much
    history they have rather than by how unusual it is.

    Holding length roughly constant is what separates the two. A layer that still separates
    fraud *within* a band has learned something about behaviour; one whose PR-AUC collapses
    to the band's base rate was reading depth.

    Returns:
        ``(band, accounts, positives, base_rate, pr_auc)`` per band.
    """
    table = pd.DataFrame(
        {
            "account_id": pd.Series(list(split.account_id)),
            "score": np.where(split.scoreable(min_length), window_scores, np.nan),
            "is_fraud": split.labels,
            "length": split.lengths,
        }
    )
    grouped = table.groupby("account_id", sort=True).agg(
        score=("score", "max"), is_fraud=("is_fraud", "any"), length=("length", "max")
    )
    grouped = grouped[grouped["score"].notna()]

    rows: list[tuple[str, int, int, float, float]] = []
    for label, low, high in LENGTH_BANDS:
        band = grouped[(grouped["length"] >= low) & (grouped["length"] <= high)]
        labels = band["is_fraud"].to_numpy(dtype=bool)
        scores = band["score"].to_numpy(dtype="float64")
        rows.append(
            (
                label,
                int(len(band)),
                int(labels.sum()),
                float(labels.mean()) if labels.size else 0.0,
                pr_auc(labels, scores),
            )
        )
    return rows


def decile_table(
    title: str,
    groups: dict[str, npt.NDArray[np.float64]],
) -> str:
    """Return a text table of the reconstruction-error distributions.

    Printed beside the plot the phase gates on, because a distribution that exists only as a
    PNG cannot be checked, diffed or quoted — ``notebooks/README.md`` says as much.
    """
    header = "  " + "".join(f"{100 * q:>10.0f}%" for q in DECILES)
    lines = [title, f"{'':<28}{header}", f"{'':<28}  " + "-" * (11 * len(DECILES))]
    for name, values in groups.items():
        if values.size == 0:
            lines.append(f"{name:<28}  (no rows)")
            continue
        cells = "".join(f"{np.quantile(values, q):>11.5f}" for q in DECILES)
        lines.append(f"{name:<28} n={values.size:<8,}{cells}")
    return "\n".join(lines)


def plot_error_distribution(
    normal: npt.NDArray[np.float64],
    fraud: npt.NDArray[np.float64],
    threshold: float,
    path: Path,
) -> None:
    """Write the reconstruction-error distribution plot the phase verification requires."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positive = np.concatenate([normal[normal > 0], fraud[fraud > 0]])
    if positive.size == 0:
        return
    bins = np.logspace(np.log10(max(positive.min(), 1e-9)), np.log10(positive.max()), 80).tolist()

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(normal, bins=bins, density=True, alpha=0.6, label=f"legitimate (n={normal.size:,})")
    axis.hist(fraud, bins=bins, density=True, alpha=0.6, label=f"fraud (n={fraud.size:,})")
    # Only draw the operating point when it lands inside the plotted range. On a log axis a
    # non-positive threshold cannot be drawn at all, and the cost-minimising account-level
    # threshold collapses to "flag everything" — so without this guard the legend advertises
    # a line that is not on the chart, on one of the two artefacts the phase is gated on.
    if np.isfinite(threshold) and threshold > float(bins[0]):
        axis.axvline(threshold, linestyle="--", linewidth=1.4, label=f"threshold {threshold:.4f}")
    else:
        axis.set_xlabel(
            "masked reconstruction error (log scale)\n"
            "cost-minimising threshold flags every account and is off-scale; "
            "see the capacity-constrained operating point"
        )
    axis.set_xscale("log")
    if axis.get_xlabel() == "":
        axis.set_xlabel("masked reconstruction error (log scale)")
    axis.set_ylabel("density")
    axis.set_title("Tier-2 reconstruction error — held-out test accounts")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    logger.info("wrote %s", path)


def plot_loss_curve(curve: Sequence[TrainingEpoch], path: Path) -> None:
    """Write the training/validation loss curve, for the convergence check."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    epochs = [point.epoch for point in curve]
    axis.plot(epochs, [point.train_loss for point in curve], marker="o", label="train (fraud-free)")
    axis.plot(
        epochs,
        [point.validation_loss for point in curve],
        marker="s",
        label="validation (clean accounts)",
    )
    axis.set_xlabel("epoch")
    axis.set_ylabel("masked mean squared reconstruction error")
    axis.set_title("Tier-2 training curve — selected model")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    logger.info("wrote %s", path)


@dataclass(frozen=True)
class ComplementarityProfile:
    """Whether Tier-2 sees anything Tier-1 does not.

    The head-to-head comparison answers "is Tier-2 better than Tier-1", and the answer is
    almost bound to be no: Tier-1 is supervised on the label and Tier-2 is one-class. But
    Tier-2 is not built to replace Tier-1 — Phase 5 **fuses** them, and a weaker signal that
    is uncorrelated with a stronger one can still add ranking power to the pair while losing
    to it alone.

    So the question that actually decides whether this layer earned its place is not which
    wins, but whether Tier-2 separates fraud **among the accounts Tier-1 has already decided
    are safe**. That is the number below, and it is what Phase 5 inherits.
    """

    rank_correlation: float
    accounts_below_tier1_threshold: int
    positives_below_tier1_threshold: int
    base_rate_below_tier1_threshold: float
    tier2_pr_auc_below_tier1_threshold: float

    @property
    def lift_on_tier1_misses(self) -> float:
        """Return Tier-2's PR-AUC as a multiple of the floor, among Tier-1's non-flags."""
        rate = self.base_rate_below_tier1_threshold
        return self.tier2_pr_auc_below_tier1_threshold / rate if rate else 0.0

    def describe(self) -> list[str]:
        """Return plain-language lines for the report."""
        return [
            f"Spearman rank correlation between the two layers' account scores: "
            f"**{self.rank_correlation:+.4f}**. Near zero means the layers are looking at "
            "different things, which is the precondition for fusion adding anything.",
            f"Among the {self.accounts_below_tier1_threshold:,} test accounts Tier-1 leaves "
            f"below its own capacity threshold — the accounts a Tier-1-only system does not "
            f"review — {self.positives_below_tier1_threshold:,} are fraudulent "
            f"({100 * self.base_rate_below_tier1_threshold:.2f}%).",
            f"Tier-2 ranks that residual at PR-AUC "
            f"**{self.tier2_pr_auc_below_tier1_threshold:.4f}** against a floor of "
            f"{self.base_rate_below_tier1_threshold:.4f} — **{self.lift_on_tier1_misses:.1f}x "
            "lift on the fraud Tier-1 would not have looked at.**",
        ]


def profile_complementarity(
    tier2_accounts: AccountScores,
    tier1_accounts: AccountScores,
    max_flag_rate: float = MAX_REVIEW_FLAG_RATE,
) -> ComplementarityProfile:
    """Measure what Tier-2 adds on the accounts Tier-1 does not flag.

    Both inputs must be aggregated from the same split over the same account set, so their
    rows line up — :func:`aggregate_to_accounts` sorts by account id, which is what makes
    that true rather than hoped for.
    """
    if tier2_accounts.n_accounts != tier1_accounts.n_accounts:
        raise RuntimeError(
            f"Tier-1 and Tier-2 account sets differ ({tier1_accounts.n_accounts} vs "
            f"{tier2_accounts.n_accounts}); the complementarity comparison would be aligning "
            "different accounts."
        )
    tier1_threshold = threshold_for_flag_rate(tier1_accounts.scores, max_flag_rate)
    residual = tier1_accounts.scores < tier1_threshold
    labels = tier2_accounts.labels[residual]
    scores = tier2_accounts.scores[residual]
    return ComplementarityProfile(
        rank_correlation=spearman(tier2_accounts.scores, tier1_accounts.scores),
        accounts_below_tier1_threshold=int(residual.sum()),
        positives_below_tier1_threshold=int(labels.sum()),
        base_rate_below_tier1_threshold=float(labels.mean()) if labels.size else 0.0,
        tier2_pr_auc_below_tier1_threshold=pr_auc(labels, scores),
    )


@dataclass(frozen=True)
class FalseNegativeProfile:
    """What the model actually missed, measured rather than imagined.

    ml-evaluation-standards section 4 requires a "what this does NOT catch" section written
    from observed false negatives. BUILD_LOG records that in Phase 2 both such claims were
    written from reasoning and both turned out false, so this is computed.
    """

    total: int
    abstained: int
    scored_but_below_threshold: int
    median_missed_amount: float
    median_caught_amount: float
    median_missed_length: float
    median_caught_length: float
    missed_value_share: float

    def describe(self) -> list[str]:
        """Return plain-language lines for the README and the report."""
        if self.total == 0:
            return ["No fraudulent account went undetected on the test split."]
        return [
            f"{self.abstained:,} of {self.total:,} missed fraudulent accounts "
            f"({100 * self.abstained / self.total:.1f}%) were **abstentions** — too little "
            "history for Tier-2 to hold an opinion at all. Tier-1 is the only layer covering "
            "them.",
            f"{self.scored_but_below_threshold:,} were scored and fell below the threshold: "
            "their sequences reconstructed as normally as legitimate ones.",
            f"Missed fraudulent accounts have a median transaction amount of "
            f"{self.median_missed_amount:,.2f} against {self.median_caught_amount:,.2f} for "
            "caught ones.",
            f"Missed accounts hold a median of {self.median_missed_length:.0f} transactions "
            f"in the window against {self.median_caught_length:.0f} for caught ones.",
            f"By value, {100 * self.missed_value_share:.1f}% of test fraud value sits in the "
            "accounts this layer missed.",
        ]


def profile_false_negatives(
    accounts: AccountScores,
    threshold: float,
    lengths_by_account: dict[Any, int],
    account_ids: Sequence[Any],
) -> FalseNegativeProfile:
    """Measure the fraudulent accounts the model failed to flag."""
    flagged = accounts.scores >= threshold
    missed = accounts.labels & ~flagged
    caught = accounts.labels & flagged
    abstained = missed & (accounts.scores == ABSTAINED_RANK_SENTINEL)

    lengths = np.array([lengths_by_account.get(name, 0) for name in account_ids], dtype="float64")
    total_value = float(accounts.amounts[accounts.labels].sum())
    return FalseNegativeProfile(
        total=int(missed.sum()),
        abstained=int(abstained.sum()),
        scored_but_below_threshold=int((missed & ~abstained).sum()),
        median_missed_amount=float(np.median(accounts.amounts[missed])) if missed.any() else 0.0,
        median_caught_amount=float(np.median(accounts.amounts[caught])) if caught.any() else 0.0,
        median_missed_length=float(np.median(lengths[missed])) if missed.any() else 0.0,
        median_caught_length=float(np.median(lengths[caught])) if caught.any() else 0.0,
        missed_value_share=(
            float(accounts.amounts[missed].sum() / total_value) if total_value > 0 else 0.0
        ),
    )


def build_scoring_sequences(
    corpus: AssembledCorpus,
    split: SplitWindows,
    model: Tier2Model,
    count: int,
) -> list[list[TransactionFeatures]]:
    """Build assembled scoring windows from test rows, for the latency benchmark.

    Mirrors what Phase 7 will assemble: for each anchor, the account's trailing window as a
    list of ``TransactionFeatures`` carrying the **unscaled** Tier-2 vector and the Tier-2
    ``feature_version``. Non-finite floats become None, as the pipeline does, because that is
    what survives a JSONB round-trip.
    """
    ordered = corpus.history.iloc[corpus.order].reset_index(drop=True)
    raw = derive_timestep_frame(ordered)

    scoreable = np.flatnonzero(split.scoreable(model.spec.min_length))[:count]
    sequences: list[list[TransactionFeatures]] = []
    for position in scoreable:
        gather = split.windows.gather[position][split.windows.mask[position]]
        window: list[TransactionFeatures] = []
        for row_index in gather:
            row = ordered.iloc[int(row_index)]
            values = raw.iloc[int(row_index)]
            vector: dict[str, Any] = {}
            for name in model.feature_names:
                value = float(values[name])
                vector[name] = None if not np.isfinite(value) else value
            window.append(
                TransactionFeatures(
                    transaction_id=str(row["transaction_id"]),
                    source_dataset=str(row["source_dataset"]),  # type: ignore[arg-type]
                    event_time=row["event_time"].to_pydatetime(),
                    amount=Decimal(str(round(float(row["amount"]), 4))),
                    account_id=str(row["account_id"]),
                    counterparty_id=None,
                    transaction_type=(
                        None if pd.isna(row["transaction_type"]) else str(row["transaction_type"])
                    ),
                    feature_version=model.feature_version,
                    features=vector,
                )
            )
        sequences.append(window)
    return sequences


# ==========================================================================================
# The run
# ==========================================================================================


@dataclass
class CorpusReport:
    """Everything measured for Tier-2 on IEEE-CIS."""

    spec: Tier2SequenceSpec
    rows: dict[str, int]
    candidates: list[Candidate]
    winner: Candidate
    runner_up: Candidate
    tier1_baseline: Candidate | None
    tier1_delta: tuple[float, float] | None
    runner_up_interval: tuple[float, float]
    min_length_sweep: list[tuple[int, float]]
    scoreable_only: EvaluationResult
    capacity: EvaluationResult
    sensitivity: str
    latency: dict[str, float]
    coverage: dict[str, float]
    coverage_by_split: dict[str, dict[str, float]]
    error_length_correlation: float
    length_strata: list[tuple[str, int, int, float, float]]
    decile_text: str
    false_negatives: FalseNegativeProfile
    complementarity: ComplementarityProfile | None
    reaches_earlier_split: dict[str, int]
    training_window: dict[str, str]
    cost_model: CostModel
    plots: dict[str, Path]
    notes: list[str] = field(default_factory=list)

    @property
    def winner_is_significantly_better(self) -> bool:
        """Return whether the winner beats the runner-up beyond sampling noise."""
        return self.runner_up_interval[0] > 0.0

    @property
    def beats_tier1(self) -> bool | None:
        """Return whether Tier-2 beats Tier-1-aggregated beyond noise, or None if untested."""
        return None if self.tier1_delta is None else self.tier1_delta[0] > 0.0


def _index_of(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.intp]:
    """Return positional indices where ``mask`` is True."""
    return np.flatnonzero(mask).astype(np.intp)


def run_corpus(
    frames: dict[str, pd.DataFrame],
    moment: datetime,
    *,
    epochs: int,
    artifact_dir: Path,
    registry_path: Path,
    reports_dir: Path,
) -> CorpusReport:
    """Fit every candidate, compare them, and pick the production model."""
    cost_model = CostModel(units=IEEE_COST_UNITS)
    logger.info(
        "cost model — review=%.2f per flagged account, chargeback fee=%.2f [%s]",
        cost_model.review_cost,
        cost_model.chargeback_fee,
        cost_model.units,
    )

    corpus = assemble(frames, DEFAULT_WINDOW)
    candidates: list[Candidate] = []

    def account_pair(
        model: Tier2Model, min_length: int
    ) -> tuple[AccountScores, AccountScores, npt.NDArray[np.float64]]:
        """Score validation and test at the account level for one model and one N."""
        validation_scores = model.score_windows(corpus.per_split["val"].windows)
        test_scores = model.score_windows(corpus.per_split["test"].windows)
        return (
            to_accounts(
                corpus.per_split["val"], validation_scores, min_length, scoreable_only=False
            ),
            to_accounts(corpus.per_split["test"], test_scores, min_length, scoreable_only=False),
            test_scores,
        )

    # --- 0: the references ---------------------------------------------------------------
    # The references are scored with **no abstention** (`NO_ABSTENTION`), because neither of
    # them has one: an amount is available on every transaction and Tier-1 scores every row it
    # is given. Handicapping them with Tier-2's minimum-history rule would compare Tier-2
    # against a deliberately crippled alternative and inflate its apparent contribution. The
    # account set stays identical either way — abstained accounts are kept at the sentinel —
    # so the paired bootstrap against them remains valid.
    amount_validation = to_accounts(
        corpus.per_split["val"],
        amount_max_scores(corpus.per_split["val"]),
        NO_ABSTENTION,
        scoreable_only=False,
    )
    amount_test = to_accounts(
        corpus.per_split["test"],
        amount_max_scores(corpus.per_split["test"]),
        NO_ABSTENTION,
        scoreable_only=False,
    )
    candidates.append(
        choose_and_evaluate(
            "amount-only account ranking (reference)",
            None,
            amount_validation,
            amount_test,
            cost_model,
            NO_ABSTENTION,
            notes=[
                "Not a model. The score is the account's largest `log1p(amount)`, on every "
                "account — this reference does not abstain.",
            ],
        )
    )

    tier1_scores = load_tier1_account_scores(corpus, artifact_dir, registry_path)
    tier1_baseline: Candidate | None = None
    if tier1_scores is not None:
        tier1_baseline = choose_and_evaluate(
            "Tier-1, aggregated to account (baseline)",
            None,
            to_accounts(
                corpus.per_split["val"],
                tier1_scores["val"],
                NO_ABSTENTION,
                scoreable_only=False,
            ),
            to_accounts(
                corpus.per_split["test"],
                tier1_scores["test"],
                NO_ABSTENTION,
                scoreable_only=False,
            ),
            cost_model,
            NO_ABSTENTION,
            notes=[
                "The registered Tier-1 model's per-transaction scores, aggregated to the "
                "account with `max`. The layer that already ships. Tier-2 has to beat this "
                "to have earned a second model, a second feature definition and a serving "
                "path that needs an account's history.",
                "Scored on **every** account, with no minimum-history abstention: Tier-1 "
                "does not have one. Applying Tier-2's abstention rule here would compare "
                "against a crippled alternative and overstate what Tier-2 adds.",
            ],
        )
        candidates.append(tier1_baseline)

    # --- 1: the latent sweep, at the default window --------------------------------------
    # `train_eligible` and `clean_val` index the *full* window set, not a split's subset: the
    # fits read them against `corpus.all_windows` so a training batch can reference any base
    # row, including one whose own anchor was dropped.
    all_train_index = _index_of(corpus.train_eligible)
    all_clean_val_index = _index_of(corpus.clean_val)

    best_latent, best_validation = LATENT_GRID[0], -1.0
    for latent in LATENT_GRID:
        model, curve = fit_autoencoder(
            "lstm_autoencoder",
            corpus.spec,
            corpus.all_windows,
            all_train_index,
            all_clean_val_index,
            model_id=build_model_id("tier2-behavioral", f"lstm-latent{latent}", SOURCE, moment),
            latent_size=latent,
            epochs=epochs,
        )
        validation, test, _ = account_pair(model, MIN_SEQUENCE_LENGTH)
        candidate = choose_and_evaluate(
            f"LSTM autoencoder (latent={latent}, W={corpus.spec.window})",
            model,
            validation,
            test,
            cost_model,
            MIN_SEQUENCE_LENGTH,
            curve=curve,
            corpus=corpus,
        )
        candidates.append(candidate)
        if candidate.validation_pr_auc > best_validation:
            best_latent, best_validation = latent, candidate.validation_pr_auc
    logger.info("latent sweep: best latent=%d (val PR-AUC %.4f)", best_latent, best_validation)

    # --- 2: the window sweep, at the winning latent --------------------------------------
    for window in WINDOW_GRID:
        windowed = assemble(frames, window)
        model, curve = fit_autoencoder(
            "lstm_autoencoder",
            windowed.spec,
            windowed.all_windows,
            _index_of(windowed.train_eligible),
            _index_of(windowed.clean_val),
            model_id=build_model_id(
                "tier2-behavioral", f"lstm-w{window}-latent{best_latent}", SOURCE, moment
            ),
            latent_size=best_latent,
            epochs=epochs,
        )
        validation = to_accounts(
            windowed.per_split["val"],
            model.score_windows(windowed.per_split["val"].windows),
            MIN_SEQUENCE_LENGTH,
            scoreable_only=False,
        )
        test = to_accounts(
            windowed.per_split["test"],
            model.score_windows(windowed.per_split["test"].windows),
            MIN_SEQUENCE_LENGTH,
            scoreable_only=False,
        )
        candidates.append(
            choose_and_evaluate(
                f"LSTM autoencoder (latent={best_latent}, W={window})",
                model,
                validation,
                test,
                cost_model,
                MIN_SEQUENCE_LENGTH,
                curve=curve,
                corpus=windowed,
                notes=[
                    "Windowed corpora are assembled independently, so this candidate's "
                    "account set matches the others' but its window contents do not.",
                ],
            )
        )

    # --- 3: the non-recurrent control ----------------------------------------------------
    dense, dense_curve = fit_autoencoder(
        "dense_autoencoder",
        corpus.spec,
        corpus.all_windows,
        all_train_index,
        all_clean_val_index,
        model_id=build_model_id("tier2-behavioral", "dense-autoencoder", SOURCE, moment),
        latent_size=best_latent,
        epochs=epochs,
    )
    dense_validation, dense_test, _ = account_pair(dense, MIN_SEQUENCE_LENGTH)
    candidates.append(
        choose_and_evaluate(
            f"Dense autoencoder (latent={best_latent}, no recurrence)",
            dense,
            dense_validation,
            dense_test,
            cost_model,
            MIN_SEQUENCE_LENGTH,
            curve=dense_curve,
            corpus=corpus,
            notes=[
                "Same features, same window, same mask, same bottleneck — no recurrence. The "
                "gap against the LSTM is what the sequence *model* contributes over the "
                "account-relative features alone.",
            ],
        )
    )

    # --- Pick the winner on VALIDATION account PR-AUC ------------------------------------
    trained = [candidate for candidate in candidates if candidate.model is not None]
    ranked = sorted(trained, key=lambda candidate: candidate.validation_pr_auc, reverse=True)
    winner, runner_up = ranked[0], ranked[1]
    assert winner.model is not None
    # Everything from here on re-measures the winner, so it must read the winner's own
    # windowing: a W=20 model scored against W=10 windows is a shape mismatch, not a
    # degraded number.
    selected = winner.corpus if winner.corpus is not None else corpus
    logger.info(
        "winner is %s (selected on val PR-AUC %.4f; test PR-AUC %.4f)",
        winner.name,
        winner.validation_pr_auc,
        winner.result.pr_auc,
    )

    test_labels = winner.test_accounts.labels
    runner_up_interval = bootstrap_pr_auc_delta(
        test_labels, winner.test_accounts.scores, runner_up.test_accounts.scores, seed=RANDOM_SEED
    )
    tier1_delta = (
        bootstrap_pr_auc_delta(
            test_labels,
            winner.test_accounts.scores,
            tier1_baseline.test_accounts.scores,
            seed=RANDOM_SEED,
        )
        if tier1_baseline is not None
        else None
    )
    if tier1_delta is not None and tier1_delta[0] <= 0.0:
        logger.warning(
            "Tier-2 does not beat Tier-1-aggregated beyond noise (delta 95%% CI [%.4f, %.4f]). "
            "That is the finding; it must be reported as one.",
            *tier1_delta,
        )

    # --- The abstention threshold N, swept on validation ---------------------------------
    winner_validation_scores = winner.model.score_windows(selected.per_split["val"].windows)
    winner_test_scores = winner.model.score_windows(selected.per_split["test"].windows)
    min_length_sweep: list[tuple[int, float]] = []
    for candidate_n in MIN_LENGTH_GRID:
        accounts = to_accounts(
            selected.per_split["val"], winner_validation_scores, candidate_n, scoreable_only=False
        )
        min_length_sweep.append((candidate_n, pr_auc(accounts.labels, accounts.scores)))
    best_n = max(min_length_sweep, key=lambda pair: pair[1])[0]
    logger.info(
        "abstention sweep on validation: %s -> N=%d",
        ", ".join(f"N={n}: {value:.4f}" for n, value in min_length_sweep),
        best_n,
    )
    winner.model.spec = Tier2SequenceSpec(
        source_dataset=winner.model.spec.source_dataset,
        feature_names=winner.model.spec.feature_names,
        window=winner.model.spec.window,
        min_length=best_n,
        means=winner.model.spec.means,
        stds=winner.model.spec.stds,
        clip=winner.model.spec.clip,
    )
    winner.min_length = best_n

    # --- The winner, re-measured at the chosen N -----------------------------------------
    validation_accounts = to_accounts(
        selected.per_split["val"], winner_validation_scores, best_n, scoreable_only=False
    )
    test_accounts = to_accounts(
        selected.per_split["test"], winner_test_scores, best_n, scoreable_only=False
    )
    threshold, _ = choose_threshold_by_cost(
        validation_accounts.labels,
        validation_accounts.scores,
        validation_accounts.amounts,
        cost_model,
    )
    winner.model.threshold = threshold
    winner.validation_threshold = threshold
    winner.validation_pr_auc = pr_auc(validation_accounts.labels, validation_accounts.scores)
    winner.test_accounts = test_accounts
    winner.result = evaluate(
        winner.name,
        "test",
        test_accounts.labels,
        test_accounts.scores,
        threshold=threshold,
        threshold_criterion="minimising estimated cost on validation accounts",
        interval=bootstrap_pr_auc(test_accounts.labels, test_accounts.scores, seed=RANDOM_SEED),
        cost=cost_at_threshold(
            test_accounts.labels,
            test_accounts.scores,
            test_accounts.amounts,
            threshold,
            cost_model,
        ),
        notes=[
            f"All test accounts (n={test_accounts.n_accounts:,}), with the "
            f"{test_accounts.n_abstained:,} Tier-2 abstained on counted as never flagged. "
            "**This is the deployed system's number.** The model's own number, over the "
            "accounts it can actually score, is reported separately and is higher.",
        ],
    )

    # --- The same model over only the accounts it can score ------------------------------
    scoreable_accounts = to_accounts(
        selected.per_split["test"], winner_test_scores, best_n, scoreable_only=True
    )
    scoreable_only = evaluate(
        f"{winner.name} — scoreable accounts only",
        "test",
        scoreable_accounts.labels,
        scoreable_accounts.scores,
        threshold=threshold,
        threshold_criterion="minimising estimated cost on validation accounts",
        interval=bootstrap_pr_auc(
            scoreable_accounts.labels, scoreable_accounts.scores, seed=RANDOM_SEED
        ),
        cost=cost_at_threshold(
            scoreable_accounts.labels,
            scoreable_accounts.scores,
            scoreable_accounts.amounts,
            threshold,
            cost_model,
        ),
        notes=[
            "The model's own performance, over the accounts with enough history to score. "
            "Higher than the figure above by construction; quoting it as the system's number "
            "would describe a product that abstains on nothing.",
        ],
    )

    # --- Capacity, sensitivity, diagnostics ----------------------------------------------
    capacity_threshold = threshold_for_flag_rate(validation_accounts.scores, MAX_REVIEW_FLAG_RATE)
    capacity = evaluate(
        f"{winner.name} @ capacity",
        "test",
        test_accounts.labels,
        test_accounts.scores,
        threshold=capacity_threshold,
        threshold_criterion=(
            f"flagging at most {100 * MAX_REVIEW_FLAG_RATE:.1f}% of validation accounts"
        ),
        cost=cost_at_threshold(
            test_accounts.labels,
            test_accounts.scores,
            test_accounts.amounts,
            capacity_threshold,
            cost_model,
        ),
        notes=[
            "The same model at a staffable operating point. Reported because the "
            "cost-minimising threshold above assumes unbounded review capacity.",
        ],
    )
    sensitivity = "\n\n".join(
        (
            render_sensitivity(
                sensitivity_sweep(
                    validation_accounts.labels,
                    validation_accounts.scores,
                    validation_accounts.amounts,
                    cost_model,
                ),
                "Cost sensitivity, +/-50% on BOTH parameters (threshold re-chosen on "
                "validation):",
            ),
            render_sensitivity(
                review_cost_sweep(
                    validation_accounts.labels,
                    validation_accounts.scores,
                    validation_accounts.amounts,
                    cost_model,
                ),
                "Cost sensitivity, review cost ONLY (the direction that moves the "
                "recommendation):",
            ),
        )
    )

    test_split = selected.per_split["test"]
    scoreable_mask = test_split.scoreable(best_n)
    correlation = spearman(
        winner_test_scores[scoreable_mask],
        test_split.lengths[scoreable_mask].astype("float64"),
    )
    logger.info(
        "reconstruction error vs sequence length, Spearman rho = %+.4f (test, scoreable)",
        correlation,
    )

    window_coverage = coverage(test_split.lengths, test_split.labels, best_n)
    window_coverage["account_coverage"] = 1.0 - test_accounts.abstention_rate
    window_coverage["accounts_total"] = float(test_accounts.n_accounts)
    window_coverage["accounts_abstained"] = float(test_accounts.n_abstained)

    # How much of the system-level ranking is the abstention rule rather than the model?
    #
    # Abstained accounts sit at the bottom of the ranking, so if they are also less likely to
    # be fraudulent, the deployed PR-AUC gets credit for a separation the autoencoder never
    # produced — "we did not look" scoring as "we judged it safe". The two base rates below
    # are what makes that visible, and they belong beside the headline rather than in a
    # footnote. The error-vs-length correlation catches the same effect *within* the
    # scoreable set; this catches it across the abstention boundary.
    abstained = test_accounts.scores == ABSTAINED_RANK_SENTINEL
    window_coverage["abstained_account_fraud_rate"] = (
        float(test_accounts.labels[abstained].mean()) if abstained.any() else 0.0
    )
    window_coverage["scoreable_account_fraud_rate"] = (
        float(test_accounts.labels[~abstained].mean()) if (~abstained).any() else 0.0
    )
    logger.info(
        "fraud rate: %.4f%% among abstained accounts, %.4f%% among scoreable ones",
        100 * window_coverage["abstained_account_fraud_rate"],
        100 * window_coverage["scoreable_account_fraud_rate"],
    )

    # Coverage per split, not test alone. Train and validation are reported because an
    # abstention rate that moves across the timeline would mean the chosen N describes the
    # validation period rather than a stable property of the corpus — and the operating
    # threshold was chosen under that same N.
    coverage_by_split: dict[str, dict[str, float]] = {}
    for split in SPLIT_ORDER:
        window_split = selected.per_split[split]
        measured = coverage(window_split.lengths, window_split.labels, best_n)
        accounts = to_accounts(
            window_split,
            np.zeros(len(window_split.windows), dtype="float64"),
            best_n,
            scoreable_only=False,
        )
        measured["accounts_total"] = float(accounts.n_accounts)
        measured["accounts_abstained"] = float(accounts.n_abstained)
        measured["account_coverage"] = 1.0 - accounts.abstention_rate
        coverage_by_split[split] = measured
        logger.info(
            "%s coverage at N=%d: %.1f%% of accounts, %.1f%% of rows, %.1f%% of fraud",
            split,
            best_n,
            100 * measured["account_coverage"],
            100 * measured["row_coverage"],
            100 * measured["fraud_coverage"],
        )

    normal_errors = scoreable_accounts.scores[~scoreable_accounts.labels]
    fraud_errors = scoreable_accounts.scores[scoreable_accounts.labels]
    decile_text = decile_table(
        "Account reconstruction error by class (held-out test, scoreable accounts only):",
        {"legitimate accounts": normal_errors, "fraudulent accounts": fraud_errors},
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    plots = {
        "error_distribution": reports_dir / "tier2_error_distribution.png",
        "loss_curve": reports_dir / "tier2_loss_curve.png",
    }
    plot_error_distribution(normal_errors, fraud_errors, threshold, plots["error_distribution"])
    plot_loss_curve(winner.curve, plots["loss_curve"])

    lengths_by_account: dict[Any, int] = (
        pd.DataFrame({"account_id": test_split.account_id, "length": test_split.lengths})
        .groupby("account_id")["length"]
        .max()
        .to_dict()
    )
    account_order = sorted(set(test_split.account_id))
    # Profiled at the **capacity-constrained** threshold, not the cost-minimising one. Under
    # an unbounded-review-queue assumption the cost-optimal account-level threshold collapses
    # to flagging everything, which leaves zero false negatives and a "what this does NOT
    # catch" section that says nothing. The staffable operating point is the one a reviewer
    # would actually run, so it is the one whose misses are worth characterising.
    false_negatives = profile_false_negatives(
        test_accounts, capacity_threshold, lengths_by_account, account_order
    )
    # Stratified at MIN_SEQUENCE_LENGTH, not at the chosen N. The point of this diagnostic is
    # to see whether the model separates *within* a length band; evaluating it at an N of 10
    # empties the two shorter bands and leaves nothing to compare against, which is exactly
    # the comparison that answers whether the layer learned behaviour or depth.
    strata = length_stratified_pr_auc(test_split, winner_test_scores, MIN_SEQUENCE_LENGTH)
    complementarity: ComplementarityProfile | None = None
    if tier1_scores is not None:
        tier1_test_accounts = to_accounts(
            test_split, tier1_scores["test"], NO_ABSTENTION, scoreable_only=False
        )
        complementarity = profile_complementarity(test_accounts, tier1_test_accounts)
        logger.info(
            "complementarity: rho=%+.4f with Tier-1; on Tier-1's %d non-flagged accounts "
            "(%d fraud) Tier-2 scores PR-AUC %.4f (%.1fx lift)",
            complementarity.rank_correlation,
            complementarity.accounts_below_tier1_threshold,
            complementarity.positives_below_tier1_threshold,
            complementarity.tier2_pr_auc_below_tier1_threshold,
            complementarity.lift_on_tier1_misses,
        )

    for band, band_accounts, band_positives, band_base_rate, band_pr_auc in strata:
        logger.info(
            "length band %s: n=%d, positives=%d, base rate %.4f, PR-AUC %.4f",
            band,
            band_accounts,
            band_positives,
            band_base_rate,
            band_pr_auc,
        )

    sequences = build_scoring_sequences(
        selected, test_split, winner.model, count=LATENCY_BENCHMARK_CALLS
    )
    latency = benchmark_latency(winner.model, sequences)
    logger.info(
        "latency p50=%.3fms p95=%.3fms p99=%.3fms (budget p95 < %.0fms)",
        latency["p50_ms"],
        latency["p95_ms"],
        latency["p99_ms"],
        LATENCY_BUDGET_P95_MS,
    )

    return CorpusReport(
        spec=winner.model.spec,
        rows=selected.rows,
        candidates=candidates,
        winner=winner,
        runner_up=runner_up,
        tier1_baseline=tier1_baseline,
        tier1_delta=tier1_delta,
        runner_up_interval=runner_up_interval,
        min_length_sweep=min_length_sweep,
        scoreable_only=scoreable_only,
        capacity=capacity,
        sensitivity=sensitivity,
        latency=latency,
        coverage=window_coverage,
        coverage_by_split=coverage_by_split,
        error_length_correlation=correlation,
        length_strata=strata,
        decile_text=decile_text,
        false_negatives=false_negatives,
        complementarity=complementarity,
        reaches_earlier_split={
            split: selected.per_split[split].reaches_earlier_split for split in SPLIT_ORDER
        },
        training_window=selected.training_window,
        cost_model=cost_model,
        plots=plots,
        notes=(
            []
            if tier1_scores is not None
            else [
                "The Tier-1 baseline was NOT measured: its registry artefact is absent from "
                "models/artifacts. Without it, no claim that Tier-2 adds anything over the "
                "layer that already ships is supported by this run."
            ]
        ),
    )


def register(report: CorpusReport, registry_path: Path, artifact_dir: Path) -> RegistryEntry:
    """Save the winning model and append its record to the registry."""
    winner = report.winner
    assert winner.model is not None
    artifact = winner.model.save(artifact_dir)

    caveats: list[str] = []
    # A reader of registry.json alone sees `precision` equal to the base rate and `recall`
    # equal to 1.0 and has no way to know why. Prepended so the explanation arrives before
    # the numbers do.
    if winner.result.confusion.recall >= 1.0 or not np.isfinite(winner.result.threshold):
        caveats.append(
            "THE HEADLINE CONFUSION MATRIX IS DEGENERATE. The cost-minimising threshold "
            "collapsed to flagging every account, so `precision` here equals the base rate "
            "and `recall` is 1.0; neither carries information about the model. That is the "
            "honest arithmetic of the cost model, not a bug: at the account level a false "
            f"negative costs the account's fraud value plus {DEFAULT_CHARGEBACK_FEE:.2f} "
            f"while a review costs {DEFAULT_REVIEW_COST:.2f}, and the model assumes unbounded "
            "review capacity. PR-AUC is threshold-free and is unaffected. For a deployable "
            "operating point read `capacity_constrained_operating_point` below."
        )
    if winner.result.is_leak_suspicious:
        caveats.append(
            f"DO NOT QUOTE AS A HEADLINE. Test PR-AUC {winner.result.pr_auc:.4f} exceeds "
            f"{LEAK_SUSPICION_PR_AUC} on fraud data, which ml-evaluation-standards section 4 "
            "treats as a leak signal until disproven. See app/models/README.md and "
            "BUILD_LOG.md for the investigation and the diagnosis before using this number."
        )
    if report.beats_tier1 is False:
        residual = (
            f" Measured complementarity: on the accounts Tier-1 leaves below its own capacity "
            f"threshold, Tier-2 ranks the residual fraud at PR-AUC "
            f"{report.complementarity.tier2_pr_auc_below_tier1_threshold:.4f} against a floor "
            f"of {report.complementarity.base_rate_below_tier1_threshold:.4f} "
            f"({report.complementarity.lift_on_tier1_misses:.1f}x), with a rank correlation "
            f"of {report.complementarity.rank_correlation:+.4f} against Tier-1."
            if report.complementarity is not None
            else ""
        )
        caveats.append(
            "DO NOT DEPLOY STANDALONE. Tier-2 does not beat Tier-1's per-transaction scores "
            "aggregated to the account on this test split, so it is not a standalone "
            "improvement over the layer that already ships. It is registered as an input to "
            "the Phase 5 meta-learner, where an uncorrelated weaker signal can still "
            f"contribute to the pair.{residual}"
        )

    entry = RegistryEntry(
        model_id=winner.model.model_id,
        layer="tier2_behavioral",
        algorithm=winner.model.algorithm,
        source_dataset=SOURCE,
        feature_version=winner.model.feature_version,
        training_window=report.training_window,
        hyperparameters={
            **winner.model.hyperparameters,
            "min_sequence_length": winner.model.spec.min_length,
            "standardised_clip": winner.model.spec.clip,
            "latent_grid": list(LATENT_GRID),
            "window_grid": [DEFAULT_WINDOW, *WINDOW_GRID],
            "min_length_grid": list(MIN_LENGTH_GRID),
        },
        random_seed=RANDOM_SEED,
        heldout_test={
            **winner.result.to_dict(),
            "scoreable_accounts_only": report.scoreable_only.to_dict(),
            "capacity_constrained_operating_point": report.capacity.to_dict(),
            "coverage": {key: round(value, 6) for key, value in report.coverage.items()},
            "coverage_by_split": {
                split: {key: round(value, 6) for key, value in measured.items()}
                for split, measured in report.coverage_by_split.items()
            },
            "error_vs_sequence_length_spearman": round(report.error_length_correlation, 4),
            "pr_auc_by_window_length": [
                {
                    "band": band,
                    "accounts": accounts,
                    "positives": positives,
                    "base_rate": round(base_rate, 6),
                    "pr_auc": round(value, 6),
                }
                for band, accounts, positives, base_rate, value in report.length_strata
            ],
            "latency": report.latency,
            "windows_reaching_an_earlier_split": report.reaches_earlier_split,
            "complementarity_with_tier1": (
                {
                    "rank_correlation": round(report.complementarity.rank_correlation, 4),
                    "accounts_below_tier1_threshold": (
                        report.complementarity.accounts_below_tier1_threshold
                    ),
                    "positives_below_tier1_threshold": (
                        report.complementarity.positives_below_tier1_threshold
                    ),
                    "base_rate": round(report.complementarity.base_rate_below_tier1_threshold, 6),
                    "tier2_pr_auc": round(
                        report.complementarity.tier2_pr_auc_below_tier1_threshold, 6
                    ),
                    "lift": round(report.complementarity.lift_on_tier1_misses, 3),
                }
                if report.complementarity is not None
                else None
            ),
        },
        baseline_comparison=[
            candidate.result.to_dict() for candidate in report.candidates if candidate is not winner
        ],
        artifact=artifact.relative_to(artifact_dir.parent).as_posix(),
        notes=[
            *caveats,
            f"Selected on VALIDATION account PR-AUC ({winner.validation_pr_auc:.4f}) from "
            f"{len(report.candidates)} candidates; test was scored once afterwards, for "
            "reporting only.",
            "Evaluated **per account, not per transaction**. IEEE-CIS propagates a chargeback "
            "across an account's later transactions, so a per-transaction PR-AUC counts one "
            "compromised account's correlated rows as independent correct calls and its "
            "bootstrap interval comes out far too tight.",
            (
                (
                    f"Advantage over Tier-1 aggregated to account: 95% CI "
                    f"[{report.tier1_delta[0]:.4f}, {report.tier1_delta[1]:.4f}]"
                    + ("." if report.beats_tier1 else " — includes zero.")
                )
                if report.tier1_delta is not None
                else "Tier-1 baseline not measured; see notes in notebooks/tier2_report.md."
            ),
            (
                f"Advantage over the runner-up ({report.runner_up.result.model_name}): 95% CI "
                f"[{report.runner_up_interval[0]:.4f}, {report.runner_up_interval[1]:.4f}]"
            )
            + (
                "."
                if report.winner_is_significantly_better
                else " — includes zero, so this selection is inside sampling noise and the "
                "runner-up is an equally defensible choice."
            ),
            f"Coverage: Tier-2 scores {100 * report.coverage['account_coverage']:.1f}% of test "
            f"accounts, {100 * report.coverage['row_coverage']:.1f}% of test transactions and "
            f"{100 * report.coverage['fraud_coverage']:.1f}% of test fraud. It abstains on the "
            "rest rather than returning a zero.",
            f"Reconstruction error vs sequence length, Spearman rho "
            f"{report.error_length_correlation:+.4f} — the check that the masked loss is not "
            "letting the score become a proxy for how much history an account has.",
            "One-class supervised, not unsupervised: the label defines the training set "
            "(fraud-free accounts only) even though it is never a target.",
            "Tier-2 mints its own feature_version: its input is a (window, feature) matrix "
            "with its own derivations, padding rule and fitted scaler, so neither the Phase 1 "
            "pipeline hash nor Tier-1's describes what produced a prediction.",
            *winner.notes,
            *report.notes,
        ],
    )
    append_entry(entry, registry_path)
    logger.info("appended %s to %s", entry.model_id, registry_path)
    return entry


def render_report(report: CorpusReport, sample: int | None, epochs: int) -> str:
    """Render the Phase 3 metrics report."""
    winner = report.winner
    lines = [
        "# RiskIQ — Phase 3 Tier-2 Evaluation",
        "",
        "Generated by `python -m app.models.train_tier2`. Every headline number is measured "
        "on the held-out **test** split, **per account**, which was scored exactly once after "
        "the operating threshold, the latent size, the window and the abstention threshold "
        "had all been chosen on validation.",
        "",
        f"Random seed: `{RANDOM_SEED}`. Device: CPU (cuDNN's LSTM kernel is "
        f"non-deterministic). Latency budget: p95 < {LATENCY_BUDGET_P95_MS:.0f}ms.",
        "",
        "## Why IEEE-CIS only",
        "",
        "Phase 1 measured PaySim at 2,768,630 accounts of which 99.9% hold a single "
        "transaction, a maximum of 3, and `seconds_since_prior_txn` 99.94% null. A sequence "
        "model needs sequences. Running it there would produce a number, and the number "
        "would describe the simulator rather than any behaviour — the same trap Phase 2 "
        "documented for PaySim's Tier-1 result.",
    ]
    if sample is not None:
        lines += [
            "",
            f"> **Sampled run**: only the earliest {sample:,} rows of each split were used, "
            f"at {epochs} epochs. These numbers are for iteration and are not reportable.",
        ]

    lines += [
        "",
        "## Why the headline is per account, not per transaction",
        "",
        "IEEE-CIS propagates a chargeback label across an account's later transactions. One "
        "compromised account holding 300 rows therefore contributes 300 correlated "
        "positives, and a per-transaction PR-AUC counts them as 300 independent correct "
        "calls when the model made one. The bootstrap has the same problem in reverse: "
        "resampling rows as though they were independent produces an interval far tighter "
        "than the evidence supports. At the account level the resampling unit genuinely is "
        "independent.",
        "",
        "A straddling account appears as separate account-split units rather than carrying "
        "one global label, so its test outcome cannot describe its validation rows.",
        "",
        "## Tier-2 input definition",
        "",
        report.spec.describe(),
        "",
        f"Rows: train {report.rows['train']:,}, val {report.rows['val']:,}, "
        f"test {report.rows['test']:,}.",
        "",
        "### Windows reaching into an earlier split",
        "",
        "A window is built from the account's real history and assigned to its **anchor's** "
        "split, so a test window may contain train-period transactions. That is what a live "
        "scorer has in front of it; refusing to look would fabricate a serving condition "
        "that does not exist. The reach is one-directional and asserted against the "
        "timestamps, not merely intended:",
        "",
        "| split | windows reaching back into an earlier split |",
        "|---|---|",
    ]
    lines += [f"| {split} | {count:,} |" for split, count in report.reaches_earlier_split.items()]

    lines += [
        "",
        "## Coverage — what Tier-2 can score at all",
        "",
        "Not a footnote. Phase 1 measured 57.7% of IEEE-CIS accounts holding a single "
        "transaction, so a large share of traffic has no behavioural baseline and Tier-2 "
        "**abstains** on it rather than returning a zero. The fraud figure is the decisive "
        "one: a layer that can only score the transactions nobody was worried about has not "
        "earned its place.",
        "",
        "| unit | scoreable | total | coverage |",
        "|---|---|---|---|",
    ]
    scoreable_accounts_count = (
        report.coverage["accounts_total"] - report.coverage["accounts_abstained"]
    )
    lines += [
        f"| test accounts | {scoreable_accounts_count:,.0f} "
        f"| {report.coverage['accounts_total']:,.0f} | "
        f"{100 * report.coverage['account_coverage']:.1f}% |",
        f"| test transactions | {report.coverage['rows_scoreable']:,.0f} | "
        f"{report.coverage['rows_total']:,.0f} | {100 * report.coverage['row_coverage']:.1f}% |",
        f"| test fraud transactions | {report.coverage['fraud_scoreable']:,.0f} | "
        f"{report.coverage['fraud_total']:,.0f} | "
        f"{100 * report.coverage['fraud_coverage']:.1f}% |",
        "",
        "### Coverage by split",
        "",
        "Reported for all three splits, not test alone: an abstention rate that drifted along "
        "the timeline would mean the chosen N describes the validation period rather than a "
        "stable property of the corpus — and the operating threshold was chosen under that "
        "same N.",
        "",
        "| split | accounts scored | account coverage | row coverage | fraud coverage |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {split} | "
        f"{measured['accounts_total'] - measured['accounts_abstained']:,.0f} / "
        f"{measured['accounts_total']:,.0f} | "
        f"{100 * measured['account_coverage']:.1f}% | "
        f"{100 * measured['row_coverage']:.1f}% | "
        f"{100 * measured['fraud_coverage']:.1f}% |"
        for split, measured in report.coverage_by_split.items()
    ]
    lines += [
        "",
        "### Is the abstention rule doing the ranking?",
        "",
        "Abstained accounts sit at the bottom of the deployed ranking. If they are also less "
        "likely to be fraudulent, the system-level PR-AUC collects credit for a separation "
        "the autoencoder never produced — *we did not look* scoring as *we judged it safe*. "
        "The two base rates say how much of that is happening:",
        "",
        "| account group | fraud base rate |",
        "|---|---|",
        f"| Tier-2 abstained (ranked last) | "
        f"{100 * report.coverage['abstained_account_fraud_rate']:.4f}% |",
        f"| Tier-2 scored | {100 * report.coverage['scoreable_account_fraud_rate']:.4f}% |",
        "",
        "The further apart these are, the more of the deployed number is the abstention "
        "policy rather than the model, and the more weight the scoreable-accounts-only block "
        "below carries as the measure of what the autoencoder actually learned.",
        "",
        "## Candidate comparison — validation selects, test reports",
        "",
        "**Selection reads the `val PR-AUC` column, never the test column.** Test is scored "
        "once, after the winner is already chosen, for reporting only — choosing on it would "
        "make it a validation set and invalidate the headline.",
        "",
        "| candidate | val PR-AUC | test PR-AUC | 95% CI | lift | precision | recall | F1 |",
        "|---|---|---|---|---|---|---|---|",
        f"| _no-skill floor (random ranker)_ | — | {winner.result.base_rate:.4f} | — | 1.0x "
        "| — | — | — |",
    ]
    for candidate in report.candidates:
        result = candidate.result
        interval = (
            f"{result.pr_auc_interval[0]:.4f}–{result.pr_auc_interval[1]:.4f}"
            if result.pr_auc_interval
            else "—"
        )
        mark = " **(selected)**" if candidate is winner else ""
        validation = f"{candidate.validation_pr_auc:.4f}" if candidate.model is not None else "—"
        lines.append(
            f"| {result.model_name}{mark} | {validation} | {result.pr_auc:.4f} | {interval} | "
            f"{result.lift_over_no_skill:.1f}x | {result.confusion.precision:.4f} | "
            f"{result.confusion.recall:.4f} | {result.confusion.f1:.4f} |"
        )

    lines += [
        "",
        "### Does the bottleneck bite?",
        "",
        "The phase's main architectural risk was an autoencoder wide enough to learn the "
        "identity map: it reconstructs fraud exactly as faithfully as normal behaviour and "
        "its two error distributions land on top of each other. Reconstruction loss alone "
        "cannot detect that — a wider latent always reconstructs *better*. Separation can:",
        "",
        "| candidate | clean-val loss | median error, legit | median error, fraud | ratio |",
        "|---|---|---|---|---|",
    ]
    for candidate in report.candidates:
        if candidate.model is None:
            continue
        accounts = candidate.test_accounts
        scored = accounts.scores > ABSTAINED_RANK_SENTINEL
        legit = accounts.scores[scored & ~accounts.labels]
        fraud = accounts.scores[scored & accounts.labels]
        if legit.size == 0 or fraud.size == 0:
            continue
        legit_median, fraud_median = float(np.median(legit)), float(np.median(fraud))
        loss = candidate.model.hyperparameters.get("best_clean_validation_loss", float("nan"))
        lines.append(
            f"| {candidate.name} | {loss:.5f} | {legit_median:.5f} | {fraud_median:.5f} | "
            f"{fraud_median / legit_median if legit_median else 0.0:.3f}x |"
        )
    lines += [
        "",
        "A latent that reconstructs everything well shows a low loss and a ratio near 1.0 — "
        "faithful, and useless. The selected candidate is the one whose *ratio* is largest on "
        "validation, not the one whose loss is lowest, and the two do not agree.",
        "",
        "### Does Tier-2 beat the layer that already ships?",
        "",
    ]
    if report.tier1_delta is None:
        lines += [
            "**Not measured.** The Tier-1 registry artefact was absent, so no claim that "
            "Tier-2 adds ranking power over Tier-1 is supported by this run.",
        ]
    else:
        low, high = report.tier1_delta
        lines += [
            f"PR-AUC delta against Tier-1 aggregated to account: **95% CI "
            f"[{low:.4f}, {high:.4f}]**.",
            "",
            (
                "The interval excludes zero, so on this test split the sequence layer ranks "
                "accounts better than the per-transaction model does."
                if low > 0
                else "**Tier-2 loses this comparison.** On this test split it does not rank "
                "accounts better than Tier-1's existing per-transaction scores aggregated "
                "the same way. That is the finding, and it is not a surprising one: Tier-1 "
                "is supervised on the label and Tier-2 is one-class, so the head-to-head was "
                "always the wrong question to hang the layer on. The right one is below."
            ),
        ]

    if report.complementarity is not None:
        lines += [
            "",
            "### What Tier-2 adds that Tier-1 does not",
            "",
            "Tier-2 is not built to replace Tier-1 — Phase 5 **fuses** them. A weaker signal "
            "that is uncorrelated with a stronger one can still add ranking power to the "
            "pair while losing to it alone, so the question that decides whether this layer "
            "earned its place is whether it separates fraud **among the accounts Tier-1 has "
            "already decided are safe**.",
            "",
        ]
        lines += [f"- {line}" for line in report.complementarity.describe()]
        # Deliberately not a thresholded verdict. An editorial call that flips on a
        # hard-coded lift boundary reads as a finding when it is really a constant, and the
        # first draft of this report tipped from "unproven" to "supported" on a lift of
        # 1.5125 against a boundary of 1.5. The numbers are stated, the decision rule is
        # stated, and Phase 5 measures it rather than inheriting an adjective.
        lift = report.complementarity.lift_on_tier1_misses
        lines += [
            "",
            f"A {lift:.1f}x residual lift is **not** on its own evidence that the layer "
            "earns its place, and this report does not claim it is. What it establishes is "
            "narrower: the two signals are close to orthogonal, and Tier-2 ranks Tier-1's "
            "residual above chance. Whether that converts into anything is a question only "
            "the fused model can answer.",
            "",
            "**The test Phase 5 should run**, stated here so it is not quietly skipped: fit "
            "the meta-learner with and without the Tier-2 feature, on validation, and keep "
            "the feature only if the paired PR-AUC delta excludes zero. If it does not, this "
            "layer is a negative result and should be written up as one rather than carried "
            "because it was built.",
        ]

    lines += [
        "",
        f"**Selection over the runner-up** ({report.runner_up.result.model_name}): delta 95% "
        f"CI [{report.runner_up_interval[0]:.4f}, {report.runner_up_interval[1]:.4f}]. "
        + (
            "The interval excludes zero, so the selection is justified."
            if report.winner_is_significantly_better
            else "**The interval includes zero**: on this test split the runner-up performs "
            "as well, and the selection rests on a point estimate inside the noise."
        ),
        "",
        "## Reconstruction-error distribution",
        "",
        f"![reconstruction error]({report.plots['error_distribution'].name})",
        "",
        "```",
        report.decile_text,
        "```",
        "",
        "### Is the score a sequence-length sensor?",
        "",
        f"Spearman rank correlation between reconstruction error and window length, on "
        f"held-out test scoreable accounts: **{report.error_length_correlation:+.4f}**.",
        "",
        "Padding is right-aligned and every error divides by the count of real timesteps "
        "rather than by W; had it divided by W, a 3-step window in a 10-step slot would "
        "receive seven free perfectly-reconstructed steps and every short-history account "
        "would score as normal. `tests/test_tier2.py` pins that arithmetic in both "
        "directions, so a correlation here is **not** a masking bug — it is the "
        "architecture: regenerating more timesteps from one fixed latent vector is genuinely "
        "harder, so longer windows reconstruct worse. Since fraudulent accounts also tend to "
        "hold more transactions, part of the headline may be the model ranking accounts by "
        "history depth rather than by how unusual that history is.",
        "",
        "Holding length roughly constant is what separates the two:",
        "",
        "| window length | accounts | fraudulent | base rate | PR-AUC | lift |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {band} | {accounts:,} | {positives:,} | {100 * base_rate:.2f}% | {value:.4f} | "
        f"{value / base_rate if base_rate else 0.0:.1f}x |"
        for band, accounts, positives, base_rate, value in report.length_strata
    ]
    lines += [
        "",
        "A band whose PR-AUC collapses to its own base rate (1.0x lift) is one where the "
        "layer has learned nothing about behaviour and was reading depth. A band that keeps "
        "its lift is evidence the sequence signal is real.",
        "",
        "## Training curve",
        "",
        f"![training curve]({report.plots['loss_curve'].name})",
        "",
        "| epoch | train loss (fraud-free) | clean-validation loss |",
        "|---|---|---|",
    ]
    lines += [
        f"| {point.epoch} | {point.train_loss:.6f} | {point.validation_loss:.6f} |"
        for point in winner.curve
    ]

    lines += [
        "",
        "Early stopping reads the clean-validation **reconstruction loss** — the "
        "autoencoder's own objective — not PR-AUC. Stopping on PR-AUC would tune the model "
        "against the metric it is judged by. Which candidate ships is a separate decision "
        "and reads validation PR-AUC.",
        "",
        "## Abstention threshold N, swept on validation",
        "",
        "N is a scoring-time policy, not a training one: the autoencoder fits on every "
        f"eligible window of at least {MIN_SEQUENCE_LENGTH} transactions and N only decides "
        "which windows get an opinion. The sweep therefore costs no extra training and the "
        "weights are identical across it.",
        "",
        "| N | validation account PR-AUC |",
        "|---|---|",
    ]
    lines += [
        f"| {value} | {score:.4f} |{' **(chosen)**' if value == winner.min_length else ''}"
        for value, score in report.min_length_sweep
    ]

    lines += [
        "",
        "## Full results",
        "",
        "### Deployed system — all test accounts, abstentions counted as never flagged",
        "",
        "```",
        winner.result.render(),
        "```",
        "",
        "### The model alone — only accounts it has enough history to score",
        "",
        "```",
        report.scoreable_only.render(),
        "```",
        "",
        "The gap between these two blocks is the cost of abstention, and the first is the "
        "honest basis for any system-level claim.",
        "",
    ]
    if not np.isfinite(winner.result.threshold) or winner.result.confusion.recall >= 1.0:
        lines += [
            "> **The cost-minimising threshold has collapsed to flagging everything.** That "
            "is the honest arithmetic, not a bug: at the account level a false negative "
            f"costs the account's fraud value plus {report.cost_model.chargeback_fee:,.2f} "
            f"while a review costs {report.cost_model.review_cost:,.2f}, and under the cost "
            "model's stated assumption of **unbounded review capacity** the ratio makes "
            "reviewing every account cheaper than missing any. The precision and recall in "
            "the block above are therefore degenerate — precision equals the base rate and "
            "recall is 1.0 — and carry no information about the model. **PR-AUC is "
            "threshold-free and is unaffected; the operating point to quote for a product is "
            "the capacity-constrained one below.** The review-cost-only sensitivity sweep "
            "further down shows where the recommendation stops collapsing.",
            "",
        ]
    lines += [
        "### At a staffable operating point",
        "",
        "```",
        report.capacity.render(),
        "```",
        "",
        "### Cost sensitivity — thresholds re-chosen on validation",
        "",
        "```",
        report.sensitivity,
        "```",
        "",
        "### Latency",
        "",
        "```",
        f"{int(report.latency['calls'])} sequential score() calls",
        f"  p50  {report.latency['p50_ms']:.3f} ms",
        f"  p95  {report.latency['p95_ms']:.3f} ms   (budget {LATENCY_BUDGET_P95_MS:.0f} ms)",
        f"  p99  {report.latency['p99_ms']:.3f} ms",
        f"  max  {report.latency['max_ms']:.3f} ms",
        "```",
        "",
        "Measures the scoring call only — window in, decision out. Assembling the window from "
        "the account's history is Phase 7's scoring endpoint and is budgeted separately.",
        "",
        "## What this does NOT catch",
        "",
        "Written from the false negatives actually observed on the held-out test split at "
        "the **capacity-constrained** operating point — the one a review team could actually "
        "run — not from reasoning about the architecture. BUILD_LOG records that in Phase 2 "
        "both such claims were written from reasoning and both turned out to be false.",
        "",
    ]
    lines += [f"- {line}" for line in report.false_negatives.describe()]
    lines += [
        "",
        "## Other candidates in full",
        "",
    ]
    for candidate in report.candidates:
        lines += ["```", candidate.result.render(), "```", ""]
    return "\n".join(lines) + "\n"


def run(
    settings: Settings,
    *,
    sample: int | None = None,
    epochs: int = TRAINING_EPOCHS,
    skip_registry: bool = False,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_path: Path | None = None,
) -> CorpusReport:
    """Train and evaluate Tier-2 on IEEE-CIS."""
    logger.info("random seed = %d", RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.use_deterministic_algorithms(True)
    # Fixed thread count for the same reason LightGBM gets `deterministic=True` and
    # `num_threads=4` in Tier-1: without it the reduction order varies between runs and the
    # reproducibility claim the registry makes about this model would be false.
    torch.set_num_threads(4)
    logger.info("device = cpu, threads = 4, torch = %s", torch.__version__)

    frames = load_splits(settings.processed_data_dir, SOURCE, sample)
    report = run_corpus(
        frames,
        datetime.now(UTC),
        epochs=epochs,
        artifact_dir=artifact_dir,
        registry_path=registry_path,
        reports_dir=settings.reports_dir,
    )

    print()
    print("================ tier-2, ieee_cis ================")
    print()
    print(report.winner.result.render())
    print()
    print(report.scoreable_only.render())
    print()
    print(report.decile_text)
    print()
    print(f"error vs length, Spearman rho = {report.error_length_correlation:+.4f}")
    print()
    print(report.sensitivity)
    print()
    print(f"Coverage by split at N={report.spec.min_length}:")
    for split, measured in report.coverage_by_split.items():
        print(
            f"  {split:<6} accounts {100 * measured['account_coverage']:>5.1f}%   "
            f"rows {100 * measured['row_coverage']:>5.1f}%   "
            f"fraud {100 * measured['fraud_coverage']:>5.1f}%   "
            f"({measured['accounts_total'] - measured['accounts_abstained']:,.0f} of "
            f"{measured['accounts_total']:,.0f} accounts scored)"
        )
    print()
    print(
        f"Latency: p50 {report.latency['p50_ms']:.3f}ms  "
        f"p95 {report.latency['p95_ms']:.3f}ms  "
        f"p99 {report.latency['p99_ms']:.3f}ms  "
        f"(budget p95 < {LATENCY_BUDGET_P95_MS:.0f}ms)"
    )
    print()
    for band, accounts, positives, base_rate, value in report.length_strata:
        print(
            f"  {band:<20} n={accounts:>7,}  positives={positives:>5,}  "
            f"base rate {100 * base_rate:>6.2f}%  PR-AUC {value:.4f}  "
            f"({value / base_rate if base_rate else 0.0:.1f}x)"
        )
    print()
    if report.complementarity is not None:
        for line in report.complementarity.describe():
            print(f"  - {line}")
        print()
    for line in report.false_negatives.describe():
        print(f"  - {line}")

    if report.latency["p95_ms"] >= LATENCY_BUDGET_P95_MS:
        logger.error(
            "p95 latency %.3fms exceeds the %.0fms budget",
            report.latency["p95_ms"],
            LATENCY_BUDGET_P95_MS,
        )
    if report.winner.result.is_leak_suspicious:
        logger.warning(
            "test PR-AUC %.4f exceeds %.2f. Treat as a suspected leak until disproven "
            "(ml-evaluation-standards section 4).",
            report.winner.result.pr_auc,
            LEAK_SUSPICION_PR_AUC,
        )
    if not skip_registry:
        register(report, registry_path, artifact_dir)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(report, sample, epochs), encoding="utf-8")
        logger.info("wrote %s", report_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run Tier-2 training from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m app.models.train_tier2",
        description="Train, compare and register the RiskIQ Tier-2 behavioural layer.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Override DATA_DIR.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Earliest N rows per split. For iteration only; results are not reportable.",
    )
    parser.add_argument(
        "--epochs", type=int, default=TRAINING_EPOCHS, help="Maximum epochs per candidate."
    )
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="Do not save artefacts or append to models/registry.json.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Where to write the metrics report (default: notebooks/tier2_report.md).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        stream=sys.stdout,
    )

    settings = get_settings()
    if args.data_dir is not None:
        settings = settings.model_copy(update={"data_dir": args.data_dir})

    try:
        run(
            settings,
            sample=args.sample,
            epochs=args.epochs,
            skip_registry=args.skip_registry,
            report_path=args.report or (settings.reports_dir / "tier2_report.md"),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("tier-2 training failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
