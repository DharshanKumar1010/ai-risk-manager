"""What Tier-2 sees: per-account trailing windows, and how they are padded and scaled.

Tier-1 asks "is this transaction unusual?". Tier-2 asks "is this transaction unusual *for
this account, right now*?", and the unit that carries that question is a **trailing window**:
the last ``W`` transactions of an account ending at, and including, the one being scored.

Three properties of that construction are load-bearing.

**The window is anchored at the scored transaction, not at the account.** One window per
transaction, so Tier-2 emits a per-transaction reconstruction error — which is what
``AuditRecord.tier2_reconstruction_error`` stores and what Phase 5's meta-learner joins on.
The account-level headline is then an aggregation over an account's windows, computed in
``train_tier2``; it is not what this module produces.

**Windows read the account's whole history, and are assigned to the anchor's split.** A test
window may therefore reach back across the split boundary into train rows. That is not
leakage: at serving time the account's real history is exactly what is available, and
refusing to look at it would fabricate a condition that does not exist in deployment. The
guard is that the reach is *one-directional* — :func:`assemble_windows` builds every window
from strictly-earlier-or-equal rows and :func:`find_future_reads` re-checks it against the
timestamps rather than trusting the construction.

**Padding is never free.** Right-padding to ``W`` and then averaging the reconstruction error
over ``W`` would hand a 3-transaction window seven perfectly-reconstructed pad steps and
deflate its error by 7/10. Every short-history account would score as normal, and Tier-2
would be a sequence-length sensor wearing an autoencoder's clothes. Hence
:attr:`SequenceWindows.mask`, which every loss and every score in
``app/models/tier2_behavioral.py`` divides by.

Nothing here reads a column from :data:`app.models.tier1_features.DENIED_COLUMNS`. The
sequence features are deltas and account-relative quantities — ``seconds_since_prior_txn``,
not ``event_time``; ``amount_zscore_vs_own_history``, not ``TransactionDT``. Under a
chronological split an absolute clock separates train from test perfectly and teaches
nothing, which is the failure ``tier1_features`` documents at length and
``tests/test_tier2.py`` re-checks here.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from app.data.feature_store import FeatureDefinition, digest_encoders
from app.data.features import sort_for_engineering
from app.data.raw_spec import SourceDataset
from app.models.tier1_features import denied_columns_present

#: Shortest window Tier-2 will score. Below this it abstains — see
#: ``Tier2Result.is_scoreable``.
#:
#: Two transactions give one inter-arrival interval and no rhythm to be unusual against;
#: reconstructing them is an identity map on a point, which Tier-1 already does better and
#: with a calibrated probability. Phase 1 measured 118,361 IEEE-CIS accounts (57.7%) holding
#: exactly one transaction, so the share of traffic this excludes is large and is reported as
#: a coverage figure rather than left implicit.
MIN_SEQUENCE_LENGTH = 3

#: Default trailing-window length. The phase brief suggests 20; Phase 1 measured the median
#: account at 1 transaction and p75 at 6, which would make a 20-step window overwhelmingly
#: padding. Swept against {5, 10, 20} and selected on validation, never fixed by assertion.
DEFAULT_WINDOW = 10

#: Standardised features are clipped to +/- this many standard deviations.
#:
#: The reconstruction loss is a mean square, so it is dominated by whichever feature has the
#: widest tail. IEEE-CIS ``amount`` spans six orders of magnitude and
#: ``amount_zscore_vs_own_history`` is unbounded by construction; without a clip the LSTM
#: spends its capacity reconstructing one outlier and learns nothing about ``device_is_new``.
STANDARDISED_CLIP = 5.0

#: Columns of the Phase 1 parquet the per-timestep features are derived from. Named
#: explicitly so ``tests/test_tier2.py`` can check this list against ``DENIED_COLUMNS``
#: rather than checking the derived names, which would miss a leak introduced upstream.
SOURCE_COLUMNS: tuple[str, ...] = (
    "amount_log",
    "amount_zscore_vs_own_history",
    "seconds_since_prior_txn",
    "hour_of_day",
    "day_of_week",
    "velocity_count_1h",
    "velocity_count_24h",
    "velocity_sum_24h",
    "device_is_new",
    "device_mismatch",
    "addr_is_new",
    "addr_mismatch",
    "account_prior_txn_count",
    "freq_ProductCD",
    "freq_card4",
    "freq_card6",
    "freq_P_emaildomain",
)

#: The per-timestep feature vector, in matrix column order.
#:
#: Deliberately small and behavioural. Tier-1 already reads the 129-column matrix including
#: Vesta's ``C1``-``C14`` and ``D1``-``D15`` per-entity aggregates; repeating them here would
#: make Tier-2 a worse copy of a layer that exists rather than a different view of the same
#: transaction. What Tier-2 adds is *rhythm* — timing, escalation, and departure from an
#: account's own established pattern.
SEQUENCE_FEATURE_NAMES: tuple[str, ...] = (
    "amount_log",
    "amount_zscore",
    "log_seconds_since_prior",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "velocity_count_1h",
    "velocity_count_24h",
    "log_velocity_sum_24h",
    "device_is_new",
    "device_mismatch",
    "addr_is_new",
    "addr_mismatch",
    "log_account_prior_txn_count",
    "freq_ProductCD",
    "freq_card4",
    "freq_card6",
    "freq_P_emaildomain",
    "has_prior",
    "has_zscore",
)

#: Hours per day and days per week, for the cyclical encodings.
HOURS_PER_DAY = 24.0
DAYS_PER_WEEK = 7.0


def derive_timestep_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the per-timestep feature matrix, unscaled, in :data:`SEQUENCE_FEATURE_NAMES` order.

    Pure with respect to its input and row-local: every column reads only the row it
    describes. The Phase 1 pipeline already did the temporal work — ``velocity_*`` over a
    ``(t-W, t]`` trailing window, ``amount_zscore_vs_own_history`` against strictly prior
    transactions — so nothing here needs to look sideways, and the leakage surface is
    inherited rather than re-created.

    Two encodings are worth naming:

    ``hour_sin``/``hour_cos`` and ``day_sin``/``day_cos``
        Cyclical rather than ordinal. Hour 23 and hour 0 are one hour apart; as raw integers
        they are the two furthest-apart values in the column, and a model reconstructing them
        would treat "just after midnight" as maximally distant from "just before".

    ``has_prior``/``has_zscore``
        Explicit missingness indicators. ``seconds_since_prior_txn`` is null on an account's
        first transaction and ``amount_zscore_vs_own_history`` needs two priors, so nulls
        here are *structural* — they say "this account has no history yet", which is a real
        state and a risk-relevant one. The nulls themselves are imputed to the train mean by
        :meth:`Tier2SequenceSpec.transform`; these two columns are what stop that imputation
        from erasing the fact.

    Args:
        frame: Rows from the Phase 1 IEEE-CIS parquet, any order.

    Returns:
        A float64 frame with :data:`SEQUENCE_FEATURE_NAMES` as columns, sharing ``frame``'s
        index.
    """
    missing = sorted(set(SOURCE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(
            f"Tier-2 sequence features need {len(missing)} column(s) absent from the frame: "
            f"{missing}. Run `python -m app.data.pipeline` to rebuild the feature store."
        )

    hour = frame["hour_of_day"].astype("float64")
    day = frame["day_of_week"].astype("float64")
    prior_count = frame["account_prior_txn_count"].astype("float64")
    zscore = frame["amount_zscore_vs_own_history"].astype("float64")

    # `dict[str, Any]` for the same reason `Tier1InputSpec.transform` uses it: pandas-stubs
    # types a numpy ufunc over a Series as returning `ndarray`, though at runtime it returns
    # a Series. Both are valid DataFrame column inputs and the column order is pinned below.
    columns: dict[str, Any] = {
        "amount_log": frame["amount_log"].astype("float64"),
        "amount_zscore": zscore,
        # Inter-arrival time spans seconds to weeks. log1p keeps "20 seconds vs 2 minutes"
        # and "3 days vs 30 days" both visible; on a linear scale the second pair swamps the
        # first, and it is the first that distinguishes a scripted burst from a human.
        "log_seconds_since_prior": np.log1p(
            frame["seconds_since_prior_txn"].astype("float64").clip(lower=0.0)
        ),
        "hour_sin": np.sin(2.0 * np.pi * hour / HOURS_PER_DAY),
        "hour_cos": np.cos(2.0 * np.pi * hour / HOURS_PER_DAY),
        "day_sin": np.sin(2.0 * np.pi * day / DAYS_PER_WEEK),
        "day_cos": np.cos(2.0 * np.pi * day / DAYS_PER_WEEK),
        "velocity_count_1h": frame["velocity_count_1h"].astype("float64"),
        "velocity_count_24h": frame["velocity_count_24h"].astype("float64"),
        "log_velocity_sum_24h": np.log1p(
            frame["velocity_sum_24h"].astype("float64").clip(lower=0.0)
        ),
        "device_is_new": frame["device_is_new"].astype("float64"),
        "device_mismatch": frame["device_mismatch"].astype("float64"),
        "addr_is_new": frame["addr_is_new"].astype("float64"),
        "addr_mismatch": frame["addr_mismatch"].astype("float64"),
        "log_account_prior_txn_count": np.log1p(prior_count),
        "freq_ProductCD": frame["freq_ProductCD"].astype("float64"),
        "freq_card4": frame["freq_card4"].astype("float64"),
        "freq_card6": frame["freq_card6"].astype("float64"),
        "freq_P_emaildomain": frame["freq_P_emaildomain"].astype("float64"),
        "has_prior": (prior_count > 0).astype("float64"),
        "has_zscore": zscore.notna().astype("float64"),
    }
    return pd.DataFrame(columns, index=frame.index)[list(SEQUENCE_FEATURE_NAMES)]


@dataclass(frozen=True)
class SequenceWindows:
    """Trailing windows over one ordered frame, held as gather indices rather than a tensor.

    Materialising ``(n_windows, W, n_features)`` directly would cost roughly 470MB at
    ``W=10`` and 940MB at ``W=20`` on the 590,540-row IEEE-CIS frame, for data that is
    almost entirely repetition — each row appears in up to ``W`` windows. Storing the base
    matrix once plus an index array keeps memory flat as ``W`` grows and lets
    :meth:`batch` build only the minibatch actually being trained on.

    Attributes:
        base: ``(n_rows, n_features)`` scaled per-timestep matrix, in account-then-time order.
        gather: ``(n_windows, window)`` row indices into ``base``. Positions at or beyond a
            window's true length hold a repeat of its first row and are masked out; they are
            never read as data.
        mask: ``(n_windows, window)`` True on real timesteps. Right-padded, so each row is a
            run of True followed by a run of False.
        lengths: ``(n_windows,)`` true window length, in ``[1, window]``.
        anchor_row: ``(n_windows,)`` index into ``base`` of the transaction being scored —
            always the window's last real timestep.
        window: The ``W`` these were built at.
    """

    base: npt.NDArray[np.float32]
    gather: npt.NDArray[np.int32]
    mask: npt.NDArray[np.bool_]
    lengths: npt.NDArray[np.int32]
    anchor_row: npt.NDArray[np.int32]
    window: int

    def __len__(self) -> int:
        """Return the number of windows."""
        return int(self.gather.shape[0])

    @property
    def n_features(self) -> int:
        """Return the per-timestep feature count."""
        return int(self.base.shape[1])

    def select(self, keep: npt.NDArray[np.bool_]) -> "SequenceWindows":
        """Return the subset of windows where ``keep`` is True, sharing the same base matrix.

        Used to take one split's anchors, or the fraud-free subset the autoencoder trains on.
        ``base`` is deliberately not subset: a kept window may legitimately reference rows
        whose own anchors were dropped, which is the whole point of letting a test window
        reach back into the train period.
        """
        return SequenceWindows(
            base=self.base,
            gather=self.gather[keep],
            mask=self.mask[keep],
            lengths=self.lengths[keep],
            anchor_row=self.anchor_row[keep],
            window=self.window,
        )

    def batch(
        self, index: npt.NDArray[np.intp]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Materialise ``(values, mask)`` for the windows at ``index``.

        Returns:
            ``values`` of shape ``(len(index), window, n_features)`` with padded positions
            zeroed, and ``mask`` of shape ``(len(index), window)`` as float32 — float rather
            than bool because it is multiplied into the loss, and a bool tensor would promote
            on every use.
        """
        mask = self.mask[index]
        values = self.base[self.gather[index]]
        values[~mask] = 0.0
        return values, mask.astype(np.float32)


def assemble_windows(
    matrix: npt.NDArray[np.float32],
    account_id: "pd.Series[Any]",
    window: int,
) -> tuple[SequenceWindows, npt.NDArray[np.intp]]:
    """Build one trailing window per row, vectorised.

    The frame must already be in event-time order (:func:`app.data.features.sort_for_engineering`).
    Rows are re-ordered here into account-then-time blocks, which makes each account's history
    contiguous and turns "the previous ``W-1`` transactions of this account" into a slice —
    the step that lets 590,540 windows be built without a Python loop.

    Args:
        matrix: ``(n_rows, n_features)`` per-timestep features in event-time order.
        account_id: Account identifier per row, in the same order as ``matrix``.
        window: ``W``, the maximum trailing length.

    Returns:
        ``(windows, order)``. ``order`` maps a position in the account-then-time base matrix
        back to its row position in the caller's event-time frame, so labels, timestamps and
        splits can be aligned to ``anchor_row`` without re-sorting them.

    Raises:
        ValueError: If ``window`` is below :data:`MIN_SEQUENCE_LENGTH`, which would make every
            assembled window shorter than the model will agree to score.
    """
    if window < MIN_SEQUENCE_LENGTH:
        raise ValueError(
            f"window={window} is below MIN_SEQUENCE_LENGTH={MIN_SEQUENCE_LENGTH}; every "
            "assembled window would be too short to score."
        )

    # Stable sort on the account alone: the frame is already in event-time order, so a stable
    # sort leaves each account's rows in time order within its block. Sorting on the pair
    # would be equivalent and slower.
    order = np.asarray(
        account_id.reset_index(drop=True).sort_values(kind="mergesort").index, dtype=np.intp
    )
    ordered_accounts = account_id.to_numpy()[order]

    position = np.arange(order.size, dtype=np.int64)
    # First row of each contiguous account block, broadcast back over the block.
    is_block_start = np.empty(order.size, dtype=bool)
    if order.size:
        is_block_start[0] = True
        is_block_start[1:] = ordered_accounts[1:] != ordered_accounts[:-1]
    block_start = np.maximum.accumulate(np.where(is_block_start, position, 0))

    low = np.maximum(block_start, position - window + 1)
    lengths = (position - low + 1).astype(np.int32)

    offsets = np.arange(window, dtype=np.int64)
    mask = offsets[None, :] < lengths[:, None]
    # Out-of-range slots are clamped onto the window's first row so the gather stays in
    # bounds. They are zeroed by `batch` and excluded by `mask`; the clamp is arithmetic
    # hygiene, never a value the model sees.
    gather = np.where(mask, low[:, None] + offsets[None, :], low[:, None])

    return (
        SequenceWindows(
            base=np.ascontiguousarray(matrix[order], dtype=np.float32),
            gather=gather.astype(np.int32),
            mask=mask,
            lengths=lengths,
            anchor_row=position.astype(np.int32),
            window=window,
        ),
        order,
    )


def find_future_reads(
    windows: SequenceWindows,
    event_time: npt.NDArray[np.int64],
) -> int:
    """Return how many windows contain a timestep later than their own anchor.

    The answer must be zero. :func:`assemble_windows` constructs windows that cannot look
    forward, so this checks the *result* against the timestamps rather than re-stating the
    construction — which is the difference between an assertion and a comment, and the form
    ``ml-evaluation-standards`` section 1 asks leakage checks to take.

    Args:
        windows: Assembled windows.
        event_time: Timestamps as integer nanoseconds, indexed like ``windows.base``.
    """
    times = np.where(windows.mask, event_time[windows.gather], np.iinfo(np.int64).min)
    anchor_times = event_time[windows.anchor_row]
    return int(np.sum(np.any(times > anchor_times[:, None], axis=1)))


@dataclass(frozen=True)
class Tier2SequenceSpec:
    """The fitted Tier-2 input definition: what a timestep is, and how it is scaled.

    Everything here is derived from **fraud-free train windows alone**. Applying it to
    validation and test is a pure transformation, which is what makes the held-out numbers
    mean anything.

    Attributes:
        source_dataset: The corpus this spec applies to. IEEE-CIS in Phase 3 — PaySim was
            measured at 99.9% single-transaction accounts, and a sequence model over a corpus
            with no sequences produces a number that means nothing.
        feature_names: Per-timestep features, in matrix column order.
        window: Trailing window length ``W``.
        min_length: Shortest window the model will score; below it Tier-2 abstains.
        means: Per-feature mean over eligible train timesteps.
        stds: Per-feature standard deviation over the same. Zero-variance features are stored
            as 1.0 so the division is a no-op rather than a NaN.
        clip: Standardised values are clipped to +/- this.
    """

    source_dataset: SourceDataset
    feature_names: tuple[str, ...]
    window: int
    min_length: int
    means: tuple[float, ...]
    stds: tuple[float, ...]
    clip: float

    def __post_init__(self) -> None:
        """Reject a spec whose scaler does not line up with its feature list."""
        if not (len(self.feature_names) == len(self.means) == len(self.stds)):
            raise ValueError(
                f"Tier-2 spec has {len(self.feature_names)} features but "
                f"{len(self.means)} means and {len(self.stds)} standard deviations."
            )
        if self.min_length > self.window:
            raise ValueError(
                f"min_length={self.min_length} exceeds window={self.window}; no window could "
                "ever be long enough to score."
            )

    @property
    def n_features(self) -> int:
        """Return the per-timestep feature count."""
        return len(self.feature_names)

    def transform(self, frame: pd.DataFrame) -> npt.NDArray[np.float32]:
        """Return the scaled per-timestep matrix for ``frame``.

        Standardise, clip, then impute remaining nulls to zero — which after standardisation
        *is* the train mean. That imputation is only honest because ``has_prior`` and
        ``has_zscore`` travel alongside as explicit indicators: without them, "this account
        has no history yet" and "this account is exactly average" would arrive at the model
        as the same vector, and they are opposite statements about risk.
        """
        raw = derive_timestep_frame(frame).to_numpy(dtype="float64")
        standardised = (raw - np.asarray(self.means)) / np.asarray(self.stds)
        clipped = np.clip(standardised, -self.clip, self.clip)
        return np.ascontiguousarray(np.nan_to_num(clipped, nan=0.0), dtype=np.float32)

    def engineering_parameters(self) -> dict[str, Any]:
        """Return the parameters that define this input set, for the version hash."""
        return {
            "tier": "tier2_behavioral",
            "window": self.window,
            "min_sequence_length": self.min_length,
            "standardised_clip": self.clip,
            "source_columns": list(SOURCE_COLUMNS),
            "feature_names": list(self.feature_names),
            "padding": "right; masked out of the loss and of every reported error",
            "nan_policy": (
                "imputed to the train mean after standardisation, with has_prior/has_zscore "
                "carrying the missingness explicitly"
            ),
            "training_eligibility": (
                "windows whose account has no fraud anywhere in the train split; one-class "
                "supervised, not unsupervised"
            ),
            "denied_columns": sorted(denied_columns_present(SOURCE_COLUMNS)),
        }

    def to_feature_definition(self) -> FeatureDefinition:
        """Return the Tier-2 feature definition, whose hash is this model's feature_version.

        Distinct from both the Phase 1 pipeline hash and Tier-1's. Tier-2's input is not a
        row, it is a ``(window, feature)`` matrix with its own derivations, its own padding
        rule and its own fitted scaler — so giving it either existing version would be a
        false claim about what produced a prediction.

        The fitted scaler goes through ``digest_encoders`` for exactly the reason that
        function exists: two runs over different training data produce different means and
        standard deviations under identical code, and a version blind to that would call two
        genuinely different feature sets the same one.
        """
        scaler: Mapping[str, Mapping[str, float]] = {
            "scaler_mean": dict(zip(self.feature_names, self.means, strict=True)),
            "scaler_std": dict(zip(self.feature_names, self.stds, strict=True)),
        }
        return FeatureDefinition(
            source_dataset=self.source_dataset,
            feature_names=self.feature_names,
            parameters=self.engineering_parameters(),
            encoder_digest=digest_encoders(scaler),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the sidecar shape written next to the saved weights."""
        return {
            "source_dataset": self.source_dataset,
            "feature_names": list(self.feature_names),
            "window": self.window,
            "min_length": self.min_length,
            "means": list(self.means),
            "stds": list(self.stds),
            "clip": self.clip,
            "feature_version": self.to_feature_definition().feature_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Tier2SequenceSpec":
        """Rebuild a spec from its sidecar."""
        return cls(
            source_dataset=payload["source_dataset"],
            feature_names=tuple(payload["feature_names"]),
            window=int(payload["window"]),
            min_length=int(payload["min_length"]),
            means=tuple(float(value) for value in payload["means"]),
            stds=tuple(float(value) for value in payload["stds"]),
            clip=float(payload["clip"]),
        )

    def describe(self) -> str:
        """Return a human-readable summary for the metrics report."""
        return "\n".join(
            (
                f"- **source_dataset**: `{self.source_dataset}`",
                f"- **tier2_feature_version**: `{self.to_feature_definition().feature_version}`",
                f"- **window (W)**: {self.window} transactions, trailing, anchored at the "
                "transaction being scored",
                f"- **minimum length (N)**: {self.min_length} — shorter histories abstain "
                "rather than score",
                f"- **per-timestep features**: {self.n_features} "
                f"({', '.join(f'`{name}`' for name in self.feature_names)})",
                f"- **scaling**: standardised on fraud-free train timesteps, clipped to "
                f"+/-{self.clip:g} sd",
            )
        )


def fit_sequence_spec(
    train_frame: pd.DataFrame,
    eligible: npt.NDArray[np.bool_],
    source_dataset: SourceDataset,
    *,
    window: int = DEFAULT_WINDOW,
    min_length: int = MIN_SEQUENCE_LENGTH,
) -> Tier2SequenceSpec:
    """Fit the Tier-2 input definition on the eligible train rows.

    Args:
        train_frame: The train split, straight from ``data/processed/ieee_cis_train.parquet``.
        eligible: True for rows the autoencoder is allowed to learn "normal" from — see
            :func:`eligible_training_rows`. The scaler is fitted on these alone: standardising
            against a mean that includes known fraud would centre the distribution partly on
            the behaviour the model is supposed to find surprising.
        source_dataset: Which corpus.
        window: Trailing window length.
        min_length: Shortest scoreable window.

    Raises:
        ValueError: If no train row is eligible, or if a denied column reached
            :data:`SOURCE_COLUMNS`. The second is the bug class that produces a spectacular
            and completely fake result, so it fails loudly here rather than quietly later.
    """
    violations = denied_columns_present(SOURCE_COLUMNS)
    if violations:
        raise ValueError(f"Denied columns reached the Tier-2 source set: {violations}")
    if not eligible.any():
        raise ValueError("No eligible train rows: the autoencoder would have nothing to fit.")

    raw = derive_timestep_frame(train_frame.loc[eligible]).to_numpy(dtype="float64")
    means = np.nanmean(raw, axis=0)
    stds = np.nanstd(raw, axis=0)
    # A constant feature carries no information and its standard deviation is zero. Storing
    # 1.0 makes the division a no-op, leaving the column at a constant offset from its mean,
    # rather than emitting NaN across the whole matrix.
    stds = np.where(stds > 0.0, stds, 1.0)

    return Tier2SequenceSpec(
        source_dataset=source_dataset,
        feature_names=SEQUENCE_FEATURE_NAMES,
        window=window,
        min_length=min_length,
        means=tuple(float(value) for value in np.nan_to_num(means, nan=0.0)),
        stds=tuple(float(value) for value in stds),
        clip=STANDARDISED_CLIP,
    )


def eligible_training_rows(train_frame: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Return which train rows the autoencoder may learn "normal" from.

    A row is eligible when **its account has no fraud anywhere in the train split** — not
    merely when the row itself is unlabelled fraud.

    The distinction matters on this corpus specifically. IEEE-CIS propagates a chargeback
    label forward across an account's subsequent transactions for roughly 120 days, so an
    account carrying any fraud is a *compromised* account, and its clean-looking earlier rows
    are ambiguous: plausibly already under someone else's control, simply not yet disputed.
    Training on them would teach the autoencoder that takeover behaviour is normal, which is
    the one thing it must not learn.

    This reads the label, and Tier-2 is therefore **one-class supervised, not unsupervised**.
    Tier-1's Isolation Forest is this project's unsupervised baseline; Tier-2 is not one, and
    the report says so rather than letting "autoencoder" imply it.
    """
    fraud_accounts = set(train_frame.loc[train_frame["is_fraud"].astype(bool), "account_id"])
    return ~train_frame["account_id"].isin(fraud_accounts).to_numpy(dtype=bool)


def order_full_history(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate the three splits into one event-time-ordered history.

    Windows are built over the account's whole history and assigned to the anchor's split, so
    the splits have to be seen together — a test window whose account first appeared during
    the train period genuinely has that history available at serving time.

    The ``split`` column survives the concatenation and is what
    ``train_tier2`` filters anchors on afterwards. Ordering uses
    :func:`app.data.features.sort_for_engineering` — a stable mergesort — rather than a fresh
    ``sort_values``: the Phase 1 pipeline's determinism guarantee is specifically about that
    sort, and re-sorting with a different ``kind`` would quietly break it.
    """
    combined = pd.concat(
        [frame.assign(split=split) for split, frame in frames.items()], ignore_index=True
    )
    return sort_for_engineering(combined)


def coverage(
    lengths: npt.NDArray[np.int32],
    labels: npt.NDArray[np.bool_],
    min_length: int,
) -> dict[str, float]:
    """Return what share of a split Tier-2 can score at all.

    A first-class number, not a footnote. Phase 1 measured 57.7% of IEEE-CIS accounts holding
    a single transaction, so the abstention rate is large by construction, and a precision
    figure quoted over the scoreable subset alone describes a different product from the one
    that would actually be deployed. The **fraud** coverage is the decisive one: a layer that
    can only score the transactions nobody was worried about has not earned its place.

    Returns:
        Row, fraud and flaggable-fraud shares, all in ``[0, 1]``.
    """
    scoreable = lengths >= min_length
    positives = int(labels.sum())
    return {
        "rows_total": float(labels.size),
        "rows_scoreable": float(scoreable.sum()),
        "row_coverage": float(scoreable.mean()) if labels.size else 0.0,
        "fraud_total": float(positives),
        "fraud_scoreable": float((scoreable & labels).sum()),
        "fraud_coverage": float((scoreable & labels).sum() / positives) if positives else 0.0,
    }


def aggregate_to_accounts(
    account_id: "Sequence[Any] | npt.NDArray[np.object_]",
    scores: npt.NDArray[np.float64],
    labels: npt.NDArray[np.bool_],
    amounts: npt.NDArray[np.float64],
    *,
    how: str = "max",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.float64], int]:
    """Collapse per-transaction scores to one row per account, within a single split.

    **This is the unit Tier-2's headline is reported at**, and the reason is the label
    structure rather than a preference. IEEE-CIS propagates a chargeback across an account's
    later transactions, so one compromised account with 300 rows contributes 300 correlated
    positives; a per-transaction PR-AUC counts those as 300 independent correct calls when
    the model made one, and ``bootstrap_pr_auc``'s interval — which resamples rows as though
    they were independent — comes out far too tight. At the account level the resampling unit
    genuinely is independent, so the existing bootstrap is correct here unmodified.

    Called once per split. A straddling account (Phase 1 measured 29,321, 14.3%) therefore
    appears as separate account-split units rather than being given one global label, which
    would let its test outcome describe its validation rows.

    Args:
        account_id: Account per scored transaction.
        scores: Per-transaction anomaly score, higher meaning more suspicious.
        labels: True where the transaction is fraud.
        amounts: Transaction amount, for the account-level false-negative cost.
        how: ``"max"`` or ``"mean"``. Max is primary — a takeover is one abnormal stretch,
            not a uniformly elevated average, and averaging dilutes it across however many
            ordinary transactions the account also made.

    Returns:
        ``(account_scores, account_labels, account_amounts, n_accounts)``. The amount is the
        summed value of that account's fraudulent transactions in this split, which is what a
        missed account actually costs; it is zero for a legitimate account, which never
        enters the false-negative term.
    """
    if how not in ("max", "mean"):
        raise ValueError(f"how must be 'max' or 'mean', got {how!r}")
    table = pd.DataFrame(
        {
            "account_id": pd.Series(list(account_id)),
            "score": scores,
            "is_fraud": labels,
            "fraud_amount": np.where(labels, amounts, 0.0),
        }
    )
    grouped = table.groupby("account_id", sort=True).agg(
        score=("score", how),
        is_fraud=("is_fraud", "any"),
        fraud_amount=("fraud_amount", "sum"),
    )
    return (
        grouped["score"].to_numpy(dtype="float64"),
        grouped["is_fraud"].to_numpy(dtype=bool),
        grouped["fraud_amount"].to_numpy(dtype="float64"),
        int(len(grouped)),
    )
