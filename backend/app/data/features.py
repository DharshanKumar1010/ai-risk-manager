"""Feature engineering over the canonical frame.

Every function here obeys one rule: **a feature describing a transaction may only read data
from that transaction or from strictly earlier ones**. That is not a style preference, it is
what makes the held-out metric mean anything, and ``tests/test_pipeline.py`` enforces it by
recomputing features against a truncated history rather than by reading the code.

How the rule is kept, concretely:

- The frame is sorted by ``event_time`` once, so every ``cumsum`` / ``cumcount`` / ``shift``
  within an account walks forward in time.
- Trailing windows use ``groupby().rolling(<offset>)``, whose window is ``(t - W, t]`` —
  the current row is included, earlier rows are included, later rows never are.
- The amount z-score compares against strictly prior transactions only. Including the
  current amount in its own baseline would flatten exactly the outlier it exists to detect.
- Frequency encodings are fitted on the **train split only** and applied everywhere.
  Fitting on the full frame would leak the test period's category distribution backwards.
- "The account's usual device/geo" is operationalised as *what this account has used
  before*, never as a mode or first-value over the whole account. ``GroupBy.first()`` and
  ``mode()`` skip nulls, so on an account whose earliest device is missing they reach
  forward to a later row — a subtle leak that reads as innocuous code.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from app.data.raw_spec import SourceDataset

#: Trailing windows for velocity, as ``feature-name label -> pandas offset alias``.
#:
#: The two differ on purpose. pandas deprecated ``H`` in favour of ``h`` for hours while days
#: stay ``D``, so the aliases are not the labels uppercased and cannot be derived from them.
#: Holding the labels separate keeps the engineered feature names stable — and therefore the
#: ``feature_version`` stable — across a pandas alias change.
VELOCITY_WINDOWS: dict[str, str] = {"1h": "1h", "24h": "24h", "7d": "7D"}

#: Categorical columns frequency-encoded per corpus. IEEE-CIS has no merchant-category
#: field, so ``ProductCD`` (product line) and the card network/type stand in for it.
IEEE_FREQUENCY_COLUMNS: tuple[str, ...] = ("ProductCD", "card4", "card6", "P_emaildomain")
PAYSIM_FREQUENCY_COLUMNS: tuple[str, ...] = ("transaction_type",)

#: Entity columns compared against an account's own history. ``addr1`` is the billing
#: region — IEEE-CIS carries no latitude/longitude and no IP, so this is the geo signal.
IEEE_FAMILIARITY_COLUMNS: dict[str, str] = {"device": "DeviceInfo", "addr": "addr1"}

#: Missing categoricals get an explicit level rather than being dropped. For the identity
#: block the missingness *is* the signal — roughly 76% of IEEE-CIS transactions have no
#: identity row at all, and imputing a "typical" device would fabricate one.
MISSING_SENTINEL = "__missing__"

#: A z-score needs at least two prior observations to have a standard deviation at all.
ZSCORE_MIN_PRIOR = 2


def sort_for_engineering(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frame sorted by event time, with a fresh positional index.

    Every cumulative operation below depends on this ordering. A stable sort keeps rows
    sharing a timestamp in their original relative order, so the pipeline is deterministic
    across runs.
    """
    return frame.sort_values("event_time", kind="mergesort").reset_index(drop=True)


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add hour-of-day and day-of-week.

    Both read only the transaction's own timestamp. For IEEE-CIS the calendar *date* rests
    on an assumed epoch, but hour-of-day is derived from ``TransactionDT`` modulo one day
    and is therefore correct regardless of where the epoch is anchored.
    """
    frame["hour_of_day"] = frame["event_time"].dt.hour.astype("int16")
    frame["day_of_week"] = frame["event_time"].dt.dayofweek.astype("int16")
    return frame


def add_amount_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add per-account history depth, recency, and the amount z-score against own history.

    The z-score uses prior transactions only. Sums of priors are derived by subtracting the
    current row from the inclusive cumulative sums, which keeps the whole computation
    vectorised over ~590k rows and ~400k accounts.
    """
    accounts = frame["account_id"]
    amount = frame["amount"].astype("float64")

    prior_count = frame.groupby(accounts, sort=False).cumcount()
    frame["account_prior_txn_count"] = prior_count.astype("int32")

    prior_time = frame.groupby(accounts, sort=False)["event_time"].shift(1)
    frame["seconds_since_prior_txn"] = (frame["event_time"] - prior_time).dt.total_seconds()

    sum_prior = amount.groupby(accounts, sort=False).cumsum() - amount
    squares = amount**2
    sumsq_prior = squares.groupby(accounts, sort=False).cumsum() - squares

    countable = prior_count.where(prior_count > 0)
    mean_prior = sum_prior / countable
    # Sample variance from raw moments: (sum(x^2) - n * mean^2) / (n - 1). Clipped at zero
    # because floating-point cancellation can push a genuinely-zero variance slightly below.
    variance_prior = (sumsq_prior - prior_count * mean_prior**2) / (prior_count - 1).where(
        prior_count >= ZSCORE_MIN_PRIOR
    )
    # `.pow(0.5)` rather than `np.sqrt` so the result stays a Series with `.replace`.
    std_prior = variance_prior.clip(lower=0).pow(0.5)

    frame["amount_prior_mean"] = mean_prior
    # A zero standard deviation (every prior amount identical) would divide to infinity;
    # NaN is the honest answer, meaning "no dispersion to compare against".
    frame["amount_zscore_vs_own_history"] = (amount - mean_prior) / std_prior.replace(0, np.nan)
    frame["amount_log"] = np.log1p(amount)
    return frame


def add_velocity_features(
    frame: pd.DataFrame,
    windows: Mapping[str, str] = VELOCITY_WINDOWS,
) -> pd.DataFrame:
    """Add trailing per-account transaction count and amount sum over each window.

    The window is ``(t - W, t]``: earlier transactions and the current one, never a later
    one. Rows sharing a timestamp within an account each see the others, which is correct —
    at scoring time all of them have arrived.

    Args:
        frame: Sorted by ``event_time``, with a positional index.
        windows: Feature-name label to pandas offset alias, e.g. ``{"7d": "7D"}``.
    """
    work = frame.sort_values(["account_id", "event_time"], kind="mergesort")
    grouped = work.groupby("account_id", sort=True)[["event_time", "amount"]]

    for label, offset in windows.items():
        rolled = grouped.rolling(offset, on="event_time")["amount"]
        # groupby().rolling() emits rows in the grouped-and-sorted order of `work`, so a
        # positional assignment lines up. Pinned by test_velocity_matches_hand_computation.
        work[f"velocity_count_{label}"] = rolled.count().to_numpy()
        work[f"velocity_sum_{label}"] = rolled.sum().to_numpy()

    restored = work.sort_index()
    for label in windows:
        frame[f"velocity_count_{label}"] = restored[f"velocity_count_{label}"].astype("float64")
        frame[f"velocity_sum_{label}"] = restored[f"velocity_sum_{label}"].astype("float64")
    frame["velocity_available"] = True
    return frame


def add_absent_velocity_features(
    frame: pd.DataFrame,
    windows: Mapping[str, str] = VELOCITY_WINDOWS,
) -> pd.DataFrame:
    """Mark velocity as unavailable rather than computing a misleading value.

    PaySim's clock has one-hour granularity, which makes a 1h trailing window degenerate
    ("the same step"), and its ``nameOrig`` is near-unique so there is almost no account
    history to accumulate anyway. The columns are written as nulls with an explicit flag —
    a zero here would read as "this account was inactive", which is a different claim.
    """
    for label in windows:
        frame[f"velocity_count_{label}"] = np.nan
        frame[f"velocity_sum_{label}"] = np.nan
    frame["velocity_available"] = False
    return frame


def add_familiarity_features(frame: pd.DataFrame, columns: Mapping[str, str]) -> pd.DataFrame:
    """Add "is this entity unusual for this account?" flags, from prior rows only.

    For each mapping of ``prefix -> column``:

    ``<prefix>_is_new``
        True when this account has not been seen with this value before.
    ``<prefix>_mismatch``
        One minus the share of the account's *prior* transactions that used this value.
        1.0 means never seen before, 0.0 means it is all this account has ever used, and
        null means there is no prior history to compare against.
    """
    accounts = frame["account_id"]
    prior_account_rows = frame.groupby(accounts, sort=False).cumcount()

    for prefix, column in columns.items():
        values = frame[column].astype("string").fillna(MISSING_SENTINEL)
        prior_pair_rows = frame.groupby([accounts, values], sort=False, dropna=False).cumcount()
        share = prior_pair_rows / prior_account_rows.where(prior_account_rows > 0)
        frame[f"{prefix}_is_new"] = (prior_pair_rows == 0).astype("bool")
        frame[f"{prefix}_mismatch"] = 1.0 - share
    return frame


def fit_frequency_encoders(
    frame: pd.DataFrame,
    train_mask: "pd.Series[bool]",
    columns: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Fit category frequencies on the train split only.

    Fitting on the whole frame would let each row's encoding reflect how common its category
    turns out to be across the test period — a quiet, and quite effective, leak.

    Args:
        frame: The canonical frame.
        train_mask: True for rows in the train split.
        columns: Categorical columns to encode.

    Returns:
        Column name to {category: relative frequency in train}.
    """
    encoders: dict[str, dict[str, float]] = {}
    training_rows = frame.loc[train_mask]
    for column in columns:
        values = training_rows[column].astype("string").fillna(MISSING_SENTINEL)
        counts = values.value_counts()
        total = int(counts.sum())
        encoders[column] = (
            {str(category): int(count) / total for category, count in counts.items()}
            if total
            else {}
        )
    return encoders


def apply_frequency_encoders(
    frame: pd.DataFrame,
    encoders: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Apply fitted frequency tables, mapping categories unseen in train to zero.

    Zero is meaningful here rather than merely convenient: it says "this category never
    occurred in training", which is itself a risk signal for a card network or email domain
    appearing for the first time in the scoring period.
    """
    for column, table in encoders.items():
        values = frame[column].astype("string").fillna(MISSING_SENTINEL)
        frame[f"freq_{column}"] = values.map(table).astype("float64").fillna(0.0)
    return frame


def add_paysim_balance_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add PaySim's balance-consistency signals and counterparty history depth.

    The balance columns do not always reconcile, and the discrepancy is informative rather
    than merely dirty. ``drains_origin_balance`` captures the simulator's takeover
    signature: a transfer that empties the originating account.

    ``counterparty_prior_txn_count`` is the one genuinely useful history feature in this
    corpus — ``nameOrig`` is near-unique but mule accounts recur as ``nameDest``, which is
    the structure Tier-3 is built on.
    """
    amount = frame["amount"].astype("float64")
    frame["balance_error_orig"] = (
        frame["oldbalanceOrg"].astype("float64")
        - amount
        - frame["newbalanceOrig"].astype("float64")
    )
    frame["balance_error_dest"] = (
        frame["oldbalanceDest"].astype("float64")
        + amount
        - frame["newbalanceDest"].astype("float64")
    )
    frame["drains_origin_balance"] = (
        (frame["newbalanceOrig"].astype("float64") == 0.0)
        & (frame["oldbalanceOrg"].astype("float64") > 0.0)
    ).astype("bool")
    # Merchant destinations carry no balance information at all, so their balance_error_dest
    # is structurally meaningless rather than anomalous.
    frame["dest_is_merchant"] = frame["counterparty_id"].astype("string").str.startswith("M")
    frame["dest_is_merchant"] = frame["dest_is_merchant"].fillna(False).astype("bool")
    frame["counterparty_prior_txn_count"] = (
        frame.groupby(frame["counterparty_id"], sort=False, dropna=False).cumcount().astype("int32")
    )
    return frame


def ieee_feature_names() -> tuple[str, ...]:
    """Return every engineered feature key for IEEE-CIS."""
    velocity = tuple(
        f"velocity_{stat}_{window}" for window in VELOCITY_WINDOWS for stat in ("count", "sum")
    )
    familiarity = tuple(
        f"{prefix}_{suffix}"
        for prefix in IEEE_FAMILIARITY_COLUMNS
        for suffix in ("is_new", "mismatch")
    )
    return (
        "account_prior_txn_count",
        "seconds_since_prior_txn",
        "amount_log",
        "amount_prior_mean",
        "amount_zscore_vs_own_history",
        "hour_of_day",
        "day_of_week",
        "velocity_available",
        "has_identity",
        *velocity,
        *familiarity,
        *(f"freq_{column}" for column in IEEE_FREQUENCY_COLUMNS),
    )


def paysim_feature_names() -> tuple[str, ...]:
    """Return every engineered feature key for PaySim."""
    velocity = tuple(
        f"velocity_{stat}_{window}" for window in VELOCITY_WINDOWS for stat in ("count", "sum")
    )
    return (
        "account_prior_txn_count",
        "counterparty_prior_txn_count",
        "seconds_since_prior_txn",
        "amount_log",
        "amount_prior_mean",
        "amount_zscore_vs_own_history",
        "hour_of_day",
        "day_of_week",
        "velocity_available",
        "balance_error_orig",
        "balance_error_dest",
        "drains_origin_balance",
        "dest_is_merchant",
        *velocity,
        *(f"freq_{column}" for column in PAYSIM_FREQUENCY_COLUMNS),
    )


def feature_names_for(source_dataset: SourceDataset) -> tuple[str, ...]:
    """Return the engineered feature keys for one corpus."""
    return ieee_feature_names() if source_dataset == "ieee_cis" else paysim_feature_names()


def engineering_parameters(source_dataset: SourceDataset) -> dict[str, Any]:
    """Return the parameters that define this corpus's feature semantics.

    These go into the ``feature_version`` hash, so changing a window from 24h to 12h moves
    the version even though no feature is renamed.
    """
    shared: dict[str, Any] = {
        "velocity_windows": list(VELOCITY_WINDOWS),
        "zscore_min_prior": ZSCORE_MIN_PRIOR,
        "missing_sentinel": MISSING_SENTINEL,
        "velocity_window_closed": "(t-W, t]",
    }
    if source_dataset == "ieee_cis":
        return {
            **shared,
            "frequency_columns": list(IEEE_FREQUENCY_COLUMNS),
            "familiarity_columns": dict(IEEE_FAMILIARITY_COLUMNS),
            "velocity_computed": True,
        }
    return {
        **shared,
        "frequency_columns": list(PAYSIM_FREQUENCY_COLUMNS),
        "familiarity_columns": {},
        "velocity_computed": False,
        "velocity_omitted_because": "one-hour clock granularity and near-unique nameOrig",
    }


def engineer_features(
    frame: pd.DataFrame,
    *,
    source_dataset: SourceDataset,
    encoders: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Engineer every feature for one corpus.

    Pure with respect to its inputs: given the same frame and encoders it returns the same
    features, which is what lets the leakage test recompute against a truncated frame and
    compare.

    Args:
        frame: Canonical frame for a single corpus, any row order.
        source_dataset: Selects the corpus-specific feature set.
        encoders: Frequency tables from :func:`fit_frequency_encoders`.

    Returns:
        A new frame, sorted by ``event_time``, with the engineered columns added.
    """
    engineered = sort_for_engineering(frame.copy())
    engineered = add_calendar_features(engineered)
    engineered = add_amount_history_features(engineered)

    if source_dataset == "ieee_cis":
        engineered = add_velocity_features(engineered)
        engineered = add_familiarity_features(engineered, IEEE_FAMILIARITY_COLUMNS)
    else:
        engineered = add_absent_velocity_features(engineered)
        engineered = add_paysim_balance_features(engineered)

    return apply_frequency_encoders(engineered, encoders)
