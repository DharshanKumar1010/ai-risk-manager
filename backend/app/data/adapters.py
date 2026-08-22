"""Per-corpus translation into the canonical frame.

Each adapter answers the same question — what are the canonical columns for this row? —
from a very different starting point. The canonical columns are those in
:class:`app.data.schema.TransactionFeatures` plus ``is_fraud`` and ``uid_strategy``; each
adapter additionally carries through the raw columns its own feature engineering needs.

Phase 9's Razorpay webhook adapter plugs into this same seam.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.data.raw_spec import (
    IEEE_CIS_IDENTITY_SPEC,
    IEEE_CIS_TRANSACTION_COLUMNS,
    IEEE_CIS_TRANSACTION_SPEC,
    PAYSIM_FRAUD_BEARING_TYPES,
    PAYSIM_SPEC,
    SCAN_CHUNK_ROWS,
    normalise_column,
    numbered_columns,
)
from app.data.schema import ieee_cis_event_time, paysim_event_time
from app.data.validate_raw import resolve_path

#: The IEEE-CIS raw columns carried into feature engineering and the parquet output. The
#: ``V1``-``V339`` block is deliberately excluded — see :func:`iter_ieee_cis_v_block`.
IEEE_CIS_MODEL_COLUMNS: tuple[str, ...] = (
    "ProductCD",
    *numbered_columns("card", 6),
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    *numbered_columns("C", 14),
    *numbered_columns("D", 15),
    *numbered_columns("M", 9),
)

#: Columns read from the identity sidecar. The full 41 are carried; they are cheap.
IEEE_CIS_IDENTITY_MODEL_COLUMNS: tuple[str, ...] = (
    *numbered_columns("id_", 38, width=2),
    "DeviceType",
    "DeviceInfo",
)

#: Columns that must be present for the pipeline to run at all.
IEEE_CIS_CORE_COLUMNS: tuple[str, ...] = (
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
)

_SECONDS_PER_DAY = 86_400

#: Numeric blocks read as float32. At 590,540 rows this roughly halves the frame's memory
#: against the float64 default, and none of these columns needs more precision.
#:
#: Note which card columns are absent: ``card4`` is the network (visa/mastercard/amex/
#: discover) and ``card6`` the funding type (debit/credit). They sit inside a numerically
#: named block but are categorical, and coercing them to float fails outright.
_FLOAT32_COLUMNS: frozenset[str] = frozenset(
    (
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
        *numbered_columns("C", 14),
        *numbered_columns("D", 15),
    )
)


@dataclass(frozen=True)
class AdapterResult:
    """A canonical frame plus what had to be decided to produce it."""

    frame: pd.DataFrame
    notes: dict[str, Any]


def _ieee_cis_dtypes() -> dict[str, Any]:
    """Return an explicit dtype map for the transaction file."""
    dtypes: dict[str, Any] = {
        "TransactionID": "int32",
        "isFraud": "int8",
        "TransactionDT": "int32",
        "TransactionAmt": "float32",
    }
    for column in _FLOAT32_COLUMNS:
        dtypes[column] = "float32"
    return dtypes


def build_ieee_cis_uid(frame: pd.DataFrame) -> tuple["pd.Series[Any]", "pd.Series[Any]"]:
    """Construct an account identifier for IEEE-CIS, which has no account column.

    IEEE-CIS records no customer id, yet per-account velocity, the amount z-score against an
    account's own history, and Tier-2's sequences all require one. The primary key follows
    the approach the Kaggle community converged on:

        ``D1`` is days since the card account began, so ``floor(TransactionDT / 86400) - D1``
        is a constant per client — a stable client identifier when combined with the card
        and billing region.

    A fallback ladder handles rows missing a component, and every row records which rung
    produced it so the data-quality report can state how much of the corpus has usable
    history at all. Rows reaching ``singleton`` have none: their per-account features are
    noise, and that share is a reported number rather than a hidden assumption.

    This is an inference, not an observation. It is leakage-safe — assigning an identity
    reads no future row — but the identity itself may be wrong, and aggregates computed over
    it are only as good as it is.

    Returns:
        ``(account_id, uid_strategy)``, both aligned to ``frame``'s index.
    """
    card1 = frame["card1"]
    addr1 = frame["addr1"]
    d1 = frame["D1"]
    email = frame["P_emaildomain"].astype("string")

    day = frame["TransactionDT"] // _SECONDS_PER_DAY
    d1n = (day - d1).round()

    def numeric_key(series: "pd.Series[Any]", prefix: str) -> "pd.Series[Any]":
        return prefix + series.round().astype("Int64").astype("string")

    card_key = numeric_key(card1, "c")
    primary = card_key + numeric_key(addr1, "_a") + numeric_key(d1n, "_d")
    email_key = card_key + "_e" + email
    singleton = "txn" + frame["TransactionID"].astype("string")

    has_card = card1.notna()
    has_primary = has_card & addr1.notna() & d1.notna()
    has_email = has_card & email.notna()

    conditions = [has_primary.to_numpy(), has_email.to_numpy(), has_card.to_numpy()]
    account_id = pd.Series(
        np.select(
            conditions,
            [
                primary.to_numpy(dtype=object),
                email_key.to_numpy(dtype=object),
                card_key.to_numpy(dtype=object),
            ],
            default=singleton.to_numpy(dtype=object),
        ),
        index=frame.index,
        dtype="string",
    )
    strategy = pd.Series(
        np.select(
            conditions,
            ["card_addr_d1n", "card_email", "card_only"],
            default="singleton",
        ),
        index=frame.index,
        dtype="string",
    )
    return account_id, strategy


def load_ieee_cis(raw_dir: Path, *, sample: int | None = None) -> AdapterResult:
    """Load IEEE-CIS into the canonical frame, joining the identity sidecar.

    Args:
        raw_dir: Directory holding the untouched downloads.
        sample: Keep only the earliest N transactions. For fast iteration; a chronological
            head rather than a random sample, so the time ordering stays intact.

    Raises:
        FileNotFoundError: If either required file is absent.
    """
    transaction_path = resolve_path(IEEE_CIS_TRANSACTION_SPEC, raw_dir)
    identity_path = resolve_path(IEEE_CIS_IDENTITY_SPEC, raw_dir)
    if transaction_path is None or identity_path is None:
        raise FileNotFoundError(
            f"IEEE-CIS files not found in {raw_dir}. Run `python -m app.data.validate_raw`."
        )

    usecols = [*IEEE_CIS_CORE_COLUMNS, *IEEE_CIS_MODEL_COLUMNS]
    transactions = pd.read_csv(
        transaction_path,
        usecols=usecols,
        dtype=_ieee_cis_dtypes(),
        nrows=sample,
    )
    identity = pd.read_csv(identity_path)
    identity.columns = pd.Index([normalise_column(str(name)) for name in identity.columns])

    merged = transactions.merge(identity, on="TransactionID", how="left", indicator=True)
    has_identity = (merged.pop("_merge") == "both").to_numpy()

    account_id, uid_strategy = build_ieee_cis_uid(merged)
    frame = merged.assign(
        transaction_id=merged["TransactionID"].astype("string"),
        source_dataset="ieee_cis",
        event_time=[ieee_cis_event_time(int(value)) for value in merged["TransactionDT"]],
        amount=merged["TransactionAmt"].astype("float64"),
        account_id=account_id,
        counterparty_id=pd.Series(pd.NA, index=merged.index, dtype="string"),
        transaction_type=merged["ProductCD"].astype("string"),
        is_fraud=merged["isFraud"].astype("bool"),
        uid_strategy=uid_strategy,
        has_identity=has_identity,
    )
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)

    strategy_counts = uid_strategy.value_counts().to_dict()
    return AdapterResult(
        frame=frame,
        notes={
            "rows": len(frame),
            "identity_join_rate": float(has_identity.mean()),
            "uid_strategy_counts": {str(k): int(v) for k, v in strategy_counts.items()},
            "distinct_accounts": int(account_id.nunique()),
        },
    )


def iter_ieee_cis_v_block(raw_dir: Path, *, sample: int | None = None) -> Iterator[pd.DataFrame]:
    """Stream the ``V1``-``V339`` block, keyed by ``TransactionID``.

    Held apart from the main frame for two reasons. It is 339 of the file's 394 columns, so
    carrying it through engineering would roughly triple peak memory for columns nothing in
    Phase 1 reads. And it needs its own treatment before a model should use it: the block is
    missing in correlated groups, and the standard reduction is to cluster by NaN-pattern and
    keep one representative per correlated group — a decision that must be fitted on train
    only, which makes it Phase 2's to make rather than something to do implicitly here.

    Yields:
        Chunks of ``TransactionID`` plus the V columns, as float32.
    """
    path = resolve_path(IEEE_CIS_TRANSACTION_SPEC, raw_dir)
    if path is None:
        raise FileNotFoundError(f"train_transaction.csv not found in {raw_dir}")

    v_columns = [name for name in IEEE_CIS_TRANSACTION_COLUMNS if name.startswith("V")]
    dtypes: dict[str, Any] = {name: "float32" for name in v_columns}
    dtypes["TransactionID"] = "int32"

    remaining = sample
    with pd.read_csv(
        path,
        usecols=["TransactionID", *v_columns],
        dtype=dtypes,
        chunksize=SCAN_CHUNK_ROWS,
    ) as reader:
        for chunk in reader:
            if remaining is not None:
                if remaining <= 0:
                    return
                chunk = chunk.head(remaining)
                remaining -= len(chunk)
            yield chunk


def load_paysim(raw_dir: Path, *, sample: int | None = None) -> AdapterResult:
    """Load PaySim into the canonical frame, scoped to the fraud-bearing transaction types.

    The scope filter is the one substantive decision here. Every one of PaySim's 8,213 fraud
    rows is a ``TRANSFER`` or a ``CASH_OUT`` — verified, not assumed, by
    ``app.data.validate_raw``, which treats a violation as an error. Keeping only those two
    types retains 100% of the positive class while dropping ~3.6M rows that contribute no
    fraud and whose merchant destinations are high-degree hubs that would dominate Tier-3's
    centrality metrics without being rings.

    The filter is recorded in the returned notes and reported in the data-quality report; it
    is an explicit, justified scope, not a silent subsample.

    Args:
        raw_dir: Directory holding the untouched download.
        sample: Keep only the earliest N rows after filtering.
    """
    path = resolve_path(PAYSIM_SPEC, raw_dir)
    if path is None:
        raise FileNotFoundError(
            f"PaySim log not found in {raw_dir}. Run `python -m app.data.validate_raw`."
        )

    raw = pd.read_csv(
        path,
        dtype={
            "step": "int16",
            "type": "string",
            "amount": "float64",
            "nameOrig": "string",
            "nameDest": "string",
            "oldbalanceOrg": "float64",
            "newbalanceOrig": "float64",
            "oldbalanceDest": "float64",
            "newbalanceDest": "float64",
            "isFraud": "int8",
            "isFlaggedFraud": "int8",
        },
    )
    total_rows = len(raw)
    total_positives = int((raw["isFraud"] == 1).sum())

    # The row's position in the untouched file becomes its identifier, so a row in Postgres
    # can be traced back to its source line after filtering.
    raw = raw.assign(source_row=np.arange(total_rows, dtype="int64"))
    kept = raw.loc[raw["type"].isin(PAYSIM_FRAUD_BEARING_TYPES)].copy()
    if sample is not None:
        # Evenly spaced across the timeline rather than a chronological head. PaySim's clock
        # advances one step per hour, so the first 20k rows occupy about nine hours — far too
        # narrow to cut a 70/15/15 split from, and the split would (correctly) refuse. A
        # stride keeps the full 30-day span, which is what makes a sampled run representative.
        kept = kept.sort_values("step", kind="mergesort")
        stride = max(1, len(kept) // sample)
        kept = kept.iloc[::stride].head(sample)

    frame = kept.assign(
        transaction_id=kept["source_row"].astype("string"),
        source_dataset="paysim",
        event_time=pd.to_datetime(
            [paysim_event_time(int(step)) for step in kept["step"]], utc=True
        ),
        account_id=kept["nameOrig"],
        counterparty_id=kept["nameDest"],
        transaction_type=kept["type"],
        is_fraud=kept["isFraud"].astype("bool"),
        uid_strategy="native",
    )
    # isFlaggedFraud is the simulator's own naive rule and therefore a leaked downstream
    # decision. Dropped here so it cannot become a feature by accident.
    frame = frame.drop(columns=["isFlaggedFraud", "isFraud", "type", "source_row"])

    kept_positives = int(frame["is_fraud"].sum())
    return AdapterResult(
        frame=frame,
        notes={
            "rows_in_file": total_rows,
            "rows_kept": len(frame),
            "scope_filter": sorted(PAYSIM_FRAUD_BEARING_TYPES),
            "positives_in_file": total_positives,
            "positives_kept": kept_positives,
            "positive_retention": (kept_positives / total_positives if total_positives else 0.0),
            "distinct_origins": int(frame["account_id"].nunique()),
            "distinct_destinations": int(frame["counterparty_id"].nunique()),
        },
    )
