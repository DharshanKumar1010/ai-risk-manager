"""What Tier-1 is allowed to see, and how the matrix is built from it.

Tier-1 must be scoreable at ingestion with no dependency on Tier-2 or Tier-3. Reading the
Phase 1 code, that constraint sorts the available columns into four classes:

**A — row-only.** Computable from the arriving transaction alone. ``amount_log``,
``hour_of_day``, ``has_identity``, the ``freq_*`` encodings (an O(1) dict lookup against a
table shipped with the model), and every raw IEEE-CIS column: ``C1``-``C14``, ``D1``-``D15``,
``M1``-``M9``, the card/address/distance block and the identity sidecar.

**B — account state.** ``velocity_*``, ``amount_zscore_vs_own_history``, the ``*_mismatch``
familiarity features, ``counterparty_prior_txn_count``. These need the account's prior rows,
but they are **Phase 1 feature-store outputs, not Tier-2/3 model outputs** — so using them
leaves Tier-1 independently scoreable, which is what the constraint actually asks. One indexed
range scan against ``ix_transactions_account_time`` supplies them at serving time.

**A' — post-settlement.** PaySim's ``balance_error_*`` and ``drains_origin_balance`` read
``newbalance*``, the balance *after* the transaction settles. They are in the corpus, they are
not temporally leaky, and a real ingestion-time scorer still would not have them. They are
marked, and Tier-1 trains a second PaySim model without them so the gap is measured rather
than asserted.

**Denied.** Labels, split metadata, and — the one that would actually produce a fake result —
anything monotone in time. ``TransactionDT``, ``TransactionID`` and PaySim's ``step`` all
increase along the timeline, and under a chronological split every test value exceeds every
train value. A tree splits on them, scores beautifully in-sample, and sends the whole test set
down one branch. :data:`DENIED_COLUMNS` is enforced by ``tests/test_tier1.py``, not by care.

The remaining decisions — which categoricals LightGBM takes natively, which are too
high-cardinality and get frequency-encoded instead, and which columns are degenerate — are all
**fitted on the train split** and recorded in the returned :class:`Tier1InputSpec`, which
hashes into a Tier-1 ``feature_version`` of its own.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.data.feature_store import FeatureDefinition, digest_encoders
from app.data.features import (
    MISSING_SENTINEL,
    apply_frequency_encoders,
    fit_frequency_encoders,
)
from app.data.raw_spec import SourceDataset

#: Columns that must never reach a model matrix, whatever else changes.
#:
#: Three groups, for three different reasons:
#:
#: *Labels and training metadata* — ``is_fraud``/``isFraud`` are the target; ``split`` and
#: ``feature_version`` describe how the row was processed, not the transaction.
#:
#: *Monotone in time* — ``TransactionDT``, ``TransactionID`` (issued in time order) and
#: PaySim's ``step`` and ``transaction_id`` (the source row index). Under a chronological
#: split these separate train from test perfectly and teach nothing.
#:
#: *Identifiers* — ``account_id``, ``nameOrig``, ``nameDest``, ``counterparty_id``. A tree
#: given a near-unique key memorises the training rows.
#:
#: ``uid_strategy`` is denied because it describes *our* account-inference fallback ladder
#: rather than the transaction; its content is already visible through the nullness of
#: ``addr1`` and ``D1``. ``amount``/``TransactionAmt`` are denied as exact duplicates of
#: ``amount_log``, which is monotone in them and therefore gives a tree identical splits.
DENIED_COLUMNS: frozenset[str] = frozenset(
    {
        # Labels and training metadata.
        "is_fraud",
        "isFraud",
        "isFlaggedFraud",
        "split",
        "feature_version",
        "source_dataset",
        # Monotone in time.
        "TransactionDT",
        "TransactionID",
        "step",
        "transaction_id",
        "event_time",
        # Identifiers.
        "account_id",
        "counterparty_id",
        "nameOrig",
        "nameDest",
        # Provenance of our own inference, not a property of the transaction.
        "uid_strategy",
        # Duplicates of amount_log.
        "amount",
        "TransactionAmt",
    }
)

#: IEEE-CIS ``transaction_type`` is a verbatim copy of ``ProductCD``; PaySim's is the real
#: field. Denied per corpus rather than globally.
IEEE_EXTRA_DENIED: frozenset[str] = frozenset({"transaction_type"})

#: PaySim columns that read the settled balance. Available in the corpus, absent at ingestion.
PAYSIM_POST_SETTLEMENT_COLUMNS: tuple[str, ...] = (
    "newbalanceOrig",
    "newbalanceDest",
    "balance_error_orig",
    "balance_error_dest",
    "drains_origin_balance",
)

#: Above this many distinct values on train, a categorical is frequency-encoded rather than
#: handed to LightGBM as a native category. LightGBM's categorical splits partition the level
#: set directly, which overfits badly in a long tail — ``DeviceInfo`` alone carries well over a
#: thousand levels, most seen a handful of times.
MAX_NATIVE_CARDINALITY = 64

#: A column null for at least this share of the train split carries no usable signal. PaySim's
#: ``seconds_since_prior_txn`` (99.94% null) and ``amount_zscore_vs_own_history`` (3 non-null
#: rows in 2.77M) are the cases this exists for. Measured on train, never assumed.
MAX_NULL_RATE = 0.999


@dataclass(frozen=True)
class DroppedColumn:
    """One column excluded by measurement, with the measurement that excluded it."""

    column: str
    reason: str

    def __str__(self) -> str:
        """Return the report line."""
        return f"{self.column} ({self.reason})"


def denied_columns_present(columns: Sequence[str]) -> list[str]:
    """Return any denied column found in ``columns``.

    The guard behind ``tests/test_tier1.py``'s leak check. Returns rather than raises so the
    caller can report every violation at once.
    """
    return sorted(set(columns) & DENIED_COLUMNS)


def candidate_columns(frame: pd.DataFrame, source_dataset: SourceDataset) -> list[str]:
    """Return every column of the parquet frame Tier-1 is permitted to consider."""
    denied = DENIED_COLUMNS | (IEEE_EXTRA_DENIED if source_dataset == "ieee_cis" else frozenset())
    return [column for column in frame.columns if column not in denied]


def find_degenerate_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[list[str], list[DroppedColumn]]:
    """Split columns into those carrying variation on train and those that do not.

    Measured on the train split only, so the decision cannot reflect the test period. Two
    exclusions, both of which fire on real Phase 1 columns:

    - **Effectively all-null** at or above :data:`MAX_NULL_RATE`.
    - **Constant.** ``velocity_available`` is True on every IEEE-CIS row and False on every
      PaySim row; within a single-corpus model that is no information at all. Likewise
      ``dest_is_merchant``, which the TRANSFER+CASH_OUT scope filter leaves permanently False.

    Returns:
        ``(kept, dropped)``.
    """
    kept: list[str] = []
    dropped: list[DroppedColumn] = []
    for column in columns:
        series = frame[column]
        null_rate = float(series.isna().mean())
        if null_rate >= MAX_NULL_RATE:
            dropped.append(DroppedColumn(column, f"{100 * null_rate:.2f}% null on train"))
            continue
        if series.nunique(dropna=True) <= 1:
            dropped.append(DroppedColumn(column, "constant on train"))
            continue
        kept.append(column)
    return kept, dropped


def classify_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    """Sort columns into numeric, native-categorical and frequency-encoded.

    Cardinality is measured on the train split, so the same column is classified identically
    for val and test however their level sets differ.

    Returns:
        ``(numeric, native_categorical, frequency_encoded)``.
    """
    numeric: list[str] = []
    native: list[str] = []
    frequency: list[str] = []
    for column in columns:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
        elif series.nunique(dropna=True) <= MAX_NATIVE_CARDINALITY:
            native.append(column)
        else:
            frequency.append(column)
    return numeric, native, frequency


@dataclass(frozen=True)
class Tier1InputSpec:
    """The fitted Tier-1 input definition: what the model sees and how to reproduce it.

    Everything here is derived from the **train split alone**. Applying it to val or test is a
    pure transformation, which is what makes the held-out numbers mean something.

    Attributes:
        source_dataset: The corpus this spec applies to.
        numeric_columns: Numeric and boolean columns, passed through as-is with NaN preserved.
        native_categorical_columns: Low-cardinality categoricals, given to LightGBM as pandas
            ``category`` dtype with the level set fixed here.
        frequency_columns: High-cardinality categoricals, replaced by a ``freq_<column>``
            encoding fitted on train.
        categories: The fixed level set per native categorical.
        encoders: Fitted frequency tables, keyed by source column.
        post_settlement_columns: Columns present in this spec that a real ingestion-time
            scorer would not have. Empty for IEEE-CIS.
        dropped: Columns excluded by measurement, kept for the report.
    """

    source_dataset: SourceDataset
    numeric_columns: tuple[str, ...]
    native_categorical_columns: tuple[str, ...]
    frequency_columns: tuple[str, ...]
    categories: Mapping[str, tuple[str, ...]]
    encoders: Mapping[str, Mapping[str, float]]
    post_settlement_columns: tuple[str, ...]
    dropped: tuple[DroppedColumn, ...]

    @property
    def encoded_frequency_names(self) -> tuple[str, ...]:
        """Return the generated ``freq_<column>`` feature names."""
        return tuple(f"freq_{column}" for column in self.frequency_columns)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return every feature the full (LightGBM-native) matrix carries."""
        return (
            *self.numeric_columns,
            *self.encoded_frequency_names,
            *self.native_categorical_columns,
        )

    @property
    def numeric_feature_names(self) -> tuple[str, ...]:
        """Return the columns of the numeric-only matrix.

        This is the matrix Isolation Forest takes — it cannot consume categoricals, which
        reach it only through their frequency encodings. LightGBM is also fitted on exactly
        this matrix as a controlled comparison, so that "the supervised model won" is not
        confounded with "the supervised model got richer inputs".
        """
        return (*self.numeric_columns, *self.encoded_frequency_names)

    def without_post_settlement(self) -> "Tier1InputSpec":
        """Return a copy restricted to features available at ingestion time.

        The PaySim ablation. The difference between this spec's model and the full one is the
        measured cost of the corpus recording balances only after settlement.
        """
        excluded = set(self.post_settlement_columns)
        return Tier1InputSpec(
            source_dataset=self.source_dataset,
            numeric_columns=tuple(c for c in self.numeric_columns if c not in excluded),
            native_categorical_columns=tuple(
                c for c in self.native_categorical_columns if c not in excluded
            ),
            frequency_columns=tuple(c for c in self.frequency_columns if c not in excluded),
            categories=dict(self.categories),
            encoders={k: v for k, v in self.encoders.items() if k not in excluded},
            post_settlement_columns=(),
            dropped=self.dropped,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the full model matrix for ``frame``.

        NaN is preserved rather than imputed. LightGBM routes missing values natively, and
        scikit-learn 1.9's ``IsolationForest`` accepts them too — so both models see the same
        missingness, and no imputer has to be fitted, versioned and kept in step with the
        model. In IEEE-CIS missingness is signal (76% of rows have no identity record at all),
        and imputing a plausible value would fabricate one.
        """
        encoded = apply_frequency_encoders(frame.copy(), self.encoders)
        columns: dict[str, Any] = {}
        for column in self.numeric_columns:
            series = encoded[column]
            columns[column] = (
                series.astype("float64")
                if not pd.api.types.is_bool_dtype(series)
                else series.astype("float64")
            )
        for name in self.encoded_frequency_names:
            columns[name] = encoded[name].astype("float64")
        for column in self.native_categorical_columns:
            values = encoded[column].astype("string").fillna(MISSING_SENTINEL)
            columns[column] = pd.Categorical(values, categories=list(self.categories[column]))
        return pd.DataFrame(columns, index=frame.index)[list(self.feature_names)]

    def transform_numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the numeric-only matrix — Isolation Forest's input."""
        return self.transform(frame)[list(self.numeric_feature_names)]

    def engineering_parameters(self) -> dict[str, Any]:
        """Return the parameters that define this input set, for the version hash."""
        return {
            "tier": "tier1_anomaly",
            "numeric_columns": list(self.numeric_columns),
            "native_categorical_columns": list(self.native_categorical_columns),
            "frequency_columns": list(self.frequency_columns),
            "post_settlement_columns": list(self.post_settlement_columns),
            "max_native_cardinality": MAX_NATIVE_CARDINALITY,
            "max_null_rate": MAX_NULL_RATE,
            "missing_sentinel": MISSING_SENTINEL,
            "denied_columns": sorted(DENIED_COLUMNS),
            "dropped_on_train": [str(item) for item in self.dropped],
            "nan_policy": "preserved; no imputation fitted",
        }

    def to_feature_definition(self) -> FeatureDefinition:
        """Return the Tier-1 feature definition, whose hash is this model's feature_version.

        Distinct from the Phase 1 pipeline's ``feature_version`` on purpose. Tier-1 reads the
        engineered vector *and* the raw row columns the pipeline deliberately kept out of
        ``transactions.features`` — chiefly ``C1``-``C14`` and ``D1``-``D15``, Vesta's own
        per-entity aggregates, which are the strongest signal this corpus carries. So the
        model's input set is genuinely a different set, and giving it the pipeline's hash
        would be a false claim about what produced a prediction.
        """
        return FeatureDefinition(
            source_dataset=self.source_dataset,
            feature_names=self.feature_names,
            parameters=self.engineering_parameters(),
            encoder_digest=digest_encoders(self.encoders),
        )

    def describe(self) -> str:
        """Return a human-readable summary for the metrics report."""
        lines = [
            f"- **source_dataset**: `{self.source_dataset}`",
            f"- **tier1_feature_version**: `{self.to_feature_definition().feature_version}`",
            f"- **features**: {len(self.feature_names)} "
            f"({len(self.numeric_columns)} numeric, "
            f"{len(self.frequency_columns)} frequency-encoded, "
            f"{len(self.native_categorical_columns)} native categorical)",
            f"- **numeric-only matrix** (Isolation Forest / matched LightGBM): "
            f"{len(self.numeric_feature_names)} features",
        ]
        if self.post_settlement_columns:
            lines.append(
                "- **post-settlement features** (present in corpus, absent at ingestion): "
                + ", ".join(f"`{c}`" for c in self.post_settlement_columns)
            )
        if self.dropped:
            lines.append(
                f"- **dropped on train measurement** ({len(self.dropped)}): "
                + ", ".join(f"`{item}`" for item in self.dropped)
            )
        return "\n".join(lines)


def fit_tier1_input_spec(
    train_frame: pd.DataFrame,
    source_dataset: SourceDataset,
) -> Tier1InputSpec:
    """Fit the Tier-1 input definition on the train split.

    Every decision made here — which columns are degenerate, which categoricals are native,
    the frequency tables, the fixed level sets — is derived from ``train_frame`` alone and
    then applied unchanged to validation and test.

    Args:
        train_frame: The train split, straight from ``data/processed/<corpus>_train.parquet``.
        source_dataset: Which corpus this is.

    Raises:
        ValueError: If a denied column survived into the candidate set. That would be a bug in
            :data:`DENIED_COLUMNS` or in the caller, and it is the kind of bug that produces a
            spectacular and completely fake result.
    """
    candidates = candidate_columns(train_frame, source_dataset)
    violations = denied_columns_present(candidates)
    if violations:
        raise ValueError(f"Denied columns reached the Tier-1 candidate set: {violations}")

    kept, dropped = find_degenerate_columns(train_frame, candidates)
    numeric, native, frequency = classify_columns(train_frame, kept)

    encoders = fit_frequency_encoders(
        train_frame,
        pd.Series(True, index=train_frame.index),
        frequency,
    )

    categories: dict[str, tuple[str, ...]] = {}
    for column in native:
        values = train_frame[column].astype("string").fillna(MISSING_SENTINEL)
        categories[column] = tuple(sorted(str(level) for level in values.unique()))

    post_settlement = (
        tuple(
            column
            for column in (*numeric, *native, *frequency)
            if column in PAYSIM_POST_SETTLEMENT_COLUMNS
        )
        if source_dataset == "paysim"
        else ()
    )

    return Tier1InputSpec(
        source_dataset=source_dataset,
        numeric_columns=tuple(numeric),
        native_categorical_columns=tuple(native),
        frequency_columns=tuple(frequency),
        categories=categories,
        encoders=encoders,
        post_settlement_columns=post_settlement,
        dropped=tuple(dropped),
    )


def labels_of(frame: pd.DataFrame) -> "np.ndarray[Any, np.dtype[np.bool_]]":
    """Return the fraud label as a boolean array."""
    return frame["is_fraud"].to_numpy(dtype=bool)


def amounts_of(frame: pd.DataFrame) -> "np.ndarray[Any, np.dtype[np.float64]]":
    """Return transaction amounts, for the amount-weighted false-negative cost."""
    return frame["amount"].to_numpy(dtype="float64")
