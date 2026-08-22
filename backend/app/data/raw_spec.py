"""What each raw dataset file is expected to contain.

These constants are the reference the Phase 1 plan's dataset figures are checked against.
Every count here was recorded from the published Kaggle releases and is **asserted**, not
assumed — :mod:`app.data.validate_raw` fails loudly when a download disagrees.

If your copy legitimately differs (a mirror, a re-release), update the constants in this
module and record the change in ``BUILD_LOG.md``. Do not loosen the check: the point of
these numbers is that every later metric can name the exact corpus it came from.

Sources
-------
IEEE-CIS Fraud Detection (Vesta, Kaggle 2019)
    ``train_transaction.csv`` + ``train_identity.csv``, joined on ``TransactionID``.
    The competition's ``test_*.csv`` files are deliberately **not** specified here: Kaggle
    withheld their labels, so they cannot be evaluated on. Our held-out test split is
    carved chronologically out of ``train_transaction.csv``.

PaySim (Lopez-Rojas, "Synthetic Financial Datasets for Fraud Detection")
    A single simulated log. Used for the Tier-3 graph layer only — its ``nameOrig`` is
    near-unique, so it carries network structure but effectively no account history.
"""

from dataclasses import dataclass
from typing import Literal

SourceDataset = Literal["ieee_cis", "paysim"]

#: Row-group size for chunked scans of the large CSVs. 500k rows of the few columns a
#: scan actually needs is a small buffer; the point is to never hold 652MB at once.
SCAN_CHUNK_ROWS = 500_000


def numbered_columns(prefix: str, count: int, *, width: int = 1) -> tuple[str, ...]:
    """Return ``count`` 1-indexed column names sharing a prefix.

    Args:
        prefix: Literal text before the number, e.g. ``"V"`` or ``"id_"``.
        count: How many columns, numbered 1..count inclusive.
        width: Zero-padded width of the number. IEEE-CIS uses width 1 for ``C``/``D``/
            ``M``/``V``/``card`` and width 2 for ``id_``.

    Returns:
        The column names in dataset order.
    """
    return tuple(f"{prefix}{index:0{width}d}" for index in range(1, count + 1))


# --- IEEE-CIS ----------------------------------------------------------------------

#: 394 columns. Order matters only for readability; validation compares as a set.
IEEE_CIS_TRANSACTION_COLUMNS: tuple[str, ...] = (
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
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
    *numbered_columns("V", 339),
)

#: 41 columns. Only ~24% of transactions have an identity row.
IEEE_CIS_IDENTITY_COLUMNS: tuple[str, ...] = (
    "TransactionID",
    *numbered_columns("id_", 38, width=2),
    "DeviceType",
    "DeviceInfo",
)

#: The five product codes. Not a fraud label — a transaction-type discriminator.
IEEE_CIS_PRODUCT_CODES: frozenset[str] = frozenset({"W", "C", "R", "H", "S"})

#: ``TransactionDT`` is an offset in seconds from an unstated reference, not a timestamp.
#: The reference is anchored in :mod:`app.data.schema`; only ordering and hour-of-day are
#: load-bearing, so a wrong anchor changes no downstream result.
IEEE_CIS_DT_SPAN_DAYS = 182

#: Fraction of transactions carrying an identity row: 144,233 / 590,540.
IEEE_CIS_IDENTITY_JOIN_RATE = 0.2442


# --- PaySim ------------------------------------------------------------------------

PAYSIM_COLUMNS: tuple[str, ...] = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)

PAYSIM_TRANSACTION_TYPES: frozenset[str] = frozenset(
    {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
)

#: Every fraudulent row is one of these two types. The Tier-3 scope filter rests entirely
#: on this claim, so :mod:`app.data.validate_raw` treats a violation as an error rather
#: than a warning — it would mean the graph layer is scoped to the wrong subset.
PAYSIM_FRAUD_BEARING_TYPES: frozenset[str] = frozenset({"TRANSFER", "CASH_OUT"})

#: ``step`` is an hour index over a 30-day simulation.
PAYSIM_MIN_STEP = 1
PAYSIM_MAX_STEP = 744

#: A naive simulator rule (single transfer over 200,000) that fires on 16 rows. It is a
#: leaked downstream decision, never a feature and never a label — the pipeline drops it.
PAYSIM_FLAGGED_FRAUD_ROWS = 16


@dataclass(frozen=True)
class RawFileSpec:
    """The expected shape of one raw dataset file.

    Attributes:
        key: Stable identifier used in reports and test ids.
        filenames: Accepted filenames, canonical first. Aliases exist because the PaySim
            download is commonly renamed on the way out of Kaggle.
        source_dataset: Which corpus this file belongs to.
        columns: Every column the file must contain, exactly — a missing column means the
            file is unusable, an extra one means it is not the file we think it is.
        key_column: Column that must be unique, or None if the file has no natural key.
        label_column: Binary fraud label, or None for the identity sidecar.
        profile_columns: Extra columns read during the scan to verify the structural
            claims the Phase 1 plan depends on.
        expected_rows: Exact row count of the published release.
        expected_positives: Exact count of ``label_column == 1``, or None when unlabelled.
        approximate_bytes: Rough on-disk size, used only to explain the download budget.
    """

    key: str
    filenames: tuple[str, ...]
    source_dataset: SourceDataset
    columns: tuple[str, ...]
    key_column: str | None
    label_column: str | None
    profile_columns: tuple[str, ...]
    expected_rows: int
    expected_positives: int | None
    approximate_bytes: int

    @property
    def canonical_filename(self) -> str:
        """Return the filename this file is expected to be downloaded as."""
        return self.filenames[0]

    @property
    def expected_base_rate(self) -> float | None:
        """Return the positive-class base rate, or None when the file is unlabelled.

        Every metrics report in this project states a base rate alongside its precision,
        per ``.claude/skills/ml-evaluation-standards/SKILL.md`` section 2.
        """
        if self.expected_positives is None:
            return None
        return self.expected_positives / self.expected_rows

    @property
    def scan_columns(self) -> tuple[str, ...]:
        """Return the columns a validation scan needs to read, de-duplicated."""
        wanted = (self.key_column, self.label_column, *self.profile_columns)
        seen: dict[str, None] = {}
        for column in wanted:
            if column is not None:
                seen[column] = None
        return tuple(seen)


IEEE_CIS_TRANSACTION_SPEC = RawFileSpec(
    key="ieee_cis_transaction",
    filenames=("train_transaction.csv",),
    source_dataset="ieee_cis",
    columns=IEEE_CIS_TRANSACTION_COLUMNS,
    key_column="TransactionID",
    label_column="isFraud",
    profile_columns=("TransactionDT", "ProductCD"),
    expected_rows=590_540,
    expected_positives=20_663,
    approximate_bytes=652_000_000,
)

IEEE_CIS_IDENTITY_SPEC = RawFileSpec(
    key="ieee_cis_identity",
    filenames=("train_identity.csv",),
    source_dataset="ieee_cis",
    columns=IEEE_CIS_IDENTITY_COLUMNS,
    key_column="TransactionID",
    label_column=None,
    profile_columns=("DeviceType",),
    expected_rows=144_233,
    expected_positives=None,
    approximate_bytes=26_000_000,
)

PAYSIM_SPEC = RawFileSpec(
    key="paysim",
    filenames=(
        "PS_20174392719_1491204439457_log.csv",
        "paysim.csv",
        "Synthetic_Financial_datasets_log.csv",
    ),
    source_dataset="paysim",
    columns=PAYSIM_COLUMNS,
    key_column=None,
    label_column="isFraud",
    profile_columns=("type", "step", "isFlaggedFraud"),
    expected_rows=6_362_620,
    expected_positives=8_213,
    approximate_bytes=470_000_000,
)

#: Every file Phase 1 requires. Validation runs over all of them.
RAW_FILE_SPECS: tuple[RawFileSpec, ...] = (
    IEEE_CIS_TRANSACTION_SPEC,
    IEEE_CIS_IDENTITY_SPEC,
    PAYSIM_SPEC,
)


def normalise_column(name: str) -> str:
    """Return a column name in the underscore form the pipeline uses internally.

    The IEEE-CIS release is inconsistent: ``train_identity.csv`` names its columns
    ``id_01``..``id_38`` while ``test_identity.csv`` uses ``id-01``..``id-38``. We only
    load the train file, but normalising on read means a copy sourced from a mirror that
    applied the test-file convention still joins correctly instead of silently producing
    an all-null identity block.
    """
    return name.strip().replace("-", "_")
