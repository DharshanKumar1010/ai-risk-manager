"""Verify a raw dataset download before any pipeline stage reads it.

This runs *before* Phase 1's pipeline and exists to convert every dataset figure in the
Phase 1 plan from an expectation into a verified fact. It reads the files, it never writes
them, and it never modifies anything on disk.

Two levels of check:

**Structural** (errors — the file is unusable as-is)
    file present and non-empty; the column set matches :mod:`app.data.raw_spec` exactly;
    the row count and positive-label count match the published release; the key column is
    unique; PaySim fraud appears only in ``TRANSFER``/``CASH_OUT``.

**Profile** (warnings — the file is usable but a documented claim has drifted)
    the IEEE-CIS time span, the identity join rate, the PaySim ``step`` range and
    ``isFlaggedFraud`` count, and the categorical levels of the type columns.

A structural failure is not something to work around by loosening the constant. Every
downstream metric names the corpus it was measured on, so a corpus that does not match its
specification invalidates that naming. Update :mod:`app.data.raw_spec` and record the
change in ``BUILD_LOG.md`` instead.

Usage::

    python -m app.data.validate_raw
    python -m app.data.validate_raw --data-dir /data --strict
    python -m app.data.validate_raw --headers-only     # skip the full scan
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from app.config import get_settings
from app.data.raw_spec import (
    IEEE_CIS_DT_SPAN_DAYS,
    IEEE_CIS_IDENTITY_JOIN_RATE,
    IEEE_CIS_IDENTITY_SPEC,
    IEEE_CIS_PRODUCT_CODES,
    IEEE_CIS_TRANSACTION_SPEC,
    PAYSIM_FLAGGED_FRAUD_ROWS,
    PAYSIM_FRAUD_BEARING_TYPES,
    PAYSIM_MAX_STEP,
    PAYSIM_MIN_STEP,
    PAYSIM_SPEC,
    PAYSIM_TRANSACTION_TYPES,
    RAW_FILE_SPECS,
    SCAN_CHUNK_ROWS,
    RawFileSpec,
    normalise_column,
)

Severity = Literal["error", "warning"]

#: Above this many distinct values a profile column is reported as high-cardinality rather
#: than enumerated. Guards against accumulating a dictionary of 590k device strings.
MAX_PROFILE_CARDINALITY = 50

#: Tolerance on derived rates (identity join rate, time span) before a warning fires.
RATE_TOLERANCE = 0.01

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class Finding:
    """One problem found in a raw file.

    Attributes:
        level: ``"error"`` if the file is unusable as specified, ``"warning"`` if a
            documented claim has drifted but the file can still be processed.
        code: Stable machine-readable identifier, e.g. ``"row_count_mismatch"``.
        message: Human-readable detail naming observed and expected values.
    """

    level: Severity
    code: str
    message: str


@dataclass(frozen=True)
class ScanResult:
    """Statistics accumulated from one streaming pass over a file."""

    rows: int
    positives: int | None
    duplicate_keys: int
    numeric_ranges: Mapping[str, tuple[float, float]]
    numeric_sums: Mapping[str, float]
    category_levels: Mapping[str, Mapping[str, int]]
    positives_by_category: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True)
class FileReport:
    """The outcome of validating one raw file."""

    spec: RawFileSpec
    path: Path | None
    scan: ScanResult | None
    findings: tuple[Finding, ...]
    profile: Mapping[str, str] = field(default_factory=dict)

    @property
    def errors(self) -> tuple[Finding, ...]:
        """Return only the blocking findings."""
        return tuple(f for f in self.findings if f.level == "error")

    @property
    def warnings(self) -> tuple[Finding, ...]:
        """Return only the non-blocking findings."""
        return tuple(f for f in self.findings if f.level == "warning")


@dataclass(frozen=True)
class ValidationReport:
    """The outcome of validating every raw file, plus cross-file checks."""

    files: tuple[FileReport, ...]
    cross_file: tuple[Finding, ...]

    @property
    def errors(self) -> tuple[Finding, ...]:
        """Return every blocking finding across all files."""
        return tuple(f for report in self.files for f in report.errors) + tuple(
            f for f in self.cross_file if f.level == "error"
        )

    @property
    def warnings(self) -> tuple[Finding, ...]:
        """Return every non-blocking finding across all files."""
        return tuple(f for report in self.files for f in report.warnings) + tuple(
            f for f in self.cross_file if f.level == "warning"
        )

    def ok(self, *, strict: bool = False) -> bool:
        """Return whether the datasets are fit to build on.

        Args:
            strict: Treat warnings as blocking too. CI uses this; interactive runs
                generally do not, so that a drifted-but-usable corpus still reports its
                real structural state rather than stopping at the first warning.
        """
        if self.errors:
            return False
        return not (strict and self.warnings)


def resolve_path(spec: RawFileSpec, raw_dir: Path) -> Path | None:
    """Return the first accepted filename for ``spec`` that exists in ``raw_dir``.

    Args:
        spec: The file specification, whose ``filenames`` are tried in order.
        raw_dir: Directory holding untouched downloads.

    Returns:
        The resolved path, or None if no accepted filename is present.
    """
    for filename in spec.filenames:
        candidate = raw_dir / filename
        if candidate.is_file():
            return candidate
    return None


def read_header_mapping(path: Path) -> dict[str, str]:
    """Return a mapping of each raw CSV column name to its normalised form.

    Both halves are needed. Comparison against the spec happens on normalised names, but
    ``usecols`` has to name the columns exactly as the file spells them — reading a
    hyphenated ``id-01`` file with a normalised ``id_01`` in ``usecols`` raises rather than
    selecting the column.

    Args:
        path: The CSV to inspect.

    Returns:
        Raw column name to normalised column name, in file order.

    Raises:
        ValueError: If the file has no header row.
    """
    header = pd.read_csv(path, nrows=0)
    mapping = {str(name): normalise_column(str(name)) for name in header.columns}
    if not mapping:
        raise ValueError(f"{path} has no header row")
    return mapping


def read_header(path: Path) -> tuple[str, ...]:
    """Return the normalised column names of a CSV without reading its body."""
    return tuple(read_header_mapping(path).values())


def check_columns(spec: RawFileSpec, observed: Sequence[str]) -> list[Finding]:
    """Compare an observed header against the specification.

    Both directions matter. A missing column means the file cannot produce the features
    the pipeline promises; an unexpected column means this is not the file we think it is,
    which is the more dangerous of the two because it fails silently later.
    """
    findings: list[Finding] = []
    expected_set = frozenset(spec.columns)
    observed_set = frozenset(observed)

    missing = sorted(expected_set - observed_set)
    if missing:
        findings.append(
            Finding(
                level="error",
                code="missing_columns",
                message=(
                    f"{len(missing)} expected column(s) absent: {_summarise(missing)}. "
                    f"Expected the {len(spec.columns)}-column {spec.canonical_filename}."
                ),
            )
        )

    unexpected = sorted(observed_set - expected_set)
    if unexpected:
        findings.append(
            Finding(
                level="error",
                code="unexpected_columns",
                message=(
                    f"{len(unexpected)} unexpected column(s): {_summarise(unexpected)}. "
                    "This is probably a different file or release than the spec describes."
                ),
            )
        )

    if len(observed) != len(observed_set):
        findings.append(
            Finding(
                level="error",
                code="duplicate_columns",
                message="Header contains duplicate column names after normalisation.",
            )
        )

    return findings


def scan_file(path: Path, spec: RawFileSpec, header: Mapping[str, str]) -> ScanResult:
    """Stream a CSV once, accumulating the statistics validation needs.

    Only the handful of columns named by ``spec.scan_columns`` are read, in chunks, so
    peak memory stays a few megabytes regardless of the 652MB file on disk.

    Args:
        path: The CSV to scan.
        spec: Specification naming the key, label and profile columns.
        header: Raw-to-normalised column mapping from :func:`read_header_mapping`. Columns
            absent from it are skipped, so a partially-wrong file is still scanned for
            whatever it does have.
    """
    to_raw = {normalised: raw for raw, normalised in header.items()}
    columns = [name for name in spec.scan_columns if name in to_raw]
    usecols = [to_raw[name] for name in columns]
    label = spec.label_column if spec.label_column in columns else None
    key = spec.key_column if spec.key_column in columns else None

    rows = 0
    positives = 0 if label is not None else None
    seen_keys: set[object] = set()
    duplicate_keys = 0
    numeric_ranges: dict[str, tuple[float, float]] = {}
    numeric_sums: dict[str, float] = {}
    category_levels: dict[str, dict[str, int]] = {}
    positives_by_category: dict[str, dict[str, int]] = {}

    with pd.read_csv(path, usecols=usecols, chunksize=SCAN_CHUNK_ROWS) as reader:
        for chunk in reader:
            chunk = chunk.rename(columns=lambda name: normalise_column(str(name)))
            rows += len(chunk)

            is_positive = None
            if label is not None:
                is_positive = pd.to_numeric(chunk[label], errors="coerce") == 1
                positives = (positives or 0) + int(is_positive.sum())

            if key is not None:
                before = len(seen_keys)
                seen_keys.update(chunk[key].tolist())
                duplicate_keys += len(chunk) - (len(seen_keys) - before)

            for name in columns:
                if name == key:
                    continue
                series = chunk[name]
                if pd.api.types.is_numeric_dtype(series):
                    _accumulate_numeric(name, series, numeric_ranges, numeric_sums)
                else:
                    _accumulate_category(
                        name, series, is_positive, category_levels, positives_by_category
                    )

    return ScanResult(
        rows=rows,
        positives=positives,
        duplicate_keys=duplicate_keys,
        numeric_ranges=numeric_ranges,
        numeric_sums=numeric_sums,
        category_levels=category_levels,
        positives_by_category=positives_by_category,
    )


def _accumulate_numeric(
    name: str,
    series: "pd.Series[Any]",
    ranges: dict[str, tuple[float, float]],
    sums: dict[str, float],
) -> None:
    """Fold one chunk's numeric column into the running min/max and sum."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return
    low, high = float(values.min()), float(values.max())
    if name in ranges:
        previous_low, previous_high = ranges[name]
        low, high = min(low, previous_low), max(high, previous_high)
    ranges[name] = (low, high)
    sums[name] = sums.get(name, 0.0) + float(values.sum())


def _accumulate_category(
    name: str,
    series: "pd.Series[Any]",
    is_positive: "pd.Series[bool] | None",
    levels: dict[str, dict[str, int]],
    positives_by_category: dict[str, dict[str, int]],
) -> None:
    """Fold one chunk's categorical column into the running level counts.

    High-cardinality columns are abandoned rather than enumerated — the marker value is
    read back by :func:`_describe_categories`.
    """
    bucket = levels.setdefault(name, {})
    if bucket.get(_HIGH_CARDINALITY) is not None:
        return

    for value, count in series.astype("string").value_counts(dropna=False).items():
        bucket[str(value)] = bucket.get(str(value), 0) + int(count)

    if len(bucket) > MAX_PROFILE_CARDINALITY:
        levels[name] = {_HIGH_CARDINALITY: len(bucket)}
        return

    if is_positive is not None:
        positive_bucket = positives_by_category.setdefault(name, {})
        for value, count in series[is_positive].astype("string").value_counts().items():
            positive_bucket[str(value)] = positive_bucket.get(str(value), 0) + int(count)


_HIGH_CARDINALITY = "__high_cardinality__"


def check_counts(spec: RawFileSpec, scan: ScanResult) -> list[Finding]:
    """Compare observed row, label and key counts against the published release."""
    findings: list[Finding] = []

    if scan.rows != spec.expected_rows:
        findings.append(
            Finding(
                level="error",
                code="row_count_mismatch",
                message=(
                    f"{scan.rows:,} rows, expected {spec.expected_rows:,} "
                    f"(difference {scan.rows - spec.expected_rows:+,}). If this copy is a "
                    "legitimate re-release, update app/data/raw_spec.py and BUILD_LOG.md."
                ),
            )
        )

    if spec.expected_positives is not None and scan.positives is not None:
        if scan.positives != spec.expected_positives:
            observed_rate = scan.positives / scan.rows if scan.rows else 0.0
            findings.append(
                Finding(
                    level="error",
                    code="positive_count_mismatch",
                    message=(
                        f"{scan.positives:,} positives ({observed_rate:.4%}), expected "
                        f"{spec.expected_positives:,} "
                        f"({spec.expected_base_rate or 0.0:.4%}). Every metrics report in "
                        "this project states its base rate, so this must be exact."
                    ),
                )
            )

    if scan.duplicate_keys:
        findings.append(
            Finding(
                level="error",
                code="duplicate_keys",
                message=(
                    f"{scan.duplicate_keys:,} duplicate values in key column "
                    f"{spec.key_column!r}. The identity join and the account-level split "
                    "both assume this column is unique."
                ),
            )
        )

    return findings


def check_profile(spec: RawFileSpec, scan: ScanResult) -> list[Finding]:
    """Verify the per-dataset structural claims the Phase 1 plan is built on."""
    if spec.key == IEEE_CIS_TRANSACTION_SPEC.key:
        return _check_ieee_transaction_profile(scan)
    if spec.key == PAYSIM_SPEC.key:
        return _check_paysim_profile(scan)
    return []


def _check_ieee_transaction_profile(scan: ScanResult) -> list[Finding]:
    """Check the IEEE-CIS time span and product codes."""
    findings: list[Finding] = []

    span = scan.numeric_ranges.get("TransactionDT")
    if span is not None:
        observed_days = (span[1] - span[0]) / SECONDS_PER_DAY
        if abs(observed_days - IEEE_CIS_DT_SPAN_DAYS) > 1.0:
            findings.append(
                Finding(
                    level="warning",
                    code="dt_span_drift",
                    message=(
                        f"TransactionDT spans {observed_days:.1f} days, expected about "
                        f"{IEEE_CIS_DT_SPAN_DAYS}. The chronological split boundaries "
                        "shift with this."
                    ),
                )
            )

    findings.extend(
        _check_category_levels("ProductCD", scan, IEEE_CIS_PRODUCT_CODES, level="warning")
    )
    return findings


def _check_paysim_profile(scan: ScanResult) -> list[Finding]:
    """Check the PaySim step range, flagged-fraud count, and fraud-by-type claim."""
    findings: list[Finding] = []

    findings.extend(_check_category_levels("type", scan, PAYSIM_TRANSACTION_TYPES, level="warning"))

    # The load-bearing claim: the Tier-3 scope filter keeps only TRANSFER and CASH_OUT on
    # the grounds that no other type carries fraud. If that is false the graph layer would
    # be scoped to a subset that silently drops positives, so this is an error.
    fraud_by_type = scan.positives_by_category.get("type", {})
    unexpected = {
        name: count
        for name, count in fraud_by_type.items()
        if count > 0 and name not in PAYSIM_FRAUD_BEARING_TYPES
    }
    if unexpected:
        findings.append(
            Finding(
                level="error",
                code="fraud_outside_expected_types",
                message=(
                    f"Fraud found in transaction types outside "
                    f"{sorted(PAYSIM_FRAUD_BEARING_TYPES)}: {unexpected}. The Tier-3 scope "
                    "filter would drop these positives — re-scope it before building."
                ),
            )
        )

    step = scan.numeric_ranges.get("step")
    if step is not None and (step[0] < PAYSIM_MIN_STEP or step[1] > PAYSIM_MAX_STEP):
        findings.append(
            Finding(
                level="warning",
                code="step_range_drift",
                message=(
                    f"step ranges {step[0]:.0f}..{step[1]:.0f}, expected "
                    f"{PAYSIM_MIN_STEP}..{PAYSIM_MAX_STEP}."
                ),
            )
        )

    flagged = scan.numeric_sums.get("isFlaggedFraud")
    if flagged is not None and int(flagged) != PAYSIM_FLAGGED_FRAUD_ROWS:
        findings.append(
            Finding(
                level="warning",
                code="flagged_fraud_drift",
                message=(
                    f"isFlaggedFraud sums to {int(flagged)}, expected "
                    f"{PAYSIM_FLAGGED_FRAUD_ROWS}. The column is dropped by the pipeline "
                    "either way — it is a leaked downstream decision, not a feature."
                ),
            )
        )

    return findings


def _check_category_levels(
    column: str,
    scan: ScanResult,
    expected: frozenset[str],
    *,
    level: Severity,
) -> list[Finding]:
    """Report categorical values outside the documented set."""
    observed = scan.category_levels.get(column)
    if not observed or _HIGH_CARDINALITY in observed:
        return []
    unexpected = sorted(set(observed) - expected - {"<NA>", "nan"})
    if not unexpected:
        return []
    return [
        Finding(
            level=level,
            code="unexpected_category_levels",
            message=f"{column} contains undocumented values: {_summarise(unexpected)}.",
        )
    ]


def validate_file(spec: RawFileSpec, raw_dir: Path, *, headers_only: bool = False) -> FileReport:
    """Validate one raw file against its specification.

    Args:
        spec: What the file is expected to contain.
        raw_dir: Directory holding untouched downloads.
        headers_only: Skip the streaming scan and check only presence and columns. Useful
            as a fast pre-flight; it cannot verify row counts or base rates.
    """
    path = resolve_path(spec, raw_dir)
    if path is None:
        return FileReport(
            spec=spec,
            path=None,
            scan=None,
            findings=(
                Finding(
                    level="error",
                    code="file_missing",
                    message=(
                        f"Not found in {raw_dir}. Expected {spec.canonical_filename} "
                        f"(~{spec.approximate_bytes / 1e6:.0f}MB)"
                        + (
                            f", or one of {list(spec.filenames[1:])}."
                            if len(spec.filenames) > 1
                            else "."
                        )
                    ),
                ),
            ),
        )

    if path.stat().st_size == 0:
        return FileReport(
            spec=spec,
            path=path,
            scan=None,
            findings=(Finding(level="error", code="file_empty", message=f"{path} is empty."),),
        )

    try:
        header = read_header_mapping(path)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return FileReport(
            spec=spec,
            path=path,
            scan=None,
            findings=(
                Finding(
                    level="error",
                    code="header_unreadable",
                    message=f"Could not read a CSV header from {path}: {exc}",
                ),
            ),
        )

    findings = check_columns(spec, tuple(header.values()))
    if headers_only:
        return FileReport(spec=spec, path=path, scan=None, findings=tuple(findings))

    scan = scan_file(path, spec, header)
    findings.extend(check_counts(spec, scan))
    findings.extend(check_profile(spec, scan))

    return FileReport(
        spec=spec,
        path=path,
        scan=scan,
        findings=tuple(findings),
        profile=_describe(spec, scan),
    )


def check_identity_join(raw_dir: Path) -> list[Finding]:
    """Verify that the IEEE-CIS identity sidecar joins onto the transaction file.

    The Phase 1 plan states that only ~24% of transactions carry an identity row, and the
    whole ``has_identity`` feature and its missing-value strategy rest on that. A join rate
    near zero would mean the two files come from different releases — which would otherwise
    surface as an all-null identity block rather than as an error.
    """
    transaction_path = resolve_path(IEEE_CIS_TRANSACTION_SPEC, raw_dir)
    identity_path = resolve_path(IEEE_CIS_IDENTITY_SPEC, raw_dir)
    if transaction_path is None or identity_path is None:
        return []

    key = "TransactionID"
    try:
        transaction_ids = pd.read_csv(transaction_path, usecols=[key])[key]
        identity_ids = pd.read_csv(identity_path, usecols=[key])[key]
    except (ValueError, pd.errors.ParserError) as exc:
        return [
            Finding(
                level="error",
                code="identity_join_unreadable",
                message=f"Could not read {key} from both IEEE-CIS files: {exc}",
            )
        ]

    if len(transaction_ids) == 0:
        return []

    matched = int(identity_ids.isin(set(transaction_ids.tolist())).sum())
    orphaned = len(identity_ids) - matched
    join_rate = matched / len(transaction_ids)

    findings: list[Finding] = []
    if orphaned:
        findings.append(
            Finding(
                level="error",
                code="orphaned_identity_rows",
                message=(
                    f"{orphaned:,} identity rows have no matching transaction. The two "
                    "files are probably from different releases."
                ),
            )
        )
    if abs(join_rate - IEEE_CIS_IDENTITY_JOIN_RATE) > RATE_TOLERANCE:
        findings.append(
            Finding(
                level="warning",
                code="identity_join_rate_drift",
                message=(
                    f"Identity join rate {join_rate:.2%}, expected about "
                    f"{IEEE_CIS_IDENTITY_JOIN_RATE:.2%}. The has_identity feature and the "
                    "identity-block missing-value strategy are calibrated to this."
                ),
            )
        )
    return findings


def validate_raw_data(
    raw_dir: Path,
    specs: Sequence[RawFileSpec] = RAW_FILE_SPECS,
    *,
    headers_only: bool = False,
) -> ValidationReport:
    """Validate every required raw file, plus the cross-file identity join.

    Args:
        raw_dir: Directory holding untouched downloads.
        specs: Which files to validate. Defaults to everything Phase 1 requires.
        headers_only: Skip streaming scans; presence and columns only.

    Returns:
        A report whose ``ok()`` answers whether the datasets are fit to build on.
    """
    files = tuple(validate_file(spec, raw_dir, headers_only=headers_only) for spec in specs)
    cross_file: tuple[Finding, ...] = ()
    if not headers_only and all(report.path is not None for report in files):
        cross_file = tuple(check_identity_join(raw_dir))
    return ValidationReport(files=files, cross_file=cross_file)


def _describe(spec: RawFileSpec, scan: ScanResult) -> dict[str, str]:
    """Render the observed facts worth printing even when everything passes."""
    profile: dict[str, str] = {"rows": f"{scan.rows:,}"}

    if scan.positives is not None:
        rate = scan.positives / scan.rows if scan.rows else 0.0
        profile["positives"] = f"{scan.positives:,} ({rate:.4%} base rate)"

    if spec.key == IEEE_CIS_TRANSACTION_SPEC.key:
        span = scan.numeric_ranges.get("TransactionDT")
        if span is not None:
            profile["time span"] = f"{(span[1] - span[0]) / SECONDS_PER_DAY:.1f} days"

    if spec.key == PAYSIM_SPEC.key:
        step = scan.numeric_ranges.get("step")
        if step is not None:
            profile["step range"] = f"{step[0]:.0f}..{step[1]:.0f}"
        fraud_by_type = scan.positives_by_category.get("type")
        if fraud_by_type:
            profile["fraud by type"] = ", ".join(
                f"{name}={count:,}" for name, count in sorted(fraud_by_type.items())
            )

    for column, levels in scan.category_levels.items():
        profile[f"{column} levels"] = _describe_categories(levels)

    return profile


def _describe_categories(levels: Mapping[str, int]) -> str:
    """Render a categorical column's levels, or its cardinality if it has too many."""
    if _HIGH_CARDINALITY in levels:
        return f"{levels[_HIGH_CARDINALITY]}+ distinct values (not enumerated)"
    return ", ".join(f"{name}={count:,}" for name, count in sorted(levels.items()))


def _summarise(values: Sequence[str], limit: int = 8) -> str:
    """Render a possibly-long list of column names without flooding the terminal."""
    shown = list(values[:limit])
    if len(values) > limit:
        shown.append(f"... and {len(values) - limit} more")
    return ", ".join(shown)


def render_report(report: ValidationReport, *, strict: bool = False) -> str:
    """Render a validation report as plain text for the terminal."""
    lines: list[str] = ["RiskIQ raw dataset validation", "=" * 60, ""]

    for file_report in report.files:
        spec = file_report.spec
        status = "FAIL" if file_report.errors else ("WARN" if file_report.warnings else "OK")
        location = str(file_report.path) if file_report.path else spec.canonical_filename
        lines.append(f"[{status:4}] {spec.key}  ({spec.source_dataset})")
        lines.append(f"        {location}")

        for name, value in file_report.profile.items():
            lines.append(f"        {name}: {value}")
        for finding in file_report.findings:
            lines.append(f"        {finding.level.upper()} [{finding.code}] {finding.message}")
        lines.append("")

    if report.cross_file:
        lines.append("Cross-file checks")
        for finding in report.cross_file:
            lines.append(f"        {finding.level.upper()} [{finding.code}] {finding.message}")
        lines.append("")

    errors, warnings = len(report.errors), len(report.warnings)
    lines.append("-" * 60)
    lines.append(f"{errors} error(s), {warnings} warning(s)")
    if report.ok(strict=strict):
        lines.append("Datasets validated. Phase 1's pipeline can read these.")
    else:
        lines.append("Datasets are NOT ready. Resolve the errors above before building.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the raw datasets and print a report.

    Returns:
        0 when the datasets are fit to build on, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.data.validate_raw",
        description="Verify the IEEE-CIS and PaySim downloads before Phase 1 reads them.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Dataset root. Defaults to the configured DATA_DIR.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures. Used by CI.",
    )
    parser.add_argument(
        "--headers-only",
        action="store_true",
        help="Check presence and columns only; skip the full scan.",
    )
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir if args.data_dir is not None else get_settings().data_dir
    raw_dir = data_dir / "raw"

    if not raw_dir.is_dir():
        print(f"Raw data directory does not exist: {raw_dir}", file=sys.stderr)
        return 1

    report = validate_raw_data(raw_dir, headers_only=args.headers_only)
    print(render_report(report, strict=args.strict))
    return 0 if report.ok(strict=args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
