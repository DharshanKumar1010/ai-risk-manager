"""Tests for raw-dataset validation.

These exercise the validator's mechanism against tiny synthetic files, never the real
datasets: the corpora are gitignored, absent in CI, and far too large to fixture. A
purpose-built spec with small expected counts stands in for the published releases.
"""

import csv
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.data.raw_spec import RawFileSpec
from app.data.validate_raw import (
    Finding,
    check_columns,
    render_report,
    validate_file,
    validate_raw_data,
)

TINY_COLUMNS = ("row_id", "label", "kind", "value")

TINY_SPEC = RawFileSpec(
    key="tiny",
    filenames=("tiny.csv", "tiny_alias.csv"),
    source_dataset="paysim",
    columns=TINY_COLUMNS,
    key_column="row_id",
    label_column="label",
    profile_columns=("kind", "value"),
    expected_rows=4,
    expected_positives=1,
    approximate_bytes=1_000,
)

GOOD_ROWS: tuple[tuple[object, ...], ...] = (
    (1, 0, "PAYMENT", 10),
    (2, 0, "TRANSFER", 20),
    (3, 1, "TRANSFER", 30),
    (4, 0, "CASH_OUT", 40),
)


def write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> Path:
    """Write a small CSV and return its path."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Return an empty raw-data directory."""
    directory = tmp_path / "raw"
    directory.mkdir()
    return directory


def codes(findings: Sequence[Finding]) -> set[str]:
    """Return the finding codes, for order-independent assertions."""
    return {finding.code for finding in findings}


class TestMissingAndUnreadableFiles:
    def test_reports_a_missing_file_with_the_expected_name(self, raw_dir: Path) -> None:
        report = validate_file(TINY_SPEC, raw_dir)
        assert codes(report.errors) == {"file_missing"}
        assert "tiny.csv" in report.errors[0].message

    def test_missing_file_message_lists_accepted_aliases(self, raw_dir: Path) -> None:
        report = validate_file(TINY_SPEC, raw_dir)
        assert "tiny_alias.csv" in report.errors[0].message

    def test_accepts_an_alias_filename(self, raw_dir: Path) -> None:
        write_csv(raw_dir / "tiny_alias.csv", TINY_COLUMNS, GOOD_ROWS)
        report = validate_file(TINY_SPEC, raw_dir)
        assert report.errors == ()

    def test_reports_an_empty_file(self, raw_dir: Path) -> None:
        (raw_dir / "tiny.csv").touch()
        report = validate_file(TINY_SPEC, raw_dir)
        assert codes(report.errors) == {"file_empty"}


class TestColumnChecks:
    def test_a_correct_header_produces_no_findings(self) -> None:
        assert check_columns(TINY_SPEC, TINY_COLUMNS) == []

    def test_missing_column_is_an_error(self) -> None:
        findings = check_columns(TINY_SPEC, ("row_id", "label", "kind"))
        assert codes(findings) == {"missing_columns"}
        assert "value" in findings[0].message

    def test_unexpected_column_is_an_error(self) -> None:
        """An extra column means this is not the file the spec describes."""
        findings = check_columns(TINY_SPEC, (*TINY_COLUMNS, "surprise"))
        assert codes(findings) == {"unexpected_columns"}
        assert "surprise" in findings[0].message

    def test_duplicate_columns_are_an_error(self) -> None:
        findings = check_columns(TINY_SPEC, (*TINY_COLUMNS, "value"))
        assert "duplicate_columns" in codes(findings)

    def test_header_is_normalised_before_comparison(self, raw_dir: Path) -> None:
        """A mirror applying the test-file hyphen convention still validates."""
        hyphenated = ("row-id", "label", "kind", "value")
        write_csv(raw_dir / "tiny.csv", hyphenated, GOOD_ROWS)
        report = validate_file(TINY_SPEC, raw_dir)
        assert "missing_columns" not in codes(report.errors)


class TestCountChecks:
    def test_a_matching_file_passes_cleanly(self, raw_dir: Path) -> None:
        write_csv(raw_dir / "tiny.csv", TINY_COLUMNS, GOOD_ROWS)
        report = validate_file(TINY_SPEC, raw_dir)
        assert report.findings == ()
        assert report.scan is not None
        assert report.scan.rows == 4
        assert report.scan.positives == 1

    def test_row_count_mismatch_is_an_error_naming_both_numbers(self, raw_dir: Path) -> None:
        write_csv(raw_dir / "tiny.csv", TINY_COLUMNS, GOOD_ROWS[:3])
        report = validate_file(TINY_SPEC, raw_dir)
        assert "row_count_mismatch" in codes(report.errors)
        message = next(f.message for f in report.errors if f.code == "row_count_mismatch")
        assert "3" in message and "4" in message

    def test_positive_count_mismatch_is_an_error(self, raw_dir: Path) -> None:
        """Base rates are quoted in every metrics report, so this must be exact."""
        rows = [(1, 1, "TRANSFER", 10), (2, 1, "TRANSFER", 20), *GOOD_ROWS[2:]]
        write_csv(raw_dir / "tiny.csv", TINY_COLUMNS, rows)
        report = validate_file(TINY_SPEC, raw_dir)
        assert "positive_count_mismatch" in codes(report.errors)

    def test_duplicate_keys_are_an_error(self, raw_dir: Path) -> None:
        rows = [(1, 0, "PAYMENT", 10), (1, 0, "TRANSFER", 20), *GOOD_ROWS[2:]]
        write_csv(raw_dir / "tiny.csv", TINY_COLUMNS, rows)
        report = validate_file(TINY_SPEC, raw_dir)
        assert "duplicate_keys" in codes(report.errors)

    def test_scan_spans_multiple_chunks(
        self, raw_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chunked accumulation must total the same as a single-pass read."""
        monkeypatch.setattr("app.data.validate_raw.SCAN_CHUNK_ROWS", 2)
        write_csv(raw_dir / "tiny.csv", TINY_COLUMNS, GOOD_ROWS)
        report = validate_file(TINY_SPEC, raw_dir)
        assert report.scan is not None
        assert report.scan.rows == 4
        assert report.scan.positives == 1
        assert report.scan.duplicate_keys == 0


class TestHeadersOnlyMode:
    def test_skips_the_scan(self, raw_dir: Path) -> None:
        """A truncated file passes headers-only, because counts are not checked."""
        write_csv(raw_dir / "tiny.csv", TINY_COLUMNS, GOOD_ROWS[:1])
        report = validate_file(TINY_SPEC, raw_dir, headers_only=True)
        assert report.scan is None
        assert report.errors == ()

    def test_still_catches_a_wrong_header(self, raw_dir: Path) -> None:
        write_csv(raw_dir / "tiny.csv", ("row_id", "label"), [(1, 0)])
        report = validate_file(TINY_SPEC, raw_dir, headers_only=True)
        assert "missing_columns" in codes(report.errors)


class TestReport:
    def test_report_is_not_ok_when_a_file_is_missing(self, raw_dir: Path) -> None:
        report = validate_raw_data(raw_dir, specs=(TINY_SPEC,))
        assert not report.ok()

    def test_report_is_ok_when_everything_matches(self, raw_dir: Path) -> None:
        write_csv(raw_dir / "tiny.csv", TINY_COLUMNS, GOOD_ROWS)
        report = validate_raw_data(raw_dir, specs=(TINY_SPEC,))
        assert report.ok()
        assert report.ok(strict=True)

    def test_strict_mode_promotes_warnings_to_failure(self, raw_dir: Path) -> None:
        write_csv(raw_dir / "tiny.csv", TINY_COLUMNS, GOOD_ROWS)
        report = validate_raw_data(raw_dir, specs=(TINY_SPEC,))
        warned = type(report)(
            files=report.files,
            cross_file=(Finding(level="warning", code="drift", message="drifted"),),
        )
        assert warned.ok()
        assert not warned.ok(strict=True)

    def test_render_names_the_failing_file(self, raw_dir: Path) -> None:
        report = validate_raw_data(raw_dir, specs=(TINY_SPEC,))
        rendered = render_report(report)
        assert "FAIL" in rendered
        assert "tiny" in rendered
        assert "NOT ready" in rendered

    def test_render_reports_observed_profile_on_success(self, raw_dir: Path) -> None:
        write_csv(raw_dir / "tiny.csv", TINY_COLUMNS, GOOD_ROWS)
        rendered = render_report(validate_raw_data(raw_dir, specs=(TINY_SPEC,)))
        assert "OK" in rendered
        assert "rows: 4" in rendered
        assert "base rate" in rendered
