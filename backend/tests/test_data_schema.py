"""Tests for the canonical transaction contract and the raw-file specifications."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from app.data.raw_spec import (
    IEEE_CIS_IDENTITY_COLUMNS,
    IEEE_CIS_IDENTITY_SPEC,
    IEEE_CIS_TRANSACTION_COLUMNS,
    IEEE_CIS_TRANSACTION_SPEC,
    PAYSIM_COLUMNS,
    PAYSIM_FRAUD_BEARING_TYPES,
    PAYSIM_SPEC,
    PAYSIM_TRANSACTION_TYPES,
    RAW_FILE_SPECS,
    SourceDataset,
    normalise_column,
)
from app.data.schema import (
    IEEE_CIS_EPOCH,
    MAX_TRANSACTION_AMOUNT,
    PAYSIM_EPOCH,
    SPLIT_FRACTIONS,
    SPLIT_ORDER,
    LabelledTransaction,
    TransactionFeatures,
    ieee_cis_event_time,
    paysim_event_time,
)


def _features(**overrides: Any) -> dict[str, Any]:
    """Return a valid TransactionFeatures payload with optional overrides."""
    payload: dict[str, Any] = {
        "transaction_id": "2987000",
        "source_dataset": "ieee_cis",
        "event_time": datetime(2017, 12, 2, 12, 0, tzinfo=UTC),
        "amount": Decimal("68.50"),
        "account_id": "13926_299_1234",
        "transaction_type": "W",
        "feature_version": "fv_abc123",
        "features": {"amount_zscore": 1.5, "has_identity": False},
    }
    payload.update(overrides)
    return payload


class TestTransactionFeatures:
    """The scoring contract every adapter targets and every model reads."""

    def test_accepts_a_well_formed_transaction(self) -> None:
        transaction = TransactionFeatures(**_features())
        assert transaction.source_dataset == "ieee_cis"
        assert transaction.counterparty_id is None

    def test_rejects_naive_event_time(self) -> None:
        """A naive datetime survives until it meets an aware one deep inside a window."""
        with pytest.raises(ValidationError, match="timezone-aware"):
            TransactionFeatures(**_features(event_time=datetime(2017, 12, 2, 12, 0)))

    def test_rejects_unknown_fields(self) -> None:
        """extra='forbid' — security-checklist section 4."""
        with pytest.raises(ValidationError):
            TransactionFeatures(**_features(is_fraud=True))

    def test_rejects_negative_amount(self) -> None:
        with pytest.raises(ValidationError):
            TransactionFeatures(**_features(amount=Decimal("-0.01")))

    def test_rejects_amount_above_bound(self) -> None:
        """An unbounded amount is a model-poisoning vector, not just a validation hole."""
        with pytest.raises(ValidationError):
            TransactionFeatures(**_features(amount=MAX_TRANSACTION_AMOUNT + Decimal(1)))

    def test_accepts_paysim_largest_known_transfer(self) -> None:
        """The bound must clear the real corpus it is supposed to admit."""
        transaction = TransactionFeatures(
            **_features(
                source_dataset="paysim",
                account_id="C1231006815",
                counterparty_id="C1666544295",
                amount=Decimal("92445516.64"),
                transaction_type="TRANSFER",
            )
        )
        assert transaction.amount == Decimal("92445516.64")

    def test_rejects_counterparty_on_ieee_cis(self) -> None:
        """IEEE-CIS records no money-flow edge; inventing one would feed Tier-3 a fiction."""
        with pytest.raises(ValidationError, match="no origin/destination structure"):
            TransactionFeatures(**_features(counterparty_id="C1666544295"))

    def test_allows_counterparty_on_paysim(self) -> None:
        transaction = TransactionFeatures(
            **_features(
                source_dataset="paysim",
                account_id="C1231006815",
                counterparty_id="C1666544295",
            )
        )
        assert transaction.counterparty_id == "C1666544295"

    def test_is_immutable(self) -> None:
        transaction = TransactionFeatures(**_features())
        with pytest.raises(ValidationError):
            transaction.amount = Decimal("1.00")  # type: ignore[misc]

    def test_carries_no_label_field(self) -> None:
        """The label lives on LabelledTransaction so a scoring path cannot reach it."""
        assert "is_fraud" not in TransactionFeatures.model_fields
        assert "split" not in TransactionFeatures.model_fields


class TestLabelledTransaction:
    """Training metadata, kept structurally separate from the scoring contract."""

    def test_wraps_a_transaction_with_label_and_split(self) -> None:
        labelled = LabelledTransaction(
            transaction=TransactionFeatures(**_features()),
            is_fraud=True,
            split="train",
        )
        assert labelled.is_fraud is True
        assert labelled.split == "train"

    def test_rejects_an_unknown_split(self) -> None:
        with pytest.raises(ValidationError):
            LabelledTransaction(
                transaction=TransactionFeatures(**_features()),
                is_fraud=False,
                split="holdout",  # type: ignore[arg-type]
            )


class TestSplitDefinition:
    """The chronological split is 70/15/15 per source — never a shuffle."""

    def test_fractions_sum_to_one(self) -> None:
        assert sum(SPLIT_FRACTIONS.values()) == pytest.approx(1.0)

    def test_fractions_cover_exactly_the_declared_splits(self) -> None:
        assert set(SPLIT_FRACTIONS) == set(SPLIT_ORDER)

    def test_train_precedes_val_precedes_test(self) -> None:
        assert SPLIT_ORDER == ("train", "val", "test")


class TestTimeAnchors:
    """Each source's clock is an offset; the anchors make it sortable, nothing more."""

    def test_ieee_cis_offset_is_seconds_from_its_epoch(self) -> None:
        assert ieee_cis_event_time(0) == IEEE_CIS_EPOCH
        assert ieee_cis_event_time(86_400) == IEEE_CIS_EPOCH + timedelta(days=1)

    def test_ieee_cis_preserves_ordering(self) -> None:
        assert ieee_cis_event_time(86_400) < ieee_cis_event_time(15_811_131)

    def test_paysim_step_is_a_one_based_hour_index(self) -> None:
        assert paysim_event_time(1) == PAYSIM_EPOCH
        assert paysim_event_time(2) == PAYSIM_EPOCH + timedelta(hours=1)
        assert paysim_event_time(744) == PAYSIM_EPOCH + timedelta(hours=743)

    def test_both_anchors_are_timezone_aware(self) -> None:
        assert ieee_cis_event_time(0).tzinfo is not None
        assert paysim_event_time(1).tzinfo is not None


class TestRawSpecColumns:
    """The column lists are generated, so their shape is worth pinning down."""

    def test_ieee_cis_transaction_has_394_unique_columns(self) -> None:
        assert len(IEEE_CIS_TRANSACTION_COLUMNS) == 394
        assert len(set(IEEE_CIS_TRANSACTION_COLUMNS)) == 394

    def test_ieee_cis_identity_has_41_unique_columns(self) -> None:
        assert len(IEEE_CIS_IDENTITY_COLUMNS) == 41
        assert len(set(IEEE_CIS_IDENTITY_COLUMNS)) == 41

    def test_paysim_has_11_unique_columns(self) -> None:
        assert len(PAYSIM_COLUMNS) == 11
        assert len(set(PAYSIM_COLUMNS)) == 11

    @pytest.mark.parametrize(
        ("column", "present"),
        [
            ("V1", True),
            ("V339", True),
            ("V340", False),
            ("C14", True),
            ("C15", False),
            ("D15", True),
            ("M9", True),
            ("card6", True),
            ("card7", False),
        ],
    )
    def test_transaction_series_boundaries(self, column: str, present: bool) -> None:
        assert (column in IEEE_CIS_TRANSACTION_COLUMNS) is present

    @pytest.mark.parametrize(
        ("column", "present"),
        [("id_01", True), ("id_09", True), ("id_38", True), ("id_39", False), ("id_1", False)],
    )
    def test_identity_columns_are_zero_padded(self, column: str, present: bool) -> None:
        """``id_01`` not ``id_1`` — the release pads to two digits."""
        assert (column in IEEE_CIS_IDENTITY_COLUMNS) is present


class TestRawFileSpecs:
    """The specs are the reference every dataset figure is checked against."""

    def test_every_spec_declares_a_known_source(self) -> None:
        known = set(get_args(SourceDataset))
        assert {spec.source_dataset for spec in RAW_FILE_SPECS} <= known

    def test_spec_keys_are_unique(self) -> None:
        keys = [spec.key for spec in RAW_FILE_SPECS]
        assert len(keys) == len(set(keys))

    def test_scan_columns_are_deduplicated_and_present_in_the_schema(self) -> None:
        for spec in RAW_FILE_SPECS:
            scan = spec.scan_columns
            assert len(scan) == len(set(scan))
            assert set(scan) <= set(spec.columns)

    def test_ieee_cis_base_rate_is_about_three_and_a_half_percent(self) -> None:
        rate = IEEE_CIS_TRANSACTION_SPEC.expected_base_rate
        assert rate is not None
        assert rate == pytest.approx(0.0350, abs=0.0005)

    def test_paysim_base_rate_is_about_one_eighth_of_a_percent(self) -> None:
        rate = PAYSIM_SPEC.expected_base_rate
        assert rate is not None
        assert rate == pytest.approx(0.00129, abs=0.00005)

    def test_base_rates_differ_enough_that_pooling_them_is_meaningless(self) -> None:
        """The reason models never train across sources — roughly a 27x gap."""
        ieee = IEEE_CIS_TRANSACTION_SPEC.expected_base_rate
        paysim = PAYSIM_SPEC.expected_base_rate
        assert ieee is not None and paysim is not None
        assert ieee / paysim > 20

    def test_identity_sidecar_is_unlabelled(self) -> None:
        assert IEEE_CIS_IDENTITY_SPEC.label_column is None
        assert IEEE_CIS_IDENTITY_SPEC.expected_base_rate is None

    def test_identity_covers_a_minority_of_transactions(self) -> None:
        """~24%. The missingness is itself a feature, not something to impute away."""
        ratio = IEEE_CIS_IDENTITY_SPEC.expected_rows / IEEE_CIS_TRANSACTION_SPEC.expected_rows
        assert 0.20 < ratio < 0.30

    def test_paysim_has_no_natural_key(self) -> None:
        """No transaction id column — the adapter synthesises one from the row index."""
        assert PAYSIM_SPEC.key_column is None

    def test_fraud_bearing_types_are_real_transaction_types(self) -> None:
        assert PAYSIM_FRAUD_BEARING_TYPES < PAYSIM_TRANSACTION_TYPES


class TestColumnNormalisation:
    """train_identity uses ``id_01``; test_identity uses ``id-01``. Normalise on read."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("id-01", "id_01"), ("id_01", "id_01"), (" TransactionID ", "TransactionID")],
    )
    def test_normalises_hyphens_and_whitespace(self, raw: str, expected: str) -> None:
        assert normalise_column(raw) == expected
