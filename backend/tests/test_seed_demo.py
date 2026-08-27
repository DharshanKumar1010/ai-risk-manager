"""``app.data.seed_demo`` — the pure sampling and coercion logic, no database required.

The DB-touching halves (``write_transactions_and_accounts``, ``score_demo_subset``) are
integration code by nature and are exercised by hand against a real database and the real
artefacts, not here -- a fake session standing in for Postgres would be asserting its own
behaviour rather than the script's, the same reasoning ``tests/conftest.py``'s ``FakeSession``
docstring gives for the equivalent choice elsewhere in this suite. What belongs in a fast,
DB-free suite is the sampling determinism, the ordering guarantee velocity features depend on,
and the raw-column extraction that feeds the real scoring path.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from app.data.seed_demo import (
    MAX_ACCOUNT_TRANSACTIONS,
    MIN_ACCOUNT_TRANSACTIONS,
    build_raw_columns,
    select_demo_sample,
    select_scoring_subset,
    to_decimal_amount,
    to_utc_datetime,
)


def _synthetic_frame(accounts_and_counts: dict[str, int]) -> pd.DataFrame:
    """Build a minimal frame with the columns `select_demo_sample` reads."""
    rows = []
    for account_id, count in accounts_and_counts.items():
        for index in range(count):
            rows.append(
                {
                    "account_id": account_id,
                    "event_time": datetime(2018, 5, 1, tzinfo=UTC).replace(
                        hour=index % 24, minute=index // 24
                    ),
                    "amount": 10.0 + index,
                    "transaction_id": f"{account_id}-{index}",
                    "source_dataset": "ieee_cis",
                    "is_fraud": False,
                    "split": "test",
                    "feature_version": "fv_test",
                    "uid_strategy": "card_addr_d1n",
                    "transaction_type": "W",
                    "counterparty_id": None,
                }
            )
    return pd.DataFrame(rows)


class TestSelectDemoSample:
    """Which accounts enter the sample, and the ordering guarantee within each."""

    def test_only_accounts_within_bounds_are_eligible(self) -> None:
        frame = _synthetic_frame(
            {"too-small": MIN_ACCOUNT_TRANSACTIONS - 1, "just-right": 5, "too-big": 50}
        )
        sample = select_demo_sample(frame, account_sample=10)
        assert set(sample["account_id"]) == {"just-right"}

    def test_the_sample_is_deterministic_across_runs(self) -> None:
        frame = _synthetic_frame({f"acct-{i}": 3 for i in range(50)})
        first = select_demo_sample(frame, account_sample=5, seed=7)
        second = select_demo_sample(frame, account_sample=5, seed=7)
        assert first["transaction_id"].tolist() == second["transaction_id"].tolist()

    def test_a_different_seed_can_choose_different_accounts(self) -> None:
        frame = _synthetic_frame({f"acct-{i}": 3 for i in range(50)})
        first = select_demo_sample(frame, account_sample=5, seed=1)
        second = select_demo_sample(frame, account_sample=5, seed=2)
        assert set(first["account_id"]) != set(second["account_id"])

    def test_rows_within_one_account_stay_in_chronological_order(self) -> None:
        """Velocity and familiarity features depend on this; a shuffle would silently
        corrupt every history-dependent feature computed from the seeded rows.
        """
        frame = _synthetic_frame({"acct-1": 10})
        sample = select_demo_sample(frame, account_sample=1)
        times = sample["event_time"].tolist()
        assert times == sorted(times)

    def test_never_exceeds_the_requested_account_count(self) -> None:
        frame = _synthetic_frame({f"acct-{i}": 3 for i in range(50)})
        sample = select_demo_sample(frame, account_sample=5)
        assert sample["account_id"].nunique() == 5

    def test_raises_rather_than_returning_an_empty_sample_when_nothing_is_eligible(
        self,
    ) -> None:
        frame = _synthetic_frame({"lonely": 1})
        with pytest.raises(ValueError, match="no accounts"):
            select_demo_sample(frame, account_sample=10)

    def test_requesting_more_accounts_than_exist_returns_what_is_available(self) -> None:
        frame = _synthetic_frame({f"acct-{i}": 3 for i in range(5)})
        sample = select_demo_sample(frame, account_sample=1000)
        assert sample["account_id"].nunique() == 5


class TestSelectScoringSubset:
    """The rows chosen to actually go through the real scoring path."""

    def test_prefers_rows_with_history_over_an_accounts_first_row(self) -> None:
        """A first transaction has no prior history to show; later ones do."""
        frame = _synthetic_frame({f"acct-{i}": MAX_ACCOUNT_TRANSACTIONS for i in range(20)})
        sample = select_demo_sample(frame, account_sample=20)
        subset = select_scoring_subset(sample, score_sample=50)

        first_per_account = sample.groupby("account_id")["event_time"].idxmin()
        first_ids = set(sample.loc[first_per_account, "transaction_id"])
        assert not (set(subset["transaction_id"]) & first_ids)

    def test_falls_back_to_first_rows_when_there_are_not_enough_others(self) -> None:
        """Every account here has exactly the minimum -- one non-first row each."""
        frame = _synthetic_frame({f"acct-{i}": MIN_ACCOUNT_TRANSACTIONS for i in range(5)})
        sample = select_demo_sample(frame, account_sample=5)
        subset = select_scoring_subset(sample, score_sample=100)
        # 5 accounts * MIN_ACCOUNT_TRANSACTIONS rows total; asking for more than exist
        # must not raise and must not duplicate any row.
        assert len(subset) == len(sample)
        assert subset["transaction_id"].is_unique

    def test_is_deterministic_across_runs(self) -> None:
        frame = _synthetic_frame({f"acct-{i}": 10 for i in range(20)})
        sample = select_demo_sample(frame, account_sample=20)
        first = select_scoring_subset(sample, score_sample=30, seed=3)
        second = select_scoring_subset(sample, score_sample=30, seed=3)
        assert first["transaction_id"].tolist() == second["transaction_id"].tolist()

    def test_never_returns_duplicate_rows(self) -> None:
        frame = _synthetic_frame({f"acct-{i}": 15 for i in range(10)})
        sample = select_demo_sample(frame, account_sample=10)
        subset = select_scoring_subset(sample, score_sample=200)
        assert subset["transaction_id"].is_unique


class TestBuildRawColumns:
    """What actually reaches the real scoring path -- must match its allowlist exactly."""

    def test_only_allowed_columns_are_forwarded(self) -> None:
        record = {"ProductCD": "W", "amount_log": 5.0, "not_a_real_column": 1}
        allowed = frozenset({"ProductCD"})
        raw = build_raw_columns(record, allowed)
        assert raw == {"ProductCD": "W"}

    def test_a_missing_allowed_column_is_simply_absent_not_none(self) -> None:
        """Matches assemble_tier1_vector's contract: a missing raw column means genuinely
        absent, not a supplied null -- the two are handled differently downstream.
        """
        raw = build_raw_columns({}, frozenset({"ProductCD"}))
        assert raw == {}

    def test_nan_and_non_finite_floats_are_dropped_rather_than_forwarded(self) -> None:
        record = {"C1": np.nan, "C2": np.inf, "C3": -np.inf, "C4": 3.5}
        raw = build_raw_columns(record, frozenset({"C1", "C2", "C3", "C4"}))
        assert raw == {"C4": 3.5}

    def test_numpy_scalar_types_are_coerced_to_plain_python(self) -> None:
        """A real HTTP client sends JSON, which has no numpy types -- this is what a live
        POST /score body would carry, and pydantic validation should see the same shapes.
        """
        record = {
            "card1": np.int64(13926),
            "C1": np.float64(1.5),
            "device_is_new": np.bool_(True),
        }
        raw = build_raw_columns(record, frozenset(record.keys()))
        assert raw == {"card1": 13926, "C1": 1.5, "device_is_new": True}
        assert isinstance(raw["card1"], int)
        assert isinstance(raw["C1"], float)
        assert isinstance(raw["device_is_new"], bool)

    def test_a_pandas_na_string_value_is_dropped(self) -> None:
        record = {"ProductCD": pd.NA}
        raw = build_raw_columns(record, frozenset({"ProductCD"}))
        assert raw == {}


class TestAmountAndTimeCoercion:
    """Small helpers, but a rounding or timezone mistake here changes a real decision."""

    def test_amount_rounds_to_the_columns_precision(self) -> None:
        assert to_decimal_amount(19.999999) == to_decimal_amount(19.999999)
        result = to_decimal_amount(10.00005)
        assert result.as_tuple().exponent == -4

    def test_utc_datetime_is_always_timezone_aware(self) -> None:
        naive = pd.Timestamp("2018-05-05 12:00:00")
        result = to_utc_datetime(naive)
        assert result.tzinfo is not None

    def test_an_already_aware_timestamp_is_preserved(self) -> None:
        aware = pd.Timestamp("2018-05-05 12:00:00", tz="UTC")
        result = to_utc_datetime(aware)
        assert result.tzinfo is not None
        assert result.hour == 12
