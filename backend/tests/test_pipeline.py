"""Tests for the Phase 1 feature pipeline.

Run against synthetic frames, never the real corpora: the datasets are gitignored, absent in
CI, and 1.15GB. The synthetic frames are built to contain the situations that actually break
this code — accounts with one transaction and accounts with many, repeated devices, duplicate
timestamps, categories that appear only after the train boundary.

The leakage test is the important one. It does not read the code looking for suspicious
calls; it recomputes each feature against a frame truncated to that row's own point in time
and asserts the value is unchanged. Any feature that consulted a later row moves when the
later rows are removed.
"""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.data.adapters import AdapterResult
from app.data.feature_store import FeatureDefinition, digest_encoders
from app.data.features import (
    IEEE_FAMILIARITY_COLUMNS,
    IEEE_FREQUENCY_COLUMNS,
    MISSING_SENTINEL,
    PAYSIM_FREQUENCY_COLUMNS,
    VELOCITY_WINDOWS,
    add_velocity_features,
    engineer_features,
    feature_names_for,
    fit_frequency_encoders,
    sort_for_engineering,
)
from app.data.pipeline import build_feature_vectors, process_source
from app.data.raw_spec import SourceDataset
from app.data.report import class_balance_drift, paysim_graph_viability
from app.data.schema import SPLIT_ORDER
from app.data.splitting import (
    SplitWindow,
    assign_splits,
    find_boundary_overlaps,
    summarise_splits,
)

SEED = 20260822
BASE_TIME = pd.Timestamp("2017-12-01T00:00:00Z")


def synthetic_ieee_frame(rows: int = 600, accounts: int = 40, seed: int = SEED) -> pd.DataFrame:
    """Build an IEEE-CIS-shaped canonical frame.

    Deliberately includes single-transaction accounts, repeated and novel devices, nulls in
    the identity block, and a category (`R`) that only appears late enough to fall outside
    the train split.
    """
    rng = np.random.default_rng(seed)
    account_ids = [f"c{index}_a100_d50" for index in range(accounts)]
    # Skewed so some accounts get many transactions and some get exactly one.
    weights = rng.dirichlet(np.full(accounts, 0.4))
    chosen = rng.choice(account_ids, size=rows, p=weights)

    offsets = np.sort(rng.integers(0, 120 * 86_400, size=rows))
    event_time = BASE_TIME + pd.to_timedelta(offsets, unit="s")

    # Nulls are introduced by masking rather than by putting None in the choice list, which
    # keeps the arrays typed and matches how the real corpus arrives: a value or a gap.
    devices = pd.array(
        rng.choice(["Windows", "iOS Device", "MacOS"], size=rows, p=[0.45, 0.33, 0.22]),
        dtype="string",
    )
    devices[rng.random(rows) < 0.10] = pd.NA

    emails = pd.array(rng.choice(["gmail.com", "yahoo.com"], size=rows), dtype="string")
    emails[rng.random(rows) < 0.20] = pd.NA

    product = rng.choice(["W", "C", "H"], size=rows, p=[0.6, 0.3, 0.1])
    # 'R' only in the final tenth, so it lands outside train and exercises unseen-category
    # handling in the frequency encoder.
    product[int(rows * 0.9) :] = "R"

    return pd.DataFrame(
        {
            "transaction_id": [str(2_987_000 + index) for index in range(rows)],
            "source_dataset": "ieee_cis",
            "event_time": event_time,
            "amount": np.round(rng.lognormal(4.0, 1.1, size=rows), 3),
            "account_id": pd.array(chosen, dtype="string"),
            "counterparty_id": pd.array([pd.NA] * rows, dtype="string"),
            "transaction_type": pd.array(product, dtype="string"),
            "is_fraud": rng.random(rows) < 0.035,
            "uid_strategy": "card_addr_d1n",
            "has_identity": rng.random(rows) < 0.24,
            "ProductCD": pd.array(product, dtype="string"),
            "card4": pd.array(rng.choice(["visa", "mastercard"], size=rows), dtype="string"),
            "card6": pd.array(rng.choice(["debit", "credit"], size=rows), dtype="string"),
            "P_emaildomain": emails,
            "DeviceInfo": devices,
            "addr1": rng.choice([204.0, 299.0, np.nan], size=rows),
        }
    )


def synthetic_paysim_frame(rows: int = 400, seed: int = SEED) -> pd.DataFrame:
    """Build a PaySim-shaped canonical frame, at one-hour clock granularity."""
    rng = np.random.default_rng(seed)
    steps = np.sort(rng.integers(1, 200, size=rows))
    amounts = np.round(rng.lognormal(9.0, 1.5, size=rows), 2)
    old_orig = amounts * rng.uniform(1.0, 3.0, size=rows)
    return pd.DataFrame(
        {
            "transaction_id": [str(index) for index in range(rows)],
            "source_dataset": "paysim",
            "event_time": BASE_TIME + pd.to_timedelta(steps - 1, unit="h"),
            "amount": amounts,
            "account_id": pd.array(
                [f"C{rng.integers(0, 10**9)}" for _ in range(rows)], dtype="string"
            ),
            "counterparty_id": pd.array(
                rng.choice(
                    [f"C{index}" for index in range(40)] + [f"M{index}" for index in range(5)],
                    size=rows,
                ),
                dtype="string",
            ),
            "transaction_type": pd.array(
                rng.choice(["TRANSFER", "CASH_OUT"], size=rows), dtype="string"
            ),
            "is_fraud": rng.random(rows) < 0.02,
            "uid_strategy": "native",
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": np.maximum(old_orig - amounts, 0.0),
            "oldbalanceDest": rng.uniform(0, 10_000, size=rows),
            "newbalanceDest": rng.uniform(0, 10_000, size=rows),
        }
    )


def encoders_for(frame: pd.DataFrame, source: SourceDataset) -> dict[str, dict[str, float]]:
    """Fit encoders on the train split of a frame, the way the pipeline does."""
    sorted_frame = sort_for_engineering(frame)
    split, _ = assign_splits(sorted_frame["event_time"], source)
    columns = IEEE_FREQUENCY_COLUMNS if source == "ieee_cis" else PAYSIM_FREQUENCY_COLUMNS
    return fit_frequency_encoders(sorted_frame, split == "train", columns)


@pytest.fixture
def ieee_frame() -> pd.DataFrame:
    return synthetic_ieee_frame()


@pytest.fixture
def paysim_frame() -> pd.DataFrame:
    return synthetic_paysim_frame()


class TestChronologicalSplit:
    """(a) Split boundaries are strictly time-ordered with zero overlap."""

    @pytest.mark.parametrize("source", ["ieee_cis", "paysim"])
    def test_boundaries_do_not_overlap(self, source: SourceDataset) -> None:
        frame = synthetic_ieee_frame() if source == "ieee_cis" else synthetic_paysim_frame()
        frame = sort_for_engineering(frame)
        frame["split"], _ = assign_splits(frame["event_time"], source)
        assert find_boundary_overlaps(frame) == []

    @pytest.mark.parametrize("source", ["ieee_cis", "paysim"])
    def test_every_train_row_precedes_every_val_and_test_row(self, source: SourceDataset) -> None:
        frame = synthetic_ieee_frame() if source == "ieee_cis" else synthetic_paysim_frame()
        frame = sort_for_engineering(frame)
        frame["split"], _ = assign_splits(frame["event_time"], source)
        windows = {window.split: window for window in summarise_splits(frame)}
        assert windows["train"].last_event < windows["val"].first_event
        assert windows["val"].last_event < windows["test"].first_event

    def test_splits_partition_every_row_exactly_once(self, ieee_frame: pd.DataFrame) -> None:
        split, boundaries = assign_splits(ieee_frame["event_time"], "ieee_cis")
        assert set(split.unique()) <= set(SPLIT_ORDER)
        assert boundaries.total == len(ieee_frame)
        assert split.notna().all()

    def test_fractions_are_close_to_the_target(self, ieee_frame: pd.DataFrame) -> None:
        _, boundaries = assign_splits(ieee_frame["event_time"], "ieee_cis")
        fractions = boundaries.fractions
        assert fractions["train"] == pytest.approx(0.70, abs=0.02)
        assert fractions["val"] == pytest.approx(0.15, abs=0.02)
        assert fractions["test"] == pytest.approx(0.15, abs=0.02)

    def test_is_not_a_shuffle_row_order_cannot_change_the_answer(
        self, ieee_frame: pd.DataFrame
    ) -> None:
        """The assignment is a pure function of the timestamps, not of row position."""
        ordered, _ = assign_splits(ieee_frame["event_time"], "ieee_cis")
        shuffled_frame = ieee_frame.sample(frac=1.0, random_state=7)
        shuffled, _ = assign_splits(shuffled_frame["event_time"], "ieee_cis")
        pd.testing.assert_series_equal(
            ordered.sort_index(), shuffled.sort_index(), check_names=False
        )

    def test_is_deterministic_across_repeated_calls(self, ieee_frame: pd.DataFrame) -> None:
        first, _ = assign_splits(ieee_frame["event_time"], "ieee_cis")
        second, _ = assign_splits(ieee_frame["event_time"], "ieee_cis")
        pd.testing.assert_series_equal(first, second)

    def test_rows_sharing_a_boundary_timestamp_all_fall_on_the_later_side(self) -> None:
        """Splitting by row position would cut a shared timestamp in half — an overlap."""
        # Heavy ties on every day, as PaySim has at one-hour granularity. Enough distinct
        # days that the cut is feasible — the point under test is where the ties land, not
        # whether a too-coarse timeline is rejected (covered separately below).
        times = pd.to_datetime(
            ["2020-01-01"] * 7 + ["2020-01-02"] * 7 + ["2020-01-03"] * 3 + ["2020-01-04"] * 3,
            utc=True,
        )
        frame = pd.DataFrame({"event_time": times})
        split, boundaries = assign_splits(frame["event_time"], "paysim")
        frame["split"] = split
        for _, group in frame.groupby("event_time"):
            assert group["split"].nunique() == 1, "one timestamp landed in two splits"
        assert find_boundary_overlaps(frame.assign(is_fraud=False)) == []

    def test_refuses_to_produce_an_empty_validation_split(self) -> None:
        """A silently empty val split would invalidate every threshold chosen on it."""
        times = pd.to_datetime(["2020-01-01"] * 9 + ["2020-01-02"], utc=True)
        with pytest.raises(ValueError, match="same timestamp"):
            assign_splits(pd.Series(times), "paysim")

    def test_rejects_null_timestamps(self) -> None:
        times = pd.Series(
            [
                pd.Timestamp("2020-01-01", tz="UTC"),
                pd.NaT,
                pd.Timestamp("2020-01-03", tz="UTC"),
            ]
        )
        with pytest.raises(ValueError, match="nulls"):
            assign_splits(times, "paysim")


class TestNoFutureLeakage:
    """(b) No engineered feature reads a timestamp later than the row it describes."""

    @staticmethod
    def _compare_row(
        full: pd.DataFrame,
        truncated: pd.DataFrame,
        position: int,
        feature_names: Sequence[str],
    ) -> list[str]:
        """Return the names of features that changed when the future was removed."""
        differing: list[str] = []
        for name in feature_names:
            with_future = full.iloc[position][name]
            without_future = truncated.iloc[-1][name]
            if pd.isna(with_future) and pd.isna(without_future):
                continue
            if isinstance(with_future, (float, np.floating)):
                if not np.isclose(float(with_future), float(without_future), equal_nan=True):
                    differing.append(name)
            elif with_future != without_future:
                differing.append(name)
        return differing

    @pytest.mark.parametrize("source", ["ieee_cis", "paysim"])
    def test_features_are_unchanged_when_later_rows_are_removed(
        self, source: SourceDataset
    ) -> None:
        frame = synthetic_ieee_frame() if source == "ieee_cis" else synthetic_paysim_frame()
        frame = sort_for_engineering(frame)
        encoders = encoders_for(frame, source)
        feature_names = feature_names_for(source)

        full = engineer_features(frame, source_dataset=source, encoders=encoders)

        # Spread the probes across the timeline, including the very end.
        positions = [len(frame) // 8, len(frame) // 3, len(frame) // 2, len(frame) - 1]
        for position in positions:
            truncated = engineer_features(
                frame.iloc[: position + 1].copy(), source_dataset=source, encoders=encoders
            )
            differing = self._compare_row(full, truncated, position, feature_names)
            assert not differing, (
                f"{source}: at row {position} these features changed when later rows were "
                f"removed, so they were reading the future: {differing}"
            )

    def test_the_check_would_catch_a_deliberately_leaky_feature(
        self, ieee_frame: pd.DataFrame
    ) -> None:
        """A guard on the guard: the comparison must actually fail on a known leak."""
        frame = sort_for_engineering(ieee_frame)
        encoders = encoders_for(frame, "ieee_cis")

        def leaky(subset: pd.DataFrame) -> pd.DataFrame:
            engineered = engineer_features(subset, source_dataset="ieee_cis", encoders=encoders)
            # The classic mistake: a whole-column aggregate, which on the full frame sees the
            # test period and on a truncated frame does not.
            engineered["account_mean_amount"] = engineered.groupby("account_id")[
                "amount"
            ].transform("mean")
            return engineered

        position = len(frame) // 2
        differing = self._compare_row(
            leaky(frame),
            leaky(frame.iloc[: position + 1].copy()),
            position,
            ["account_mean_amount"],
        )
        assert differing == ["account_mean_amount"]

    def test_amount_zscore_excludes_the_transaction_it_describes(self) -> None:
        """Including the current amount in its own baseline flattens the outlier."""
        times = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"], utc=True)
        frame = pd.DataFrame(
            {
                "transaction_id": ["1", "2", "3", "4"],
                "source_dataset": "paysim",
                "event_time": times,
                "amount": [100.0, 100.0, 100.0, 10_000.0],
                "account_id": pd.array(["A"] * 4, dtype="string"),
                "counterparty_id": pd.array(["C1"] * 4, dtype="string"),
                "transaction_type": pd.array(["TRANSFER"] * 4, dtype="string"),
                "is_fraud": [False, False, False, True],
                "uid_strategy": "native",
                "oldbalanceOrg": [1e6] * 4,
                "newbalanceOrig": [1e6] * 4,
                "oldbalanceDest": [0.0] * 4,
                "newbalanceDest": [0.0] * 4,
            }
        )
        engineered = engineer_features(frame, source_dataset="paysim", encoders={})
        scores = engineered["amount_zscore_vs_own_history"]
        assert pd.isna(scores.iloc[0]), "first transaction has no history to compare against"
        assert pd.isna(scores.iloc[1]), "one prior observation has no standard deviation"
        # Priors are three identical amounts, so dispersion is zero and the score is undefined
        # rather than infinite.
        assert pd.isna(scores.iloc[3])

    def test_amount_zscore_flags_a_genuine_outlier(self) -> None:
        times = pd.to_datetime([f"2020-01-{day:02d}" for day in range(1, 7)], utc=True)
        frame = pd.DataFrame(
            {
                "transaction_id": [str(index) for index in range(6)],
                "source_dataset": "paysim",
                "event_time": times,
                "amount": [100.0, 110.0, 90.0, 105.0, 95.0, 5_000.0],
                "account_id": pd.array(["A"] * 6, dtype="string"),
                "counterparty_id": pd.array(["C1"] * 6, dtype="string"),
                "transaction_type": pd.array(["TRANSFER"] * 6, dtype="string"),
                "is_fraud": [False] * 5 + [True],
                "uid_strategy": "native",
                "oldbalanceOrg": [1e6] * 6,
                "newbalanceOrig": [1e6] * 6,
                "oldbalanceDest": [0.0] * 6,
                "newbalanceDest": [0.0] * 6,
            }
        )
        engineered = engineer_features(frame, source_dataset="paysim", encoders={})
        assert engineered["amount_zscore_vs_own_history"].iloc[-1] > 10


class TestVelocityWindows:
    """The trailing window is (t - W, t]: earlier rows and this one, never a later one."""

    def test_velocity_matches_hand_computation(self) -> None:
        """Pins groupby().rolling() ordering — a misalignment here would be silent."""
        frame = pd.DataFrame(
            {
                "account_id": pd.array(["A", "B", "A", "A", "B", "A"], dtype="string"),
                "event_time": pd.to_datetime(
                    [
                        "2020-01-01 00:00",
                        "2020-01-01 00:10",
                        "2020-01-01 00:30",
                        "2020-01-01 00:30",
                        "2020-01-01 05:00",
                        "2020-01-01 09:00",
                    ],
                    utc=True,
                ),
                "amount": [10.0, 100.0, 20.0, 30.0, 200.0, 40.0],
            }
        )
        result = add_velocity_features(sort_for_engineering(frame), windows={"1h": "1h"})
        assert result["velocity_count_1h"].tolist() == [1.0, 1.0, 2.0, 3.0, 1.0, 1.0]
        assert result["velocity_sum_1h"].tolist() == [10.0, 100.0, 30.0, 60.0, 200.0, 40.0]

    def test_windows_are_nested(self, ieee_frame: pd.DataFrame) -> None:
        engineered = engineer_features(
            sort_for_engineering(ieee_frame),
            source_dataset="ieee_cis",
            encoders=encoders_for(ieee_frame, "ieee_cis"),
        )
        assert (engineered["velocity_count_1h"] <= engineered["velocity_count_24h"]).all()
        assert (engineered["velocity_count_24h"] <= engineered["velocity_count_7d"]).all()

    def test_paysim_velocity_is_null_not_zero(self, paysim_frame: pd.DataFrame) -> None:
        """A zero would read as 'this account was inactive', which is a different claim."""
        engineered = engineer_features(
            sort_for_engineering(paysim_frame), source_dataset="paysim", encoders={}
        )
        assert not engineered["velocity_available"].any()
        for window in VELOCITY_WINDOWS:
            assert engineered[f"velocity_count_{window}"].isna().all()


class TestFrequencyEncoders:
    """Fitted on train only — fitting on the whole frame leaks the test distribution back."""

    def test_categories_only_present_after_train_encode_to_zero(
        self, ieee_frame: pd.DataFrame
    ) -> None:
        frame = sort_for_engineering(ieee_frame)
        split, _ = assign_splits(frame["event_time"], "ieee_cis")
        encoders = fit_frequency_encoders(frame, split == "train", IEEE_FREQUENCY_COLUMNS)
        assert "R" not in encoders["ProductCD"], "R appears only after the train boundary"

        engineered = engineer_features(frame, source_dataset="ieee_cis", encoders=encoders)
        assert (engineered.loc[frame["ProductCD"] == "R", "freq_ProductCD"] == 0.0).all()

    def test_frequencies_sum_to_one_over_train(self, ieee_frame: pd.DataFrame) -> None:
        frame = sort_for_engineering(ieee_frame)
        split, _ = assign_splits(frame["event_time"], "ieee_cis")
        encoders = fit_frequency_encoders(frame, split == "train", IEEE_FREQUENCY_COLUMNS)
        for table in encoders.values():
            assert sum(table.values()) == pytest.approx(1.0)

    def test_missing_categories_get_an_explicit_level(self, ieee_frame: pd.DataFrame) -> None:
        """Nulls are a level, not a dropped row — for the identity block the gap is signal."""
        frame = sort_for_engineering(ieee_frame)
        split, _ = assign_splits(frame["event_time"], "ieee_cis")
        encoders = fit_frequency_encoders(frame, split == "train", ("P_emaildomain",))
        assert MISSING_SENTINEL in encoders["P_emaildomain"]


class TestFamiliarityFeatures:
    """'The account's usual device' means what it has used before, from prior rows only."""

    def test_first_sighting_is_new_and_fully_mismatched(self) -> None:
        frame = pd.DataFrame(
            {
                "account_id": pd.array(["A", "A", "A", "A"], dtype="string"),
                "event_time": pd.to_datetime(
                    ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"], utc=True
                ),
                "amount": [1.0, 2.0, 3.0, 4.0],
                "DeviceInfo": pd.array(["Windows", "Windows", "Windows", "iOS"], dtype="string"),
                "addr1": [1.0, 1.0, 1.0, 1.0],
            }
        )
        from app.data.features import add_familiarity_features

        result = add_familiarity_features(sort_for_engineering(frame), IEEE_FAMILIARITY_COLUMNS)
        assert result["device_is_new"].tolist() == [True, False, False, True]
        # Row 3 has three priors, none on iOS, so the device is entirely unlike its history.
        assert result["device_mismatch"].iloc[3] == pytest.approx(1.0)
        # Row 2 has two priors, both on Windows.
        assert result["device_mismatch"].iloc[2] == pytest.approx(0.0)
        assert pd.isna(result["device_mismatch"].iloc[0])


class TestFeatureStore:
    """The feature_version must move when meaning moves, and only then."""

    def _definition(self, **overrides: Any) -> FeatureDefinition:
        base: dict[str, Any] = {
            "source_dataset": "ieee_cis",
            "feature_names": ("amount_log", "hour_of_day"),
            "parameters": {"velocity_windows": ["1h", "24h"]},
            "encoder_digest": "abc123",
        }
        base.update(overrides)
        return FeatureDefinition(**base)

    def test_version_is_stable_across_runs(self) -> None:
        assert self._definition().feature_version == self._definition().feature_version

    def test_version_ignores_feature_ordering(self) -> None:
        reordered = self._definition(feature_names=("hour_of_day", "amount_log"))
        assert reordered.feature_version == self._definition().feature_version

    def test_version_moves_when_a_window_changes(self) -> None:
        changed = self._definition(parameters={"velocity_windows": ["1h", "12h"]})
        assert changed.feature_version != self._definition().feature_version

    def test_version_moves_when_the_fitted_encoders_change(self) -> None:
        """Same code, different training data, genuinely different features."""
        changed = self._definition(encoder_digest="def456")
        assert changed.feature_version != self._definition().feature_version

    def test_version_differs_between_corpora(self) -> None:
        assert self._definition(source_dataset="paysim").feature_version != (
            self._definition().feature_version
        )

    def test_rejects_duplicate_feature_names(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            self._definition(feature_names=("amount_log", "amount_log"))

    def test_rejects_parameters_that_cannot_be_hashed_stably(self) -> None:
        with pytest.raises(TypeError, match="JSON-native"):
            _ = self._definition(parameters={"windows": {"1h", "24h"}}).feature_version

    def test_encoder_digest_is_order_independent(self) -> None:
        first = digest_encoders({"a": {"x": 0.5, "y": 0.5}})
        second = digest_encoders({"a": {"y": 0.5, "x": 0.5}})
        assert first == second


class TestProcessSourceAndLogging:
    """(c) Row counts and class balance are logged."""

    def test_logs_row_counts_and_class_balance(
        self, ieee_frame: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="riskiq.pipeline"):
            process_source("ieee_cis", AdapterResult(frame=ieee_frame, notes={}))

        messages = "\n".join(record.getMessage() for record in caplog.records)
        assert "base rate" in messages
        assert "split assigned" in messages
        assert "frequency encoders fitted on train only" in messages
        for split in SPLIT_ORDER:
            assert f"ieee_cis/{split}" in messages

    def test_logged_counts_match_the_frame(
        self, ieee_frame: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="riskiq.pipeline"):
            processed = process_source("ieee_cis", AdapterResult(frame=ieee_frame, notes={}))
        messages = "\n".join(record.getMessage() for record in caplog.records)
        assert f"{len(processed.frame)} rows" in messages
        assert f"{int(processed.frame['is_fraud'].sum())} positives" in messages

    def test_produces_a_feature_version_on_every_row(self, ieee_frame: pd.DataFrame) -> None:
        processed = process_source("ieee_cis", AdapterResult(frame=ieee_frame, notes={}))
        assert processed.frame["feature_version"].nunique() == 1
        assert processed.frame["feature_version"].iloc[0] == processed.definition.feature_version

    def test_is_reproducible(self, ieee_frame: pd.DataFrame) -> None:
        """Same inputs, same feature_version — the claim the audit trail rests on."""
        first = process_source("ieee_cis", AdapterResult(frame=ieee_frame, notes={}))
        second = process_source("ieee_cis", AdapterResult(frame=ieee_frame, notes={}))
        assert first.definition.feature_version == second.definition.feature_version

    def test_every_declared_feature_is_actually_produced(self, ieee_frame: pd.DataFrame) -> None:
        processed = process_source("ieee_cis", AdapterResult(frame=ieee_frame, notes={}))
        missing = set(processed.definition.feature_names) - set(processed.frame.columns)
        assert not missing, f"declared but never engineered: {sorted(missing)}"


class TestAccounts:
    """The accounts table carries the two measures that qualify per-account features."""

    def test_straddling_accounts_are_flagged(self, ieee_frame: pd.DataFrame) -> None:
        processed = process_source("ieee_cis", AdapterResult(frame=ieee_frame, notes={}))
        accounts = processed.accounts
        for _, row in accounts.iterrows():
            rows = processed.frame.loc[processed.frame["account_id"] == row["account_id"]]
            assert row["straddles_split"] == (rows["split"].nunique() > 1)

    def test_counts_reconcile_with_the_transaction_frame(self, ieee_frame: pd.DataFrame) -> None:
        processed = process_source("ieee_cis", AdapterResult(frame=ieee_frame, notes={}))
        assert processed.accounts["transaction_count"].sum() == len(processed.frame)
        assert processed.accounts["fraud_count"].sum() == int(processed.frame["is_fraud"].sum())

    def test_fraud_count_never_exceeds_transaction_count(self, ieee_frame: pd.DataFrame) -> None:
        """Mirrors the database check constraint."""
        processed = process_source("ieee_cis", AdapterResult(frame=ieee_frame, notes={}))
        assert (processed.accounts["fraud_count"] <= processed.accounts["transaction_count"]).all()

    def test_singleton_accounts_hold_exactly_one_transaction(self) -> None:
        frame = synthetic_ieee_frame()
        frame["uid_strategy"] = "singleton"
        frame["account_id"] = pd.array(
            [f"txn{index}" for index in range(len(frame))], dtype="string"
        )
        processed = process_source("ieee_cis", AdapterResult(frame=frame, notes={}))
        singletons = processed.accounts.loc[processed.accounts["uid_strategy"] == "singleton"]
        assert (singletons["transaction_count"] == 1).all()


class TestClassBalanceDrift:
    """A chronological split does not promise a stationary class balance."""

    @staticmethod
    def _window(split: str, rows: int, positives: int) -> SplitWindow:
        return SplitWindow(
            split=split,  # type: ignore[arg-type]
            rows=rows,
            first_event=BASE_TIME,
            last_event=BASE_TIME,
            positives=positives,
        )

    def test_silent_when_balance_is_stable(self) -> None:
        windows = [
            self._window("train", 7000, 245),
            self._window("val", 1500, 52),
            self._window("test", 1500, 53),
        ]
        assert class_balance_drift(windows) is None

    def test_flags_a_test_split_concentrated_with_positives(self) -> None:
        """The real PaySim shape: about half the fraud in the final 15% of the timeline."""
        windows = [
            self._window("train", 1_938_484, 3_633),
            self._window("val", 415_628, 560),
            self._window("test", 416_297, 4_020),
        ]
        message = class_balance_drift(windows)
        assert message is not None
        assert "**test**" in message
        assert "not stationary" in message
        assert "quote the split's own base rate" in message

    def test_handles_a_corpus_with_no_positives(self) -> None:
        assert class_balance_drift([self._window("train", 100, 0)]) is None


class TestGraphViability:
    """Phase 4's linking strategy is chosen from this measurement, so it must be honest."""

    @staticmethod
    def _frame(chainable: bool) -> pd.DataFrame:
        """Build transfer/cash-out pairs that either do or do not chain by account name."""
        transfers = pd.DataFrame(
            {
                "account_id": pd.array([f"C{i}" for i in range(20)], dtype="string"),
                "counterparty_id": pd.array([f"M{i}" for i in range(20)], dtype="string"),
                "transaction_type": pd.array(["TRANSFER"] * 20, dtype="string"),
                "is_fraud": [True] * 20,
            }
        )
        # When chainable, the cash-out origin is the transfer's destination.
        origins = [f"M{i}" for i in range(20)] if chainable else [f"Z{i}" for i in range(20)]
        cashouts = pd.DataFrame(
            {
                "account_id": pd.array(origins, dtype="string"),
                "counterparty_id": pd.array([f"M{i}" for i in range(20)], dtype="string"),
                "transaction_type": pd.array(["CASH_OUT"] * 20, dtype="string"),
                "is_fraud": [True] * 20,
            }
        )
        return pd.concat([transfers, cashouts], ignore_index=True)

    def test_reports_an_absent_link_as_absent(self) -> None:
        """The measured PaySim case: 0% chaining. 'A minority' would overstate it."""
        body = paysim_graph_viability(self._frame(chainable=False))
        assert "recovers essentially nothing" in body
        assert "must not link transfers to cash-outs by account name" in body

    def test_stays_quiet_when_chaining_works(self) -> None:
        body = paysim_graph_viability(self._frame(chainable=True))
        assert "must not link" not in body

    def test_separates_the_graph_being_viable_from_the_link_being_absent(self) -> None:
        """Recurrence and chaining are different questions; only one of them failed."""
        body = paysim_graph_viability(self._frame(chainable=False))
        assert "graph itself is still viable" in body


class TestPersistenceParity:
    """Postgres and parquet must describe the same rows.

    Two stores exist because they answer different questions: Postgres is the serving source
    of truth and carries the engineered vector as JSONB, while parquet carries the same rows
    plus the raw model columns and is what training reads. That split is only safe while the
    two agree, so the agreement is asserted rather than assumed.

    Skipped unless a populated database and processed parquet are both present — CI has
    neither, and these are integration checks over a real pipeline run.
    """

    @staticmethod
    def _database_counts() -> dict[tuple[str, str], tuple[int, int, str]]:
        """Return {(source, split): (rows, positives, feature_version)} from Postgres."""
        import asyncio

        import asyncpg

        from app.config import get_settings
        from app.data.pipeline import _asyncpg_dsn

        async def query() -> list[Any]:
            connection = await asyncpg.connect(_asyncpg_dsn(get_settings().database_url))
            try:
                fetched = await connection.fetch(
                    "SELECT source_dataset, split, count(*) AS rows, "
                    "count(*) FILTER (WHERE is_fraud) AS positives, "
                    "min(feature_version) AS feature_version, "
                    "count(DISTINCT feature_version) AS versions "
                    "FROM transactions GROUP BY source_dataset, split"
                )
                return list(fetched)
            finally:
                await connection.close()

        try:
            rows = asyncio.run(query())
        except Exception as exc:  # noqa: BLE001 - any connection problem means "skip"
            pytest.skip(f"database unavailable: {exc}")

        if not rows:
            pytest.skip("transactions table is empty; run the pipeline first")
        for row in rows:
            assert row["versions"] == 1, (
                f"{row['source_dataset']}/{row['split']} holds "
                f"{row['versions']} feature versions; a split must be one definition"
            )
        return {
            (row["source_dataset"], row["split"]): (
                row["rows"],
                row["positives"],
                row["feature_version"],
            )
            for row in rows
        }

    def test_parquet_and_postgres_agree(self) -> None:
        from app.config import get_settings

        processed_dir = get_settings().processed_data_dir
        if not processed_dir.is_dir():
            pytest.skip(f"no processed directory at {processed_dir}")

        database = self._database_counts()
        compared = 0
        for (source, split), (rows, positives, version) in sorted(database.items()):
            path = processed_dir / f"{source}_{split}.parquet"
            if not path.is_file():
                pytest.skip(f"missing {path}; database and parquet are from different runs")
            frame = pd.read_parquet(path, columns=["is_fraud", "feature_version"])
            assert len(frame) == rows, f"{source}/{split}: parquet {len(frame)} vs db {rows}"
            assert int(frame["is_fraud"].sum()) == positives, f"{source}/{split}: positives differ"
            assert frame["feature_version"].unique().tolist() == [version]
            compared += 1
        assert compared, "nothing compared"


class TestFeatureVectors:
    """What lands in JSONB must survive the round trip unchanged."""

    def test_values_are_json_native(self, ieee_frame: pd.DataFrame) -> None:
        processed = process_source("ieee_cis", AdapterResult(frame=ieee_frame, notes={}))
        vectors = build_feature_vectors(
            processed.frame.head(50), processed.definition.feature_names
        )
        allowed = (bool, int, float, str, type(None))
        for vector in vectors:
            for key, value in vector.items():
                assert isinstance(value, allowed), f"{key} is {type(value).__name__}"

    def test_non_finite_floats_become_null(self) -> None:
        """NaN and Infinity are not valid JSON and JSONB rejects them."""
        frame = pd.DataFrame({"f": [1.0, np.nan, np.inf, -np.inf]})
        vectors = build_feature_vectors(frame, ["f"])
        assert [vector["f"] for vector in vectors] == [1.0, None, None, None]

    def test_every_vector_has_the_declared_keys(self, ieee_frame: pd.DataFrame) -> None:
        processed = process_source("ieee_cis", AdapterResult(frame=ieee_frame, notes={}))
        expected = set(processed.definition.feature_names)
        vectors = build_feature_vectors(
            processed.frame.head(20), processed.definition.feature_names
        )
        assert all(set(vector) == expected for vector in vectors)
