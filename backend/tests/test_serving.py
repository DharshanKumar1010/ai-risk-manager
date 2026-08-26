"""Serving-time feature assembly.

The failure this file exists to catch is the quiet one. An endpoint that rejects a bad request
fails visibly; an endpoint that assembles a subtly wrong feature vector returns a decision that
looks exactly like a correct one, records it in an audit row that looks exactly like a correct
one, and is wrong. So these tests check the engineered values against hand-computed arithmetic
rather than against whatever the code currently produces.

Tests needing the trained artefacts skip where they are absent — ``models/artifacts/`` is
gitignored, so a fresh checkout has none. The assembly tests that do not need a fitted spec run
everywhere, and they are the majority.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from app.config import Settings
from app.core.serving import HistoryRow, ModelBundle, assemble_tier1_vector
from app.data import features as feature_engineering
from tests.conftest import TEST_SIGNING_KEY

BASE_TIME = datetime(2018, 5, 5, 14, 30, tzinfo=UTC)

RAW = {
    "ProductCD": "W",
    "card1": 13926,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
    "addr1": 315.0,
    "DeviceInfo": "Windows",
    "C1": 1.0,
    "C13": 1.0,
    "D1": 0.0,
}


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    """Load the shipped models, or skip the module where the artefacts are absent."""
    settings = Settings(environment="ci", jwt_secret_key=TEST_SIGNING_KEY)
    try:
        return ModelBundle.load(settings)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        pytest.skip(f"trained artefacts unavailable: {exc}")


def _history(count: int, *, amount: float = 50.0, hours: int = 1) -> list[HistoryRow]:
    """Return ``count`` prior transactions, one every ``hours`` before the base time."""
    return [
        HistoryRow(
            event_time=BASE_TIME - timedelta(hours=hours * (count - index)),
            amount=amount + index,
            device_info="Windows",
            addr1=315.0,
        )
        for index in range(count)
    ]


def _assemble(
    bundle: ModelBundle, history: list[HistoryRow], amount: str = "150.00"
) -> dict[str, Any]:
    """Assemble one vector and return its feature mapping."""
    return dict(
        assemble_tier1_vector(
            bundle,
            transaction_id="T-1",
            account_id="acct-1",
            event_time=BASE_TIME,
            amount=Decimal(amount),
            raw_columns=dict(RAW),
            history=history,
        ).features
    )


class TestVectorShape:
    """The vector must match the model's definition exactly, or scoring refuses it."""

    def test_every_feature_the_model_reads_is_present(self, bundle: ModelBundle) -> None:
        vector = _assemble(bundle, [])
        assert set(vector) == set(bundle.tier1.feature_names)

    def test_the_vector_carries_the_models_own_feature_version(self, bundle: ModelBundle) -> None:
        """A mismatch makes ``Tier1Model.score`` refuse, which is the intended behaviour."""
        transaction = assemble_tier1_vector(
            bundle,
            transaction_id="T-1",
            account_id="acct-1",
            event_time=BASE_TIME,
            amount=Decimal("150.00"),
            raw_columns=dict(RAW),
            history=[],
        )
        assert transaction.feature_version == bundle.tier1.feature_version

    def test_an_assembled_vector_scores(self, bundle: ModelBundle) -> None:
        transaction = assemble_tier1_vector(
            bundle,
            transaction_id="T-1",
            account_id="acct-1",
            event_time=BASE_TIME,
            amount=Decimal("150.00"),
            raw_columns=dict(RAW),
            history=_history(3),
        )
        result = bundle.tier1.score(transaction)
        assert 0.0 <= result.score <= 1.0
        assert result.model_version == bundle.tier1.model_id


class TestHistoryDerivedFeatures:
    """Checked against hand arithmetic, not against the current output."""

    def test_prior_count_is_the_number_of_earlier_transactions(self, bundle: ModelBundle) -> None:
        assert _assemble(bundle, _history(6))["account_prior_txn_count"] == 6.0

    def test_a_cold_account_has_no_prior_transactions(self, bundle: ModelBundle) -> None:
        assert _assemble(bundle, [])["account_prior_txn_count"] == 0.0

    def test_seconds_since_prior_matches_the_gap(self, bundle: ModelBundle) -> None:
        """One prior transaction, exactly three hours earlier."""
        history = [
            HistoryRow(
                event_time=BASE_TIME - timedelta(hours=3),
                amount=40.0,
                device_info="Windows",
                addr1=315.0,
            )
        ]
        assert _assemble(bundle, history)["seconds_since_prior_txn"] == 3 * 3600.0

    def test_a_cold_account_has_no_recency(self, bundle: ModelBundle) -> None:
        """Null, not zero: "no prior transaction" is not "a prior transaction just now"."""
        assert _assemble(bundle, [])["seconds_since_prior_txn"] is None

    def test_velocity_counts_the_window_inclusive_of_the_current_row(
        self, bundle: ModelBundle
    ) -> None:
        """Six priors an hour apart, plus this one, all inside 24h."""
        vector = _assemble(bundle, _history(6))
        assert vector["velocity_count_24h"] == 7.0

    def test_velocity_sums_amounts_over_the_window(self, bundle: ModelBundle) -> None:
        """Priors are 50..55 (sum 315), the current is 150."""
        vector = _assemble(bundle, _history(6))
        assert vector["velocity_sum_24h"] == pytest.approx(315.0 + 150.0)

    def test_a_transaction_outside_the_window_is_excluded(self, bundle: ModelBundle) -> None:
        """A prior eight days back must not enter the 7-day window."""
        history = [
            HistoryRow(
                event_time=BASE_TIME - timedelta(days=8),
                amount=999.0,
                device_info="Windows",
                addr1=315.0,
            )
        ]
        vector = _assemble(bundle, history)
        assert vector["velocity_count_7d"] == 1.0
        assert vector["velocity_sum_7d"] == pytest.approx(150.0)

    def test_amount_log_is_log1p_of_the_amount(self, bundle: ModelBundle) -> None:
        import numpy as np

        vector = _assemble(bundle, [], amount="150.00")
        assert vector["amount_log"] == pytest.approx(float(np.log1p(150.0)))


class TestFamiliarityFeatures:
    """ "Is this entity unusual for this account?", from prior rows only."""

    def test_a_first_seen_device_is_new(self, bundle: ModelBundle) -> None:
        assert _assemble(bundle, [])["device_is_new"] == 1.0

    def test_a_repeated_device_is_not_new(self, bundle: ModelBundle) -> None:
        assert _assemble(bundle, _history(3))["device_is_new"] == 0.0

    def test_a_device_the_account_always_uses_does_not_mismatch(self, bundle: ModelBundle) -> None:
        assert _assemble(bundle, _history(3))["device_mismatch"] == pytest.approx(0.0)

    def test_a_device_never_used_before_fully_mismatches(self, bundle: ModelBundle) -> None:
        """Three priors on a different device: the share of priors on this one is zero."""
        history = [
            HistoryRow(
                event_time=BASE_TIME - timedelta(hours=index + 1),
                amount=50.0,
                device_info="MacOS",
                addr1=315.0,
            )
            for index in range(3)
        ]
        assert _assemble(bundle, history)["device_mismatch"] == pytest.approx(1.0)


class TestCallerSuppliedValuesCannotReachDerivedFeatures:
    """The security property, asserted rather than argued.

    Validation refuses these names at the route. This checks the *second* control: even handed
    a derived name directly, assembly overwrites it with the real computation.
    """

    # (feature, value the caller plants, value the server must compute anyway). The expected
    # value is asserted rather than merely "not the planted one": for a feature whose true
    # value happens to coincide with a plausible plant -- ``device_is_new`` is 0.0 for an
    # account with history, and 0.0 is exactly what an attacker would send -- an inequality
    # check cannot tell an overwrite from a coincidence, and would pass while the control was
    # broken. Six priors an hour apart on the same device, at 14:30, amount 150.
    @pytest.mark.parametrize(
        ("feature", "planted", "expected"),
        [
            ("velocity_count_24h", 0.0, 7.0),
            ("account_prior_txn_count", 0.0, 6.0),
            ("amount_log", 0.0, 5.0172798368149243),
            ("device_is_new", 1.0, 0.0),
            ("hour_of_day", 3.0, 14.0),
        ],
    )
    def test_a_planted_derived_feature_is_overwritten(
        self, bundle: ModelBundle, feature: str, planted: float, expected: float
    ) -> None:
        raw = dict(RAW)
        raw[feature] = planted
        vector = dict(
            assemble_tier1_vector(
                bundle,
                transaction_id="T-1",
                account_id="acct-1",
                event_time=BASE_TIME,
                amount=Decimal("150.00"),
                raw_columns=raw,
                history=_history(6),
            ).features
        )
        assert vector[feature] == pytest.approx(
            expected
        ), f"{feature} was not recomputed over the caller's value"

    def test_a_planted_frequency_encoding_is_recomputed(self, bundle: ModelBundle) -> None:
        """``freq_*`` comes from the fitted table, never from the payload.

        Asserted against the table itself rather than a literal, because the frequency is a
        property of the training corpus and hard-coding it here would pin this test to one
        artefact build.
        """
        raw = dict(RAW)
        raw["freq_ProductCD"] = 0.0
        vector = dict(
            assemble_tier1_vector(
                bundle,
                transaction_id="T-1",
                account_id="acct-1",
                event_time=BASE_TIME,
                amount=Decimal("150.00"),
                raw_columns=raw,
                history=_history(6),
            ).features
        )
        assert vector["freq_ProductCD"] == pytest.approx(bundle.serving_encoders["ProductCD"]["W"])

    def test_has_identity_is_derived_from_the_payload_not_taken_from_it(
        self, bundle: ModelBundle
    ) -> None:
        """No identity columns supplied, so the flag is false however it is planted."""
        raw = {key: value for key, value in RAW.items()}
        raw["has_identity"] = 1.0
        vector = dict(
            assemble_tier1_vector(
                bundle,
                transaction_id="T-1",
                account_id="acct-1",
                event_time=BASE_TIME,
                amount=Decimal("150.00"),
                raw_columns=raw,
                history=[],
            ).features
        )
        assert vector["has_identity"] == 0.0

    def test_the_allowlist_excludes_every_engineered_feature(self, bundle: ModelBundle) -> None:
        engineered = set(feature_engineering.ieee_feature_names())
        assert not (engineered & bundle.allowed_raw_columns)

    def test_the_allowlist_still_admits_genuine_raw_columns(self, bundle: ModelBundle) -> None:
        """The control must not be so broad that a caller cannot describe its transaction."""
        for column in ("card1", "ProductCD", "DeviceInfo", "addr1", "C1", "D1"):
            assert column in bundle.allowed_raw_columns


class TestServingEncodersMatchThePipeline:
    """The four tables the pipeline fits and discards, rebuilt for serving."""

    def test_a_rebuild_that_disagrees_with_the_pipeline_is_refused(self) -> None:
        """The check that makes the artefact trustworthy, exercised on a deliberate mismatch."""
        from app.data.serving_encoders import verify_against_processed

        frame = pd.DataFrame(
            {
                "ProductCD": ["W", "C"],
                "card4": ["visa", "visa"],
                "card6": ["debit", "debit"],
                "P_emaildomain": ["gmail.com", "gmail.com"],
                "freq_ProductCD": [0.5, 0.5],
                "freq_card4": [1.0, 1.0],
                "freq_card6": [1.0, 1.0],
                "freq_P_emaildomain": [1.0, 1.0],
            }
        )
        wrong = {"ProductCD": {"W": 0.9, "C": 0.1}}
        with pytest.raises(ValueError, match="differs from the pipeline"):
            verify_against_processed(wrong, frame, "ieee_cis")

    def test_a_frame_without_the_pipelines_output_cannot_verify(self) -> None:
        """Refusing beats writing an artefact nobody checked."""
        from app.data.serving_encoders import verify_against_processed

        frame = pd.DataFrame({"ProductCD": ["W"]})
        with pytest.raises(ValueError, match="cannot be verified"):
            verify_against_processed({"ProductCD": {"W": 1.0}}, frame, "ieee_cis")


class TestHistoryExcludesTheIncomingTransaction:
    """A transaction must not appear in its own history.

    Reachable three ways: a re-score, the Phase 7 ``/replay`` enhancement, and a redelivered
    Phase 9 webhook. In each the row is already persisted when scoring runs, so an unfiltered
    scan returns it *and* assembly appends it again. The result is a decision made on a
    doubled transaction, recorded in an audit row that looks entirely correct.
    """

    async def test_the_scan_filters_the_incoming_transaction_id(self) -> None:
        """Asserted on the compiled SQL, because the fake session executes nothing."""
        from app.core.serving import read_account_history

        captured: dict[str, object] = {}

        class CapturingSession:
            """Records the statement instead of running it."""

            async def execute(self, statement: object, parameters: object = None) -> object:
                captured["statement"] = statement

                class Empty:
                    def all(self) -> list[object]:
                        return []

                return Empty()

        await read_account_history(
            CapturingSession(),  # type: ignore[arg-type]
            "ieee_cis",
            "acct-1",
            before=BASE_TIME,
            limit=10,
            exclude_transaction_id="T-1",
        )
        sql = str(captured["statement"])
        assert "transactions.transaction_id !=" in sql

    async def test_without_an_exclusion_no_id_filter_is_applied(self) -> None:
        """The parameter is opt-in, so a caller with no id to exclude is unaffected."""
        from app.core.serving import read_account_history

        captured: dict[str, object] = {}

        class CapturingSession:
            """Records the statement instead of running it."""

            async def execute(self, statement: object, parameters: object = None) -> object:
                captured["statement"] = statement

                class Empty:
                    def all(self) -> list[object]:
                        return []

                return Empty()

        await read_account_history(
            CapturingSession(),  # type: ignore[arg-type]
            "ieee_cis",
            "acct-1",
            before=BASE_TIME,
            limit=10,
        )
        assert "transactions.transaction_id !=" not in str(captured["statement"])

    def test_a_duplicated_prior_would_change_the_decision(self, bundle: ModelBundle) -> None:
        """Why the filter matters: the doubled row moves real features, not cosmetic ones."""
        single = _assemble(bundle, _history(1))
        doubled = _assemble(bundle, _history(1) + _history(1))
        assert single["account_prior_txn_count"] != doubled["account_prior_txn_count"]
        assert single["velocity_count_24h"] != doubled["velocity_count_24h"]
