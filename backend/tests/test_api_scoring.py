"""``POST /score``: the decision path, the audit write, and what the response may carry.

The disclosure tests in this file are the ones worth reading. Phase 6 widened the carried gate
from "must not return ``top_features``" to "must not return any field of ``DecisionCost``",
because the sign of ``expected_saving_from_blocking`` *is* the decision boundary. These assert
that gate against the real response body rather than against a reviewer's memory of it.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response

from app.api.schemas import (
    RISK_BAND_ELEVATED,
    RISK_BAND_HIGH,
    ScoreResponse,
    risk_band,
)
from app.config import Settings
from app.core.security import SCOPE_SCORE
from app.core.serving import (
    DEGRADED_TIER3_TIMEOUT,
    DEGRADED_TIER3_UNAVAILABLE,
    ModelBundle,
    _tier3_with_timeout,
)
from app.data.schema import TransactionFeatures
from tests.conftest import FakeSession, auth_header, score_payload

#: Field names that must never appear anywhere in a scoring response body.
FORBIDDEN_RESPONSE_FIELDS = (
    "top_features",
    "cost_estimate",
    "expected_cost",
    "cost_if_blocked",
    "cost_if_allowed",
    "expected_saving_from_blocking",
    "fraud_probability",
    "risk_probability",
    "threshold",
    "assumptions",
)


def _score(client: TestClient, settings: Settings, **overrides: Any) -> Response:
    """Issue a scoring request as a merchant holding ``score:write``."""
    return client.post(
        "/score",
        json=score_payload(**overrides),
        headers=auth_header(settings, account_id="acct-1", scopes=(SCOPE_SCORE,)),
    )


class TestRiskBanding:
    """The response reports a band, deliberately not the calibrated probability."""

    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.0, "low"),
            (RISK_BAND_ELEVATED - 1e-9, "low"),
            (RISK_BAND_ELEVATED, "elevated"),
            (RISK_BAND_HIGH - 1e-9, "elevated"),
            (RISK_BAND_HIGH, "high"),
            (1.0, "high"),
        ],
    )
    def test_band_edges_are_inclusive_below(self, probability: float, expected: str) -> None:
        assert risk_band(probability) == expected

    def test_the_response_schema_has_no_probability_field(self) -> None:
        """Structural, not incidental: there is no field for a caller to read."""
        assert "risk_probability" not in ScoreResponse.model_fields
        assert "probability" not in ScoreResponse.model_fields


class TestResponseDisclosure:
    """The Phase 6 gate: no field of DecisionCost, and no attribution, may reach a caller."""

    def test_score_response_schema_excludes_every_forbidden_field(self) -> None:
        """Asserted on the schema, so it holds for every response the route can produce."""
        for field in FORBIDDEN_RESPONSE_FIELDS:
            assert field not in ScoreResponse.model_fields, f"{field} is exposed by ScoreResponse"

    def test_score_response_body_excludes_every_forbidden_field(
        self, client: TestClient, settings: Settings
    ) -> None:
        """And asserted against a real body, in case a route ever returns something else."""
        response = _score(client, settings)
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        assert response.status_code == 200
        body = response.text.lower()
        for field in FORBIDDEN_RESPONSE_FIELDS:
            assert field not in body, f"{field} leaked into the scoring response"

    def test_public_projection_of_the_meta_result_omits_attribution(self) -> None:
        """``MetaResult.public()`` is the mechanism the carried gate names by name."""
        from app.models.meta_learner import MetaResult

        result = MetaResult(
            probability=0.4,
            decision="review",
            top_features=(("amount_log", 0.3),),
            model_version="m1",
            latency_ms=1.0,
        )
        assert "top_features" not in result.public()


class TestRawColumnValidation:
    """A caller supplies raw source columns and cannot reach a derived feature."""

    @pytest.mark.parametrize(
        "derived",
        [
            "velocity_count_24h",
            "amount_zscore_vs_own_history",
            "account_prior_txn_count",
            "freq_ProductCD",
            "has_identity",
            "device_is_new",
        ],
    )
    def test_derived_features_are_refused(
        self, client: TestClient, settings: Settings, derived: str
    ) -> None:
        """The features an attacker would want to set are exactly the refused ones."""
        response = _score(client, settings, raw_columns={derived: 0.0})
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "unknown_raw_columns"

    def test_the_error_names_columns_but_never_echoes_values(
        self, client: TestClient, settings: Settings
    ) -> None:
        """item 4.4: an error response must not echo the raw input back."""
        secret = "a-value-that-must-not-be-echoed"
        response = _score(client, settings, raw_columns={"not_a_column": secret})
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        assert response.status_code == 422
        assert secret not in response.text

    def test_an_unbounded_amount_is_refused(self, client: TestClient, settings: Settings) -> None:
        """item 4.2, and the Phase 6 finding: a negative amount switches blocking off."""
        assert _score(client, settings, amount="-1.00").status_code == 422

    def test_a_naive_event_time_is_refused(self, client: TestClient, settings: Settings) -> None:
        """Every trailing window is a time comparison; a naive datetime breaks one deep inside."""
        assert _score(client, settings, event_time="2018-05-05T14:30:00").status_code == 422


class TestDegradedMode:
    """PHASE_PROMPTS item 5: the Tier-3 timeout must fall back *and say so*."""

    async def test_a_missing_snapshot_degrades_with_a_reason(self) -> None:
        """The service runs without a ring snapshot, and records that it did."""
        bundle = _bundle_without_tier3()
        result, reason = await _tier3_with_timeout(bundle, _transaction(), timeout_ms=50)
        assert result is None
        assert reason == DEGRADED_TIER3_UNAVAILABLE

    async def test_a_slow_lookup_times_out_and_degrades(self) -> None:
        """A lookup that hangs must not hang the request."""
        import time as time_module

        class HangingTier3:
            """A snapshot whose lookup takes far longer than any budget."""

            def score(self, _transaction: TransactionFeatures) -> None:
                """Block well past the timeout."""
                time_module.sleep(0.5)

        bundle = _bundle_without_tier3(tier3=HangingTier3())
        result, reason = await _tier3_with_timeout(bundle, _transaction(), timeout_ms=20)
        assert result is None
        assert reason == DEGRADED_TIER3_TIMEOUT

    async def test_a_raising_lookup_degrades_rather_than_failing_the_request(self) -> None:
        """An enrichment that errors is still an enrichment."""

        class BrokenTier3:
            """A snapshot that raises on every lookup."""

            def score(self, _transaction: TransactionFeatures) -> None:
                """Fail."""
                raise RuntimeError("snapshot corrupt")

        bundle = _bundle_without_tier3(tier3=BrokenTier3())
        result, reason = await _tier3_with_timeout(bundle, _transaction(), timeout_ms=50)
        assert result is None
        assert reason == DEGRADED_TIER3_UNAVAILABLE

    def test_the_audit_record_refuses_to_degrade_without_a_reason(self) -> None:
        """A row saying a layer was missing, without saying which, cannot reconstruct itself."""
        from pydantic import ValidationError

        from app.core.audit import AuditRecord

        with pytest.raises(ValidationError, match="degraded_reason is required"):
            AuditRecord(
                transaction_id="T-1",
                account_id="acct-1",
                decided_at=datetime(2018, 5, 5, tzinfo=UTC),
                decision="allow",
                risk_probability=0.1,
                feature_version="fv_test",
                degraded=True,
            )


class TestAuditIsWritten:
    """security-checklist section 7: no decision reaches a caller without its audit row."""

    def test_a_successful_score_commits_exactly_one_audit_row(
        self, client: TestClient, settings: Settings, session: FakeSession
    ) -> None:
        response = _score(client, settings)
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        assert response.status_code == 200
        assert len(session.added) == 1
        assert session.committed is True

    def test_the_audit_row_carries_the_provenance_a_replay_would_need(
        self, client: TestClient, settings: Settings, session: FakeSession
    ) -> None:
        """model_versions and feature_version are what make a past decision reconstructable."""
        response = _score(client, settings)
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        row = session.added[0]
        assert row.feature_version.startswith("fv_")
        assert "tier1" in row.model_versions
        assert "causal_cost" in row.model_versions
        assert row.account_id == "acct-1"

    def test_the_audit_row_keeps_what_the_response_withholds(
        self, client: TestClient, settings: Settings, session: FakeSession
    ) -> None:
        """The record is strictly richer than the response. That asymmetry is the design."""
        response = _score(client, settings)
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        row = session.added[0]
        assert row.top_features, "attribution must be recorded even though it is not served"
        assert row.cost_estimate is not None
        assert "expected_saving_from_blocking" in row.cost_estimate

    def test_the_response_returns_the_audit_handle(
        self, client: TestClient, settings: Settings
    ) -> None:
        response = _score(client, settings)
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        assert response.json()["audit_id"] >= 1


def _transaction() -> TransactionFeatures:
    """Return a minimal transaction for the Tier-3 timeout tests."""
    return TransactionFeatures(
        transaction_id="T-1",
        source_dataset="ieee_cis",
        event_time=datetime(2018, 5, 5, 14, 30, tzinfo=UTC),
        amount=Decimal("150.00"),
        account_id="acct-1",
        feature_version="fv_test",
        features={},
    )


def _bundle_without_tier3(tier3: Any = None) -> ModelBundle:
    """Return a bundle carrying only what the Tier-3 timeout path reads.

    Built by ``__new__`` rather than by loading artefacts: these tests are about the timeout
    wrapper, and requiring a 13 MB booster to test a ``wait_for`` would tie a unit test to a
    gitignored build output.
    """
    bundle = object.__new__(ModelBundle)
    object.__setattr__(bundle, "tier3", tier3)
    return bundle


class TestDegradedModeThroughTheEndpoint:
    """The phase's verify step: degraded-mode fallback demonstrated end to end.

    The unit tests above cover the timeout wrapper. These cover what a caller and an auditor
    actually see when it fires — that the request still succeeds, that the response says it
    degraded, and that the audit row records *why*.
    """

    def test_a_hanging_ring_lookup_still_returns_a_decision(
        self, client: TestClient, settings: Settings, app: Any, session: FakeSession
    ) -> None:
        """A stalled enrichment must not stall the request."""
        import time as time_module

        bundle = getattr(app.state, "model_bundle", None)
        if bundle is None:
            pytest.skip("scoring models are not loaded in this environment")

        class HangingTier3:
            """A ring lookup far slower than any budget."""

            def score(self, _transaction: TransactionFeatures) -> None:
                """Block past the timeout."""
                time_module.sleep(0.5)

        object.__setattr__(bundle, "tier3", HangingTier3())
        app.state.settings = settings.model_copy(update={"tier3_timeout_ms": 20})

        response = _score(client, settings)

        assert response.status_code == 200
        assert response.json()["degraded"] is True

    def test_the_audit_row_records_why_it_degraded(
        self, client: TestClient, settings: Settings, app: Any, session: FakeSession
    ) -> None:
        """ "A layer was missing" without "which, and why" is not a reconstructable record."""
        import time as time_module

        bundle = getattr(app.state, "model_bundle", None)
        if bundle is None:
            pytest.skip("scoring models are not loaded in this environment")

        class HangingTier3:
            """A ring lookup far slower than any budget."""

            def score(self, _transaction: TransactionFeatures) -> None:
                """Block past the timeout."""
                time_module.sleep(0.5)

        object.__setattr__(bundle, "tier3", HangingTier3())
        app.state.settings = settings.model_copy(update={"tier3_timeout_ms": 20})

        _score(client, settings)

        row = session.added[0]
        assert row.degraded is True
        assert row.degraded_reason == DEGRADED_TIER3_TIMEOUT

    def test_the_response_does_not_say_which_layer_degraded(
        self, client: TestClient, settings: Settings, app: Any
    ) -> None:
        """The caller learns the decision was degraded, not which control was unavailable.

        Naming the layer tells a caller which defence is currently down, which is an
        operational disclosure that belongs in the audit row and not in a response body.
        """
        import time as time_module

        bundle = getattr(app.state, "model_bundle", None)
        if bundle is None:
            pytest.skip("scoring models are not loaded in this environment")

        class HangingTier3:
            """A ring lookup far slower than any budget."""

            def score(self, _transaction: TransactionFeatures) -> None:
                """Block past the timeout."""
                time_module.sleep(0.5)

        object.__setattr__(bundle, "tier3", HangingTier3())
        app.state.settings = settings.model_copy(update={"tier3_timeout_ms": 20})

        body = _score(client, settings).text.lower()
        assert "tier3" not in body
        assert "degraded_reason" not in body


class TestExplainRouteDisclosure:
    """Regression cover for the Phase 7 security review's B3.

    The explain route returned `cost_estimate` — the whole of `DecisionCost.to_audit_dict()`,
    whose own docstring says it is "for the server-side audit trail only. Never a response
    body. Every field here is an evasion oracle, and together they are complete." That is a
    direct violation of the gate Phase 6 widened to cover the entire object.
    """

    def test_the_explanation_schema_carries_no_cost_field(self) -> None:
        """Structural: there is no field for a future edit to populate."""
        from app.api.schemas import ExplanationResponse

        assert "cost_estimate" not in ExplanationResponse.model_fields
        for field in (
            "expected_cost",
            "cost_if_blocked",
            "cost_if_allowed",
            "expected_saving_from_blocking",
            "assumptions",
            "amount",
        ):
            assert field not in ExplanationResponse.model_fields

    def test_the_explanation_route_requires_analyst_not_just_explain_read(
        self, client: TestClient, settings: Settings
    ) -> None:
        """Owner match must not be enough: the owner is who this must never reach.

        Gating on `explain:read` alone would have made the boundary depend on an issuance
        policy that does not exist in this repo — there is no token endpoint and nothing
        constrains which scopes a merchant is granted.
        """
        from app.core.security import SCOPE_EXPLAIN_READ

        response = client.get(
            "/audit/entry/1/explain",
            headers=auth_header(settings, account_id="acct-1", scopes=(SCOPE_EXPLAIN_READ,)),
        )
        assert response.status_code == 403

    def test_an_analyst_reaches_the_attribution(
        self, client: TestClient, settings: Settings, session: FakeSession
    ) -> None:
        """The route still has to work for the reviewer it exists for."""
        from app.core.security import SCOPE_ANALYST, SCOPE_EXPLAIN_READ

        session.rows = [_audit_row_with_attribution()]
        response = client.get(
            "/audit/entry/1/explain",
            headers=auth_header(
                settings, account_id="acct-1", scopes=(SCOPE_EXPLAIN_READ, SCOPE_ANALYST)
            ),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["top_features"][0]["feature"] == "amount_log"
        assert "cost_estimate" not in body
        assert "expected_saving_from_blocking" not in response.text


class TestScoringResponseIsNotABandOracle:
    """Regression cover for the review's B2.

    The response carried a three-value `risk_band` whose edges were published constants. The
    band is monotone in the probability, so binary search over the amount locates an edge in
    O(log n) — the docstring's claim that coarsening forced O(n) was wrong. The band is gone
    from the scoring response; the decision itself is the irreducible one-bit disclosure.
    """

    def test_the_scoring_schema_has_no_risk_band(self) -> None:
        assert "risk_band" not in ScoreResponse.model_fields

    def test_the_scoring_body_has_no_risk_band(
        self, client: TestClient, settings: Settings
    ) -> None:
        response = _score(client, settings)
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        assert response.status_code == 200
        assert "risk_band" not in response.json()
        for band in ("low", "elevated", "high"):
            assert f'"{band}"' not in response.text

    def test_no_response_schema_anywhere_carries_a_band(self) -> None:
        """The first fix removed it from /score only, and that was not enough.

        The account holder can read its own audit rows, so leaving the band on
        `AuditEntryResponse` reassembled the probe loop across two calls: post a transaction,
        read back its audit row, bisect on the amount. Magnitude is now available only as
        `risk_probability` on the analyst-only explain route.
        """
        from app.api.schemas import (
            AuditEntryResponse,
            AuditListResponse,
            ExplanationResponse,
            RingListResponse,
            ScoreResponse,
            TransactionListResponse,
        )

        for schema in (
            ScoreResponse,
            AuditEntryResponse,
            AuditListResponse,
            ExplanationResponse,
            RingListResponse,
            TransactionListResponse,
        ):
            assert "risk_band" not in schema.model_fields, f"{schema.__name__} exposes a band"


def _audit_row_with_attribution() -> Any:
    """Return an audit row carrying attribution and a cost estimate."""

    class Row:
        """One audit_log row, as stored."""

        audit_id = 1
        transaction_id = "T-1"
        account_id = "acct-1"
        decided_at = datetime(2018, 5, 5, tzinfo=UTC)
        decision = "allow"
        risk_probability = 0.01
        model_versions: dict[str, str] = {"tier1": "m1"}
        feature_version = "fv_test"
        degraded = False
        degraded_reason = None
        top_features: list[Any] = [["amount_log", 0.31]]
        # Stored, and deliberately unreachable through any response.
        cost_estimate: dict[str, Any] = {"expected_saving_from_blocking": -1.23}

    return Row()
