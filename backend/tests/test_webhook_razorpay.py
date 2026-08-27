"""``POST /webhooks/razorpay/transaction``: signature verification, the response's deliberate
disclosure exception, merchant_context, and the redelivery/persistence gaps BUILD_LOG.md
records as Phase 9 decisions rather than bugs.

See ``app/api/schemas.py``'s module docstring for the argument this file's disclosure tests
verify: this is the one route on the service allowed to return ``risk_score``, ``cost_estimate``
and ``merchant_context``, because its caller is authenticated by a shared HMAC secret rather
than a merchant's own JWT.
"""

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx2 import Response

from app.config import Settings
from app.core.merchant_context import compute_merchant_context
from app.core.serving import HistoryAnomalyFeatures
from app.core.webhook_security import INVALID_SIGNATURE_DETAIL
from tests.conftest import FakeSession, razorpay_webhook_payload, signed_webhook_body

#: Mirrors test_api_scoring.py's FORBIDDEN_RESPONSE_FIELDS, minus the three fields this one
#: route's response is deliberately authorized to carry.
STILL_FORBIDDEN_RESPONSE_FIELDS = ("top_features",)

WEBHOOK_PATH = "/webhooks/razorpay/transaction"


def _post_webhook(
    client: TestClient, settings: Settings, entity_overrides: dict[str, Any] | None = None
) -> Response:
    """Post a validly-signed webhook payload."""
    payload = razorpay_webhook_payload(**(entity_overrides or {}))
    body, headers = signed_webhook_body(settings, payload)
    return client.post(WEBHOOK_PATH, content=body, headers=headers)


@pytest.fixture(autouse=True)
def _known_account(session: FakeSession) -> None:
    """Every test in this file defaults to a recognized account_id, so tests about scoring
    behaviour are not also, incidentally, tests of the unknown-account gate.

    See ``app/api/webhooks.py``'s ``_require_known_account`` for why that gate exists at all
    (security-review finding: ``notes["riskiq_account_id"]`` is signed by Razorpay but not
    verified by it) and ``TestUnknownAccountIsRefused`` below for the gate's own coverage,
    which overrides this default explicitly rather than fighting it.
    """
    session.get_result = _Account(fraud_count=2, transaction_count=50)


class TestSignatureVerification:
    def test_a_missing_signature_is_refused(self, client: TestClient) -> None:
        payload = razorpay_webhook_payload()
        response = client.post(WEBHOOK_PATH, content=json.dumps(payload).encode("utf-8"))
        assert response.status_code == 401
        assert response.json()["detail"] == INVALID_SIGNATURE_DETAIL

    def test_a_signature_from_the_wrong_secret_is_refused(
        self, client: TestClient, settings: Settings
    ) -> None:
        wrong = settings.model_copy(
            update={"razorpay_webhook_secret": "an-entirely-different-secret-of-length-32+"}
        )
        response = _post_webhook(client, wrong)
        assert response.status_code == 401

    def test_a_tampered_body_with_a_stale_signature_is_refused(
        self, client: TestClient, settings: Settings
    ) -> None:
        payload = razorpay_webhook_payload()
        body, headers = signed_webhook_body(settings, payload)
        tampered = body.replace(b"15000", b"99999")
        response = client.post(WEBHOOK_PATH, content=tampered, headers=headers)
        assert response.status_code == 401

    def test_auth_failure_does_not_disclose_the_reason(
        self, client: TestClient, settings: Settings
    ) -> None:
        """The webhook's equivalent of item 2.5: missing vs. wrong-secret vs. tampered must be
        indistinguishable, the same property test_api_security.py pins for the JWT routes."""
        missing = client.post(
            WEBHOOK_PATH, content=json.dumps(razorpay_webhook_payload()).encode("utf-8")
        )
        wrong = settings.model_copy(
            update={"razorpay_webhook_secret": "an-entirely-different-secret-of-length-32+"}
        )
        wrong_secret = _post_webhook(client, wrong)
        assert missing.status_code == wrong_secret.status_code == 401
        assert missing.text == wrong_secret.text

    def test_a_valid_signature_reaches_scoring(
        self, client: TestClient, settings: Settings
    ) -> None:
        response = _post_webhook(client, settings)
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        assert response.status_code == 200


class TestUnknownAccountIsRefused:
    """Security-review finding, closed by ``_require_known_account``:
    ``notes["riskiq_account_id"]`` is signed by Razorpay's channel but not a claim Razorpay
    itself verifies, so without this gate any signed payload could attribute a decision to, and
    read merchant_context for, an arbitrary account it invents. This is that gate's own test --
    every other test in this file overrides the ``_known_account`` fixture's default instead."""

    def test_an_unrecognized_account_id_is_refused(
        self, client: TestClient, settings: Settings, session: FakeSession
    ) -> None:
        session.get_result = None
        response = _post_webhook(client, settings)
        assert response.status_code == 404

    def test_an_unrecognized_account_never_gets_an_audit_row(
        self, client: TestClient, settings: Settings, session: FakeSession
    ) -> None:
        """The gate runs before score_transaction/write_audit_record, not after -- an
        unrecognized account_id must never even be scored, let alone written."""
        session.get_result = None
        _post_webhook(client, settings)
        assert session.added == []

    def test_a_recognized_account_still_scores(
        self, client: TestClient, settings: Settings
    ) -> None:
        """Guard on the guard: the _known_account fixture's default must not itself be broken."""
        response = _post_webhook(client, settings)
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        assert response.status_code == 200


class TestPayloadValidation:
    def test_an_unexpected_razorpay_field_does_not_break_verification_or_scoring(
        self, client: TestClient, settings: Settings
    ) -> None:
        """extra="ignore" on the envelope/entity is deliberate -- Razorpay's real payload
        carries fields this service does not read."""
        response = _post_webhook(client, settings, {"upi": {"vpa": "someone@bank"}})
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        assert response.status_code == 200

    def test_a_missing_riskiq_account_id_note_is_refused(
        self, client: TestClient, settings: Settings
    ) -> None:
        response = _post_webhook(client, settings, {"notes": {}})
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "missing_riskiq_account_id"

    def test_an_unrecognised_event_type_is_refused(
        self, client: TestClient, settings: Settings
    ) -> None:
        payload = razorpay_webhook_payload()
        payload["event"] = "payment.refunded"
        body, headers = signed_webhook_body(settings, payload)
        response = client.post(WEBHOOK_PATH, content=body, headers=headers)
        assert response.status_code == 422


class TestResponseDisclosure:
    """The deliberate exception: risk_score, cost_estimate and merchant_context are present
    here, unlike every other response on this service -- but top_features still is not."""

    def test_the_response_carries_risk_score_cost_estimate_and_merchant_context(
        self, client: TestClient, settings: Settings
    ) -> None:
        response = _post_webhook(client, settings)
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["risk_score"] <= 1.0
        assert "expected_cost" in body["cost_estimate"]
        assert "expected_saving_from_blocking" in body["cost_estimate"]
        assert "fraud_rate_last_100" in body["merchant_context"]
        assert "decision_rationale" in body["merchant_context"]

    def test_fraud_rate_last_100_carries_its_proxy_caveat_on_the_wire(
        self, client: TestClient, settings: Settings
    ) -> None:
        """ml-evaluator finding: a caveat living only in the OpenAPI schema description is not
        one a caller reading raw JSON ever sees. fraud_rate_basis puts it in the payload."""
        response = _post_webhook(client, settings)
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        assert response.status_code == 200
        assert (
            response.json()["merchant_context"]["fraud_rate_basis"]
            == "decision_proxy_no_ground_truth_label"
        )

    def test_top_features_is_still_absent(self, client: TestClient, settings: Settings) -> None:
        """Only risk_score/cost_estimate/merchant_context were reasoned about and authorized --
        attribution is a different, worse oracle. See schemas.py's module docstring."""
        response = _post_webhook(client, settings)
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        assert response.status_code == 200
        body = response.text.lower()
        for field in STILL_FORBIDDEN_RESPONSE_FIELDS:
            assert field not in body, f"{field} leaked into the webhook response"

    def test_there_is_no_confidence_field(self, client: TestClient, settings: Settings) -> None:
        """Considered and dropped -- see BUILD_LOG.md's Phase 9 entry and RazorpayWebhookResponse's
        docstring: nothing this system produces measures it distinctly from risk_score."""
        response = _post_webhook(client, settings)
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        assert "confidence" not in response.json()

    def test_an_unsigned_caller_never_sees_any_of_the_authorized_fields(
        self, client: TestClient
    ) -> None:
        response = client.post(
            WEBHOOK_PATH, content=json.dumps(razorpay_webhook_payload()).encode("utf-8")
        )
        assert response.status_code == 401
        body = response.text.lower()
        for field in ("risk_score", "cost_estimate", "merchant_context", "expected_cost"):
            assert field not in body


class TestAuditIsWritten:
    def test_a_successful_call_commits_exactly_one_audit_row(
        self, client: TestClient, settings: Settings, session: FakeSession
    ) -> None:
        response = _post_webhook(client, settings)
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        assert response.status_code == 200
        assert len(session.added) == 1
        assert session.committed is True

    def test_the_audit_row_is_attributed_to_the_notes_account_id(
        self, client: TestClient, settings: Settings, session: FakeSession
    ) -> None:
        response = _post_webhook(client, settings, {"notes": {"riskiq_account_id": "acct-42"}})
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        assert session.added[0].account_id == "acct-42"


class TestRedeliveryIsNotDeduplicated:
    """BUILD_LOG.md records this as a Phase 9 decision, not a gap: Razorpay's at-least-once
    retry semantics mean the same event can arrive twice, and audit_log.py's own docstring
    already anticipates "a transaction may be scored more than once; each is its own row"."""

    def test_the_same_payload_posted_twice_produces_two_audit_rows(
        self, client: TestClient, settings: Settings, session: FakeSession
    ) -> None:
        first = _post_webhook(client, settings)
        if first.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        second = _post_webhook(client, settings)
        assert first.status_code == second.status_code == 200
        assert first.json()["audit_id"] != second.json()["audit_id"]
        assert len(session.added) == 2


class TestRateLimiting:
    def test_the_limiter_is_consulted(
        self, app: FastAPI, client: TestClient, settings: Settings
    ) -> None:
        response = _post_webhook(client, settings)
        if response.status_code == 503:
            pytest.skip("scoring models are not loaded in this environment")
        limiter = app.state.rate_limiter
        assert any(identity.startswith("ip:") for identity in limiter.calls)

    def test_no_limiter_installed_fails_closed(
        self, app: FastAPI, client: TestClient, settings: Settings
    ) -> None:
        app.state.rate_limiter = None
        response = _post_webhook(client, settings)
        assert response.status_code == 503


class TestMerchantContextComputation:
    """Unit tests against compute_merchant_context directly. A dedicated stub, not the shared
    FakeSession: this function issues two different reads (an audit_log decision history and an
    accounts .get()) against the same session, and FakeSession answers every read from one
    shared `rows` list -- it cannot represent "these two reads return different shapes" the way
    a real Postgres session can. See app/core/merchant_context.py for what each read is for.
    """

    def _history(self, **overrides: Any) -> HistoryAnomalyFeatures:
        base = dict(
            amount_zscore_vs_own_history=None,
            velocity_count_1h=None,
            velocity_count_24h=None,
            velocity_count_7d=None,
            prior_velocity_count_1h_mean=None,
            prior_velocity_count_1h_std=None,
        )
        base.update(overrides)
        return HistoryAnomalyFeatures(**base)

    async def test_fraud_rate_last_100_counts_review_and_block_not_allow(self) -> None:
        session = _MerchantContextSession(
            decisions=["allow", "review", "block", "allow"], account=None
        )
        context = await compute_merchant_context(
            session, "ieee_cis", "acct-1", self._history(), "allow"
        )
        assert context.decisions_considered == 4
        assert context.fraud_rate_last_100 == pytest.approx(0.5)

    async def test_a_new_merchant_with_no_history_falls_back_to_baseline_stats(self) -> None:
        session = _MerchantContextSession(decisions=[], account=None)
        context = await compute_merchant_context(
            session, "ieee_cis", "acct-new", self._history(), "allow"
        )
        assert context.decisions_considered == 0
        assert context.fraud_rate_last_100 == 0.0
        assert context.baseline_fraud_rate is None
        assert context.baseline_transaction_count is None

    async def test_baseline_fraud_rate_is_computed_from_the_accounts_row_when_present(self) -> None:
        session = _MerchantContextSession(
            decisions=[], account=_Account(fraud_count=3, transaction_count=120)
        )
        context = await compute_merchant_context(
            session, "ieee_cis", "acct-1", self._history(), "allow"
        )
        assert context.baseline_fraud_rate == pytest.approx(3 / 120)
        assert context.baseline_transaction_count == 120

    async def test_amount_anomaly_fires_past_two_sigma(self) -> None:
        session = _MerchantContextSession(decisions=[], account=None)
        context = await compute_merchant_context(
            session,
            "ieee_cis",
            "acct-1",
            self._history(amount_zscore_vs_own_history=2.5),
            "review",
        )
        assert context.amount_anomaly is True
        assert "Amount unusual" in context.decision_rationale

    async def test_amount_anomaly_does_not_fire_within_two_sigma(self) -> None:
        session = _MerchantContextSession(decisions=[], account=None)
        context = await compute_merchant_context(
            session, "ieee_cis", "acct-1", self._history(amount_zscore_vs_own_history=1.0), "allow"
        )
        assert context.amount_anomaly is False

    async def test_velocity_anomaly_is_a_zscore_against_the_accounts_own_prior_counts(
        self,
    ) -> None:
        session = _MerchantContextSession(decisions=[], account=None)
        # 10 in the trailing hour, against a prior history averaging 2 +/- 1 -- a clear spike.
        context = await compute_merchant_context(
            session,
            "ieee_cis",
            "acct-1",
            self._history(
                velocity_count_1h=10.0,
                prior_velocity_count_1h_mean=2.0,
                prior_velocity_count_1h_std=1.0,
            ),
            "review",
        )
        assert context.velocity_zscore_1h == pytest.approx(8.0)
        assert context.velocity_anomaly is True

    async def test_velocity_anomaly_is_none_without_enough_prior_history(self) -> None:
        session = _MerchantContextSession(decisions=[], account=None)
        context = await compute_merchant_context(
            session, "ieee_cis", "acct-1", self._history(velocity_count_1h=10.0), "allow"
        )
        assert context.velocity_zscore_1h is None
        assert context.velocity_anomaly is False

    async def test_the_rationale_names_a_block_decision(self) -> None:
        session = _MerchantContextSession(decisions=[], account=None)
        context = await compute_merchant_context(
            session, "ieee_cis", "acct-1", self._history(), "block"
        )
        assert "Blocked" in context.decision_rationale


class _FakeScalarResult:
    """The subset of a SQLAlchemy Result compute_merchant_context calls."""

    def __init__(self, rows: list[str]) -> None:
        self._rows = rows

    def scalars(self) -> "_FakeScalarResult":
        return self

    def all(self) -> list[str]:
        return list(self._rows)


class _Account:
    """The two attributes compute_merchant_context reads off an accounts row."""

    def __init__(self, fraud_count: int, transaction_count: int) -> None:
        self.fraud_count = fraud_count
        self.transaction_count = transaction_count


class _MerchantContextSession:
    """A session stub scoped to exactly what compute_merchant_context calls: one execute() for
    the audit_log decision history, one get() for the accounts baseline."""

    def __init__(self, decisions: list[str], account: _Account | None) -> None:
        self._decisions = decisions
        self._account = account

    async def execute(self, _statement: Any) -> _FakeScalarResult:
        return _FakeScalarResult(self._decisions)

    async def get(self, _model: Any, _key: Any) -> _Account | None:
        return self._account
