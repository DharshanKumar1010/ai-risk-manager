"""``POST /webhooks/razorpay/transaction`` -- score a Razorpay payment event and return a
merchant-contextualized decision.

This route has no bearer token, ever. Its only authentication is
``X-Razorpay-Signature``, an HMAC over the raw request body keyed by
``settings.razorpay_webhook_secret`` (see ``app/core/webhook_security.py``) -- the caller is
authenticated as "whoever holds the shared webhook secret", not as any merchant's own JWT. That
distinction is what makes this route's response allowed to carry ``risk_score``,
``cost_estimate`` and ``merchant_context`` when no other response on this service may -- see
``app/api/schemas.py``'s module docstring for the full argument, and ``BUILD_LOG.md``'s Phase 9
entry for the record of it having been reasoned about rather than assumed.

Everything downstream of the signature check reuses the exact functions ``POST /score`` uses --
:func:`app.core.serving.score_transaction` and :func:`app.core.audit.write_audit_record` --
rather than a second scoring implementation. The one genuinely new piece is
:func:`app.core.merchant_context.compute_merchant_context`, read after the audit row commits so
the transaction just scored is included in its own "last 100" window.

Two things this route deliberately does **not** do, both recorded in ``BUILD_LOG.md`` as
decisions rather than gaps:

- It does not persist the transaction it scores. ``POST /score`` already doesn't (a known gap),
  and this phase does not add the ``scored_transactions`` ledger that would fix it. A
  merchant's live-webhook activity therefore will not appear in its own future
  ``amount_zscore_vs_own_history``/velocity history, or in ``GET /transactions``, until a
  future phase adds that ledger.
- It does not deduplicate a redelivered payment event. Razorpay's at-least-once retry semantics
  mean the same event can arrive twice; each arrival produces its own ``audit_log`` row, exactly
  as ``audit_log.py``'s own docstring anticipates ("a transaction may be scored more than once;
  each is its own row").

**One more thing this route deliberately does, found in security review and not merely
noted.** ``notes["riskiq_account_id"]`` is signed by Razorpay but not verified by it -- see
:func:`webhook_write_session`'s docstring and :func:`_require_known_account` for the full
argument. :func:`_require_known_account` refuses to score or write anything for an
``account_id`` that is not already present in ``accounts``, narrowing (not eliminating) the
resulting cross-account exposure. This is an interim mitigation, recorded as a bounded,
accepted residual risk in ``BUILD_LOG.md``'s Phase 9 entry -- not a claim that this route has a
real merchant/customer identity binding, which it does not.
"""

import contextlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    DecisionCostBlock,
    FeedEvent,
    MerchantRiskContext,
    RazorpayPaymentEntity,
    RazorpayWebhookEnvelope,
    RazorpayWebhookResponse,
)
from app.api.score import get_bundle
from app.config import Settings
from app.core.audit import write_audit_record
from app.core.feed import FeedBroadcaster
from app.core.merchant_context import compute_merchant_context
from app.core.rate_limit import enforce_webhook_rate_limit
from app.core.security import Principal
from app.core.serving import ModelBundle, score_transaction
from app.core.webhook_security import verify_razorpay_signature
from app.data.raw_spec import SourceDataset
from app.db.session import get_scoped_session
from app.models.account import Account

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks/razorpay", tags=["webhooks"])

#: The subject recorded for this route's audit rows and RLS scoping calls. Not a real
#: authenticated identity in the JWT sense -- there is none -- but ``get_scoped_session`` takes
#: a ``Principal``, and a stable, obviously-not-a-merchant subject string makes a
#: webhook-originated audit row distinguishable in a raw table scan.
WEBHOOK_PRINCIPAL_SUBJECT = "razorpay-webhook"

#: Matches audit_log.account_id's column width (String(128), same bound ScoreRequest.account_id
#: enforces on the JWT-authenticated path) -- see verified_webhook_payload's use of it.
MAX_ACCOUNT_ID_LENGTH = 128


@dataclass(frozen=True)
class WebhookScoringInputs:
    """What :func:`verified_webhook_payload` extracts from a verified Razorpay payload --
    exactly the keyword arguments :func:`app.core.serving.score_transaction` takes."""

    transaction_id: str
    account_id: str
    event_time: datetime
    amount: Decimal
    raw_columns: dict[str, Any]


def _extract_raw_columns(entity: RazorpayPaymentEntity) -> dict[str, Any]:
    """Map the (thin) overlap between Razorpay's real payment fields and IEEE-CIS's synthetic
    raw columns.

    Most of IEEE-CIS's ~90 raw columns -- ``C1``-``C14``, ``D1``-``D15``, the identity block --
    have no counterpart in a real processor's payload and are left absent. Tier-1 already
    tolerates this: 76% of IEEE-CIS rows carry no identity record at all, and
    ``assemble_tier1_vector`` treats an absent raw column as genuinely absent, not an error. A
    stated limitation of wiring a Kaggle-trained model to a live processor, not something this
    phase fixes -- see ``BUILD_LOG.md``'s Phase 9 entry.
    """
    raw: dict[str, Any] = {}
    if entity.method is not None:
        raw["ProductCD"] = entity.method
    card = entity.card or {}
    if card.get("network") is not None:
        raw["card4"] = card["network"]
    if card.get("type") is not None:
        raw["card6"] = card["type"]
    return raw


async def _require_known_account(
    session: AsyncSession, source_dataset: SourceDataset, account_id: str
) -> None:
    """Refuse to score or write anything for an ``account_id`` this deployment does not
    already recognize.

    **Why this check exists, and what it does not fix.** ``notes["riskiq_account_id"]`` is
    signed by Razorpay but not *verified* by Razorpay -- the signature proves the bytes came
    from Razorpay, not that the account_id claim inside them is honest. Razorpay never
    interprets that field; whoever controls what a payment's ``notes`` contain (this project's
    own checkout integration, or -- if that integration passes client input through
    unsanitized -- a malicious customer of it) controls this value entirely. Without this
    check, that gives an unverified claim the power to attribute a permanent, immutable
    ``audit_log`` row to an account it does not belong to, and to read that account's
    ``merchant_context`` in the response -- a cross-account read/write with no ownership
    binding at all, found in security review of this phase (see ``BUILD_LOG.md``'s Phase 9
    entry). ``POST /score`` avoids the equivalent mistake with
    :func:`app.core.security.require_account_ownership`, which cross-checks a body-supplied
    account_id against a verified JWT claim -- this route has no JWT to cross-check against.

    This check is the interim mitigation, not the real fix. It narrows the attack from
    "attribute a decision to *any* string an attacker invents" to "attribute a decision to an
    account_id that already exists in ``accounts``" -- a bounded, attacker-uninfluenceable set
    populated only by the offline Phase 1 pipeline, never by live traffic. It does **not**
    prevent an attacker who already knows or guesses a real, existing account_id from
    targeting that specific one; closing that requires a real merchant/customer identity
    binding this project does not yet have. Accepted as a recorded, bounded residual risk for
    this phase -- see BUILD_LOG.md.

    Raises:
        HTTPException: 404, matching :func:`app.core.security.require_account_ownership`'s
            choice of status -- indistinguishable from "this account_id does not exist" rather
            than a more specific code that would help an attacker enumerate valid ones.
    """
    account = await session.get(Account, (source_dataset, account_id))
    if account is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")


async def verified_webhook_payload(request: Request) -> WebhookScoringInputs:
    """Verify the signature over the raw body, then parse it and extract scoring inputs.

    A FastAPI dependency, not a ``request.state`` side effect written by middleware: both
    session dependencies below and the route handler itself declare
    ``Depends(verified_webhook_payload)``, and FastAPI resolves a dependency's prerequisites
    before the dependent, caching the result for the rest of the request -- the same ordering
    guarantee ``enforce_rate_limit``'s docstring relies on for auth-before-limiter. A
    ``request.state``-based version would have the same hazard silently: nothing would force
    this to run before something else reads the state it sets.

    Raises:
        HTTPException: 401 on a missing or invalid signature (via
            :func:`verify_razorpay_signature`); 422 when the verified body is not a well-formed
            Razorpay payment-event envelope, or carries no
            ``notes["riskiq_account_id"]`` -- Razorpay has no field that natively means "the
            RiskIQ account this belongs to"; a merchant's checkout integration must stamp it
            into Razorpay's own arbitrary-metadata ``notes`` field at order/payment creation.
    """
    settings: Settings = request.app.state.settings
    raw_body = await request.body()
    verify_razorpay_signature(
        raw_body, request.headers.get("X-Razorpay-Signature"), settings.razorpay_webhook_secret
    )

    try:
        envelope = RazorpayWebhookEnvelope.model_validate_json(raw_body)
        entity = RazorpayPaymentEntity.model_validate(
            envelope.payload.get("payment", {}).get("entity", {})
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"error": "invalid_webhook_payload"},
        ) from exc

    account_id = entity.notes.get("riskiq_account_id")
    if not account_id or len(account_id) > MAX_ACCOUNT_ID_LENGTH:
        # Same bound as ScoreRequest.account_id (schemas.py). audit_log.account_id is
        # String(128); an unbounded value here would raise asyncpg's
        # StringDataRightTruncation from inside write_audit_record instead of a 422 at the
        # point this service actually knows the input is malformed.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "error": "missing_riskiq_account_id",
                "hint": "payload.payment.entity.notes must carry riskiq_account_id, stamped "
                "by the checkout integration at order/payment creation.",
            },
        )

    return WebhookScoringInputs(
        transaction_id=entity.id,
        account_id=account_id,
        event_time=datetime.fromtimestamp(entity.created_at, tz=UTC),
        amount=Decimal(entity.amount) / Decimal(100),
        raw_columns=_extract_raw_columns(entity),
    )


async def webhook_write_session(
    extracted: Annotated[WebhookScoringInputs, Depends(verified_webhook_payload)],
) -> AsyncIterator[AsyncSession]:
    """Delegate to :func:`get_scoped_session`'s actual row-level-security logic, with a
    :class:`Principal` built from the payload's own ``account_id``.

    **That account_id is signed, not verified, and this function does not by itself make it
    trustworthy.** An earlier version of this docstring claimed it was "trustworthy because it
    came from inside the HMAC-covered body, exactly as a JWT claim is trustworthy because it
    came from inside a signature-checked token" -- security review of this phase found that
    analogy false and it is corrected here rather than left for the next reader to repeat. A
    JWT claim is trustworthy because the *issuer* vouches for *what it asserts about the
    bearer*; Razorpay's signature vouches only for "these bytes came from Razorpay," and
    Razorpay never interprets or validates ``notes["riskiq_account_id"]`` -- it is opaque
    metadata whoever creates the payment controls. What actually bounds this is
    :func:`app.api.webhooks._require_known_account`, called in the route handler before
    anything is scored or written: RLS scoping here is necessary (a session must be scoped to
    *some* account for the INSERT's ``WITH CHECK`` policy to pass) but not sufficient on its
    own, since the application is the one setting the scope from the same unverified claim.
    See ``BUILD_LOG.md``'s Phase 9 entry for the full finding.

    Called as a plain async generator, not through FastAPI's own resolution of
    ``get_scoped_session`` (which requires a JWT-derived principal this route does not have) --
    this is what lets the route reuse the exact ``SET LOCAL`` statement rather than
    reimplementing it. ``contextlib.aclosing`` guarantees the inner generator's own cleanup
    (session close/rollback) still runs when this wrapper is closed, which a bare
    ``async for ... yield`` pass-through would not.
    """
    principal = Principal(subject=WEBHOOK_PRINCIPAL_SUBJECT, account_id=extracted.account_id)
    # get_scoped_session is declared to return AsyncIterator[AsyncSession] (the FastAPI
    # dependency contract), but its actual body is an async generator function, which is what
    # gives it the aclose() aclosing() needs -- the cast tells mypy what is already true at
    # runtime rather than widening the public signature to leak an implementation detail.
    generator = cast(AsyncGenerator[AsyncSession, None], get_scoped_session(principal))
    async with contextlib.aclosing(generator) as session_gen:
        async for session in session_gen:
            yield session


async def webhook_read_session(
    extracted: Annotated[WebhookScoringInputs, Depends(verified_webhook_payload)],
) -> AsyncIterator[AsyncSession]:
    """The read-path counterpart of :func:`webhook_write_session`, delegating to
    :func:`get_scoped_session` the same way -- not :func:`app.db.session.get_analyst_session`.

    ``merchant_context`` only ever reads data already scoped to ``extracted.account_id``
    (this account's own decision history and its own ``accounts`` baseline), never an
    estate-wide view, so there is no read this route needs that ``get_scoped_session`` cannot
    already serve. Security review of this phase flagged the earlier choice of
    ``get_analyst_session`` as a dormant risk for exactly this reason: it branches on
    ``SCOPE_ANALYST in principal.scopes`` and, when present, issues ``SET LOCAL ROLE
    riskiq_analyst`` -- unreachable today only because :data:`WEBHOOK_PRINCIPAL_SUBJECT`'s
    principal happens to always carry ``scopes=()``, an invariant nothing enforced.
    ``get_scoped_session`` has no such branch at all, which removes the risk rather than
    merely relying on it staying unreached.
    """
    principal = Principal(subject=WEBHOOK_PRINCIPAL_SUBJECT, account_id=extracted.account_id)
    generator = cast(AsyncGenerator[AsyncSession, None], get_scoped_session(principal))
    async with contextlib.aclosing(generator) as session_gen:
        async for session in session_gen:
            yield session


@router.post(
    "/transaction",
    response_model=RazorpayWebhookResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a Razorpay payment event and return merchant-contextualized risk",
    dependencies=[Depends(enforce_webhook_rate_limit)],
)
async def score_razorpay_transaction(
    extracted: Annotated[WebhookScoringInputs, Depends(verified_webhook_payload)],
    write_session: Annotated[AsyncSession, Depends(webhook_write_session)],
    read_session: Annotated[AsyncSession, Depends(webhook_read_session)],
    bundle: Annotated[ModelBundle, Depends(get_bundle)],
    request: Request,
) -> RazorpayWebhookResponse:
    """Score, audit, contextualize, and push to the live feed.

    Ordering mirrors ``POST /score``: the audit row is written and committed before anything
    else runs, so a failed write fails the request rather than producing a decision with no
    record of it. ``merchant_context`` is read only after that commit, so the transaction just
    scored is included in its own "last 100 decisions" window.
    """
    settings: Settings = request.app.state.settings
    await _require_known_account(write_session, bundle.source_dataset, extracted.account_id)
    try:
        outcome, record = await score_transaction(
            write_session,
            bundle,
            settings,
            transaction_id=extracted.transaction_id,
            account_id=extracted.account_id,
            event_time=extracted.event_time,
            amount=extracted.amount,
            raw_columns=extracted.raw_columns,
        )
    except ValueError as exc:
        logger.error("webhook scoring refused for an assembled vector: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring is unavailable",
        ) from exc

    audit_id = await write_audit_record(write_session, record)
    await write_session.commit()

    context = await compute_merchant_context(
        read_session,
        bundle.source_dataset,
        extracted.account_id,
        outcome.history_anomaly,
        outcome.decision,
    )

    # Published strictly after commit, same discipline as POST /score: the feed must never show
    # a decision that could still roll back.
    broadcaster: FeedBroadcaster | None = getattr(request.app.state, "feed_broadcaster", None)
    if broadcaster is not None:
        broadcaster.publish(
            FeedEvent(
                audit_id=audit_id,
                transaction_id=extracted.transaction_id,
                account_id=extracted.account_id,
                decided_at=record.decided_at,
                decision=outcome.decision,
                risk_probability=outcome.probability,
                amount=str(extracted.amount),
                degraded=outcome.degraded,
                model_version=outcome.model_versions["tier1"],
            ).model_dump(mode="json")
        )

    return RazorpayWebhookResponse(
        transaction_id=extracted.transaction_id,
        account_id=extracted.account_id,
        decision=outcome.decision,
        audit_id=audit_id,
        degraded=outcome.degraded,
        decided_at=record.decided_at,
        model_version=outcome.model_versions["tier1"],
        risk_score=outcome.probability,
        cost_estimate=DecisionCostBlock(**outcome.cost.to_audit_dict()),
        merchant_context=MerchantRiskContext(**asdict(context)),
        timestamp=datetime.now(UTC),
    )
