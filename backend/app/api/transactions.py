"""``GET /transactions`` — scored transaction history.

Reads are account-scoped: the account filter comes from the verified principal, never from a
client-supplied parameter. There is deliberately no ``account_id`` query parameter on this
route — not one that is validated, not one that is ignored. A parameter that exists and is
overridden server-side still invites a reader to believe it does something, and the next person
to touch the route has to rediscover that it does not.

Defence in depth, and both halves are real. The ORM query filters on the principal's account,
*and* the session runs with ``app.current_account_id`` set so the row-level security policy
filters too. Either alone would be sufficient on a correct day; together, a mistake in one is
caught by the other.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import TransactionListResponse, TransactionSummary
from app.core.rate_limit import enforce_rate_limit
from app.core.security import (
    SCOPE_TRANSACTIONS_READ,
    AccountScope,
    Principal,
    account_filter,
    require_scopes,
)
from app.db.session import get_scoped_session
from app.models.transaction import Transaction

router = APIRouter(prefix="/transactions", tags=["transactions"])

#: Page size bounds. An unbounded limit is a denial-of-service lever on an indexed range scan.
DEFAULT_LIMIT = 50
MAX_LIMIT = 200


@router.get(
    "",
    response_model=TransactionListResponse,
    summary="List scored transactions visible to the caller",
    dependencies=[Depends(enforce_rate_limit)],
)
async def list_transactions(
    principal: Annotated[Principal, Depends(require_scopes(SCOPE_TRANSACTIONS_READ))],
    session: Annotated[AsyncSession, Depends(get_scoped_session)],
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0, le=100_000)] = 0,
) -> TransactionListResponse:
    """Return the caller's most recent transactions, newest first.

    Args:
        principal: The verified caller, holding ``transactions:read``.
        session: Session already scoped for row-level security.
        limit: Page size, bounded.
        offset: Rows to skip, bounded.

    Returns:
        A page of transactions. An account-scoped caller sees only its own; an analyst sees
        across the estate, which is what the reviewer console needs.
    """
    scope = account_filter(principal)
    if scope is AccountScope.NOTHING:
        # A principal with no account scope and no analyst grant may see nothing. Returning
        # early rather than running an unfiltered query is the fail-closed reading: the bug
        # this replaced turned "owns no account" into "sees every account".
        return TransactionListResponse(transactions=(), count=0)

    statement = select(Transaction).order_by(Transaction.event_time.desc())
    if scope is not AccountScope.UNRESTRICTED:
        statement = statement.where(Transaction.account_id == scope)
    statement = statement.limit(limit).offset(offset)

    rows = (await session.execute(statement)).scalars().all()
    summaries = tuple(
        TransactionSummary(
            transaction_id=row.transaction_id,
            account_id=row.account_id,
            event_time=row.event_time,
            amount=row.amount,
            transaction_type=row.transaction_type,
        )
        for row in rows
    )
    return TransactionListResponse(transactions=summaries, count=len(summaries))
