"""Async SQLAlchemy engine and session factory.

The engine is created lazily and cached, so importing this module never opens a
connection. That keeps unit tests and CI lint runs free of a live database dependency.
"""

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.core.security import SCOPE_ANALYST, CurrentPrincipal


@lru_cache
def get_engine() -> AsyncEngine:
    """Return the process-wide async engine, creating it on first call."""
    settings = get_settings()
    url = settings.database_url
    # Render issues postgres:// or postgresql:// — asyncpg needs +asyncpg scheme
    url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return create_async_engine(
        url,
        echo=settings.database_echo,
        pool_pre_ping=True,
    )


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide async session factory."""
    return async_sessionmaker(
        bind=get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield a database session for the lifetime of one request.

    Used as a FastAPI dependency. The session is rolled back and closed on exit, so a
    handler that raises never leaves a partial transaction behind.
    """
    async with get_sessionmaker()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def get_scoped_session(
    principal: CurrentPrincipal,
) -> AsyncIterator[AsyncSession]:
    """Yield a session scoped to the caller's account, for row-level security.

    The Phase 1 policies filter on ``current_setting('app.current_account_id', true)``, which
    is NULL when unset — so the comparison is NULL, so nothing is visible and nothing may be
    written. That is the fail-closed property, and it means this dependency is what turns those
    policies from definitions into a control.

    The setting comes from the **verified token** and from nowhere else. Reading it from a
    request body or query parameter would make row-level security a suggestion.

    **This dependency never issues ``SET ROLE``, on purpose -- see** :func:`get_analyst_session`
    **for why the two are kept apart.** The consequence for the one route that uses this
    dependency, ``POST /score``, is worth being explicit about. Scoping here is keyed on
    ``principal.account_id`` alone, with **no analyst-scope exclusion**: an earlier version
    skipped the ``set_config`` call whenever a principal held ``analyst`` scope, on the
    reasoning that analysts are handled elsewhere. That reasoning did not hold on a write. A
    principal can legitimately carry ``analyst`` scope *and* a matching ``account_id`` *and*
    ``score:write`` -- :func:`app.core.security.require_account_ownership` has no analyst
    branch, so ownership is already verified by exact match before this dependency runs, and
    analyst scope is irrelevant to whether the write should be allowed. Excluding such a
    principal from scoping left ``app.current_account_id`` unset, which made
    ``audit_log_app_insert_own_account``'s ``WITH CHECK`` compare against NULL and refuse the
    INSERT -- a bare 500 on ``POST /score`` for any dual-scoped token, caught by
    ``tests/test_db_session.py`` rather than found live.

    A principal with no ``account_id`` at all -- pure analyst or otherwise -- gets a session
    with the setting left unset, and therefore writes nothing: ``require_account_ownership``
    already refuses such a principal before this dependency is reached, so this is defence in
    depth rather than the primary control.

    ``SET LOCAL`` rather than ``SET``: the value is scoped to the surrounding transaction, so a
    pooled connection cannot carry one request's account into the next.
    """
    async with get_sessionmaker()() as session:
        try:
            if principal.account_id is not None:
                await session.execute(
                    # A bound parameter, not an f-string. ``SET LOCAL`` does not accept a
                    # placeholder directly, so the value goes through ``set_config``, which is
                    # the function form and takes one.
                    text("SELECT set_config('app.current_account_id', :account_id, true)"),
                    {"account_id": principal.account_id},
                )
            yield session
        except Exception:
            await session.rollback()
            raise


#: ``SET ROLE`` accepts no bound parameter -- there is no placeholder form of it, the way
#: ``set_config`` gives ``SET LOCAL`` one. This is not string concatenation: the role name is
#: a fixed module constant, never request input, so there is nothing here for a caller to
#: influence. Migration 0003 is what makes this safe to run at all -- ``riskiq_app``'s
#: membership in ``riskiq_analyst`` is granted ``WITH INHERIT FALSE``, so nothing changes for
#: a session that never executes this statement.
_SET_ANALYST_ROLE = "SET LOCAL ROLE riskiq_analyst"


async def get_analyst_session(
    principal: CurrentPrincipal,
) -> AsyncIterator[AsyncSession]:
    """Yield a session scoped for the reviewer console's estate-wide reads.

    **Deliberately a second function, not a branch added to** :func:`get_scoped_session`.
    The two look mergeable -- both resolve one setting from the principal and run it before
    yielding -- and merging them is the refactor that would reopen the exact gap this function
    closes. ``get_scoped_session`` is also used by ``POST /score``, and a principal can
    legitimately hold both ``analyst`` and ``score:write`` with a matching ``account_id`` --
    :func:`app.core.security.require_account_ownership` has no analyst branch, so that
    combination passes ownership on the merits and reaches the write path. If that path ran
    ``SET LOCAL ROLE riskiq_analyst``, the INSERT into ``audit_log`` a scoring call performs
    would fail outright: ``riskiq_analyst`` holds ``SELECT`` only, on all three tables, by
    design (0002's grants). Scoring would return a bare 500 for that principal, and the
    two-token demo -- an analyst token proving what a merchant token cannot see -- would break
    on its first analyst-flavoured write. Keeping this a separate dependency, wired only into
    the read routes (``/transactions``, ``/audit/{transaction_id}``,
    ``/audit`` (list), ``/audit/entry/{audit_id}/explain``, and any future analyst-only read),
    makes that failure mode unreachable rather than merely untested.

    A non-analyst principal here gets the same account-scoping ``get_scoped_session`` gives:
    this function serves every route the reviewer console reads from, including the ones a
    non-analyst caller may legitimately reach with an ownership check rather than the analyst
    bypass -- there is no route that mounts *only* for analysts except ``/rings``.

    ``SET LOCAL`` is transaction-scoped, matching the reasoning in :func:`get_scoped_session`:
    it cannot outlive the request's transaction, and ``pool_reset_on_return`` (SQLAlchemy's
    default, ``'rollback'``) means an assumed role can never survive a connection returning to
    the pool for reuse by an unrelated request.
    """
    async with get_sessionmaker()() as session:
        try:
            if SCOPE_ANALYST in principal.scopes:
                await session.execute(text(_SET_ANALYST_ROLE))
            elif principal.account_id is not None:
                await session.execute(
                    text("SELECT set_config('app.current_account_id', :account_id, true)"),
                    {"account_id": principal.account_id},
                )
            yield session
        except Exception:
            await session.rollback()
            raise
