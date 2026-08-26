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
    """Return the process-wide async engine, creating it on first use."""
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
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

    An analyst is deliberately left unscoped: the reviewer console reads across accounts. Note
    that this currently relies on application-level filtering rather than on the database — the
    connection is always ``riskiq_app`` and nothing issues ``SET ROLE riskiq_analyst``, so the
    analyst policies never apply and an analyst session sees nothing at the DB layer. That fails
    closed, and it is recorded as a Phase 8 prerequisite in ``BUILD_LOG.md``.
    A principal that is neither an analyst nor account-scoped gets a session with the setting
    left unset, and therefore sees nothing.

    ``SET LOCAL`` rather than ``SET``: the value is scoped to the surrounding transaction, so a
    pooled connection cannot carry one request's account into the next.
    """
    async with get_sessionmaker()() as session:
        try:
            if SCOPE_ANALYST not in principal.scopes and principal.account_id is not None:
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
