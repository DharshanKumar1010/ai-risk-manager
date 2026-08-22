"""Alembic migration environment.

The DSN comes from :class:`app.config.Settings`, never from ``alembic.ini`` — the URL
carries a password and the ini file is tracked.

Importing :mod:`app.models` is what populates ``Base.metadata``: the table modules register
themselves on import, so autogeneration sees nothing without it.
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

import app.models  # noqa: F401  (imported for its Base.metadata registration side effect)
from alembic import context
from app.config import get_settings
from app.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Return the async DSN from application settings."""
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting.

    Used to review a migration's DDL before it touches an environment holding real data —
    a security-checklist section 3 requirement.
    """
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations against an established synchronous connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Open an async engine and run the migrations through it."""
    engine = create_async_engine(_database_url(), poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await engine.dispose()


def run_migrations_online() -> None:
    """Entry point for a live migration run."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
