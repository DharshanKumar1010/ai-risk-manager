"""ORM table models and ML tier models.

One file per DB table and one file per architecture layer — per CLAUDE.md's repo map
these are never merged, because each tier is independently trainable, versioned and
scoreable.

The table modules are re-exported here for their side effect: importing them is what
registers them against ``Base.metadata``, and therefore what makes them visible to Alembic
autogeneration. ``alembic/env.py`` imports this package for exactly that reason.

``transaction`` and ``account`` are implemented (Phase 1); ``audit_log`` arrives in Phase 7
and joins this list then. Tier modules are implemented in Phases 2-6.
"""

from app.models.account import Account
from app.models.transaction import Transaction

__all__ = ["Account", "Transaction"]
