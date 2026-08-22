"""ORM table models and ML tier models.

One file per DB table and one file per architecture layer — per CLAUDE.md's repo map
these are never merged, because each tier is independently trainable, versioned and
scoreable.

Table modules are implemented in Phase 1 (``transaction``, ``account``) and Phase 7
(``audit_log``). Tier modules are implemented in Phases 2-6.
"""
