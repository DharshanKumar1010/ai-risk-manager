"""ORM model for the append-only ``audit_log`` table.

Phase 7 defines the schema. Every row records one scoring decision: the transaction id,
all four layer scores and flags, the final decision, the ``model_version`` of every layer
involved, the ``feature_version`` hash from the Phase 1 feature store, whether degraded
mode was used and why, and a timestamp.

This table is append-only by design. No UPDATE or DELETE path may exist in application
code — see ``.claude/skills/security-checklist/SKILL.md`` section 7.
"""
