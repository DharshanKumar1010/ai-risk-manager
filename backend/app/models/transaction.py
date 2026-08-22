"""ORM model for the ``transactions`` table.

Phase 1 defines the schema: the engineered feature columns, the ``split`` column
(train/val/test, assigned chronologically), and the ``feature_version`` hash linking each
row to the feature definition that produced it. This table is the single source of truth
every downstream tier reads from.

Row-Level Security is required on this table — see
``.claude/skills/security-checklist/SKILL.md`` section 3.
"""
