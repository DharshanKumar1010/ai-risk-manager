---
name: security-checklist
description: The RiskIQ security bar. Invoke before any commit touching auth, database access, request handling, or external payloads, and as the gate on Phases 7, 9 and 10. Expands the non-negotiables in CLAUDE.md into checkable items.
---

# RiskIQ Security Checklist

This is a fraud-detection product. Shipping it insecurely is thematically disqualifying,
not merely bad practice. Every item below is pass/fail — there is no "mostly."

Work through the checklist top to bottom against the current diff. For each item, report
**PASS**, **FAIL**, or **N/A (reason)**. Never report PASS on an item you did not actually
verify by reading the code.

## 1. Secrets

- [ ] No secret, key, token, DSN, or password appears in tracked source.
- [ ] Configuration reads from `.env` only; `.env` is in `.gitignore`.
- [ ] `.env.example` lists every required variable with a **placeholder**, never a real value.
- [ ] `git log -p` contains no secret in history, not just the working tree. A secret that
      was committed and later deleted is still leaked — it must be rotated, not just removed.
- [ ] The trufflehog pre-commit hook is present and blocking (not `--fail-on-error false`).

## 2. Authentication and authorization

- [ ] Every scoring endpoint and every write endpoint requires server-side auth.
- [ ] Roles, permissions and account ownership are resolved **server-side from the token**.
      A client-supplied `role`, `user_id`, `account_id`, or `is_admin` field is never trusted
      as an authorization input.
- [ ] JWT verification checks signature, expiry, issuer and audience. Never
      `decode(..., options={"verify_signature": False})` outside a test fixture.
- [ ] Ownership is enforced on every read of account-scoped data — an authenticated user
      must not be able to read another account's transactions by changing an ID in the URL.
- [ ] Auth failures return 401/403 without leaking whether the resource exists.

## 3. Database

- [ ] Row-Level Security is **enabled and forced** on every table holding transaction or
      account data. `ALTER TABLE ... ENABLE ROW LEVEL SECURITY` alone is insufficient for a
      table owner — `FORCE ROW LEVEL SECURITY` is required.
- [ ] Every RLS-enabled table has at least one policy. RLS with no policy denies all access
      and will look like a bug; RLS never enabled is a data-exposure hole.
- [ ] SQLAlchemy ORM or bound parameters only. No f-string, `%`, `.format()`, or `+`
      concatenation anywhere near SQL, including in migrations and analysis notebooks.
- [ ] The application database user is not a superuser and does not own the tables it reads
      (superusers and table owners bypass RLS).
- [ ] Migrations are reviewed for destructive operations before running against any
      environment holding real data.

## 4. Input handling

- [ ] Every request body is a Pydantic model with `model_config = ConfigDict(extra="forbid")`.
- [ ] Numeric fields that feed model scoring have explicit bounds — an unbounded `amount`
      is both a validation hole and a model-poisoning vector.
- [ ] Path and query parameters are typed and validated, not passed through as raw strings.
- [ ] Error responses never echo the raw input or a stack trace back to the caller.
- [ ] Externally originated payloads (webhooks) are signature-verified **before** parsing,
      using a constant-time comparison (`hmac.compare_digest`), never `==`.

## 5. Rate limiting and availability

- [ ] Redis-backed rate limiting on every public endpoint.
- [ ] The limiter fails **closed** on a Redis outage for auth-sensitive endpoints — a
      limiter that silently allows everything when its backing store is down is not a limiter.
- [ ] Expensive paths (graph queries, model scoring) have an explicit timeout.
- [ ] Degraded-mode fallbacks record *that* they degraded and *why* in the audit row.

## 6. Dependencies

- [ ] `pip-audit` clean of high/critical findings.
- [ ] `npm audit --audit-level=high` clean.
- [ ] Both run in CI and fail the build — not advisory-only.

## 7. Audit trail

- [ ] Every scoring decision passes through `app/core/audit.py`. No endpoint writes a
      decision by any other path.
- [ ] Audit rows are append-only: no UPDATE or DELETE path exists in application code.
- [ ] Each row records the model_version of every layer involved and the feature_version
      hash, so a past decision can be reconstructed exactly.

## 8. Defense-only (track disqualification rule)

- [ ] Nothing in the repo can be repurposed to **generate**, **automate**, or **evade**
      fraud detection.
- [ ] No synthetic-fraud generator is exposed through the API or shipped as a runnable
      user-facing tool. Fixtures used to seed a demo live in test/eval code, are clearly
      labelled, and are not reachable from a deployed endpoint.
- [ ] No endpoint reveals the decision threshold, per-feature weights, or "what would have
      made this pass" to an unauthenticated caller — that is an evasion oracle.
- [ ] If a piece of code could plausibly double as an attack tool: **stop and flag it to the
      user rather than building it.** This overrides any build instruction.

## Reporting format

```
## Security checklist — <scope>
| # | Item | Verdict | Evidence |
|---|------|---------|----------|
| 2.2 | Client-supplied role not trusted | FAIL | app/api/score.py:41 reads body.role |

Blocking findings: N
```

Report blocking findings explicitly. A phase gated on this checklist does not pass with any
FAIL outstanding.
