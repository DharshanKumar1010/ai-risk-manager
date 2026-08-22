# RiskIQ Build Log

Running record of what shipped, what was deferred, and what is known to be missing.
This becomes the **"Build Challenges & Technical Obstacles"** answer on the Razorpay
submission form, so entries are written honestly — real obstacles and how they were
solved read as more credible to a panel than a smoothed-over success narrative.

| Phase | Status | Notes | Known Gaps |
|-------|--------|-------|------------|
| 0 — Scaffolding & environment | **Complete, verified** | Monorepo, `.claude/` skills + agents, CI workflow, 4-service Docker stack booting healthy. All six verification items passed. Detail below. | Dependency ranges not yet exact-pinned; CI never executed on GitHub; audit writer raises by design until Phase 7 |
| 1 — Data pipeline & features | Not started | | |
| 2 — Tier-1 anomaly layer | Not started | | |
| 3 — Tier-2 behavioural layer | Not started | | |
| 4 — Tier-3 graph layer | Not started | | |
| 5 — Meta-learner + SHAP | Not started | | |
| 6 — Causal cost layer | Not started | | |
| 7 — Backend, audit, security | Not started | | |
| 8 — React dashboard | Not started | | |
| 9 — Razorpay webhook integration | Not started | | |
| 10 — Testing, CI, deployment | Not started | | |
| 11 — Docs, diagram, pitch, submission | Not started | | |

---

## Phase 0 — detail

**Verified end state.** All four services healthy (`postgres`, `redis`, `backend`,
`frontend`); `GET /health` → 200 in 4.7ms; Vite serves the shell at `:5173`;
`ruff` clean, `black --check` clean on 28 files, `mypy --strict` clean on 24 files,
`pytest` 10 passed; `npm run lint` and `npm test` (1) pass; `pre-commit run --all-files`
exit 0 across 13 hooks; `actionlint` exit 0 on the CI workflow; `pip-audit` and
`npm audit --audit-level=high` both report zero vulnerabilities.

### Obstacles hit and how they were solved

**`.claude/` was referenced everywhere and existed nowhere.** `CLAUDE.md` and four later
phase gates cite `security-checklist` and `ml-evaluation-standards` skills plus
`security-reviewer` and `ml-evaluator` subagents. None existed, and `code-reviewer.md`
sat at the repo root where it was never registered as a subagent. Phase 0's build prompt
said `.claude/` was "already present — do not overwrite", which would have preserved the
gap and blocked the Phase 2, 5, 7 and 10 gates. Authored all four from the non-negotiables
already written in `CLAUDE.md`, and `git mv`d `code-reviewer.md` into `.claude/agents/`.

**Vite 8's `react-ts` template ships without `strict`.** `CLAUDE.md` requires TypeScript
strict mode. The generated `tsconfig.app.json` and `tsconfig.node.json` enable only
`noUnusedLocals`/`noUnusedParameters`. Enabled `strict`, plus `noUncheckedIndexedAccess`
and `exactOptionalPropertyTypes`, and changed `npm run lint` to `oxlint && tsc -b --noEmit`
so a type error actually fails the lint gate instead of only failing `build`.

**Frontend container reported unhealthy while serving perfectly.** The healthcheck probed
`http://localhost:5173`; busybox `wget` resolves `localhost` to `::1` first, and Vite binds
IPv4 only, so the probe got connection-refused against a healthy server. Changed the probe
to `127.0.0.1`.

**Two latent defects surfaced only because `filterwarnings = ["error"]` is set in
pytest config.** Worth keeping for that reason alone:
- Starlette now deprecates `httpx` for its `TestClient` in favour of `httpx2`. Swapped the
  dependency rather than filtering the warning.
- PyJWT raised `InsecureKeyLengthWarning`: the HMAC key was 21 bytes, below the 32-byte
  minimum RFC 7518 §3.2 requires for HS256. PyJWT only *warns*. For a fraud-detection
  service that is not enough, so `Settings.jwt_secret_key` now enforces `min_length=32`
  and refuses to construct below it, the placeholder was lengthened to match, and
  `test_short_signing_key_is_rejected` locks the behaviour in.

**The secret scanner was scanning nothing useful.** `.trufflehog-exclude` patterns were
`^`-anchored to repo-relative paths, but trufflehog matches the absolute path it sees
inside the hook container (`/src/...`). Result: 17 findings, every one from
`node_modules/` or `.mypy_cache/`, and zero coverage confidence. Rewrote the patterns as
unanchored fragments; the scan now passes clean in ~2s instead of 16s over 126MB.

**`pre-commit run --all-files` was silently passing by doing nothing.** It only considers
files git knows about, and the entire scaffold was untracked — so `ruff`, `black`,
`check-yaml`, `check-json` and `check-toml` all reported "(no files to check) Skipped",
which reads as success at a glance. Staged the 68 files first; the hooks then found real
issues. This one is worth remembering: a green pre-commit run on an untracked tree proves
nothing.

**`check-json` rejects `tsconfig.*.json`.** Those files are JSONC and legitimately contain
comments. Excluded them from that hook; their validity is enforced by `tsc -b` in
`npm run lint` instead.

**`pip-audit` found 6 advisories, and our own version caps were blocking the fixes.**
`black` (PYSEC-2026-2120/2121), `pytest` (PYSEC-2026-1845) and `setuptools`
(PYSEC-2026-3447) were all vulnerable, and the `<26.0` / `<9.0` upper bounds in
`requirements.txt` made the patched versions unreachable. Raised the floors
(`black>=26.3.1`, `pytest>=9.0.3`) and pinned `setuptools>=83.0.0` forward even though it
is not a direct dependency — it ships vulnerable in the `python:3.11-slim` base image.
Also changed the CI step from `pip-audit --requirement requirements.txt` to auditing the
installed environment, which is what catches base-image packages at all. Re-verified the
whole backend on the upgraded toolchain: still clean. `pre-commit autoupdate` then synced
the hook revisions so pre-commit's `black` matches the container's rather than drifting a
major version behind.

**Windows/Linux line endings.** Git warned it would convert 60+ files to CRLF on checkout,
which puts CRLF into Linux containers where a CRLF entrypoint fails with an opaque
"not found". Added `.gitattributes` forcing `eol=lf`.

**Ruff `B008` on FastAPI's `Depends` default.** Resolved with the modern
`Annotated[..., Depends(...)]` form rather than adding `extend-immutable-calls` to suppress
the rule — the lint stays honest for Phase 7's routers.

### Known gaps leaving Phase 0

- **Dependency versions are bounded ranges, not exact pins,** so the scaffold builds
  reliably on a cold clone. Phase 10 pins exact versions and generates a lockfile.
- **The CI workflow has never run on GitHub.** It is `actionlint`-clean and every step was
  executed locally against the running stack, but the first real run happens on push.
- **`write_audit_record` raises `NotImplementedError` by design.** Phase 7 implements
  persistence. Raising rather than no-op'ing is deliberate: a silent audit writer would let
  Phase 7 ship a scoring endpoint that appears audited and is not.
- **No Alembic migrations yet** — the declarative base exists, no tables are defined.
  Phase 1 defines `transactions`/`accounts`, Phase 7 defines `audit_log` and the RLS
  policies. **RLS is not enabled anywhere yet** because no table exists to enable it on.
- **The backend image includes dev/CI tooling.** Phase 10 adds a slim production target.
- **`pre-commit` is installed to the dev machine's user site and is not on PATH**; invoke
  it as `python -m pre_commit` until that is fixed.
- **No dependency-health or `/status` endpoint.** `/health` is liveness only and
  deliberately does not check Postgres or Redis — a liveness probe that fails when a
  downstream is down causes the orchestrator to restart a healthy container. Phase 10 adds
  the readiness/status endpoint.
