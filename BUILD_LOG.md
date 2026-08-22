# RiskIQ Build Log

Running record of what shipped, what was deferred, and what is known to be missing.
This becomes the **"Build Challenges & Technical Obstacles"** answer on the Razorpay
submission form, so entries are written honestly — real obstacles and how they were
solved read as more credible to a panel than a smoothed-over success narrative.

| Phase | Status | Notes | Known Gaps |
|-------|--------|-------|------------|
| 0 — Scaffolding & environment | **Complete, verified** | Monorepo, `.claude/` skills + agents, CI workflow, 4-service Docker stack booting healthy. All six verification items passed. Detail below. | Dependency ranges not yet exact-pinned; CI never executed on GitHub; audit writer raises by design until Phase 7 |
| 1 — Data pipeline & features | **Complete, verified** | Full pipeline runs end to end over both corpora: 590,540 IEEE-CIS + 2,770,409 PaySim rows engineered, split chronologically, and persisted to Postgres and parquet. 157 tests green including a hard leakage check. Detail below. | RLS defined but **not yet effective** (app still connects as superuser — Phase 7); IEEE-CIS `V1`-`V339` carried to parquet but not yet reduced or used (Phase 2); PaySim class balance is non-stationary and name-chaining is measured at 0% — both constrain Phase 4 |
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

---

## Phase 1 — detail (in progress)

**The reconciliation problem.** No single public dataset has both rich transaction/identity
features and account-to-account network structure, so the project uses two: IEEE-CIS as the
transaction spine (Tiers 1, 2, 5, 6) and PaySim for the Tier-3 graph layer. The design that
settles it: **one canonical schema, two adapters, one table — but models never train across
sources.** `source_dataset` is a first-class discriminator, splits are computed per source,
and no fitted score crosses between corpora. Their clocks share no origin, so concatenating
them into one timeline would fabricate an ordering that does not exist; their base rates
differ by 27x, so a pooled metric would be uninterpretable.

Tier-3 bridges the two by being written against an abstract `EntityGraph` and instantiated
twice: on PaySim's money-flow graph, where ring-level precision/recall is measured against
real labels, and on an IEEE-CIS shared-entity graph, which produces the per-transaction ring
score Phase 5's meta-learner needs. One algorithm, two real graphs, no transported score.

**Every dataset figure in the plan is now verified, not assumed.** `app/data/validate_raw.py`
scanned both corpora and matched the specification exactly:

| File | Rows | Positives | Base rate |
|------|------|-----------|-----------|
| `train_transaction.csv` | 590,540 | 20,663 | 3.4990% |
| `train_identity.csv` | 144,233 | — | — |
| PaySim log | 6,362,620 | 8,213 | 0.1291% |

IEEE-CIS spans 182.0 days; the identity sidecar joins onto 24.4% of transactions with zero
orphaned rows. Critically, **all 8,213 PaySim fraud rows are CASH_OUT (4,116) or TRANSFER
(4,097)** — the claim the entire Tier-3 scope filter rests on, now measured rather than
asserted. The validator treats a violation of it as an error, not a warning.

### Obstacles hit and how they were solved

**A normalisation bug that only a test would have caught.** The IEEE-CIS release is
inconsistent: `train_identity.csv` uses `id_01`, `test_identity.csv` uses `id-01`. The
validator normalises hyphens on read — but it was passing the *normalised* names to pandas'
`usecols`, which matches against the *raw* header. A hyphenated file therefore passed the
column check and then crashed mid-scan. Fixed by keeping a raw-to-normalised mapping and
selecting on raw names. Worth recording because the header check passed throughout: the
failure was only reachable by actually reading the body.

**`data/` was never mounted into the backend container.** Phase 0's `docker-compose.yml`
mounted only `./backend:/srv`, so the datasets — which are gitignored and far too large to
bake into an image — were invisible to the service that needs them. Added a `./data:/data`
mount and a `DATA_DIR` setting that resolves to the repo's `data/` on a host checkout and
`/data` in the container.

**Alembic did not exist.** `alembic` was in `requirements.txt` from Phase 0 but no
`alembic.ini`, `env.py`, or `versions/` directory had ever been created, and no table was
defined. Built the whole harness, async-aware, reading the DSN from `Settings` rather than
from `alembic.ini` — the ini file is tracked and the DSN carries a password.

**mypy strict versus pandas.** Two options: `ignore_missing_imports` (cheap, but makes every
DataFrame expression `Any` and silently voids strict mode across the data layer — exactly the
layer where a leakage bug would hide) or `pandas-stubs` (real types, more friction). Chose
`pandas-stubs`, scoping `ignore_missing_imports` to scikit-learn alone, which ships no
`py.typed`. Two friction points followed: pandas-stubs rejects `pd.Series[object]` as a type
argument, and `Model.__table__` is typed as the broader `FromClause`, which exposes neither
`constraints` nor `indexes` — the tests read tables off `Base.metadata` instead.

**pandas 3.x is the current major.** Copy-on-write is enforced and string columns are
PyArrow-backed by default, which is a direct win here: `train_transaction.csv` is ~652MB with
many low-cardinality string columns. `pip-audit` is clean on the new dependency set.

**Narrowed a Phase 0 lint exemption.** `pyproject.toml` carried a blanket `F401` ignore on
`app/models/*.py` for the era when every model file was a docstring-only stub. Now that
`transaction.py` and `account.py` carry real imports, that exemption is narrowed to
`app/models/__init__.py`, where unused imports are genuinely load-bearing (importing a table
module is what registers it against `Base.metadata` for Alembic). The remaining tier stubs no
longer get a free pass.

**Verified the migration rather than trusting it.** Applied against Postgres 16, confirmed
`rowsecurity` and `relforcerowsecurity` are both true on `transactions` and `accounts`, all
four policies present, both roles created non-superuser and `NOLOGIN`. Every check constraint
was smoke-tested with a deliberately invalid insert and each rejected the row by name.
Downgrade to base and re-upgrade both run clean.

### Design decisions worth stating

- **Composite primary keys.** IEEE-CIS `TransactionID` and PaySim's synthesised row id
  occupy overlapping integer ranges, so `transaction_id` alone is not unique.
- **Engineered features in JSONB, raw `V1`-`V339` not in Postgres at all.** The corpora share
  almost no raw fields, so a wide table would be ~440 columns and ~95% NULL per row, and
  `feature_version` would be decorative because the column set could never move with it.
  Training reads a parquet materialisation instead, with a parity test against the table.
- **The fraud label is not on `TransactionFeatures`.** A live scoring call has no label, and
  an optional `is_fraud` on the scoring contract is an invitation to leak one into a feature
  vector. It lives on `LabelledTransaction` instead.
- **`counterparty_id` is rejected outright for IEEE-CIS**, in both Pydantic and a check
  constraint. IEEE-CIS records no money-flow edge; a fabricated one would feed the Tier-3
  graph as though it had been observed.
- **No foreign key from `transactions` to `accounts`.** `accounts` is derived from
  `transactions` in the same pipeline run, so a FK would impose an insert ordering for no
  integrity gain — and a FK between two RLS-forced tables can leak the existence of rows the
  policy hides.
- **`accounts.straddles_split` and `accounts.uid_strategy` exist to measure two known
  weaknesses rather than hide them.** IEEE-CIS has no account column, so a UID is constructed;
  `uid_strategy` records which rung of the fallback ladder produced it, and a `singleton`
  account has no history, making its per-account features noise. IEEE-CIS also propagates a
  chargeback label across subsequent transactions on the same account within ~120 days, so
  labels cluster by account and a straddling account is a real leak surface across the split
  boundary. Both get counted in the data-quality report.

### Known gaps leaving steps 0-2

- **RLS is defined but not yet effective.** `docker-compose.yml` still connects the backend as
  the `riskiq` superuser, which also owns the tables, and superusers bypass RLS
  unconditionally. The policies are correct and verified present; they simply do not apply to
  the current connection. Phase 7 must grant `LOGIN` to `riskiq_app` and repoint
  `DATABASE_URL`. **Until it does, security-checklist section 3's "application user is not a
  superuser and does not own the tables" item is an open FAIL and should be reported as one.**
- **No analyst-scoped read policy.** The only read policy is per-account isolation, which
  fails closed. The Phase 8 dashboard's live feed and ring views are not account-scoped and
  will be denied by design until Phase 7 adds the analyst role.
- **PaySim's observed maximum `step` is 743, not the documented 744.** Within the declared
  bound, so no warning fires; noted because the corpus is often described as 744 hours.
- **Ruff's isort classifies `alembic` as first-party** because `backend/alembic/` exists.
  Cosmetic — imports resolve correctly at runtime, since a regular package in site-packages
  wins over a namespace-package directory.
### Steps 3-8 — the pipeline itself

**Verified end state.** `python -m app.data.pipeline` runs both corpora end to end:
590,540 IEEE-CIS rows (20,663 positives, 3.4990% base rate, 204,960 inferred accounts) and
2,770,409 PaySim rows (8,213 positives — **100% of the corpus's fraud**, retained by the
TRANSFER+CASH_OUT scope filter). `ruff`, `black`, `mypy --strict` and 157 tests all green;
`pip-audit` clean. Two consecutive runs produce identical `feature_version` hashes
(`fv_afdab5ef437a`, `fv_ccb648194a72`), which is the reproducibility claim the audit trail
depends on.

### Three measured findings that change later phases

**1. PaySim's transfer-to-cash-out chain cannot be followed by account name — at all.**
Tier-3's premise was that a fraudulent `TRANSFER` to a mule can be linked to the `CASH_OUT`
that drains it. Measured: **0.00%** of fraudulent transfers have a `nameDest` matching any
fraudulent cash-out's `nameOrig`, and only 0.07% match *any* cash-out origin. The Phase 1
plan predicted this link would be weak; it is absent. **Phase 4 must link on amount-and-step
proximity, with a name match as corroboration only.** Decided on measurement before the
graph layer was built rather than discovered afterwards.

The graph is still viable, for a separate reason: destination accounts recur heavily
(151,957 appear 6-20 times, 16,233 appear 21+ times over 509,565 distinct destinations).
Community detection needs recurrence and the recurrence is there. Only the identity link is
missing.

**2. PaySim's class balance is not stationary.** Train 0.1874%, val 0.1347%, **test 0.9657%**
— the test window carries 4,020 of the 8,213 positives, 3.3x the corpus rate. This is a
property of the simulation, not a defect in the split, and the chronological boundaries
stand. It does mean any PaySim metric is partly a statement about which period the test
window covers, so every one must quote its split's own base rate. The report now detects and
flags this automatically rather than leaving it to be noticed in a table.

**3. The IEEE-CIS UID is usable, with a stated limit.** 204,960 accounts inferred from
590,540 transactions. 118,361 accounts (57.7%) hold a single transaction — but because they
are single-transaction accounts, that is **20.0% of rows**, so roughly 80% of transactions do
have prior history for velocity, z-score and Tier-2 sequences to work with. The fallback
ladder almost never fires: 97.1% of accounts resolve via the primary `card1+addr1+D1n` key,
2.7% via `card1+email`, 0.19% via `card1` alone, and **zero** reach the singleton rung.
29,321 accounts (14.3%) straddle a split boundary, of which 1,030 contain at least one
fraud — that is the measured leak surface, now a column in `accounts` rather than an
assumption.

### Obstacles hit and how they were solved

**A leakage guard needs its own guard.** The hard leakage check recomputes every feature
against a frame truncated to each row's own point in time and asserts nothing moves. A test
like that passes trivially if the comparison is broken, so there is a companion test that
introduces a deliberate leak — a whole-column `groupby().transform("mean")` — and asserts the
comparison *fails*. Without it, "the leakage test passes" would be an unverified claim.

**`GroupBy.first()` and `mode()` are quiet leaks.** The obvious way to express "the account's
usual device" is the first or most common value per account. Both skip nulls, so on an
account whose earliest transaction has no `DeviceInfo` they reach *forward* to a later row.
Replaced with cumulative counts of prior sightings, which cannot see forward by construction:
`device_mismatch` is one minus the share of the account's prior transactions on this device.

**pandas offset aliases are not the feature labels uppercased.** `7d` raised
`Pandas4Warning: 'd' is deprecated, use 'D'`, which `filterwarnings = ["error"]` turned into
16 test failures. Hours went the other way — `H` was deprecated in favour of `h`. Since the
aliases cannot be derived from the labels, `VELOCITY_WINDOWS` is now an explicit
`label -> alias` map, which also keeps the engineered feature names (and therefore the
`feature_version`) stable across a future pandas alias change.

**`copy_records_to_table` is binary COPY, and a custom jsonb codec is not.** Registering
`set_type_codec("jsonb", encoder=json.dumps, ...)` installs a *text* encoder, and binary COPY
fails with `no binary format encoder for type jsonb`. asyncpg's built-in jsonb codec accepts
`str` and does have a binary encoder, so the feature vectors are serialised to JSON strings
before the COPY.

**`card4` and `card6` sit inside a numerically-named block but are categorical.** The dtype
map read `card1`-`card6` as float32 and failed on `discover`. Only `card1`, `card2`, `card3`
and `card5` are numeric.

**`--sample` was unusable for PaySim.** A chronological head of 20k rows spans about nine
hours on a one-hour clock, so the split correctly refused to cut it. The sample is now an
evenly-spaced stride that preserves the full 30-day span. Worth noting that the *split code
was right* — the sampling was wrong.

**Non-finite floats are not valid JSON.** Amount z-scores can produce `inf` when an account's
prior amounts are identical. `NaN`/`Infinity` are rejected by JSONB, so the feature-vector
builder maps every non-finite float to null, vectorised — a per-element loop over 3.4M rows
times ~25 features was far too slow.

**Self-inflicted: a PowerShell `Get-Content -Raw | Set-Content` round-trip corrupted UTF-8.**
Every em-dash in `raw_spec.py` became mojibake and the BOM became a stray `?`, breaking the
parse. Repaired by re-encoding through CP1252 and stripping the stray byte. Use the editing
tools on source files, not shell text round-trips.

### Design decisions worth stating

- **Split before engineer.** Frequency encoders are fitted on train only, and a split derived
  purely from time depends on no feature, so there is no circularity. Categories appearing
  only after the train boundary encode to 0.0, which is itself a signal.
- **Velocity is null for PaySim, not zero.** A one-hour clock makes a 1h trailing window
  degenerate and `nameOrig` is 99.94% unique. `velocity_available = false` says "not
  measurable here"; a zero would have claimed "this account was inactive".
- **`V1`-`V339` go to their own parquet file, streamed.** 339 of 394 columns that nothing in
  Phase 1 reads. Kept because Phase 2 needs them, separate because carrying them through
  engineering would roughly triple peak memory, and unreduced because the NaN-block
  correlation reduction is a decision that must be fitted on train — Phase 2's to make.
- **COPY rather than ORM inserts.** ~3.4M rows. The scoped deletes are parameterised
  (`DELETE ... WHERE source_dataset = $1`) and COPY composes no SQL string at all.

### Known gaps leaving Phase 1

- **RLS is still not effective** — carried forward from steps 0-2. The app connects as the
  `riskiq` superuser, which owns the tables and bypasses RLS. Phase 7 must grant `LOGIN` to
  `riskiq_app` and repoint `DATABASE_URL`; security-checklist section 3 stays an open FAIL
  until then.
- **The IEEE-CIS UID is an inference, not an observation.** It is leakage-safe, and 97.1% of
  accounts resolve through the primary key, but a wrong grouping produces wrong per-account
  features and there is no ground truth to check it against.
- **The synthetic slow-drift injection (Phase 1's enhancement pass) is not built.** It is
  constrained by security-checklist section 8 and must live in eval-only code, never
  reachable from an endpoint.
- **`velocity_*` and the familiarity features are unvalidated against fraud.** They are
  leakage-safe and correctly computed; whether they carry signal is Phase 2's measurement.
