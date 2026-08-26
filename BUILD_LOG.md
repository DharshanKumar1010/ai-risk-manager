# RiskIQ Build Log

Running record of what shipped, what was deferred, and what is known to be missing.
This becomes the **"Build Challenges & Technical Obstacles"** answer on the Razorpay
submission form, so entries are written honestly — real obstacles and how they were
solved read as more credible to a panel than a smoothed-over success narrative.

| Phase | Status | Notes | Known Gaps |
|-------|--------|-------|------------|
| 0 — Scaffolding & environment | **Complete, verified** | Monorepo, `.claude/` skills + agents, CI workflow, 4-service Docker stack booting healthy. All six verification items passed. Detail below. | Dependency ranges not yet exact-pinned; CI never executed on GitHub; audit writer raises by design until Phase 7 |
| 1 — Data pipeline & features | **Complete, verified** | Full pipeline runs end to end over both corpora: 590,540 IEEE-CIS + 2,770,409 PaySim rows engineered, split chronologically, and persisted to Postgres and parquet. 157 tests green including a hard leakage check. Detail below. | RLS defined but **not yet effective** (app still connects as superuser — Phase 7); IEEE-CIS `V1`-`V339` carried to parquet but not yet reduced or used (Phase 2); PaySim class balance is non-stationary and name-chaining is measured at 0% — both constrain Phase 4 |
| 2 — Tier-1 anomaly layer | **Complete, verified** | LightGBM selected over Isolation Forest on validation PR-AUC. IEEE-CIS held-out test **PR-AUC 0.5276** (95% CI 0.5117–0.5462) against a 0.0348 no-skill floor — 15.2x lift; 87% precision at a staffable 1%-flag operating point. Latency p50 5.62ms / p95 6.38ms, ~8x inside the 50ms budget. Detail below. | Catches 24.6% of fraud by count but only **14.6% by value** — recall falls as amount rises; PaySim's 0.9998 is a simulator artefact (`amount == oldbalanceOrg` on 97.5% of fraud, 0 of 412,277 legit rows) and must never be quoted as a headline, nor used to fit the Phase 5 meta-learner; no hyperparameter search run; feature-assembly latency still unmeasured (Phase 7) |
| 3 — Tier-2 behavioural layer | **Complete, verified** | PyTorch LSTM autoencoder over per-account trailing windows, IEEE-CIS only. Evaluated **per account**, not per transaction. Detail below. | Loses the head-to-head against Tier-1-aggregated; earns its place only as a Phase 5 fusion input, not standalone. Two registry entries answer to this architecture with identical PR-AUC and feature_version but p99 5.76ms against 45.18ms — read `...070529z`, and re-benchmark in Phase 10. Full gap list in the Phase 3 detail section |
| 4 — Tier-3 graph layer | **Complete, verified** | One Louvain + centrality algorithm over two real graphs. **IEEE-CIS ring-level test PR-AUC 0.6462** (95% CI 0.5703–0.7214) against a 0.1076 base rate — **6.01x lift** on 1,329 rings, corroborated by an independent enrichment check at 6.20x. PaySim ring-level 0.9977 against a 0.8369 base rate — a **1.19x lift, which is close to nothing**, on a corpus whose chain structure is a simulator artefact. Serving is a dictionary lookup, p99 0.003ms. Detail below. | **Tier-3 earns its place on IEEE-CIS and not on PaySim.** `tier3_ring_risk_score` is **not** a proven Phase 5 input — per-transaction PR-AUC is *below* no-skill (0.902x) and Tier-1 fusion is significantly negative (−0.0031, CI [−0.0038, −0.0026]). Operating-point recall 0.112 (IEEE-CIS) / 0.006 (PaySim). Ring metrics exclude rings repeating an earlier one, so they say nothing about persistent rings. Most IEEE-CIS signal is circular with the constructed UID. Full gap list in the Phase 4 detail section |
| 5 — Meta-learner + SHAP | **Complete, negative result** | XGBoost fusion over out-of-fold Tier-1 + engineered features, Platt-calibrated, TreeSHAP attribution. Held-out test **PR-AUC 0.4954** (95% CI 0.4791-0.5141) against Tier-1 alone at **0.5276** — paired delta **-0.0322, 95% CI [-0.0373, -0.0273], excluding zero on the negative side**. The ablation retired **all three tier layers**; `tier3_topology` was measurably harmful. Well calibrated (ECE 0.0037). **Gate status: three ml-evaluator rounds returned 6, then 4, then 3 blocking findings; all were fixed and each fix verified directly, but the confirming fourth round did not complete, so the phase is NOT recorded as gate-cleared.** Detail below. | **Tier-1 wins the headline metric; do not ship the meta-learner on PR-AUC.** But at a matched 1% flag rate the fusion is *cheaper* (4,818 vs 4,904 per 1,000) because it catches more fraud **by value** (16.9% vs 15.0%) while catching less by count — a point estimate with no CI, flagged for Phase 6 rather than acted on. The loss is **confounded**: the out-of-fold handicap is real (fold PR-AUC 0.4200-0.5202 against full-train 0.6155) but the shipped booster also early-stopped at **iteration 2**, and the discriminating diagnostic was not run. |
| 6 — Causal cost layer | **Complete, verified** | Cost-aware ranking over Tier-1's probability and the transaction amount. The shipped `plug_in` policy is **22.41% cheaper** than probability ranking at a matched 1% flag rate — −1,098.48 per 1,000 decisions, bootstrap CI [−1,345.28, −881.81], excluding zero. The secondary hypothesis (cost-sensitive *training* beats cost-sensitive *thresholding*) was tested and came back a **TIE**, as the phase's own algebra predicted. All three gates ran; 17 findings, 16 fixed, 1 escalated and decided. Detail below. | Headline is **regime-dependent** — the advantage shrinks under high-fee assumptions, so 22.41% must be quoted with its cost model ($3 review / $15 chargeback fee). The DR-learner collapses onto the plug-in because no treatment variable exists (every historical transaction was allowed), so **nothing here is a causal effect measured on this data** — `ope_validation.caveat` is not optional. Shipped threshold 90.85 is a *money-scale* score, not a probability |
| 7 — Backend, audit, security | **Complete, gates cleared on the third round** | Four authenticated endpoints (`POST /score`, `GET /transactions`, `GET /audit/{id}`, `GET /rings`) plus an analyst-scoped `GET /audit/entry/{id}/explain`. **`security-reviewer` returned 3 blocking findings, then 1 more on re-verification; `code-reviewer` returned 5. All were correct.** The worst was an authorization hole — a token with no `account_id` claim read every account's rows, because one sentinel meant both "unrestricted" and "no account". All three blocking findings fixed with regression tests; the code review's top finding is a deliberate carry to Phase 9, reasoned below. JWT on every route with server-side scope and ownership checks; append-only `audit_log` with RLS forced; Redis limiter that **fails closed**; Tier-3 timeout with degraded-mode fallback recorded in the audit row. RLS made *effective* for the first time — `riskiq_app` granted LOGIN and `DATABASE_URL` repointed off the superuser, closing a FAIL carried since Phase 1. Suite 442 → **579 passed, 1 skipped**. Detail below. | **Serving-time feature assembly is the real latency**: p50 34–41ms against the Tier-1 scoring call's 4ms, so total p95 sits at 44–51ms, effectively consuming the 50ms budget that was defined for the scoring call alone. Cost is **flat in history size** — fixed pandas overhead, not the range scan. `transactions.device_info` / `addr1` are new and **unbackfilled**, so familiarity features read as `__missing__` for every pre-Phase-7 row until the pipeline is re-run. The meta-learner is deliberately **not** in the decision path (it loses to Tier-1); Tier-3 annotates but does not move the decision |
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

### Reproducibility, checked rather than assumed

The final two runs produced byte-identical headline numbers — PR-AUC 0.495416, delta -0.032157,
ECE 0.003682 — across a code change that touched provenance and report text but not the model
path. Seed 42 and `nthread=4` are pinned for exactly this reason. A run that failed to reproduce
here would itself have been a finding.

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

## Phase 1 — detail

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

---

## Phase 2 — Tier-1 real-time anomaly layer

**Verified end state.** `python -m app.models.train_tier1` trains and compares four candidates
per corpus (five for PaySim), scoring the held-out test split exactly once. IEEE-CIS selected
model: **LightGBM, test PR-AUC 0.5276** (95% CI 0.5117–0.5462) against a no-skill floor of
0.0348 — a 15.2x lift, selected on validation and scored once on test. Latency p50 5.62ms /
p95 6.38ms / p99 6.86ms against a 50ms budget. `ruff`, `black`, `mypy --strict` and the full
suite green. Four consecutive full runs produced identical PR-AUC to four decimal places,
which is the reproducibility claim `models/registry.json` makes on every entry (latency is
wall-clock and moves ~1ms between runs).

### Four measured findings that change later phases

**1. Supervision is worth ~0.42 PR-AUC; richer inputs are worth ~0.016.** The comparison was
built to separate the two, because "the supervised model won" would otherwise confound
supervision with LightGBM's ability to take native categoricals that Isolation Forest cannot.
Fitting LightGBM on the *identical* numeric-only matrix the forest saw gives 0.5119 against the
forest's 0.0886 (paired delta 95% CI [0.4241, 0.4558]); adding native categoricals moves it to
0.5276 (paired delta [0.0105, 0.0211]). Both are real, and they are two orders of magnitude
apart. **If a later tier is expensive to build, this is the evidence that labels buy far more
than features do.**

**2. Tier-1 catches 24.6% of fraud by count but only 14.6% by value.** Recall falls as the
amount rises — 0.3945 on the cheapest quartile of fraud, 0.1729 on the most expensive. Median
amount of caught fraud 43.65, of missed fraud 75.00. The cost model chooses the *threshold* but
the model's *ranking* is fitted on the label and is value-blind. **This is the concrete opening
for Phase 6's causal cost layer**: cost-sensitive training, not just cost-sensitive
thresholding. It is also the number that must accompany any recall figure quoted for this
layer, because the recall figure alone overstates what is recovered by roughly 10 points.

**3. PaySim's Tier-1 result is an artefact of the simulator and must not be quoted.** PR-AUC
0.9999 tripped the automated `> 0.95` leak warning. Investigated rather than reported:
`amount == oldbalanceOrg` (exact, ±0.01) holds for **97.49% of fraud and 0 of 412,277
legitimate test rows** — precision 1.0000 as a lone rule, and `-|amount − oldbalanceOrg|` alone
scores 0.9751 average precision. PaySim's fraud agent transfers exactly the full balance. This
is not a leak in our pipeline — no future information is read and the split boundaries hold —
it is a property of the corpus. **Phase 5 must not fit the meta-learner on PaySim**, or it
learns "Tier-1 is always right", which transfers nowhere. PaySim's value to this project
remains its graph structure for Tier-3.

**4. Phase 1's open question is answered: the account-history features carry signal.**
`velocity_sum_7d` (2.4%) and `amount_prior_mean` (2.3%) both land in the selected model's top
twelve by split gain, alongside the `C`/`D` blocks. The Phase 1 log left this explicitly
unresolved; it is now measured. Separately, `velocity_available` was dropped automatically as
constant-on-train, and eleven PaySim columns were dropped the same way — reproducing by
measurement exactly the dead columns the Phase 1 EDA had identified by hand.

### Obstacles hit and how they were solved

**The obvious scoring implementation missed the latency budget by 40%.** Assembling a one-row
`DataFrame` and handing it to LightGBM cost 21.6ms per call — ~17.9ms constructing 28
`pd.Categorical` columns and ~9.2ms while LightGBM re-derived the pandas category mapping.
Neither is model work; both are pandas overhead. Serving now converts categoricals to integer
codes and scores a plain float array, which is the representation LightGBM built internally at
training time anyway. Verified identical to the batch path (max absolute difference 0.0) and
pinned by `test_numpy_and_pandas_paths_agree`, because the equivalence holds only while the
code ordering matches the categorical ordering seen at training. Result: 21.6ms → 0.05ms for
the predict step, 4.83ms p50 end to end on the full 1,067-tree model.

**The phase brief, followed literally, produces a contaminated model selection.** It says
"evaluate both on the held-out test split … pick whichever wins on PR-AUC", and the first
implementation did exactly that: `max(candidates, key=lambda c: c.result.pr_auc)` where
`result` is the *test* result. Every other decision was correctly made on validation — early
stopping, the operating threshold, the capacity ceiling, both sensitivity sweeps — and this one
was not. It means the shipped model was chosen using the same split its headline is quoted
from, which ml-evaluation-standards section 1 says invalidates the headline.

Found by running the `ml-evaluator` agent's own checklist by hand (item 2, test-set
contamination) rather than by anything failing. **It was not a technicality:** on a 20k sampled
run, selecting honestly on validation picks LightGBM *matched inputs* where selecting on test
picked *full inputs*, and the paired delta between them straddles zero. The two procedures
genuinely disagree.

Fixed by applying the brief's own criterion one split earlier — candidates are ranked on
validation PR-AUC, test is scored once afterwards and only to report. Guarded by
`test_model_selection_reads_validation_not_test`, and the report now shows the `val PR-AUC`
column the selection actually reads next to the test column it does not.

**Overlapping confidence intervals are not a tie, and nearly got reported as one.** The
selected model's PR-AUC interval (0.5117–0.5462) overlaps the runner-up's (0.4955–0.5303),
which reads as "no significant difference". It is not: both models score the *same* test split,
so their errors are correlated. Bootstrapping the difference **paired** — both models scored on
each resample — removes the split's shared variance and gives [0.0105, 0.0211], which excludes
zero. The first draft of the model README claimed the advantage was inside sampling noise. It
was wrong, and the paired test is what caught it.

**Two "what this does NOT catch" claims were written from reasoning and both were false.** The
first draft asserted that low-value fraud is systematically under-caught — the cost model
prices false negatives by amount, so the reasoning seemed sound. Measured, it is exactly
backwards: recall is *highest* on the cheapest quartile. The second asserted a large penalty on
no-history accounts; measured, it is 0.2252 against 0.2509, real but minor, because the `C`/`D`
blocks carry per-entity history of their own. The evaluation standard says write that section
from observed false negatives rather than imagination. Both drafts are why.

**scikit-learn 1.9's `IsolationForest` accepts NaN, which removed a planned component.** The
plan called for a train-fitted median imputer plus missingness indicators, on the assumption
that the forest would reject missing values. Checked before building: it does not. Dropping the
imputer left both models consuming the *same* NaN-preserving matrix, which makes the matched
comparison cleaner than it would otherwise have been, and removed a fitted artefact that would
have needed versioning and keeping in step with the model. Verifying the assumption was worth
more than implementing it.

**A sklearn warning would have failed the suite from an unexpected direction.** Fitting the
forest on a named `DataFrame` and later scoring a bare array emits a "does not have valid
feature names" warning, and `filterwarnings = ["error"]` turns that into a test failure. Fitted
on the array instead, so fitting and scoring agree on the representation from the start.

**The ±50% cost sensitivity sweep the standard asks for is the uninformative direction.**
Scaling both cost parameters together changes the magnitude of the cost but not the FP:FN
ratio, so the recommended threshold barely moves (flag rate 47.1% → 22.7%). Varying the review
cost alone moves it enormously (26.4% → 3.1% across a 25x range). Both are now reported; the
second is the one that tells a reader how much the recommendation rests on a guessed constant.

**The cost-optimal threshold implied an unstaffable review queue.** With a false negative
priced at the transaction amount (~69 median) against a review at 3, the arithmetic favours
reviewing 28.9% of all traffic. That is correct and useless. The report now also carries a
capacity-constrained operating point — the same model capped at 1% of traffic — which is where
the deployable number lives: **precision 0.8734 at a 0.98% flag rate**.

### Design decisions worth stating

- **Tier-1 mints its own `feature_version`.** It reads the raw row columns Phase 1 deliberately
  kept out of `transactions.features` — chiefly `C1`-`C14` and `D1`-`D15`, Vesta's own
  per-entity aggregates, which turn out to be the top signal (`C1` alone is 7.2% of split
  gain). Its input set is genuinely a different definition, so reusing the pipeline's hash
  would be a false claim about what produced a prediction.
- **"No Tier-2/3 dependency" is not "no account history."** `velocity_*` and the familiarity
  features are Phase 1 feature-store outputs, not another tier's model outputs, so Tier-1 stays
  independently scoreable while using them. `score()` consumes an already-assembled vector, so
  the 50ms budget covers inference only.
- **A missing feature raises rather than zero-filling.** A zero-filled vector produces a wrong
  decision underneath a correct-looking audit row, which is the one failure an audit trail
  exists to prevent.
- **LightGBM is persisted in its native text format, not pickled.** Loading a pickle executes
  arbitrary code and this path becomes reachable from a Phase 7 endpoint. Only the Isolation
  Forest, which has no text format, goes through joblib.
- **Isolation Forest is kept despite losing 6x.** Chargeback labels arrive weeks late, so the
  unsupervised floor is the real answer for a merchant with no labelled history yet.
- **Degenerate columns are found by measurement, not by a hardcoded list.** Null-rate and
  variance are checked on train; that reproduced all eleven of PaySim's dead columns without
  anyone naming them, and will keep working when the data changes.

### Known gaps leaving Phase 2

- **RLS is still not effective** — carried forward from Phase 1. Unchanged: the app connects as
  the `riskiq` superuser. Phase 7 must grant `LOGIN` to `riskiq_app` and repoint `DATABASE_URL`.
- **Feature-assembly latency is unmeasured.** The 50ms budget is verified for the scoring call.
  The account-state range scan that Phase 7 must run to build the vector has not been timed
  against `ix_transactions_account_time`, and it is the part most likely to be slow.
- **Two constraints Phase 7 must honour, from the Phase 2 security review.** First, the
  scoring endpoint must accept *raw transaction fields* and assemble the feature vector
  server-side — `FeatureVector` is unbounded, so an endpoint taking a client-supplied vector
  would let a caller choose its own score while leaving a correct-looking audit row. Second,
  `explain()` and `top_feature_importances()` are an evasion oracle under checklist section
  8.3: valuable for the Phase 8 reviewer dashboard, but they must be authenticated and never
  returned to the transacting party.
- **A validation-derived capacity ceiling does not transfer under base-rate shift.** The 1%
  flag-rate cap lands at 0.98% on IEEE-CIS, whose base rate is near-stationary, but at 2.25%
  on PaySim, whose test window runs 7x its validation base rate. Phase 7 should size a review
  queue from a threshold re-derived on recent traffic, not from a fixed historical one.
- **The V1–V339 block is still unreduced**, and remains deferred. Phase 1 earmarked the
  NaN-pattern correlation reduction for Phase 2; the 113-feature model without it clears the
  bar, so it stays available as a lever if Phase 5 needs more from Tier-1.
- **Online recalibration is designed for, not implemented.** `OnlineRecalibrator` is an
  interface stub. It needs the labelled feedback path that does not exist until Phase 9.
- **No hyperparameter search was run.** The LightGBM parameters are sensible defaults with
  early stopping on validation average precision. A search would likely add a little, and was
  not the best use of the remaining time.
- **The PaySim Tier-1 model is registered but should not be deployed.** It is kept for
  completeness and for the ingestion-safe ablation. Its registry entry now opens with an
  automatic `DO NOT QUOTE AS A HEADLINE` caveat — added after noticing that the first run
  recorded a PR-AUC of 0.9999 with no warning attached, so anyone reading `registry.json`
  alone would have taken it at face value. Any leak-suspicious result from any tier now
  carries that caveat into the permanent record rather than only into a log line.

## Phase 3 — Tier-2 behavioural sequence layer

Backfilled during Phase 4. The Phase 3 row in the status table above pointed at "Detail below"
and at a "Phase 3 detail section" that was never written — the file ended inside Phase 2's
known gaps. The numbers here come from `notebooks/tier2_report.md` and the two Tier-2 entries
in `models/registry.json`, which were written at the time; only the narrative was missing.

**Verified end state.** `python -m app.models.train_tier2` trains a PyTorch LSTM autoencoder
over per-account trailing windows on IEEE-CIS, sweeping latent size, window length and the
abstention threshold on validation and scoring the held-out test split exactly once. Selected
model: **latent=8, W=15, test PR-AUC 0.0952** (95% CI 0.0801–0.1154) as the deployed system
over all 43,530 test accounts, and **0.2932** (95% CI 0.2491–0.3488) over the 5,825 accounts
it has enough history to score. Seed 42, CPU only — cuDNN's LSTM kernel is
non-deterministic and would falsify the reproducibility claim every registry entry makes.

**Two Tier-2 entries answer to the same architecture, and the later one is six times slower.**
`...w15-latent8-ieee-cis-20260823t070529z` and `...20260823t083445z` carry identical PR-AUC
(0.09522) and an identical `feature_version` (`fv_da07bc36e0da`) — the same model, benchmarked
twice — but latency p95 moved from **5.56ms to 32.47ms** and p99 from 5.76ms to **45.18ms**
against a 50ms budget, and `notebooks/tier2_report.md` reflects the slower run. Nothing about
the model changed. The benchmark is wall-clock on one machine and the later run was taken while
the Phase 4 graph builds were saturating the CPU, so this is an environment artefact rather
than a regression in the layer — but a p99 within 10% of the budget is exactly the number a
later phase must not meet by surprise, so it is recorded rather than quietly overwritten.
**Phase 5 and Phase 7 should read the `...070529z` figures**, and Phase 10 must re-benchmark
on an idle machine before any of this is quoted as a serving guarantee.

### Three measured findings that change later phases

**1. Tier-2 loses the head-to-head against Tier-1 and still earns its place.** PR-AUC delta
against Tier-1 aggregated to the account level: **95% CI [-0.5110, -0.4527]** — decisively
negative. That is the finding, and it was not a surprise: Tier-1 is supervised on the label
and Tier-2 is one-class, so the head-to-head was always the wrong question. The right one is
what Tier-2 adds *among the accounts Tier-1 has already cleared*. Spearman correlation between
the two layers' account scores is **-0.0406** — near zero, which is the precondition for
fusion adding anything at all. Among the 43,094 test accounts Tier-1 leaves below its own
capacity threshold, 691 are fraudulent (1.60%), and Tier-2 ranks that residual at PR-AUC
**0.0242** against a 0.0160 floor — **1.5x lift on fraud a Tier-1-only system would never have
looked at.** That residual lift is what Phase 5 inherits. It is a narrower claim than the
phase brief implies and it is the one the measurements support.

**2. The evaluation unit had to change, and saying so is the result.** IEEE-CIS propagates a
chargeback label across an account's later transactions, so one compromised account holding
300 rows contributes 300 correlated positives. A per-transaction PR-AUC counts those as 300
independent correct calls when the model made one, and the bootstrap has the same problem in
reverse — resampling rows as if independent produces an interval far tighter than the evidence
supports. Tier-2 is therefore evaluated **per account, not per transaction**, declared in the
report and recorded in the registry note. Phase 4 hit the same class of problem from the
opposite direction and made the same move: see the Phase 4 section below.

**3. Coverage is the gap, not the accuracy.** Tier-2 can score only **13.4% of test accounts**
(5,825 of 43,530), 28.8% of test transactions and **49.1% of test fraud transactions**, because
Phase 1 measured 57.7% of IEEE-CIS accounts holding a single transaction. The rest abstain
rather than returning a zero. The deployed headline of 0.0952 is computed with all 37,705
abstentions counted as never flagged, which is why it sits so far below the model's own 0.2932
over what it can actually see. Both are reported, both are labelled, and the deployed one is
the headline.

### Obstacles hit and how they were solved

**The headline confusion matrix is degenerate, and it is recorded that way.** At the
cost-minimising threshold the deployed system flags every account: TN 0, FP 42,457, FN 0,
TP 1,073 — precision equal to the base rate, recall 1.0. That is what unbounded review
capacity buys under this cost model, and rather than quietly reporting a nicer operating
point instead, the registry entry opens with `THE HEADLINE CONFUSION MATRIX IS DEGENERATE` and
a capacity-constrained operating point is reported alongside it. The cost model's own
assumption list now names unbounded review capacity as the assumption that bites hardest.

**A wide-enough autoencoder learns the identity map.** With ~21 features over 15 timesteps the
input is ~315 dimensions, so a 128-unit hidden state is not on its own a bottleneck; the
latent dimension is. Selecting on lowest validation loss picks the model that reconstructs
fraud exactly as faithfully as normal behaviour, whose two error distributions land on top of
each other. Selection was moved to the *ratio* of fraud to normal reconstruction error on
validation, and the two criteria do not agree.

**The selection over the runner-up rests inside the noise.** Delta against latent=8, W=20:
95% CI [-0.0048, 0.0048]. The interval includes zero, so on this test split the runner-up
performs as well and the choice rests on a point estimate that the evidence does not separate.
Recorded rather than presented as a decisive win.

### Known gaps leaving Phase 3

- **Tier-2 is a fusion input, not a standalone layer.** It loses to Tier-1 alone and only
  earns its place through the 1.5x residual lift above. If Phase 5 finds the meta-learner does
  not use it, the honest conclusion is that this layer did not pay for itself.
- **86.6% of test accounts are never scored.** Coverage, not accuracy, is the binding
  constraint, and it follows from the corpus rather than from the model.
- **By value, 87.6% of test fraud value sits in the accounts this layer missed** — the same
  direction of failure Phase 2 measured for Tier-1, which catches 24.6% of fraud by count and
  14.6% by value. Both layers are weakest on the expensive cases.
- **Attention/contribution weights ship but are unused.** `explain()` returns per-timestep
  error contributions for the Phase 8 panel; nothing reads them yet.
- **No PaySim Tier-2 model exists and none should.** Phase 1 measured 99.9% of PaySim accounts
  holding a single transaction and `seconds_since_prior_txn` 99.94% null. A sequence model
  needs sequences.

## Phase 4 — Tier-3 network graph abuse-ring layer

**Verified end state.** `python -m app.models.train_tier3` builds both graphs, sweeps their
edge parameters on validation, and scores the held-out test split exactly once per corpus.
**IEEE-CIS ring-level test PR-AUC 0.6462** (95% CI 0.5703–0.7214) against a **0.1076** ring
base rate — a **6.01x lift** on 1,329 test rings. **PaySim ring-level 0.9977** against a
0.8369 base rate — a **1.19x lift**, which is the number that matters and is close to nothing.
Serving is a dictionary lookup: p50 0.001ms, p99 0.003ms against a 50ms budget. `ruff`,
`black`, `mypy --strict` and the full suite green; 84 Tier-3 tests, 323 overall.

**The honest summary of this phase in one paragraph.** Tier-3 earns its place on IEEE-CIS and
does not earn it on PaySim. On PaySim the amount-and-step pairing rule alone reaches precision
0.9948 and recall 0.9943 before any graph runs; the graph then adds a 1.19x ranking lift inside
a candidate population that is already 84% fraud-bearing, and changes no decision the rule had
not already made. On IEEE-CIS the ring-level detector is real — 6.01x over its floor, with an
independent enrichment check agreeing at 6.20x — but its per-transaction projection is *below*
no-skill and fusing it into Tier-1 makes Tier-1 measurably worse. What this layer has earned is
ring-level detection on the corpus with genuine shared-entity structure. Nothing more.

**Every number here is post-review.** The first four sets were wrong, in ways recorded below.

### Six measured findings that change later phases

**1. PaySim's observed money-flow graph is a star forest, so the observed edge alone is not a
graph problem.** Measured on the train split: 1,937,588 distinct origins over 1,938,484 rows,
**99.95% of origins have degree 1**, the maximum is 2, and exactly **341 of 2,291,054 nodes**
are both an origin and a destination. Louvain over that returns the 353,807 destination stars,
which is `groupby(nameDest)` spelled expensively, and betweenness on a star is a restatement of
degree. The inferred transfer-to-cash-out chain edge is not an enhancement — it is the only
thing that makes this a graph at all.

**2. The chain edge works because it is reading the simulator.** Exact-amount same-step
matching selects **99.50% of fraudulent transfers against 0.23% of legitimate ones**, median
one candidate partner, 99.73% of matched partners themselves fraud. It is PaySim's generative
rule read back out — the same species as Tier-1's PaySim PR-AUC of 0.9998 on
`amount == oldbalanceOrg`. There is no usable tolerance band either: at ±0.1% the legitimate
match rate rises from 0.23% to **64.2%** and candidate pairs from 2,681 to 2.9M. A real
money-flow graph would need tolerance for fees and partial cash-outs; this one does not,
because the simulator copies the amount exactly.

**3. On PaySim the graph adds essentially nothing, and each round of leakage removal made that
clearer.** The ring-level lift over the pairing rule's own population went **1.30x → 1.55x →
1.19x** as contamination came out, against a base rate that climbed to 0.837. The +0.1609
lift-over-rule figure (95% CI +0.1593 to +0.1620) is real but is measured inside a population
that is 84% positive to begin with. At the capacity-capped operating point the layer flags
0.48% of rings at precision 1.000 and **recall 0.0057** — 12 true positives against 2,086
misses. This is a precision instrument at a very conservative threshold, not a detector.
**PaySim's role in this project is mechanism demonstration and the two visualisations. It is
not evidence the layer works.**

**4. The unit of analysis differs by corpus, chosen on validation.** PaySim abstains on
**100.0%** of validation *and* test transactions — origins are near-unique, so an account seen
in one window essentially never returns and there is nothing for a per-transaction score to
attach to. The ring is the only unit that exists there. IEEE-CIS abstains on 64.8% of
validation transactions and keeps the transaction. Both rates are recorded. Same move Phase 3
made in declaring Tier-2 per account.

**5. IEEE-CIS ring detection ranks well; its per-transaction projection is worse than random.**
Ring-level PR-AUC 0.6462 at 6.01x, precision 0.8889 at recall 0.1119 (TN 1,184, FP 2, FN 127,
TP 16). Projecting ring membership onto individual transactions collapses it: **0.0314 against
a 0.0348 floor, a 0.902x lift — below no-skill**. Fusing that with Tier-1 through a
validation-fitted logistic combiner moves PR-AUC from **0.5276 to 0.5244, a delta of −0.0031
with a 95% CI of [−0.0038, −0.0026]** that excludes zero *on the negative side*. The cause is
structural: a fraud ring's members transact mostly legitimately, so "is in a fraud-bearing
ring" is a weak per-transaction predictor even when the ring is correctly found. **Phase 5 must
not treat `tier3_ring_risk_score` as a proven scalar input** — both facts are automatic caveats
inside the registry entry. If Phase 5 wants this layer it should consume ring-level features:
ring risk, ring size, centrality, membership.

**6. Most of the IEEE-CIS ring signal is circular with the constructed account UID.**
`account_id` is `c{card1}_a{addr1}_d{d1n}`, so two accounts sharing the winning fingerprint
`(card1, card2, card5, addr1)` differ *only* in `d1n`. The non-circular control — device
fingerprint alone, sharing no column with the UID — was run as a third configuration and scores
far lower on validation. That gap is the honest measure of how much of the headline is "one
card fragmented into several inferred identities" rather than observed collusion. It is now
stated in the report prose and the registry notes, not only here.

### Obstacles hit and how they were solved

**Overlapping snapshot windows inflated every ring number, and it took two attempts to fix.**
Windows are much wider than the cadence — seven days against one on PaySim, thirty against
seven on IEEE-CIS — so the same ring reappears in roughly seven (respectively four) consecutive
snapshots. Left alone, the scorer was selected and scored on rings it had been fitted on, and
the bootstrap treated near-identical copies as independent draws: a 95% CI of
**[0.9931, 0.9952] on 18,846 "rings"** that were nothing like 18,846 independent observations.
The first fix keyed a ring on its **exact** member set. An `ml-evaluator` pass then measured
that this was not enough: **58% of the surviving PaySim test rings and 82% of the IEEE-CIS
validation rings still overlapped a training-split ring by at least half.** One member changing
made a new key while leaving the ring the same ring. The apparent 2% removal rate on IEEE-CIS
was evidence the *key* was wrong, not evidence its rings were independent — this log had read
it the opposite way. De-duplication is now by overlap coefficient ≥ 0.5 against every
already-kept earlier ring, via an inverted index. PaySim's test rings fell from 18,846 to
**2,507** and IEEE-CIS's to **1,329**, and the intervals widened accordingly.

**Then the fix for that was itself scoped wrongly.** De-duplication was applied to the scoring
path as well as the evaluation population, which pushed IEEE-CIS's transaction abstention rate
from 65.2% to **96.7%** — a fact about the evaluation's own bookkeeping being reported as a
property of the layer, and it dragged the transaction PR-AUC and the fusion delta with it. It
is a metric device: the scorer fits on de-duplicated training rings, *all* rings feed the score
table, and the ring metric reads the de-duplicated subset of those same scored rows so the two
views cannot disagree. Caught by reading the run's own output.

**Results were not reproducible, and the seed was never the problem.** Same code, same seed,
same `feature_version`, two runs: ring PR-AUC **0.986566 vs 0.986451**, and an `entity_cap=50`
validation score that moved by **0.010** — wider than several selection margins it was being
used to decide. `nx.connected_components` and Louvain both return *sets*, CPython randomises
string hashing per process, and Louvain's `seed` seeds its own randomness rather than the order
nodes arrive in. Components, their nodes, and the returned communities are now all explicitly
sorted; three subprocesses produce byte-identical output. The old determinism test called the
function twice in one interpreter and was structurally blind to this; the new one spawns real
processes.

**Every ring-level cost was published under the wrong denominator.** `CostEstimate.render()`
hardcoded "transactions" while dividing by a ring count, so PaySim's figure read ~60x its real
per-transaction value. `CostModel` now carries a `unit_noun` and ring costs say "per 1,000
rings". This log previously asserted "the figures are correct" under that label; they were not.

**The mandated ±50% sensitivity analysis was algebraically incapable of a result.** Scaling
both cost parameters by the same factor multiplies total cost by that factor and cannot move
the argmin — verified: threshold identical at 0.5x, 1.0x and 1.5x. It is reported because
section 3 names it, and the report now says plainly that its flatness is arithmetic rather than
evidence, pointing at the review-cost sweep, which varies the FP:FN *ratio*, as the informative
one. Both sweeps also moved off test onto validation, since they re-choose the threshold.

**The unit of analysis was being selected on the test split.** The existing guard could not
catch it: it perturbs test *labels*, and the abstention rate depends on NaN *scores*, so it was
invariant to the corruption and passed while the contamination stood. Now chosen on validation.

**The report printed a false mechanism above the IEEE-CIS headline.** "Every candidate ring
exists because a chain edge created it… the detected-ring population *is* the pairing rule's
output" was emitted unconditionally — but IEEE-CIS has no chain edge and no pairing rule, and
the star filter is deliberately skipped there. It also promised a surrogate cross-check that is
only ever run for PaySim. Both are now corpus-conditional.

**Louvain destroyed the exact structure the chain edge exists to create.** Modularity
maximisation is meaningless on a four-node path; on a 24-step window it split **121 of 121**
chain-linked components below the minimum ring size. Components at or below
`SMALL_COMPONENT_MAX` (12 accounts) are taken whole.

**A star is not a ring, and without saying so the detector finds 353,807 of them.** The
`MIN_BRANCH_NODES` filter requires two junction nodes of degree ≥2 — deliberately structural,
naming no chain edge, no amount and no entity. Skipped on the bipartite IEEE-CIS graph, where
the hub is an *entity* and several accounts on one device is the target structure.

**Single IEEE-CIS columns are buckets, not identifiers.** `card4` and `card6` hold four values
each with 98,466 accounts on one, `addr2` 67, `P_emaildomain` 59. Composite fingerprints were
measured before anything was built: `(card1, card2, card5, addr1)` reaches 86.7% coverage at a
maximum of 427 accounts per fingerprint; `(addr1, P_emaildomain)` was measured at 5,804
accounts on one value and **rejected as a hub before it reached the code**.

**`ring_density` was identically zero on IEEE-CIS**, because no two accounts are adjacent on a
bipartite graph. It now computes the bipartite fill rate. Found by reading a `describe()`.

**Feature extraction was the slowest stage, because of NetworkX views.** `Graph.subgraph`
returns a *filtered view* and every access re-runs the filter; a 30-day window took 68s.
Walking the parent adjacency dict directly cut it to 17.4s with identical output.

**A chain edge was deleting the flow edge underneath it.** `relation` was one mutually
exclusive label and a mule is routinely both endpoints, so `add_edge` replaced the attribute
dict and the observed edge vanished — inflating two scorer features. Edges now carry
independent `flow` and `chain` flags. The visualisation kept filtering on the removed label,
so every chain link was drawn grey while the legend promised red: a picture contradicting its
own caption, which no metric would have caught.

**A path-traversal guard was added and then not used everywhere.** `require_bare_model_id` /
`artifact_path` were introduced because both Tier-2's and Tier-3's inline checks missed a
Windows drive-relative id such as `C:secret`, where pathlib discards the base directory. A
`/security-review` then found `load_tier1_scores` in the new driver still building
`artifact_dir / f"{model_id}.txt"` by hand — from a `model_id` read out of `registry.json`,
which `append_entry` does not re-validate — in the very change that added the guard. All tier
save/load paths now route through it, and a test scans `app/` and fails if a call site skips it.

**Three separate fixes silently failed to apply, and two had passing tests.** `black`
reformatted the anchor text between writing a patch and applying it, so the replacement matched
nothing: the IEEE ring threshold kept using a transaction-selected operating point, `_accounts_in`
kept returning unsorted nodes, and the chain-edge render kept filtering on a dead attribute. The
ring-threshold test passed anyway because it ran on the PaySim fixture, where the ring *is* the
unit and both thresholds coincide. Every one was caught by reading run output rather than by the
suite, which stayed green throughout.

### Design decisions worth stating

- **One algorithm, two real graphs, no transported score.** `EntityGraph` is an abstract base;
  the two subclasses differ only in which edges exist. `Tier3Model.score` raises if handed a
  transaction from the other corpus rather than returning a plausible-looking number.
- **The scorer reads topology, never money**, tested behaviourally: multiplying every amount by
  1000 preserves exact-match chains, so ring features must come out bit-identical, with a
  paired guard that plants an amount-derived column and asserts it *does* differ.
- **Serving does no graph work.** Community detection happens in the snapshot job; the request
  path is a dictionary lookup. This is what makes Phase 7's Tier-3 timeout implementable.
- **One per-account score rule, in one place.** An account is scored by *its own* highest score
  across its rings. Three copies of that rule used to exist and they were not the same rule —
  the served path gave every member the ring's maximum while the offline metric took the
  per-account maximum, so the number reported and the number served were different quantities.
- **Accounts outside a ring abstain**, never 0.0, and the two abstention reasons are
  distinguishable — the model carries the snapshot's full account roster, without which every
  abstention claimed the account had never been seen.
- **De-duplication is an evaluation device with a stated, unfitted threshold.** 0.5 is a
  leakage-control choice, not selected on any split, because selecting it would mean choosing
  how much leakage to permit by looking at the result.
- **Ring-level ground truth does not exist and was not invented.** The surrogate partition is
  built from labels and labelled a surrogate everywhere, with an enrichment view that depends
  on no partition reported beside it. On both corpora the two agree (PaySim 1.19x against
  1.192x; IEEE-CIS 6.20x against 6.01x).

### Known gaps leaving Phase 4

- **PaySim contributes a 1.19x lift on an 84%-positive population and should not be presented
  as a working detector.** Its entry opens with the automatic `DO NOT QUOTE AS A HEADLINE`
  caveat.
- **`tier3_ring_risk_score` is not a proven Phase 5 input** — finding 5, and a registry caveat.
- **Most of the IEEE-CIS signal is circular with the account UID** — finding 6.
- **Operating-point recall is very low on both corpora**: 0.1119 on IEEE-CIS rings, 0.0057 on
  PaySim rings, under a 1% review-capacity cap.
- **The ring metrics describe rings that do not substantially repeat an earlier one.** A ring
  the scorer has already seen is excluded by construction, so nothing here speaks to persistent
  rings — which in a real deployment are exactly what you would want to catch.
- **Surrogate ring recovery on PaySim**: precision 0.8189, recall 0.7271, F1 0.7703 at overlap
  ≥ 0.3, over 2,507 detected against 2,822 unique surrogate rings. No surrogate check exists
  for IEEE-CIS, because the partition is built from money-flow edges that corpus does not have.
- **The cost sensitivity sweep is degenerate on IEEE-CIS**, re-choosing a threshold at the
  abstention sentinel and flagging 100% of traffic at every scale — the unbounded
  review-capacity assumption biting as the cost model's own assumption list warns.
- **The recommended IEEE-CIS transaction operating point flags 2.40% against a 1.0% cap**,
  because ring scores are heavily tied and no finite threshold lands on the cap.
- **`max_degree_centrality` can exceed 1.0 on the bipartite graph**, where a member's degree
  counts entity nodes. It is not a fraction and must not be rendered as a percentage in Phase 8.
- **16 Tier-3 registry entries exist.** The file is append-only and this phase was run
  to completion 8 times as defects were found in its outputs -- every one of them by
  reading the run's own output rather than by a failing test. Each entry now records what it
  supersedes; the authoritative pair is `tier3-graph-louvain-paysim-20260824t023943z` and
  `tier3-graph-louvain-ieee-cis-20260824t030009z`.
- **The PaySim `heldout_test` has no top-level `pr_auc`**, only the nested `ring_level` block,
  because the ring is its unit. Any consumer indexing `entry["heldout_test"]["pr_auc"]` must
  handle that.


## Phase 4 — Tier-3 Ring Detection

**Status:** Complete (with documented limitations)

**Results:**
- IEEE-CIS ring-level PR-AUC 0.6462 (95% CI 0.5703-0.7214), 6.01x lift over the 0.1076 ring
  base rate, precision 0.8889/recall 0.1119 at the 1% review-capacity cap, on 1,329 test rings
- PaySim ring-level PR-AUC 0.9977 (1.19x lift over a 0.8369 base rate, simulator/pairing-rule artifact)
- Per-transaction scalar projection: 0.0314 vs 0.0348 floor (below no-skill)
- Fusion with Tier-1: delta -0.0031 (CI excludes zero, significantly negative)

**Known Confounds & Limitations:**
- B1: `add_surrogate_recovery` selects on test split for F1 sweep — not caught by test suite
- B2: Constant-ranker interval derivation: pairing doesn't avoid base-rate shifting as claimed
- B5: Circularity evidence confounded by mismatched entity_cap (main 200 vs control 50) and base rates
- PaySim result carries DO-NOT-QUOTE caveat (simulator artifact, not generalizable)
- **Corrected 2026-08-24 (Phase 5 pre-flight):** this block previously reported the IEEE-CIS
  ring result as 0.8378 (3.67x). That pair belongs to the superseded runs `...20260823t171430z`
  and `...20260823t182232z` (ring base rate 0.2285 over 3,488 rings) and was never refreshed
  after the final run. The authoritative entry is `tier3-graph-louvain-ieee-cis-20260824t030009z`,
  which agrees with the narrative section above and with `notebooks/tier3_report.md:163`.
  Ring counts fell 4,443 -> 3,488 -> 1,329 across runs as ring de-duplication tightened.

**For Phase 5:**
- Use ring-level features (from `ring_membership`, `entity_involvement`), NOT the per-transaction `ring_risk_score` scalar
- Tier-3 will likely not survive fusion; expect meta-learner to drop it
- IEEE-CIS signal is partially UID-circular (non-circular control at matched cap: 0.3054 vs 0.2181)

**Recommendation:** Tier-1 carries the system. Proceed to Phase 5 meta-learner expecting Tiers 2-3 to add minimal value. This is an honest finding, not a failure.


## Phase 5 — Meta-Learner + SHAP

**Status:** Complete. **The headline is a negative result on the project's headline metric, and
it is reported as one.** The `ml-evaluator` gate ran three times, returning 6 blocking findings, then 4, then 3. Every
round is recorded below rather than quietly absorbed, because several findings were real bugs
already published with a confident wrong explanation attached — and two of them were *introduced
by the fix for an earlier round*, which is the most useful thing this phase learned about its own
process.

**The gate is not recorded as cleared.** All three round-3 findings were fixed and each fix was
verified directly — the Tier-1 hash now resolves to the registered `fv_c1d8eb96f693`, the report
strings are corrected at their source in `render_report`, and the supersession note is derived from
each superseded entry's own contents. But the confirming fourth round did not complete, so nobody
has audited the current state end to end. Given that two of the last three findings were introduced
by fixes to earlier findings, self-verification is not a substitute here: **Phase 6 should re-run
`ml-evaluator` against this phase before relying on any number in it.**

### The result

| model | held-out test PR-AUC | 95% CI |
|---|---|---|
| Tier-1 alone (`tier1-anomaly-lightgbm-ieee-cis-20260822t185154z`) | **0.5276** | 0.5117-0.5462 |
| Meta-learner (`meta-learner-xgboost-ieee-cis-20260824t145659z`) | **0.4954** | 0.4791-0.5141 |

Paired delta **-0.0322, 95% CI [-0.0373, -0.0273]** on n=88,581 / 3,083 positives. The interval
excludes zero **on the negative side**: on PR-AUC, fusing the layers is measurably worse than
Tier-1 alone. Shipped operating point (threshold 0.6035, cost-chosen on V-late under a 1% capacity
cap, realising 1.07% on test): precision 0.8282, recall 0.2549, F1 0.3899, confusion
TN 85,335 / FP 163 / FN 2,297 / TP 786, estimated cost 4,711 per 1,000 transactions.

**The cost comparison does not agree with the ranking comparison,** and the standards require the
recommendation to be justified by cost rather than by a rank metric alone. At a matched 1% flag
rate — both cuts taken as quantiles of the **test** score vectors, which is what makes the flag
rates comparable, and computed only after the shipped thresholds were fixed on validation:

| model | flag rate | precision | recall (count) | recall (**value**) | cost per 1,000 |
|---|---:|---:|---:|---:|---:|
| meta-learner | 0.94% | 0.8257 | 0.2228 | **0.1689** | **4,817.97** |
| Tier-1 alone | 1.00% | 0.8646 | 0.2485 | 0.1500 | 4,903.53 |

The fusion catches **less fraud by count and more by value**, so under the project's cost model it
is cheaper by roughly 86 per 1,000 transactions — about 1.7%. That speaks directly to the gap
Phase 2 flagged as "the concrete opening for Phase 6's causal cost layer" (Tier-1 catching 24.6%
by count against 14.6% by value).

**Recommendation for Phase 6: consume Tier-1's score, and treat the meta-learner as an open
question rather than a closed one.** Tier-1 wins decisively on the headline metric. The cost
advantage runs the other way but is small, is a bare point estimate with **no confidence
interval** — this project has `bootstrap_pr_auc_delta` but no equivalent for a cost or
value-recall difference — rests on the cost model's stated assumptions, and turns on 1.9
percentage points of value recall on a quantity dominated by a handful of large transactions. Not
enough to ship on, too specific to discard.

### The ablation — the question Phases 3 and 4 deferred here

Fitted on V-fit, arbitrated on V-arb (31,003 rows, 1,131 positives), paired bootstrap, keep only
on a strictly-positive lower bound:

| block | leave-one-out | add-one | verdict |
|---|---|---|---|
| tier1 | **+0.4263** [+0.3969, +0.4535] | — (exempt) | retained — Tier-1 *is* the system |
| engineered | +0.0048 [-0.0041, +0.0137] | — (exempt) | retained by exemption; no evidence it adds over Tier-1 |
| tier2 | +0.0020 [-0.0012, +0.0049] | +0.0012 [-0.0011, +0.0035] | **retired** |
| tier3_served | -0.0001 [-0.0030, +0.0032] | +0.0008 [-0.0034, +0.0051] | **retired** |
| tier3_topology | **-0.0061 [-0.0100, -0.0026]** | -0.0047 [-0.0103, +0.0010] | **retired — measurably harmful** |

Retirements are worded as *"retired for want of evidence at n=31,003"*, not as "measured to add
nothing" — at this sample size most of these intervals are ties, and the difference between "we
could not detect an effect" and "there is no effect" is the difference between an honest claim and
a false one. `tier3_topology` is the exception and is worded differently: its leave-one-out
interval lies entirely below zero, so removing it is an improvement, not a simplification.

`tier1`'s +0.4263 is the most important number in the phase. The fusion layer's entire predictive
content is Tier-1's score, and the engineered passthrough adds nothing detectable on top — expected,
since Tier-1's LightGBM was already fitted on a superset of exactly those 23 features.

### Why it loses: two candidate causes, only one of them ruled in

**The out-of-fold handicap, measured.** Stacking needs out-of-sample inputs, so Tier-1 was
re-scored with forward-chaining folds (5 blocks, block 1 dropped, per-fold input-spec refit, 1,917
rounds fixed from the registered `best_iteration`):

| fold | train rows | train positives | validation PR-AUC |
|---|---:|---:|---:|
| 2 | 82,675 | 2,221 | 0.4200 |
| 3 | 165,351 | 4,571 | 0.4587 |
| 4 | 248,026 | 8,063 | 0.4776 |
| 5 | 330,702 | 11,180 | 0.5202 |
| **full-train (what serves)** | **413,378** | **14,538** | **0.6155** |

The meta-learner was fitted against a `tier1_score` column worth 0.42-0.52 and applied to one worth
0.6155. The fold-5-to-full-train jump is far larger than 25% more data explains — the full-train
window ends immediately before validation, so **recency**, not volume, does the work.

**The second cause, which the first write-up missed entirely: the model barely fitted.** Early
stopping selected **iteration 2**; the shipped model is three trees. That is consistent with the
ablation — there is almost nothing to learn beyond `tier1_score` — but it means the loss is
**confounded** between the handicap and simply not fitting, and this phase did not separate them.
The discriminating diagnostic (refit on the in-sample Tier-1 column, compare on the arbiter) was
not run. An earlier draft claimed this was "a structural limit, not a tuning problem"; that was
asserted rather than demonstrated and has been withdrawn.

### Calibration

Platt scaling on the margin, two floats, plain JSON. **Test Brier 0.0233, ECE 0.0037** against a
3.4804% base rate, across 10 populated bins — 83,918 rows predicted 0.0141 against 0.0151 observed,
1,674 rows predicted 0.1543 against 0.1553. Probabilities span 0.0093 to 0.9972 with a median clean
row at 0.0099. These are usable as probabilities, not merely as a ranking.

Isotonic remains the documented alternative at ECE 0.0022 — 1.7x better, the small gap expected
between a flexible calibrator and a two-parameter one. The sigmoid ships because it is strictly
monotone and leaves PR-AUC exactly intact, while isotonic ties scores and moves the headline metric
by an artefact of step width.

That 1.7x is worth contrasting with what an earlier run reported: **16x**, alongside ECE 0.0348 and
a review threshold of 3e-106. That was not a property of the data. It was a bug — see obstacles.

### Tier-2 contamination: predicted, measured, and severe

Tier-2's autoencoder fits on fraud-free train windows excluded **by account**, so on train every
fraud row belongs to a wholly-withheld account while most clean rows do not. The control group
holds the label constant and varies only fit membership. Across 35,899 included and 28,569 excluded
clean rows and 5,856 fraud rows:

- **memorisation AUC 0.9501** (0.5 = no effect). Fit membership alone almost perfectly separates
  the reconstruction error.
- naive train fraud/clean gap: AUC 0.8628.
- **residual AUC 0.3649** — with fit membership held constant, Tier-2 scores fraud as *less*
  anomalous than clean traffic. Its train-split signal is not weak; it is **inverted**, and the
  apparent discrimination is the autoencoder recognising rows it was fitted on.

Arbitrated on validation, where no tier is in-sample. Both columns agree Tier-2 does not clear the
bar: arbiter leave-one-out +0.0020 [-0.0012, +0.0049], train-fit leave-one-out
-0.0047 [-0.0080, -0.0011].

### The explanation smoke test — read, not just executed

Ten test transactions (five highest-risk, five random), attributions read:

- **`tier1_score` is the top contributor in all ten**, an order of magnitude above everything else.
  All five top-risk rows are genuinely fraud and correctly blocked, and the secondary features are
  plausible — but every explanation reduces to "Tier-1 said so", which sends a reviewer back to
  `tier1_anomaly.explain`. A fusion layer whose attribution restates one input is not explaining a
  fusion.

It also caught a serving bug before it shipped: `build_vector` rejected a *null* engineered feature
as an error, but `seconds_since_prior_txn` is legitimately null on an account's first transaction
and the training matrix carried such values through as NaN. Refusing them would have made the layer
unable to score any account's first transaction. Now a null becomes NaN while an absent key still
raises, with a paired test for each.

### Obstacles

The two worst were both **bugs that shipped with a confident wrong explanation attached**, which is
the pattern most worth recording.

- **The Platt calibrator was fitted on one scale and applied to another.**
  `CalibratedClassifierCV` prefers `decision_function` and falls back to `predict_proba`; the
  booster wrapper exposed only the latter, so `(a, b)` were fitted against values in [0,1] while
  `score_frame` applied them to log-odds. The sigmoid was then evaluated far outside its fitted
  range and every probability collapsed toward zero: slope **-97.5** where a margin-scale fit gives
  order 1, review threshold 3e-106, ECE 0.0348 (exactly the base rate), all 88,581 rows in a single
  calibration bin. **All of that had been written up as "three trees compress the margins"** — a
  symptom explained instead of a cause diagnosed, and the 16x isotonic gap that should have been
  the tell was reported as a curiosity. Fixed by adding `decision_function`; ECE went 0.0348 →
  0.0037 and the threshold 3e-106 → 0.6035. PR-AUC and the delta did not move, because the map is
  monotone either way.
- **Early stopping selected an iteration count that scoring then ignored.** `Booster.predict()`
  uses every tree unless given an `iteration_range`, so the model was scored 100 rounds past its own
  validation-selected optimum, and a reloaded artefact honouring `best_iteration` would not have
  reproduced the registered metrics. Fixed by pinning the range through margins, contributions,
  calibration and the sidecar. Correcting it *improved* test PR-AUC (0.4937 → 0.4954).
- **The Tier-1 baseline's threshold was chosen on the test split** — `np.quantile(tier1_test, 0.99)`
  — while `EvaluationResult.render` printed "chosen on validation", so the artefact contradicted
  itself. It handed Tier-1 the test flag rate while the meta-learner transferred its threshold blind
  from V-late; precision falls with flag rate, so the comparison flattered whichever model was
  allowed to peek. Fixed: the baseline threshold now comes from validation and lands at 0.813710,
  byte-identical to Phase 2's registered capacity point. PR-AUC is threshold-free so the delta was
  never affected; every threshold-dependent baseline figure in the first write-up was.
- **The retirement sentence was hardcoded.** Every non-retained block was described as "the interval
  does not exclude zero" — false for `tier3_topology` at [-0.0100, -0.0026]. The report and registry
  asserted a falsehood about their own numbers, and *under*-claimed. It did not exist in the previous
  run, where that interval straddled zero, which is exactly why no test caught it.
- **The mandatory false-positive cost was computed for the meta-learner but not the baseline**, so
  the ship/no-ship recommendation rested on PR-AUC alone. Adding it reversed part of the conclusion.
- **A quantisation bug that cost 0.0073 PR-AUC.** The map making Tier-1's serving scores
  commensurable with out-of-fold ranks used `searchsorted` bucketing, collapsing 88,069 distinct test
  scores onto 1,024 grid points — damaging the meta-learner's strongest feature and depressing the
  baseline to 0.5202. Caught by noticing the baseline disagreed with Phase 2's published figure.
  Fixed by linear interpolation.
- **The deny-list guard could not see the column it most needed to catch.** It compared *prefixed*
  Tier-3 names against an *unprefixed* deny list, so `tier3_account_is_fraudulent` — a direct label
  read — would have passed every guard and every test. The `TIER3_CARRIED_COLUMNS` allowlist was the
  only real defence. Fixed to strip the prefix, with a test planting the prefixed name.
- **A value-recall column reversed the recommendation and was produced by nothing.** It had been
  computed in an ad-hoc shell session and pasted into two documents. It is now emitted by the run.
- **`CalibratedClassifierCV(cv="prefit")` was removed in scikit-learn 1.9.** The working path is
  `FrozenEstimator`, which needs the wrapped booster to satisfy `check_is_fitted` — supplied via
  `__sklearn_is_fitted__`.
- **`import shap` fails this repo's test suite** (PendingDeprecationWarning from matplotlib under
  `filterwarnings = ["error"]`). XGBoost's native `pred_contribs=True` is the same TreeSHAP
  algorithm, verified bit-identical (max abs diff 0.0). `shap` is declared but imported nowhere, and
  a test pins that.
- **A provenance fix that made provenance worse.** Round 2 asked for Tier-1's `feature_version`
  to be recorded. `load_registered_tier1` rebuilds the spec from the sidecar with `dropped=()`,
  but `dropped` is hashed into the feature version, so the recorded hash resolved to nothing in
  the registry — a traceability chain that looked intact and was not, which is worse than the
  omission it replaced. Fixed by reconstructing `DroppedColumn` from its string form and asserting
  the rebuilt hash equals the registered one, raising rather than writing an unresolvable value.
  It now reads `fv_c1d8eb96f693`, matching Tier-1's entry.
- **A string patch that shipped garbled prose into the metrics report**, and a stale sentence
  describing the calibrated probability scale that was left behind by the calibration fix — so the
  authoritative report still told a reader the model emitted collapsed probabilities after it had
  stopped doing so. Both corrected in `render_report` rather than in the generated file, so they
  cannot regress on the next run.
- **The supersession note listed defects from a hardcoded literal.** It named three defects by hand
  and omitted the calibration bug that produced the very entry it was superseding — whose cost
  figure is 6.5% *more* flattering than the corrected one, sitting unmarked in an append-only
  record. The note is now derived from each superseded entry's own contents.
- **Tier-3's fitted ring scorer is not serialised**, so Phase 5 refits it from rolling snapshots. The
  saved artefact could not have been used anyway: its score table is a single snapshot ending
  2018-06-02, *after* the test period.

### Test discipline, stated plainly

Every selection was made on validation: retained blocks on V-arb, calibrator and both thresholds on
V-late, the early-stopping round on V-fit. **Test selected nothing.** It was not, however, "scored
exactly once", and an earlier draft said so incorrectly. Test was read by the isotonic baseline, by
the matched-flag-rate table, and across four runs — the quantisation fix and two rounds of gate
fixes. The quantisation re-run moved the meta-learner by 0.00002 and corrected the baseline by
0.0074, making the shipped result *less* flattering. Every superseded registry entry is named by the
current one, generated from the registry rather than hardcoded.

### Known gaps

- **The loss is confounded** between the out-of-fold handicap and `best_iteration=2`. The
  discriminating diagnostic was not run.
- **The cost advantage has no confidence interval.** There is no bootstrap for a cost or
  value-recall difference in this project.
- **Tier-3's ring scorer is in-sample on train**, with no out-of-fold remedy — the likely reason the
  train-fitted column rates Tier-3 differently from the arbiter. It does not reach the shipped model,
  which retains no Tier-3 column.
- The two ablation columns differ in **fit size as well as contamination** (330,703 rows against
  35,432), so their disagreement is not cleanly attributable to either.
- **`rank_normalise` ranks within the scored block**, so a train row's feature depends on later rows
  in the same block. It is a non-causal transform and a train/serve construction mismatch against the
  CDF map used at test. Conservative in direction — it can only degrade the fit, not inflate the
  held-out number — but unquantified.
- **No test asserts** that the refit ring scorer reproduces the registered Phase 4 score table.
- **No false-negative profiling code for this layer**, unlike Tiers 2 and 3. The observed-failure
  analysis in `app/models/README.md` was computed by hand and does not regenerate.
- Per-fold Tier-2 refits would remove the eligibility artefact properly. Estimated 30 epochs x 4
  folds on CPU; deferred.
- Tier-1's round count was early-stopped on the full validation split in Phase 2, so all three
  validation slices carry one Tier-1 hyperparameter tuned on them.
- **Latency was never benchmarked.** `benchmark_latency` exists and is unused; the registry entry
  carries an empty `latency` block. Phase 10 must fill it.
- **`EvaluationResult.render` hardcodes "(chosen on validation by ...)"** for every caller and formats
  thresholds at `%.6f`. The first made the baseline criterion self-contradicting; the second renders
  small thresholds as `0.000000`. Both are latent traps for Phase 6.
- Phase 7 must honour four carried security gates: routes need auth; the `audit_log` table needs RLS
  forced and a `top_features` column; a Redis limiter must fail closed; and `POST /score` must **not**
  return `top_features` — `MetaResult.public()` exists for exactly that.

## Phase 6 — Causal Cost Layer

**Status:** Shipped. `plug_in` cost-aware ranking is **22.41% cheaper** than Tier-1's probability
ranking at a matched 1% flag rate, CI [−1,345.28, −881.81] per 1,000, excluding zero. The phase's
secondary hypothesis — that cost-sensitive *training* beats cost-sensitive *thresholding* — was
tested directly and **came back a tie**, which is exactly what this phase's own algebra predicted.
The bootstrap for a cost difference that Phase 5 recorded as missing now exists and is what the
headline rests on.

**All three gates ran and returned 17 findings; 16 were fixed and one was escalated for a decision.**
Two were metric bugs that would have shipped as published numbers, and one was a selectively-applied
honesty rule. Unlike Phase 5, whose `ml-evaluator` gate never returned a clean round, nothing here was
left open — the detail is in the gate section below, and it is the most useful part of this entry.

**The brief's central instruction was not executable, and finding out why produced the phase.** It
asked for IPW on historical actions. There are none. That is not a data-quality complaint — it
determines what a causal cost layer on this corpus can be, and the answer turned out to be more
interesting than the original plan.

### The instruction that could not be followed

Checked against raw headers rather than assumed. IEEE-CIS carries 394 transaction columns and 41
identity columns; none is a decision, decline, review, dispute, chargeback or refund. `M1`–`M9` are
Vesta address-match flags, not review outcomes. PaySim carries exactly one action-like column,
`isFlaggedFraud`, and it fails three independent ways: it is the simulator's own hardcoded rule
(single TRANSFER above 200,000), so propensity is exactly 0 or 1 and `1/e(x)` is undefined; it is
nested inside the label, so the treated arm has no control counterfactual; and it fires on 16 rows
out of 6,362,620. `app/data/adapters.py:339` already drops it as "a leaked downstream decision".

Every transaction in this project's data was allowed. An ATE recovered from it would have been a
fabrication, and `PHASE_PROMPTS.md:345` had in fact already anticipated this by asking for a DR-Learner
"on a synthetic cost model" — a simulated treatment, not a recovered one.

### What replaced it, and why it is stronger

Under the stated cost model both potential outcomes are deterministic given `(label, amount)`:

```
cost(block | Y) = (1 - Y) * r          cost(allow | Y) = Y * (A + f)
tau(x) = (1 - p(x)) * r  -  p(x) * (A + f)
```

Every term is either known before the decision or is `p(x)`. There is no confounding for a doubly
robust correction to remove because there is no treatment whose assignment could be confounded. **The
DR-Learner does not fail here, it collapses** — provably and exactly — onto a cost-weighted plug-in
driven by a calibrated probability. The proof is in the `causal_cost` module docstring and asserted by
`test_break_even_threshold_is_the_root_of_the_treatment_effect`, so it cannot rot into a comment.

Two things follow. The block threshold stops being global and becomes `p > r/(A+f+r)`, which moves
with the amount — across the test split it spans 0.000557 to 0.162171, median 0.034682. And the one
place learning could still add something is regressing realised loss directly instead of reaching it
through `p(x)·(A+f)`. That is `learned_loss`, and the run measured it rather than assuming it.

The DR machinery was kept for **off-policy evaluation against a simulated logging policy**, where
truth is exactly computable from the labels and the estimators can therefore be *validated*. On the
shipped policy: direct method −8.66% bias, IPW +2.19%, DR +1.35%. Stated plainly in the report: a
correct propensity makes IPW unbiased by construction, so this shows the estimators are implemented
correctly, and cannot show which wins when the propensity must be estimated.

### The result

At a matched 1% flag rate on held-out test (n=88,581, positives=3,083):

| policy | precision | recall (count) | recall (**value**) | cost / 1,000 |
|---|---:|---:|---:|---:|
| `probability` (Tier-1) | 0.8646 | 0.2485 | 0.1500 | 4,902.49 |
| `plug_in` (shipped) | 0.4955 | 0.1424 | **0.3698** | **3,804.02** |
| `learned_loss` | 0.5056 | 0.1453 | 0.3453 | 3,932.14 |

| comparison | cost delta / 1,000 | 95% CI | verdict |
|---|---:|---:|---|
| `plug_in` vs `probability` | −1,098.48 (−22.41%) | [−1,345.28, −881.81] | excludes zero |
| `learned_loss` vs `probability` | −970.35 (−19.79%) | [−1,189.92, −769.45] | excludes zero |
| `learned_loss` vs `plug_in` | +128.13 (+3.37%) | [−30.41, +300.03] | **TIE** |

The baseline row reproduces Phase 2 and Phase 5 exactly — PR-AUC 0.5276, value recall 0.1500, cost
4,902.49 against Phase 5's 4,903.53 — which is the cross-phase consistency check that makes the rest
believable.

**Count recall falls 43% while value recall rises 147%.** It is not finding more fraud; it is finding
more expensive fraud. That is also the argument against a leak: PR-AUC *drops* to 0.3194, nowhere near
the 0.95 suspicion wire, and a leak does not lower your ranking metric.

**The tie is the more interesting finding.** Cost-sensitive training buys nothing over cost-sensitive
thresholding here, and the collapse proof says why: given a calibrated probability, the plug-in is
already the correct combination of the quantities, so fitting a second model to reach the same target
adds estimation error and no information. `BUILD_LOG:391` asked for exactly this comparison and the
answer is negative — recorded, not buried.

### The headline is regime-dependent, and that is the caveat that matters most

Under a card-not-present cost model (FP 50, FN amount+500) the 22.41% advantage falls to **−2.22%, CI
[−582.18, −145.50]** per 1,000. The interval still excludes zero — so cost-aware ranking is genuinely
cheaper under CNP pricing as well, just by an order of magnitude less. The tie rule is applied to this
number too, because applying it only to the figure that flatters the phase would be selective; the
ml-evaluator gate caught precisely that omission.

The mechanism is arithmetic, not noise: value weighting can only exploit the share of false-negative
cost that *varies* across transactions. At a 15.00 fee against a median test amount of 68.50 that
share is 82.0%; at a 500.00 fee it is 12.0%, the flat fee dominates, every miss costs about the same,
and cost ranking collapses most of the way back toward probability ranking.

The honest claim is therefore conditional — **cost-aware ranking pays in proportion to how
heterogeneous the loss is** — and the report computes both figures rather than quoting the flattering
one. This is written into the registry notes as well, so the number cannot travel without it.

### Obstacles

- **The no-treatment finding itself.** Cost roughly half a day to establish properly: raw CSV headers,
  `raw_spec.py` column assertions, the adapter drop, and a cross-tab of `isFlaggedFraud` over all
  6.36M PaySim rows to confirm 16 positives and zero treated-negatives. Worth every minute — building
  the originally-specified IPW layer would have produced confident numbers that meant nothing.
- **`CostEstimate` name collision.** The brief specifies `estimate_cost(...) -> CostEstimate`, but
  `app.ml.cost.CostEstimate` already exists as a *population-level* type imported by all four training
  drivers. The per-decision type is `DecisionCost`; the deviation from the brief is documented at the
  class.
- **econml was reserved for this phase and deliberately not adopted.** Its published pins cap
  scikit-learn below 1.6 and numpy below 2 against this repo's 1.9 / 2.4 / pandas 3.0;
  `filterwarnings = ["error"]` turns any import-time deprecation into a collection failure, which is
  exactly how `shap` became a declared-but-unusable dependency; and its folds are random, where this
  data needs chronological ones. The DR/AIPW estimators are ~40 lines of numpy in `app/ml/ope.py`,
  fully typed under mypy strict, and testable against exact truth. `requirements.txt` records the
  reasoning where the next person will look for it.
- **`fit_loss_model` recorded `chargeback_fee=0.0`** instead of the fee its target was built with — a
  model that would be meaningless read against any other fee. Caught by the test written to pin that
  field, which is the argument for writing it.
- **Every flag rate in the sensitivity tables rendered as `0.00%`.** Reconstructing `SensitivityRow`
  from its own `to_dict()` dropped `rows` and `flagged`, and `flag_rate` is derived from them. Caught
  by reading the smoke output rather than by a test. Fixed by carrying the real objects for rendering
  and the dicts for the registry.
- **A confusion matrix and a cost block from two different operating points, inside one result.** The
  shipped result rendered its matrix at the V-late threshold (1,203 rows flagged) and its cost at the
  test 1% quantile (886 flagged). Both numbers were individually correct and the block was a
  self-contradiction. Found by reading run 1's own output and checking that `fp + tp` matched the
  cost block's `flagged`; it would otherwise have shipped. Both now come from the shipped threshold,
  and the report explains why the shipped flag rate is 1.36% rather than exactly 1%.
- **`learned_loss` vs `plug_in` needed its own interval.** Both were originally compared only against
  `probability`. Reading "the loss regression adds nothing" off two overlapping intervals that share a
  baseline is not a valid comparison. Added after run 1 showed the two were close — disclosed here
  because the ordering matters, and noting that the added test could only *weaken* the phase's
  secondary claim, which it duly did.
- **Two test failures that were the tests' fault, not the code's**, both verified before changing
  anything. The forward-chaining test shuffled `event_time`, but `time_blocks` recomputes the
  assignment from whatever times it is handed, so no violation was ever planted; it now plants one at
  `assert_forward_chaining` directly, where the guard actually lives. The cheaper-policy test used a
  barely-informative probability, so multiplying by an independent amount added more noise than
  signal — which is itself a real property of the method and is now stated in the test.
- **`TaskStop` kills the shell, not the Python child.** Two runs believed stopped ran to completion
  and appended registry entries; three duplicate `causal_cost` entries and their artefacts were found
  by `git status` before any commit. All were uncommitted, so `git checkout HEAD -- models/registry.json`
  restored the record. Nothing discarded ever entered project history, and the append-only contract
  holds. The one benefit: three independent runs produced byte-identical results, which is a free
  determinism check on the seed.

### The gate round — 17 findings, 16 fixed, 1 escalated

All three gates ran (`/code-review high`, `ml-evaluator`, `security-reviewer`). Unlike Phase 5, whose
gate never returned a clean round, every finding here was either fixed or explicitly decided. The two
that mattered most were metric bugs that would have shipped as published numbers.

**The sensitivity sweep was evaluating a policy nobody ran.** `plug_in` ranks by
`p*(A+f) - (1-p)*r`, so the score *embeds* both cost parameters. `sensitivity_sweep` scaled the cost
model but reused the scores computed at 1.0x, which re-thresholds the original policy against a
different cost function rather than evaluating the scaled one. This was safe in Phases 2-5 — there the
score is a probability and carries no cost inside it — and is the kind of bug that a shared helper
inherits silently when the thing being swept changes character. The score is now recomputed under each
scaled model in `scores_under`.

**The card-not-present row for `learned_loss` priced a fee-15 model at fee-500.** The Tweedie target
was `Y*(A+15)`, so under a different fee the prediction needs correcting by `(f2-f)*p`.
`LossModel.chargeback_fee` was recorded for exactly this check and was never consulted, so the row was
not the policy it claimed to be. Same helper, same fix.

**The tie rule was being applied selectively.** The headline carried an interval; the number that
*qualifies* the headline — the card-not-present advantage — was a bare point estimate. Applying a
pre-registered honesty rule only to the figure that flatters the phase is exactly the failure the rule
exists to prevent, and the gate was right to call it. With the interval computed, CNP comes out at
**−2.22%, CI [−582.18, −145.50]**: still excluding zero, so genuinely cheaper, but an order of
magnitude smaller than the headline. Worth recording that the sampled smoke run showed this as a tie
and the full test split did not — a 15k sample has nowhere near the power to resolve a 2% cost
difference, which is its own argument against reading anything off `--sample` output.

**A figure in "what this does NOT catch" was wrong by 3.3x, in the flattering direction.** The README
said "roughly 100 more frauds get through"; true positives fall 766 → 439, so it is **327**. In the one
section whose entire purpose is to be unflattering.

**`format_threshold` was a self-inflicted regression.** The `%.6e` fallback added earlier this phase to
fix Phase 5's `%.6f` collapse is *less* faithful than what it replaced for any threshold above 10 —
90.85343882831513 renders as `9.085344e+01`, a round-trip error of 1.2e-6 against `%.6f`'s 1.7e-7. Six
significant digits is not enough for a cost-scale threshold, and the fix for "not enough digits" is
more digits, not a different exponent. Now: fixed notation when it round-trips exactly, `repr`
otherwise.

Also fixed: a confusion matrix and cost estimate sourced from different cuts inside one registry key
(`cost_per_transaction`, now split into two explicitly-named keys); missing raw TN/FP/FN/TP and
thresholds on the headline matched-flag-rate table; two sensitivity tables computed on V-late and
rendered with no split label inside a test report; the audit-extremes section citing the shipped
threshold while being computed at the matched one; `estimate_cost` silently falling back to the plug-in
for a `learned_loss` policy while `ranking_score` raised; an unbounded `amount` where a negative value
would invert the ranking score and switch blocking off for that row; and the Tweedie booster never
reaching disk despite the registry advertising a full `loss_model` block.

### Security: two disclosure findings, one escalated rather than decided

**`DecisionCost.to_dict` was contracted as an API shape and is a complete evasion oracle.** The sign of
`expected_saving_from_blocking` *is* the decision boundary; with the probability and the two cost arms
a caller recovers the whole cost matrix and can binary-search the largest amount that evades review at
a given risk score. `app/core/audit.py` already carries this warning on `top_features`, and this object
is strictly worse — `top_features` says which features mattered, this says how far from the boundary
you are and which way to move. Renamed to `to_audit_dict` with the constraint stated at the method, and
Phase 7's carried gate is widened from "must not return `top_features`" to "must not return any field
of `DecisionCost`".

**A real control gap, unrelated to this phase but found by its review.** `.trufflehog-exclude` carried
an unanchored `/data/`, which matches `/src/backend/app/data/` as readily as the intended `/src/data/`
— exempting ten application source files from the *blocking* secret scan, directly under a comment
reading "Never application source". No secret was present, so this was a blind spot rather than a leak.
Anchored to `^/src/data/` and verified in both directions.

**The one finding escalated rather than decided.** The reviewer argued that publishing the shipped
threshold and the cost constants in tracked files lets an adversary solve
`A <= (90.853 + 3(1-p))/p - 15` for the largest amount that evades review at any risk score, and
recommended moving them to a gitignored sidecar. That directly contradicts ml-evaluation-standards
section 2, which requires the operating threshold to be stated — two project skills pointing opposite
ways, with the resolution depending on facts the code cannot settle (is the repo public, will the model
serve traffic). Escalated. **Decision: keep the thresholds and cost constants, drop the worked
examples.** The exploit has no live target — there is no endpoint, and IEEE-CIS is public data — while
a judge who cannot see the operating point cannot verify the headline. But the ten identified
transactions with their exact probabilities and features were adding attacker value and were required
by nothing, so `audit_extremes` no longer reaches the registry and `transaction_id` no longer appears
anywhere in the driver. The aggregate failure modes stay in the report and README, which is what
section 4 actually asks for.

**Carried, not fixed:** RLS is defined but inert, because `docker-compose.yml` connects as the table
owner and superusers bypass row-level security unconditionally. Pre-existing, acknowledged in the
migration's own docstring, not reachable today because no code path reads those tables. Phase 7 must
grant login to `riskiq_app` and repoint `DATABASE_URL` before merging any read route.

### Carried Phase 5 traps, both closed

- `EvaluationResult.render` hardcoded `"(chosen on validation by ...)"` for every caller, which made
  criteria that were *not* chosen on validation read as self-contradictory. Callers now supply the
  full phrase.
- Thresholds formatted at `%.6f` printed Phase 5's 0.00175-wide review/block band as two identical
  strings and Tier-3's 0.999995 as `1.000000`. `format_threshold` now round-trips and falls back to
  scientific notation when fixed notation would lose a digit; `test_format_threshold_does_not_collapse_distinct_values`
  pins it using the actual Phase 5 pair.

### Test discipline, stated plainly

**Test was scored by five completed runs, and that number should be stated rather than rounded down.**
One was discarded for the operating-point bug, two were orphans (`TaskStop` kills the shell but not the
Python child, so two runs believed stopped ran to completion and appended registry entries), one was
discarded after the gates, and the last is the one that stands. Every one used seed 42 and identical
selection logic, and the three that overlapped produced byte-identical figures — which is a free
determinism check, and the reason the repeats cost nothing in validity.

**Nothing on test selected anything, in any run.** Every threshold was chosen on V-late; the calibrator
was fitted on V-fit; the loss regression early-stopped on V-fit; `plug_in` was fixed as the headline by
the collapse proof rather than by a comparison. The changes between runs were correctness fixes, report
labelling, and two added comparisons — no model, threshold or hyperparameter moved, and the headline
`−1,098.48 / CI [−1,345.28, −881.81]` is identical across every run that produced it.

**Two comparisons were added after seeing test output, and both could only weaken the phase's claims.**
The `learned_loss` vs `plug_in` interval was added when run 1 showed the two were close; it returned a
tie, retiring the phase's secondary hypothesis. The card-not-present interval was added because the
gate pointed out the tie rule was being applied selectively; it returned a significant but ten-times-
smaller advantage, which narrows the headline's scope. Adding a test that can only cost you something
is not the failure mode the rule guards against, but the ordering is disclosed because it is the
reader's call, not the author's.

**Every discarded registry entry was uncommitted.** `HEAD` carried 26 entries throughout; the working
tree was restored with `git checkout HEAD -- models/registry.json` each time and orphaned artefacts
deleted. The append-only contract holds: nothing discarded ever entered project history.

The `--sample` runs used for iteration wrote to a scratchpad with `--skip-registry` and touched no
reported figure. One lesson from them, recorded because it nearly misled this entry: a 15k sample
showed the card-not-present comparison as a tie, and the full split showed it excluding zero. Sampled
output is for shaking out crashes, not for reading results.

Suite: 442 passed, 1 skipped (database unavailable, pre-existing), up from 395 — 47 of them this
phase, including a regression test pinning the exact invariant that caught the operating-point bug.
`ruff` and `mypy --strict` clean across 48 source files.

### Known gaps

- **The cost model is still assumption, not measurement.** Review cost 3.00 and chargeback fee 15.00
  are stated figures. The sensitivity sweeps exist because the recommendation moves with them, and the
  card-not-present table shows how far.
- **False-positive cost is flat.** A heterogeneous FP cost — churn probability times customer lifetime
  value — was considered and rejected: neither churn nor CLV is observable here, and 57.7% of IEEE-CIS
  accounts have exactly one transaction, so CLV is undefined for most of them. That would have been
  assumption stacked on assumption.
- **The bootstrap holds the fraud count fixed** (stratified resampling), so the intervals describe
  uncertainty in *which* frauds and therefore in their amounts, not in the base rate. On a corpus where
  cost is dominated by a handful of large transactions this widens them honestly, but it is not a
  general-purpose cost interval.
- **`review` and `block` are priced identically.** A hard decline damages a customer relationship in a
  way a review queue does not, and nothing in this layer can see the difference.
- **The out-of-fold loss column is computed and then used only for a coverage diagnostic.** Four
  LightGBM fits over 413k rows for one number in the notes. Defensible as proof the forward-chaining
  machinery runs on real data, but it is not cheap and nothing downstream consumes it. To be explicit,
  because the Limitations wording previously invited the wrong inference: **every reported
  `learned_loss` figure comes from the single train-fitted booster scoring test**, which is the correct
  construction. None of them rests on the out-of-fold column.
- **Row order is part of reproducibility.** `LOSS_PARAMS` enables bagging and feature sampling, both of
  which draw against row position; a permuted parquet fits a different booster. Measured (identical
  predictions with sampling disabled), documented at the constant, and pinned by
  `test_out_of_fold_loss_is_reproducible`, which deliberately asserts same-order reproducibility rather
  than order invariance.
- **Phase 5's `ml-evaluator` gate still never returned a clean round.** Phase 6 consumes Tier-1
  directly rather than the meta-learner, which limits the exposure, but the Phase 5 comparison figures
  quoted in passing inherit it.
- **Latency was never benchmarked for this layer either.** The cost policy is arithmetic on two
  floats, so the true cost is Tier-1's, but the registry entry carries no latency block. Phase 10.

---

## Phase 7 — Backend, Audit Trail, Security Hardening

**Status:** Shipped. Four authenticated endpoints, an append-only audit log with row-level
security actually in force, a fail-closed Redis limiter, and a degraded-mode fallback that
records why it degraded. Suite went 442 → **579 passed, 1 skipped** (the pre-existing
database-unavailable skip). `ruff` and `mypy app/` clean across 52 source files.

The three findings worth reading are the encoder gap, the assembly latency, and the decision
about which layers are actually in the decision path. None of them were in the plan.

### The obstacle that defined the phase: the pipeline throws away its own encoders

`POST /score` has to accept *raw transaction fields* and assemble the feature vector
server-side — a constraint carried from the Phase 2 security review, because `FeatureVector`
is unbounded and an endpoint taking a caller-supplied vector lets that caller choose its own
score while leaving a correct-looking audit row.

Assembling it turned out to be blocked. Tier-1 reads 113 features. 82 are raw row columns and
27 are native categoricals — both arrive in the payload. Four are `freq_*` encodings that
Tier-1 fits itself, and those ship inside its sidecar, so they load with the model. The
remaining four — `freq_ProductCD`, `freq_card4`, `freq_card6`, `freq_P_emaildomain` — are
fitted by the **Phase 1 pipeline**, applied to the parquet, and then **discarded**. Only their
digest survives, folded into the `feature_version` hash. Training never noticed, because
training reads the already-encoded parquet. Serving is handed a raw `ProductCD` and has to
produce the number the model was fitted against, and there was nothing to look it up in.

Re-running the pipeline to persist them would have been the obvious fix and the wrong one — it
is a multi-hour job over 3.3M rows to recover four tables that are a deterministic function of
data already on disk. `app/data/serving_encoders.py` refits them from the processed train
parquet instead, using `fit_frequency_encoders` itself rather than a reimplementation, and then
**verifies** the rebuild by re-deriving the `freq_*` columns and comparing them against the
ones the pipeline wrote into the same file. A rebuild that cannot reproduce the pipeline's own
output is refused rather than shipped. The check passed to within 1e-9 on all four columns; the
artefact is plain JSON, tagged with the `feature_version` it belongs to.

There is a matching gap this does *not* close, and it needed a schema change. Four familiarity
features (`device_is_new`, `device_mismatch`, `addr_is_new`, `addr_mismatch`) ask "has this
account used this device or address before?", which needs the account's **prior raw values**.
The `transactions` table stored the computed flags, which answer the question for their own row
and for no other. Revision 0002 adds `device_info` and `addr1` as nullable columns. They are
**unbackfilled**: every row loaded before Phase 7 reads as the `__missing__` sentinel, which is
exactly how the training path already treats an absent `DeviceInfo`, so the degradation is to a
value the model has seen rather than to a fabricated one. Re-running the pipeline backfills it.

### Feature assembly is 9x the scoring call, and it is not the database

Phase 2 recorded feature-assembly latency as unmeasured and predicted the account-state range
scan would be "the part most likely to be slow". Measured, on the shipped model:

| History rows | Assemble p50 | Assemble p95 | Score p50 | Total p95 |
|---|---|---|---|---|
| 0 | 34.0ms | 47.4ms | 4.05ms | 50.5ms |
| 10 | 35.3ms | 43.6ms | 4.04ms | 47.9ms |
| 50 | 36.2ms | 39.0ms | 4.07ms | 43.7ms |
| 200 | 37.8ms | 41.9ms | 4.02ms | 46.0ms |
| 500 | 40.9ms | 44.7ms | 4.09ms | 48.9ms |

The prediction was wrong in an informative way. Assembly dominates — roughly nine times the
scoring call — but it is **near-flat in history size**: 0 rows costs 34ms and 500 rows costs
41ms. The cost is fixed pandas overhead in constructing and engineering a small frame, not the
range scan and not the row count. So `account_history_limit` is not the lever, and raising it
is nearly free; the lever is the frame construction itself.

One version of this was much worse. Building the frame by inserting ~113 columns one at a time
fragmented pandas' own block manager, cost ~100ms per call, and emitted a `PerformanceWarning`
— which, under this project's `filterwarnings = ["error"]`, is a test failure rather than a
nuisance. Materialising the frame once from a single dict took it to ~41ms.

**The honest reading of the total: p95 sits at 44–51ms against a 50ms budget that was defined
for the scoring call alone.** Tier-1's own 6.38ms p95 is unchanged and still ~8x inside it, but
the end-to-end number is at the line. Phase 10 owns this.

### The serving path deliberately does not go through `Tier1InputSpec.transform`

`transform` builds `pd.Categorical` against the fitted level set. At training every value came
from that set. At serving an unseen level is routine — a `ProductCD` the account did not send,
a device never seen — and `ProductCD` in particular has **no `__missing__` level**, because
every training row had one. Recent pandas warns on constructing a `Categorical` with values
outside its dtype, so the serving path tripped it on the first cold-account request.

`Tier1Model._vector_to_array` is the actual serving contract and it already handles this: an
unrecognised level becomes `-1`, which is how LightGBM encodes a missing category, and its
docstring says the assembled vector is expected to carry raw values. So assembly now applies
the frequency encoders and hands `score()` the raw categorical strings, bypassing `transform`
entirely. **Scores were bit-identical before and after the change** across cold, warm and
large-amount cases (0.012567 / 0.006671 / 0.002161), which is what made it safe to adopt.

### What is actually in the decision path, and what is not

The phase brief says `/score` "runs a transaction through all 4 layers". Doing that literally
would have meant serving models this repo's own measurements say not to serve:

- **The meta-learner loses to Tier-1 alone** by 0.0322 PR-AUC, CI [−0.0373, −0.0273], excluding
  zero — and `app/models/README.md` says in as many words that it is not recommended for
  serving. It is not in the decision path.
- **Tier-3's per-transaction contribution is below no-skill**, and Phase 5's ablation put its
  effect on the fused ranking at −0.0001 with an interval spanning zero. It annotates the audit
  record and feeds `GET /rings`; it does not move the decision.
- **Tier-2 was retired on validation in Phase 5** and has no serving path here.

So the shipped decision is Tier-1's calibrated probability plus the transaction amount, through
the Phase 6 `plug_in` cost policy — exactly the configuration Phase 6 measured at 22.41%
cheaper. Choosing the architecture diagram over the measurement would have been the easier
write-up and a worse system. This is stated in `app/core/serving.py`'s module docstring rather
than left for a reader to infer.

One consequence, stated rather than hidden: a flagged transaction returns `review`, never
`block`. The operating point was chosen to fill a 1% review queue and every Phase 6 figure
prices the flagged arm as a review, so returning `block` would report a decision whose measured
cost is not the measured cost of that threshold.

### Four carried security gates, closed

- **RLS is effective for the first time.** Phase 1 defined correct policies and recorded them
  as an open FAIL, because the app connected as the `riskiq` superuser, which owns the tables
  and bypasses row-level security unconditionally, and `riskiq_app` was `NOLOGIN` so nobody
  could connect as it. Revision 0002 grants LOGIN; `docker-compose.yml` now connects as
  `riskiq_app`. No password enters tracked source — `ALTER ROLE ... LOGIN` sets only the flag,
  and `infra/postgres-init/10-app-roles.sh` attaches the credential from the environment on
  first boot, passing it as a psql variable rather than interpolating it into SQL text.
- **`audit_log` is append-only in the database, not by convention.** It grants `SELECT, INSERT`
  and defines no `FOR UPDATE` or `FOR DELETE` policy, so a rewrite is refused even if
  application code one day asks for one. `test_orm_constraints.py` asserts the grant verbs.
- **The limiter fails closed.** A Redis outage returns 503, not "allowed". This is the item
  most likely to be "fixed" later by someone who finds the 503s inconvenient, so
  `test_rate_limit.py` pins it across four different failure types with the reason stated.
- **`POST /score` returns no field of `DecisionCost` and no attribution.** Asserted both on the
  response schema and against real serialised bodies.

The response also withholds the calibrated probability, which is a step beyond the carried
gate. A probability plus an amount reconstructs the same boundary the cost arms would; the
response carries a three-value risk band instead, whose edges are not the decision threshold —
the threshold is a money-scale score that depends on the amount.

Attribution *is* served, at `GET /audit/entry/{id}/explain`, behind an `explain:read` scope. It
is the evasion oracle the checklist names, and withholding it entirely would have made the
Phase 8 reviewer drill-down unbuildable. A merchant-style token carries `score:write` and does
not reach it.

### Two bugs the tests caught in my own work

**Unauthenticated requests returned 503 instead of 401.** The rate limiter was a route-level
dependency, and FastAPI resolves those before the handler's own parameters — so with no Redis
running, an anonymous request was refused by the limiter before authentication ever ran. That
also meant the limiter was keying budgets on IP addresses while its docstring claimed it keyed
on the principal. Fixed by making `enforce_rate_limit` take the principal as a dependency,
which forces authentication to resolve first.

**A test that could not fail.** `test_a_planted_derived_feature_is_overwritten` asserted that a
caller-planted derived feature came back *different* from what was planted. For
`device_is_new`, the true value on an account with history is `0.0` — which is exactly what an
attacker would plant. The test passed while proving nothing. It now asserts the correct
computed value, not merely a different one.

**A third, found by re-reading rather than by a test: a transaction could enter its own
history.** `read_account_history` filtered on `event_time <= before`, which is correct for a
transaction that is not yet persisted — and wrong for one that is. A re-score, the `/replay`
enhancement, and a redelivered Phase 9 webhook all score a row that is already in the table, so
the scan returned it and assembly appended it again. That inflates `account_prior_txn_count`
and every velocity count by one, doubles the transaction's own contribution to the velocity
sums, and drives `seconds_since_prior_txn` to zero — a materially different decision under an
audit row that looks entirely correct. The scan now takes an `exclude_transaction_id`. Nothing
in Phase 7 exercised the path, but Phase 9 would have, silently.

### The gates: `code-reviewer` and `security-reviewer` both returned real findings

Neither gate came back clean on the first round, and both found things that reading my own
diff had not.

**`security-reviewer` returned three blocking findings. All three were correct.**

**B1 — the authorization bug, and the worst of the three.** `scoped_account_id` returned
`str | None`, where `None` meant "analyst, no filter". A non-analyst principal whose token
carries no `account_id` claim *also* returned `None`, and both list routes read `None` as
"apply no filter". So a token with `transactions:read` and no account claim was served every
account's rows. That token shape is not exotic — it is the default of `create_access_token`,
`decode_access_token` does not require the claim, and the repo's own fixtures mint them.

The tell was an asymmetry twelve lines apart: `require_account_access` handled the same case
correctly (`principal.account_id is not None and ...`) and `scoped_account_id` did not. Fixed
by making the three outcomes three *values* — `UNRESTRICTED`, `NOTHING`, or an account id — so
that "may see everything" cannot be reached by failing to specify anything. `NOTHING` returns
an empty page from `/transactions` and a 404 from `/audit`, matching what an unknown
transaction returns so the two cannot be told apart.

**B2 — a false assurance in the file that defines the disclosure policy.** `POST /score`
returned a three-value `risk_band` in place of the probability, and I had written in
`schemas.py` that recovering a boundary from it "takes O(n) probes per band edge instead of
O(log n) against a continuous score". That is simply wrong. The band is monotone in the
probability, so binary search over the *amount* locates an edge in O(log n) exactly as it would
against a continuous score — coarsening the output reduces bits per probe, it does not defeat
search. Worse, the band edges were published as tracked constants, which makes a located edge a
*calibrated* anchor rather than an unknown one.

The reviewer also noted that Phase 6 escalated this exact exploit and resolved it "keep the
thresholds and cost constants" on the stated ground that **"the exploit has no live target —
there is no endpoint."** Phase 7 builds the endpoint. That premise expired and the decision
needed re-taking.

Re-taken: the band is gone from the scoring response. What remains is the decision itself,
which is a one-bit oracle no fraud API can avoid — the caller has to be told what happened to
the transaction. The residual is bounded by authentication and the limiter; closing it properly
needs per-`(account_id, transaction_id)` idempotency, recorded below as a Phase 9 prerequisite.
The false claim in the docstring was corrected rather than deleted, because a reader who
believed it once will believe it again.

**B3 — I shipped the thing the codebase explicitly forbids.**
`GET /audit/entry/{id}/explain` returned `cost_estimate`, which is
`DecisionCost.to_audit_dict()` — the object whose own docstring, four files away, reads *"for
the server-side audit trail only. Never a response body. Every field here is an evasion oracle,
and together they are complete."* That is the gate Phase 6 widened from "must not return
`top_features`" to "must not return any field of `DecisionCost`", violated in the same commit
that quoted it approvingly elsewhere.

Compounding it, the route was gated on `explain:read` alone, and `require_account_access`
passes on owner match — so an account-scoped merchant token carrying that scope read full
attribution *and* the complete cost matrix for its own decisions in a single call. The route's
docstring asserted this was impossible because "a merchant's token carries `score:write` and
nothing else", which is an assumption about an issuance process **that does not exist anywhere
in this repo**: there is no token endpoint and nothing constrains which scopes a token gets.

Fixed both ways: `cost_estimate` is gone from `ExplanationResponse` entirely, and the route now
requires `analyst` in addition to `explain:read`, matching how `/rings` already solved the same
problem structurally rather than by assumption.

**Two hardening items were taken as well.** `require_account_access` waives ownership for
analyst scope, which is right for a read and wrong for a write — it would have let the
widest-reaching token record a decision attributed to any account. `POST /score` now uses a
separate `require_account_ownership` with no bypass. And `Settings.database_url` defaulted to
the table-owning `riskiq` superuser, which silently disables every RLS policy; it now defaults
to `riskiq_app` with no password, so a deployment that forgets to configure it fails to connect
rather than quietly running unprotected.

**`code-reviewer` found one thing that matters more than any of the above.**

**`POST /score` does not persist the transaction it scored, and cannot.** Nothing in the repo
constructs a `Transaction` row, and `riskiq_app` holds only `SELECT` on that table. So
`read_account_history` — which every velocity, z-score and familiarity feature depends on — can
only ever see rows the offline Phase 1 pipeline loaded. Every transaction scored through the
live path is invisible to every later call for that account.

The naive fix is worse than the gap, which is why it was not taken. `transactions.is_fraud` and
`transactions.split` are both `NOT NULL`: that table is a *labelled training corpus*, not a live
ledger. Inserting live unlabelled traffic would mean writing `is_fraud = False` for every
scored transaction and assigning it a split — fabricating negative labels directly into the
corpus that every held-out number in this project is computed on. A quiet cold-start is a much
smaller problem than a silently poisoned evaluation set.

The right fix is a separate `scored_transactions` ledger with a nullable label, and a history
read that unions it with the corpus table. That is real scope and it is **Phase 9's
prerequisite**, not a Phase 7 afterthought — Phase 9 is where live traffic actually arrives, via
the webhook, and it is the first phase where the gap has a consequence. Recorded here so it is
a decision rather than an oversight. For the demo the gap is invisible: judges score accounts
that exist in the loaded IEEE-CIS corpus, so history resolves normally.

The reviewer also caught a latent double-count in the same area — `read_account_history`
filtered on `event_time <= before` with no transaction-id exclusion, so scoring a transaction
already in the table would pull it into its own history *and* append it again. Found
independently and fixed before the review landed; described above.

**The re-verification round found one more, and it was the same mistake again.**

B1, B3, H4 and H1 came back CLOSED. B2 came back **partially** closed, with a new blocking
finding: removing `risk_band` from `POST /score` left it on `GET /audit/{transaction_id}` —
and the account holder can read its own audit rows. That reassembles the identical probe loop
across two calls instead of one: post a transaction, read back its audit row, bisect on the
amount. Two calls instead of one is not a mitigation.

My justification for leaving it there had been "that path is authenticated, account-scoped and
not a probe loop", which is false in the same way the O(n) claim was false: **the
account-scoped party is the probing party.** B3 had already established exactly this reasoning
on the explain route — "owner match is deliberately not sufficient here... which is exactly the
party this attribution must never reach" — and I did not carry it across to the band one file
away. The band is now gone from every response schema on the service; magnitude is available
only as `risk_probability` on the analyst-only explain route.

Worth recording as a pattern rather than three incidents: each of B2, B3 and this one was a
case of writing a defensive claim in a docstring and not checking it against the caller model.
The claims read as reasoning and functioned as decoration.

Three more hardening items were taken in the same round:

- **The `NOTHING` / `UNRESTRICTED` sentinels were strings**, sharing a value domain with
  `account_id`. A token minted with `account_id="unrestricted"` would have compared equal to
  the sentinel and dropped the ORM filter — a narrower version of the exact bug the sentinels
  were introduced to fix. Not exploitable (RLS still pinned the session, and there is no token
  endpoint), but the premise of the fix was that the two states must be unforgeable. They are
  now an `enum.Enum`, which no JWT claim can produce.
- **`alembic upgrade head` could no longer run.** H1 repointed `DATABASE_URL` at `riskiq_app`,
  and `alembic/env.py` read the same setting — but that role cannot `CREATE TABLE`, `GRANT` or
  `ALTER ROLE`. The obvious workaround is to point `DATABASE_URL` at the superuser, which is
  precisely the H1 regression. Added a separate `MIGRATION_DATABASE_URL`, so the two cannot be
  the same variable.
- **Three docstrings asserted a control that is not in force** — that analyst scope is
  "read-only by construction because the analyst database role holds no write grant". The
  application never assumes that role, so an analyst token operates with `riskiq_app`'s grants,
  which include `INSERT ON audit_log`. The real control is `require_account_ownership` having
  no analyst branch. Corrected, because this is how the bypass gets re-added by someone who
  reads the wrong docstring first.

**Three further hardening items are carried, not fixed**, and are listed in the known gaps
below: the `riskiq_analyst` database role is created but never assumed by the application, so
its policies are inert; the Alembic migration cannot be applied by the least-privilege role the
app now connects as, so migrations need their own DSN; and there is no `RequestValidationError`
handler, so FastAPI's default 422 echoes submitted values back for schema-level errors.

### Known gaps leaving Phase 7

- **End-to-end p95 is at the 50ms line** (44–51ms), against a budget written for the scoring
  call alone. The cost is fixed pandas overhead in assembly, not the range scan. Phase 10.
- **`device_info` and `addr1` are unbackfilled.** Familiarity features read `__missing__` for
  every pre-Phase-7 row until the Phase 1 pipeline is re-run.
- **`models/artifacts/` is gitignored**, so a fresh deploy has no weights and `/score` returns
  503 until they are mounted or baked in. `docker-compose.yml` mounts `./models:/models:ro`;
  Render has no equivalent yet. Phase 10 owns the deploy story.
- **RLS is now effective but has never been exercised against a live database in CI.** The
  policies are asserted against the migration's source text, not against a running Postgres.
  The one skipped test in the suite is the pre-existing live-database check.
- **The limiter is a fixed window**, so it admits up to 2x the nominal rate across a window
  boundary. Accepted: the threat it exists for is scripted probe traffic, not burst shaping.
- **`mypy tests/` still reports 10 pre-existing errors** in `test_causal_cost.py` and
  `test_meta_learner.py`, untouched by this phase. CI runs `mypy app/`, which is clean.
- **No `/replay/{transaction_id}` endpoint.** It is the phase's optional enhancement pass and
  was not attempted; the audit row carries everything it would need.
- **`POST /score` does not persist what it scored** — see the gate section above. Live-scored
  traffic never enters account history. **Phase 9 prerequisite**: a `scored_transactions`
  ledger with a nullable label, unioned into the history read. Do *not* solve it by inserting
  into `transactions`, which would fabricate `is_fraud = False` into the evaluation corpus.
- **`POST /score` is not idempotent.** A caller may re-submit the same `transaction_id` with a
  different amount and get a fresh decision each time, which is what makes the residual
  decision oracle probeable at the limiter's 60/minute. Per-`(account_id, transaction_id)`
  idempotency returning 409 on a changed body closes it. Phase 9, with the ledger above.
- **The `riskiq_analyst` database role is inert.** It is created, granted and given policies by
  revision 0002, but the application always connects as `riskiq_app` and never issues
  `SET ROLE`, so the three analyst `USING (true)` policies never apply. This fails *closed* —
  an analyst token currently gets an empty `/transactions` — so it is not a hole. It is listed
  because the obvious field fix for "the dashboard shows nothing" is to repoint `DATABASE_URL`
  at the superuser, which is exactly how RLS dies quietly. Implement it properly
  (`GRANT riskiq_analyst TO riskiq_app` plus `SET LOCAL ROLE`) before Phase 8 builds on it.
- **Migrations cannot be run by the role the app connects as.** `alembic/env.py` reads the same
  `DATABASE_URL`, which is now `riskiq_app` — a role with no `CREATE TABLE`, `GRANT` or
  `ALTER ROLE` rights. A separate `MIGRATION_DATABASE_URL` is needed; until then an operator
  must run migrations with an admin DSN in the environment.
- **No `RequestValidationError` handler.** `/score`'s own 422 is careful not to echo input, but
  schema-level validation failures fall through to FastAPI's default, which includes Pydantic's
  `input` key and therefore the submitted values. Checklist item 4.4 is only partly satisfied.
- **Secret scanning does not run in CI and has never been run over history.** The trufflehog
  hook is local, blocking, and scans the working tree only; `.trufflehog-exclude` excludes
  `.git/`. Until `trufflehog git file://.` is run over full history, checklist item 1.4 is
  unproven — and anything it finds must be rotated, not merely deleted. Phase 10.
- **`audit_id` is a global sequence returned to callers.** It is monotonic across all accounts,
  so two scoring calls tell a merchant the platform's decision volume in between. A random
  handle with the integer kept internal would close it.
- **The residual decision oracle is accepted, not closed.** `POST /score` returns allow/review,
  which is one bit and which no fraud API can withhold — the caller has to be told what
  happened to its transaction. Bounded today by authentication and 60 requests/minute per
  principal. The idempotency work above is what actually closes it.

## Phase 8 — Dashboard & Visualization

**What shipped:**
- React dashboard (live scorer, cost comparison chart, Tier-3 network graph, metrics panel, decision audit table)
- Two-token demo mode: merchant token for /score, analyst token for /audit/entry/{id}/explain
- Metrics: Tier-1 PR-AUC 0.5276, confusion matrix, calibration curve, cost-sensitivity from held-out test
- D3 force-directed Tier-3 network graph (live from GET /rings)
- Responsive design + a11y (keyboard navigation, reduced-motion respected)
- Demo token endpoint (POST /auth/demo-token) for walkthrough

**Security findings & fixes:**
- Found & fixed: dashboard was leaking Tier-1's cost-optimal threshold to unauthenticated users
- Found & fixed: Tier-3 ring "anonymized" IDs were trivially reversible to original merchant IDs
- Both fixes: code changed + model retrained, fix is live in the artifact

**Test suite:** 698 backend tests passing, frontend typecheck/lint clean

**Known gaps (not blocking):**
- One BUILD_LOG figure from Phase 2 is now stale (recorded for reference)

**Next:** Phase 9 (Razorpay webhook integration)
