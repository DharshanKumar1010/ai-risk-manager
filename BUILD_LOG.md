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
