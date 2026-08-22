# RiskIQ Build Plan — Razorpay AI Buildathon, Track 2: AI Risk Manager

**Deadline:** September 5, 2026. **Bar:** working detector, measured precision/recall
on a held-out test set, honest false-positive cost, full audit trail, strictly defense-only.

## How to use this file
1. Copy `CLAUDE.md` and the `.claude/` folder into your repo root before Phase 0.
2. Work through phases in order, one Claude Code session per phase. Run `/clear`
   between phases so context stays clean (see CLAUDE.md's workflow rules).
3. For each phase: enter **plan mode** (`Shift+Tab`), paste the *Explore* prompt,
   review the plan Claude proposes (edit it directly with `Ctrl+G` if needed),
   approve, then paste the *Build* prompt.
4. After Claude reports the verification passing, run `/code-review` plus any
   named subagent before moving to the next phase.
5. Update `BUILD_LOG.md` at the end of every phase — this becomes your
   "Build Challenges & Technical Obstacles" answer on the submission form.
6. Each phase ends with an **Enhancement pass** — optional stretch features.
   Add them only if the verification step already passed; never let a stretch
   feature block a phase's core deliverable this close to the deadline.

---

## Phase 0 — Scaffolding & Claude Code Environment

**Goal:** repo skeleton, `CLAUDE.md` + skills + agents in place, CI shell, and the
full stack booting end to end — before any model or business logic exists.

**Explore:**
```
Read the current directory and confirm it's empty. I'm about to scaffold a
fraud-detection project called RiskIQ: FastAPI + PostgreSQL + Redis backend,
React/TypeScript frontend, Python ML layer with four model tiers. Don't build
anything yet — just confirm you understand the plan below before I give the
build instruction.
```

**Build:**
```
Scaffold a monorepo called riskiq with this exact structure:

riskiq/
├── CLAUDE.md                        (already present — do not overwrite)
├── BUILD_LOG.md                     (empty, columns: Phase | Status | Notes | Known Gaps)
├── .claude/                         (already present — skills + agents, do not overwrite)
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py                (pydantic-settings, loads .env)
│   │   ├── db/                      (async SQLAlchemy engine, session, declarative base)
│   │   ├── models/                  (one file per DB table + one per ML tier — empty stubs)
│   │   ├── api/                     (routers — empty stubs)
│   │   ├── core/security.py         (JWT auth stub)
│   │   └── core/audit.py            (audit-log writer stub — every scoring decision
│   │                                  must go through this function)
│   ├── tests/
│   ├── requirements.txt
│   ├── pyproject.toml                (ruff rules E,F,I,N,W,B; mypy strict; black line-length 100)
│   └── .env.example
├── frontend/                         (Vite + React + TypeScript + Tailwind, no pages yet)
├── data/                             (gitignored — raw/processed dataset storage)
├── notebooks/                        (versioned notebooks; outputs gitignored)
├── docker-compose.yml                (postgres, redis, backend, frontend services)
├── .gitignore                        (must cover .env, data/*, __pycache__, node_modules,
│                                       .venv, *.pkl, *.pt, notebook outputs)
├── .pre-commit-config.yaml           (trufflehog secret scan + ruff + black; block
│                                       the commit on any failure)
└── README.md                         (placeholder — filled in Phase 11)

Also write .github/workflows/ci.yml running, in order: ruff check, mypy,
pytest, npm run lint, npm test, pip-audit, npm audit --audit-level=high —
fail the build on any error.

After scaffolding: run `docker compose up`, hit a hello-world `/health`
endpoint on the backend, and confirm the Vite dev server serves a blank page.
Show me the actual command output, not a description of what should happen.
Then stop — no model or business logic yet, that starts in Phase 1.
```

**Verify:** `docker compose up` boots cleanly; `curl localhost:8000/health` returns
200; `pre-commit run --all-files` passes on the empty scaffold; the CI workflow
file is syntactically valid (`actionlint` if available, otherwise a manual read).

**Enhancement pass:** a `Makefile` with `make dev` / `make test` / `make lint`
shortcuts so every command in `CLAUDE.md` has a one-word alias; a
`.devcontainer/devcontainer.json` if you want the environment reproducible for
anyone who clones the repo cold.

---

## Phase 1 — Data Pipeline & Feature Engineering

**Goal:** acquire the dataset(s), build a leakage-safe time-ordered split, and a
documented, versioned feature store — the foundation every later tier reads from.

**Explore:**
```
Use a subagent to investigate the IEEE-CIS Fraud Detection dataset schema and
the PaySim dataset schema (read any local copies in data/raw/ if present, or
describe what you already know about their columns). I want to use IEEE-CIS
for rich transaction/identity features (Tiers 1, 2, 5, 6) and PaySim separately
for its account-to-account structure (Tier 3's graph layer), since no single
public dataset has both rich features and network structure. Report back a
plan for reconciling the two before we build anything.
```

**Build:**
```
Implement backend/app/data/pipeline.py:

1. Load IEEE-CIS (primary) and PaySim (for Tier 3 only) from data/raw/.
2. Clean: handle missing values explicitly (document the strategy per column
   group, don't silently drop rows without logging how many and why).
3. Engineer features: per-account transaction velocity (count/sum over
   trailing 1h/24h/7d windows), amount z-score vs. that account's own
   historical distribution, merchant-category frequency encoding, device/geo
   mismatch flags vs. the account's usual device/geo.
4. Time-ordered split: first 70% chronologically = train, next 15% = val,
   last 15% = test. Never shuffle randomly — assert this in an automated test,
   not just a comment.
5. Persist processed features to Postgres in a `transactions` table with a
   `split` column, so every downstream model reads from one source of truth.
6. Write app/data/feature_store.py: a versioning layer that hashes the
   feature list + engineering parameters into a feature_version string, so
   every later prediction can cite exactly which feature definition scored it.
7. Generate a data-quality report (notebooks/eda_report.ipynb or a markdown
   report under notebooks/): class balance, missing-value rates, and feature
   distributions before and after engineering.

Write a pytest that asserts: (a) split boundaries are strictly time-ordered
with zero overlap, (b) no engineered feature reads a timestamp later than the
transaction it's describing (a hard leakage check, not just a docstring
promise), (c) row counts and class balance are logged.

Run the pipeline end to end and show me the data-quality report output and
the test results.
```

**Verify:** leakage test passes; split-boundary test passes; row counts and
class balance logged and sane; data-quality report generated.

**Enhancement pass:** inject a synthetic slow-drift fraud pattern into a
held-out slice, so Phase 3's behavioral model has a real, known-answer case to
detect — genuinely useful for demoing Tier 2 credibly, not just decorative.

---

## Phase 2 — Tier-1 Real-Time Anomaly Layer

**Goal:** a fast, per-transaction anomaly scorer, target p95 latency under 50ms.

**Explore:**
```
Read app/data/feature_store.py and the transactions table schema from Phase 1.
Confirm what feature set is available to a per-transaction (not per-account
history) model, since Tier-1 must be independently scoreable at ingestion time
with no dependency on Tier-2/3. Propose an Isolation Forest baseline vs. a
LightGBM binary classifier and how you'll compare them.
```

**Build:**
```
Implement app/models/tier1_anomaly.py:

1. Train both an Isolation Forest (unsupervised baseline) and a LightGBM
   binary classifier (supervised, using the fraud label) on the Phase 1
   train split, using only per-transaction features (no account-history
   dependency).
2. Evaluate both on the held-out test split per
   .claude/skills/ml-evaluation-standards/SKILL.md: PR-AUC, precision,
   recall, F1, full confusion matrix. Pick whichever wins on PR-AUC as the
   production Tier-1 model; keep the other as a documented baseline
   comparison in the README, not silently discarded.
3. Expose a clean interface:
   `def score(transaction: TransactionFeatures) -> Tier1Result` returning
   {score: float, is_anomaly: bool, latency_ms: float, model_version: str}.
4. Append the trained model's metadata (algorithm, training window,
   feature_version, hyperparameters, PR-AUC on held-out test) to
   models/registry.json — append, never overwrite prior entries.
5. Write a latency benchmark: 100 sequential scoring calls, assert p95 < 50ms,
   and log the actual p50/p95/p99.

Run the full evaluation and latency benchmark, and show me the confusion
matrix and latency numbers directly — don't summarize them away.
```

**Verify:** PR-AUC/precision/recall/confusion matrix reported on held-out test
only; latency benchmark passes; `models/registry.json` has a new entry.
Invoke the `ml-evaluator` subagent before moving on.

**Enhancement pass:** add a SHAP local explanation for Tier-1 now — it's a
small addition on top of an already-trained tree model and pays off directly
in Phase 5/8's audit trail and dashboard. Add an interface stub (not a full
implementation) for future online recalibration, and note in `BUILD_LOG.md`
that it's "designed for, not implemented" — architectural foresight without
overbuilding under deadline pressure.

---

## Phase 3 — Tier-2 Behavioral Sequence Layer (LSTM Autoencoder)

**Goal:** catch slow-building account-takeover and abuse patterns invisible to
single-transaction scoring.

**Explore:**
```
Use a subagent to review how the LSTM Autoencoder anomaly-detection pattern
was implemented in a prior project (PyTorch, hidden_size=128, reconstruction
error vs. a tuned threshold) purely as an architectural reference — the actual
threshold from that project does not transfer to this dataset and must be
re-derived here. Propose the windowing strategy for per-account transaction
sequences (e.g. last 20 transactions, padded/truncated).
```

**Build:**
```
Implement app/models/tier2_behavioral.py:

1. Build per-account transaction sequences from the Phase 1 train split
   (windowed, e.g. last 20 transactions per account, padded for shorter
   histories).
2. Train a PyTorch LSTM Autoencoder to reconstruct normal sequences; use
   reconstruction error as the anomaly signal.
3. On the held-out test split, plot the reconstruction-error distribution
   separately for normal vs. fraudulent sequences, and choose a threshold
   from that distribution — do not reuse a threshold from any other project.
4. Report precision/recall/PR-AUC at the chosen threshold, plus the training
   loss curve (to catch under/overfitting), per ml-evaluation-standards.
5. Expose `def score(sequence: list[TransactionFeatures]) -> Tier2Result`
   returning {reconstruction_error: float, is_anomaly: bool, model_version}.
6. Append model metadata to models/registry.json.

Show me the reconstruction-error distribution plot and the held-out metrics
directly.
```

**Verify:** reconstruction-error distributions plotted and visibly separated
for normal vs. fraud; precision/recall/PR-AUC reported on held-out test only;
training curve shows convergence, not obvious over/underfitting.

**Enhancement pass:** surface attention/contribution weights — which
transactions in the sequence drove the reconstruction error most — this feeds
directly into the Phase 8 dashboard's explainability panel.

---

## Phase 4 — Tier-3 Network/Graph Abuse-Ring Layer

**Goal:** the differentiator — surface collusion rings invisible to
per-transaction or per-account models. This is the single feature most likely
to make a panel remember your submission; if anything gets cut under time
pressure, cut elsewhere first.

**Explore:**
```
Use a subagent to inspect PaySim's account/transaction schema and confirm it
has enough origin/destination structure to build a meaningful account-device-
card-IP graph. Propose which graph library (NetworkX for clarity, igraph for
speed on larger graphs) fits our data size, and which community-detection
algorithm (Louvain vs. label propagation) to use.
```

**Build:**
```
Implement app/models/tier3_graph.py:

1. Construct a graph from PaySim data: nodes = accounts/devices/cards/IPs,
   edges = shared usage between them.
2. Run community detection (Louvain) plus centrality metrics (degree,
   betweenness) to flag suspiciously dense clusters.
3. Expose `def flag_rings(graph_snapshot) -> list[RingFlag]` where RingFlag
   includes the member node IDs, a ring risk score, and the centrality
   metrics that drove the flag.
4. Evaluate against PaySim's labeled fraud flags at the ring level (not just
   individual node level) — report precision/recall for ring-level detection
   per ml-evaluation-standards.
5. Save a visualization of at least one detected ring and one clean cluster,
   for a sanity-check screenshot to include in the pitch deck later.
6. Append model metadata to models/registry.json.

Show me the ring-level precision/recall and the two visualizations directly.
```

**Verify:** ring-level precision/recall reported on held-out data; at least one
detected-ring visualization saved and visibly distinct from a clean-cluster
visualization.

**Enhancement pass:** incremental graph updates — maintain a rolling window
graph instead of rebuilding from scratch on every transaction — if time
allows. This is the highest-leverage enhancement in the whole project; worth
the extra hours before polishing anything cosmetic elsewhere.

---

## Phase 5 — Meta-Learner + SHAP Explainability

**Goal:** fuse Tier-1/2/3 signals into one calibrated risk score with a
per-decision explanation.

**Build:**
```
Implement app/models/meta_learner.py:

1. Train an XGBoost meta-learner on [tier1_score, tier2_reconstruction_error,
   tier3_ring_flag/ring_risk_score, the original engineered features] as
   input, using the Phase 1 train split.
2. Calibrate output probabilities with CalibratedClassifierCV — raw XGBoost
   scores are not true probabilities, and an uncalibrated score undermines
   the "honest metrics" story.
3. Evaluate on held-out test per ml-evaluation-standards: PR-AUC, precision,
   recall, F1, full confusion matrix, plus a calibration curve (predicted
   probability vs. observed frequency).
4. Add SHAP TreeExplainer for per-prediction attribution. Store the top-3
   contributing features with every prediction.
5. Expose `def predict(transaction, tier1, tier2, tier3) -> MetaResult`
   returning {probability, decision, top_features: list[(name, shap_value)],
   model_version}.
6. Smoke-test explanations on 10 sample transactions — read them yourself and
   confirm they make sense, don't just check that the code doesn't crash.
7. Append model metadata to models/registry.json.

Show me the held-out confusion matrix, calibration curve, and 3 example
explanations directly.
```

**Verify:** held-out PR-AUC/precision/recall/confusion matrix reported;
calibration curve generated; 10 sample explanations reviewed and sensible.
Invoke the `ml-evaluator` subagent before moving on.

**Enhancement pass:** none required beyond the calibration curve above — a
well-calibrated model is already a stronger "honest metrics" story than a
merely accurate one; don't add complexity here, spend remaining time on
Phase 6 instead.

---

## Phase 6 — Causal Cost Layer (the track-specific differentiator)

**Goal:** quantify false-positive cost per decision — the track's explicit,
named bar, not an optional nice-to-have.

**Build:**
```
Implement app/models/causal_cost.py:

1. Frame this as a treatment-effect problem: treatment = "flag/block this
   transaction," outcome = net financial impact (lost legitimate revenue if
   blocked, fraud loss if allowed through).
2. Use a DR-Learner style meta-learner (same family as prior uplift work) on
   a synthetic cost model, calibrated from documented, publicly available
   average false-positive-cost figures for card-not-present fraud. State
   every assumption explicitly in a comment block and in the README — this
   is inherently estimated, present it as an estimate, not ground truth.
3. Expose `def estimate_cost(transaction, decision) -> CostEstimate`
   returning {expected_cost: float, cost_if_blocked: float,
   cost_if_allowed: float, assumptions: list[str]}.
4. Produce a cost curve across the full precision-recall operating range, so
   the chosen classification threshold in Phase 5 can be justified by cost,
   not just by F1.
5. Add a sensitivity analysis: show how the recommended threshold shifts if
   the assumed false-positive cost changes by ±50%.

Show me the cost curve and the sensitivity analysis directly.
```

**Verify:** cost curve generated across the operating range; sensitivity
analysis shows how the recommendation changes under different cost
assumptions; all assumptions documented in plain language.

**Enhancement pass:** none — this phase's value is in the honesty of the
estimate, not additional features. The sensitivity analysis above already
goes further than almost any other likely submission will.

---

## Phase 7 — Backend, Audit Trail, Security Hardening

**Goal:** a FastAPI service exposing `/score`, `/transactions`, `/audit`,
`/rings`, with the full security checklist applied and an immutable audit log.

**Build:**
```
Implement the FastAPI backend:

1. Endpoints: POST /score (runs a transaction through all 4 layers), GET
   /transactions, GET /audit/{transaction_id}, GET /rings. Strict Pydantic
   request/response schemas on every endpoint, extra="forbid".
2. JWT auth on every endpoint per .claude/skills/security-checklist/SKILL.md
   — apply the full checklist, not just the auth item.
3. RLS-enabled Postgres tables; Redis rate limiting on all public endpoints.
4. app/core/audit.py: every /score call writes an immutable row containing
   transaction_id, all 4 layer scores/flags, the final decision, the
   model_version of every layer involved, a timestamp, and the
   feature_version hash from Phase 1's feature store.
5. Explicit graceful degradation: set a timeout on the Tier-3 graph lookup;
   if it times out, fall back to a Tier-1+2-only decision AND log in the
   audit row that degraded mode was used and why. This is a hard
   requirement, not a nice-to-have — it satisfies the "handle one failure
   gracefully" bar from the buildathon's general rules.
6. Integration tests: auth bypass attempts, tampered JWTs, ownership-check
   bypass attempts, and the degraded-mode fallback path.

Run the integration test suite and show me the results. Then explicitly
invoke the security-reviewer subagent against the full API before you
consider this phase done.
```

**Verify:** all integration tests pass; `security-reviewer` subagent run
completes with no blocking findings; degraded-mode fallback demonstrated in a test.

**Enhancement pass:** a `GET /replay/{transaction_id}` endpoint that re-scores
a past transaction against the current model versions and diffs the result —
a strong pitch-video moment ("here's how the model's decision would differ
today vs. when it happened").

---

## Phase 8 — React Dashboard

**Goal:** a dashboard that makes the panel *feel* the system's rigor, not just
read about it. Before writing any component, load the frontend-design
guidance in your environment and treat this as a risk-operations console, not
a generic admin template — ground every design choice in that character.

**Build:**
```
Design and implement the dashboard:

1. Brainstorm a compact design token system first (color, type, layout,
   signature element) suited to a risk-ops console specifically — not the
   default cream/serif look, not a generic dark-mode SaaS dashboard. State
   the choice and why it fits this specific product before writing any code.
2. Live scoring feed (websocket): transactions appear as they're scored,
   color-coded by risk tier.
3. Per-transaction drill-down: SHAP waterfall chart, which tier(s) flagged
   it, the causal cost estimate from Phase 6, and the full audit trail entry.
4. Network graph visualization (force-directed, e.g. D3) for Tier-3 — this is
   the signature element, the one visual the panel is most likely to remember.
5. Metrics dashboard: PR curve, confusion matrix, calibration curve, and the
   cost-sensitivity chart from Phase 6 — all pulling from the real held-out
   evaluation and clearly labeled as such (never mixed with live-demo data
   without a label distinguishing the two).
6. Responsive down to mobile, visible keyboard focus, reduced motion
   respected — the quality floor, not an afterthought.

After building, take a screenshot, critique it against the design plan from
step 1, and tell me what you'd change if you had one more hour.
```

**Verify:** responsive at mobile width; keyboard focus visible; a self-critique
against the stated design plan is produced, not skipped.

**Enhancement pass:** dark/light toggle (ops tools are often run in dark
mode); empty and error states written in the interface's own voice
("No transactions scored in this window yet" rather than a bare "No data").

---

## Phase 9 — Razorpay Test-Mode Webhook Integration

**Goal:** prove this is wired to Razorpay's actual test-mode APIs, not just a
Kaggle notebook with a UI in front of it.

**Build:**
```
Implement Razorpay test-mode webhook integration:

1. Subscribe to payment.authorized, payment.captured, and payment.failed
   test-mode webhook events.
2. Verify the webhook signature Razorpay sends — never trust an unsigned
   payload (this is itself a security-checklist item, treat it as one).
3. Transform the incoming payload into the TransactionFeatures schema from
   Phase 1, run it through the full 4-layer pipeline via the /score
   endpoint, write to the audit log, and push the result to the live
   dashboard feed over the existing websocket.
4. Write an integration test that simulates a signed test-mode webhook
   payload and confirms it reaches the dashboard feed end to end.

Trigger an actual test-mode payment from the Razorpay dashboard and confirm
it appears in the live feed within a couple of seconds. Show me the result.
```

**Verify:** webhook signature verification implemented and tested; a real
test-mode payment demonstrably flows end to end into the live feed.

**Enhancement pass:** a script that fires a batch of synthetic test-mode
transactions with a known, planted fraud rate, so the pitch video can show
the system catching a deliberately seeded pattern live — clearly framed in
the video as a live demo, never presented as a substitute for the held-out
metrics.

---

## Phase 10 — Testing, CI, Deployment

**Goal:** everything green, deployed, reproducible from a clean clone.

**Build:**
```
1. Bring test coverage on app/models/ and app/api/ to a meaningful, justified
   level (e.g. >80% on core scoring logic — don't chase 100% on glue code;
   state the target and why in BUILD_LOG.md).
2. Get the full CI workflow from Phase 0 green (ruff, mypy, pytest, npm lint,
   npm test, pip-audit, npm audit).
3. Deploy backend to Render and frontend to Vercel. Check for the same
   Python-version pinning issue seen on a prior project (Render defaulting to
   an unexpectedly new Python) and pin explicitly if needed.
4. Re-run /security-checklist and /ml-eval-standards against the deployed
   system, not just the local one.

Show me the CI run output and the deployed URLs directly.
```

**Verify:** CI green on a clean clone; both deployed URLs reachable and
functional; both skill checklists re-verified against production.

**Enhancement pass:** a `GET /status` endpoint reporting which model versions
are currently live — useful operationally and a small signal of production-
mindedness to the panel.

---

## Phase 11 — Documentation, Architecture Diagram, Pitch Video, Submission

**Goal:** package everything for the actual Razorpay submission form. Once
submitted, the form cannot be edited — don't submit until every piece below
is genuinely final.

**Build:**
```
1. Write README.md: project objective, an architecture diagram (Mermaid),
   setup instructions, a metrics summary table (with the honest limitations
   list from each tier), and a clearly labeled "Assumptions" section for the
   causal cost model.
2. Finalize BUILD_LOG.md — this becomes the "Build Challenges & Technical
   Obstacles" form answer. Write it honestly; specific, real obstacles read
   as more credible to a panel than a smoothed-over success narrative.
3. Write a standalone architecture one-pager (the form asks for a "clear
   architectural breakdown" separate from the README).
4. Draft a 5-minute pitch video script in this order: (a) the live
   Razorpay test-mode demo — visceral and real, (b) the held-out metrics and
   cost curve — the rigor, (c) the graph/ring visualization — the memorable
   differentiator, (d) close with the honest limitations list, volunteered
   before being asked, since the track explicitly rewards honesty over polish.
```

**Verify:** watch the finished video back once, cold, as if you were the
panel — does it prove the bar (precision/recall on held-out data,
defense-only, false-positive cost, audit trail) within 5 minutes without the
narration outrunning the visuals.

**Enhancement pass:** none. This phase is about polish and honesty, not new
features — a known, clearly explained gap beats a rushed, undertested
last-minute addition. Resist adding anything new this late.
