# RiskIQ — Real-Time Fraud, Chargeback & Abuse-Ring Detection

Built for the Razorpay AI Buildathon 2026 — Track 2: AI Risk Manager.
Deadline: September 5, 2026. Bar: a working detector with measured precision/recall
on a held-out test set, honest false-positive cost, full audit trail, strictly defense-only.

## What this is
A four-layer fraud/risk decisioning system, not a single classifier:
1. **Tier-1** — real-time per-transaction anomaly score (Isolation Forest / LightGBM)
2. **Tier-2** — per-account behavioral sequence model (LSTM Autoencoder)
3. **Tier-3** — transaction-network graph abuse-ring detection (the differentiator)
4. **Meta-learner** — XGBoost fusing all three signals, with SHAP explanations
5. **Causal cost layer** — estimates false-positive cost per decision (DR-Learner style)

Full spec and phase-by-phase build prompts live in `PHASE_PROMPTS.md`. Always confirm
which phase we're on before starting work, and don't skip ahead — each phase's
verification step gates the next.

## Stack
- Backend: Python 3.11, FastAPI, SQLAlchemy (async), PostgreSQL, Redis
- ML: scikit-learn, LightGBM, XGBoost, PyTorch, NetworkX/igraph, SHAP, econml/CausalML
- Frontend: React + TypeScript (Vite), Tailwind, D3/Recharts
- Deploy: Render (backend), Vercel (frontend)

## Commands
- Backend dev: `uvicorn app.main:app --reload`
- Backend tests: `pytest -x` — always run single relevant test file during iteration, full suite before phase sign-off
- Lint/type: `ruff check . --fix && mypy app/`
- Frontend dev: `npm run dev` · tests `npm test` · lint `npm run lint`
- Full stack: `docker compose up`
- Before any commit: `ruff check . --fix && mypy app/ && pytest -x && npm run lint && npm test`

## IMPORTANT — security rules apply to every layer, no exceptions
This is a fraud-detection product; shipping it insecurely is thematically disqualifying,
not just bad practice. Full mapping in `.claude/skills/security-checklist/SKILL.md`
(invoke with `/security-checklist`). Non-negotiables:
- No secrets in code or git history — `.env` + `.gitignore`, enforced by pre-commit hook
- RLS enabled on every table holding transaction/account data
- Every scoring/write endpoint requires server-side auth — never trust a client-supplied role
- SQLAlchemy ORM only — no raw string-concatenated queries, ever
- Strict Pydantic schemas (`extra="forbid"`) on every request body
- Redis rate limiting on every public endpoint
- `pip-audit` / `npm audit` in CI — zero high/critical vulnerabilities allowed
- Nothing built here may be usable to generate or automate fraud — defense-only,
  per the track's disqualification rule. If a piece of code could plausibly double
  as an attack tool, stop and flag it rather than building it.

## IMPORTANT — model evaluation rules
Full detail in `.claude/skills/ml-evaluation-standards/SKILL.md` (invoke with
`/ml-eval-standards`). Non-negotiables:
- Time-ordered train/val/test split — never random shuffle on transaction data
- Headline metric is **PR-AUC** + precision/recall/F1 on the **held-out test set only**
- Every reported result ships with a false-positive cost estimate — a metrics
  section without one is incomplete for this project
- Never present one example as proof — always show the full confusion matrix
- State what each model does NOT catch, explicitly, in its own README section

## Code style
- Python: type hints on every function, docstrings on every public function/class,
  `black` formatting (line length 100), one model/table per file under `app/models/`
- TypeScript: functional components + hooks only, strict mode on
- Commits: Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`)
- Every model-training script sets and logs a random seed

## Workflow
- Enter plan mode (`Shift+Tab`) before implementing a new phase — under deadline
  pressure, solving the wrong problem is the most expensive mistake available
- After implementing a phase: run `/code-review`, then explicitly invoke the
  `security-reviewer` subagent for anything touching auth/DB/input handling, and
  the `ml-evaluator` subagent for anything touching model training or metrics
- `/compact` policy: when compacting, always preserve the current phase number,
  the list of modified files, and the last test command run and its result
- Update `BUILD_LOG.md` at the end of every phase — this becomes the "Build
  Challenges & Technical Obstacles" answer on the submission form, so log real
  obstacles and how they were solved, not a smoothed-over summary

## Repo map
(Derive everything else from the code — this is only what isn't obvious from reading it)
- `app/models/tier1_anomaly.py`, `tier2_behavioral.py`, `tier3_graph.py`,
  `meta_learner.py`, `causal_cost.py` — one file per architecture layer, never merge them
- `app/core/audit.py` — every scoring decision must go through this; it is the
  audit-trail requirement, not an optional log
- `models/registry.json` — append-only record of every trained model's version,
  training window, feature set, and hyperparameters
- `PHASE_PROMPTS.md` — the full build plan, phase by phase
- `BUILD_LOG.md` — running log of what shipped, what's deferred, known gaps
