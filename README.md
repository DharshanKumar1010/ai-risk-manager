# RiskIQ

**Real-time fraud, chargeback and abuse-ring detection.**
Razorpay AI Buildathon 2026 — Track 2: AI Risk Manager.

> **Placeholder.** This file is written properly in Phase 11: objective, architecture
> diagram, setup instructions, the metrics summary table with each tier's honest
> limitations, and the Assumptions section for the causal cost model. What is below is
> only enough to get the stack running.

## Quick start

```bash
cp backend/.env.example backend/.env    # then fill in the placeholders
docker compose up
```

| Service  | URL                            |
|----------|--------------------------------|
| Backend  | http://localhost:8000          |
| API docs | http://localhost:8000/docs     |
| Health   | http://localhost:8000/health   |
| Frontend | http://localhost:5173          |
| Postgres | `localhost:5432`               |
| Redis    | `localhost:6379`               |

## Local development without Docker

```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

## Checks

```bash
cd backend  && ruff check . --fix && mypy app/ && pytest -x
cd frontend && npm run lint && npm test

pre-commit install          # once per clone
pre-commit run --all-files
```

## Layout

| Path | What it holds |
|------|---------------|
| `backend/app/models/` | One file per DB table and one per architecture layer — never merged |
| `backend/app/core/audit.py` | The audit-trail choke point every scoring decision passes through |
| `models/registry.json` | Append-only record of every trained model version |
| `PHASE_PROMPTS.md` | The full phase-by-phase build plan |
| `BUILD_LOG.md` | What shipped, what is deferred, known gaps |
| `.claude/skills/` | The security and ML-evaluation bars this project is held to |

## Scope

Strictly defense-only. Nothing here may be used to generate, automate or evade fraud —
that is a track disqualification rule, and it is enforced in
`.claude/skills/security-checklist/SKILL.md` section 8.
