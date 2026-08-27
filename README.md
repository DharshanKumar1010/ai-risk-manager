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

## API

Every route requires server-side authentication and permissions come from that authentication,
never from anything in the request — but not every route uses the same mechanism. Most take a
bearer token, scoped by the `scopes` claim; `/health` needs none by design; the webhook is
authenticated by an HMAC signature instead, because it has no caller to issue it a token to.
Full schemas at `/docs`.

| Method | Path | Auth | Returns |
|--------|------|------|---------|
| `GET`  | `/health` | none | Liveness. Checks no dependency by design. |
| `POST` | `/score` | bearer, `score:write` | The decision, and an audit handle. |
| `GET`  | `/transactions` | bearer, `transactions:read` | The caller's scored transactions. |
| `GET`  | `/audit` | bearer, `audit:read` | Recent recorded decisions visible to the caller. |
| `GET`  | `/audit/{transaction_id}` | bearer, `audit:read` | Every recorded decision for one transaction. |
| `GET`  | `/audit/entry/{audit_id}/explain` | bearer, `explain:read` + `analyst` | Feature attribution. Analysts only. |
| `GET`  | `/rings` | bearer, `rings:read` + `analyst` | Flagged abuse rings and their membership. |
| `POST` | `/auth/ws-ticket` | bearer, `analyst` + `audit:read` + `explain:read` | A short-lived ticket for the live feed socket. |
| `GET`  | `/ws/feed` | ws-ticket (query param) | The live scoring decision feed, analyst-only. |
| `POST` | `/webhooks/razorpay/transaction` | HMAC (`X-Razorpay-Signature`), not a bearer token | Score a Razorpay payment event; returns risk context — see `BUILD_LOG.md`'s Phase 9 entry for why this one route's response is allowed to carry more than the others. |
| `POST` | `/auth/demo-token` | none — mints a token rather than requiring one | Local/CI only; not mounted in staging or production. |

`POST /score` takes **raw transaction fields**, never an engineered feature vector. The server
assembles the vector from the payload, the account's own history, and the fitted encoders — an
endpoint accepting a caller-supplied vector would let that caller choose its own score. The 22
derived features are rejected by name if supplied.

### What `/score` deliberately does not return

The response carries the decision and an opaque `audit_id`, and nothing else quantitative. No
calibrated probability, no operating threshold, no expected-cost arms, no feature attribution
— and no coarse risk band either. Each of those, combined with the amount, lets a caller
binary-search the largest transaction that evades review at a given risk score, and coarsening
a monotone score into bands does not prevent that search, it only slows it by a constant.

All of it is recorded in the audit row. Attribution is served to analysts at
`/audit/entry/{id}/explain`; the cost arms are served nowhere, because the sign of the expected
saving from blocking *is* the decision boundary.

The decision itself is the irreducible disclosure — a scoring API has to tell the caller what
happened to the transaction. That residual is bounded by authentication and the rate limiter.

### Getting a token

There is no signup flow — tokens are minted server-side by
`app.core.security.create_access_token`. For a local demo:

```bash
cd backend && python -c "
from app.config import get_settings
from app.core.security import create_access_token
print(create_access_token('demo-merchant', account_id='acct-1',
                          scopes=('score:write',), settings=get_settings()))
"
```

### Before the first score

`/score` returns 503 until it can load its models, and models are gitignored build outputs.
The serving encoders are rebuilt from the processed parquet once per corpus:

```bash
cd backend && python -m app.data.serving_encoders --source-dataset ieee_cis
```

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
