# Data

Contents are **gitignored**. This directory structure is tracked so a cold clone knows
where the downloads go.

| Path | Contents |
|------|----------|
| `data/raw/` | Untouched downloads — IEEE-CIS (Tiers 1, 2, 5, 6) and PaySim (Tier 3) |
| `data/processed/` | Pipeline output, written by Phase 1 |

Never commit a dataset file. Never commit a sample "just for testing" — transaction data
does not belong in git history, and the `check-added-large-files` hook will block it.
