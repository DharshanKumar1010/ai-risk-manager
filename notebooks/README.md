# Notebooks

Notebooks are versioned; their **outputs are not**. The `nbstripout` pre-commit hook
strips outputs before a commit lands, so diffs stay readable and no dataset rows leak
into git history through a cached cell output.

Anything a notebook proves must be reproducible from code under `backend/app/`.
A number that exists only in a notebook cell is not a result.

## What is actually here

No `.ipynb` files, despite what this file said through Phase 3. Every report in this
directory is **generated markdown**, written by a module under `backend/app/` and regenerated
from scratch on each run. That turned out to satisfy the rule above more strictly than a
notebook does: there is no cell that can hold a number the code no longer produces.

| File | Written by | Phase |
|---|---|---|
| `eda_report.md` | `python -m app.data.pipeline --report` | 1 |
| `tier1_report.md` | `python -m app.models.train_tier1` | 2 |
| `tier2_report.md`, `tier2_loss_curve.png`, `tier2_error_distribution.png` | `python -m app.models.train_tier2` | 3 |
| `tier3_report.md`, `tier3_ring_detected.png`, `tier3_cluster_clean.png` | `python -m app.models.train_tier3` | 4 |

The two Tier-3 images are the Phase 4 visual gate: a detected ring and a clean cluster have to
be visibly different, not merely differently labelled. Red nodes are accounts touching
labelled fraud, blue are clean, grey squares are shared entities, and a thick red edge is an
inferred transfer-to-cash-out chain link.
