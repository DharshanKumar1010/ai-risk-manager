# Model registry

`registry.json` is the **append-only** record of every model this project has trained.
Entries are added, never edited and never removed — a past prediction must remain
traceable to the exact model and feature definition that produced it.

Trained weights are build outputs and are gitignored (`*.pkl`, `*.pt`, `models/artifacts/`).
This registry is the part that is versioned.

Entries are written by `app.ml.registry.append_entry`, which re-reads the file, appends and
rewrites. It refuses a duplicate `model_id` and refuses to write over a file that is not a
JSON list, so an append can neither erase history nor leave two models answering to one
version.

Each entry, appended by the training script that produced it:

```json
{
  "model_id": "tier1-anomaly-lightgbm-ieee-cis-20260822t191207z",
  "layer": "tier1_anomaly",
  "algorithm": "lightgbm",
  "source_dataset": "ieee_cis",
  "trained_at": "2026-08-22T19:12:07+00:00",
  "training_window": { "start": "2017-12-02 00:00:00+00:00", "end": "2018-03-31 19:26:36+00:00" },
  "feature_version": "fv_…",
  "hyperparameters": { "num_leaves": 63, "learning_rate": 0.05, "best_iteration": 0 },
  "random_seed": 42,
  "heldout_test": {
    "pr_auc": 0.0,
    "pr_auc_ci95": [0.0, 0.0],
    "pr_auc_no_skill_floor": 0.0,
    "lift_over_no_skill": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "threshold": 0.0,
    "threshold_criterion": "minimising estimated cost on validation",
    "base_rate": 0.0,
    "confusion_matrix": { "tn": 0, "fp": 0, "fn": 0, "tp": 0 },
    "false_positive_cost": { "total_cost": 0.0, "flag_rate": 0.0, "assumptions": [] },
    "capacity_constrained_operating_point": {},
    "latency": { "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0 }
  },
  "baseline_comparison": [],
  "artifact": "artifacts/….txt",
  "notes": []
}
```

Four fields carry the evaluation bar in
`.claude/skills/ml-evaluation-standards/SKILL.md`, and an entry without them is incomplete:

- **`heldout_test`** comes from the test split only. Never a validation number.
- **`pr_auc_no_skill_floor`** is the positive base rate — the PR-AUC a random ranker scores.
  A headline PR-AUC is uninterpretable without it (section 2).
- **`false_positive_cost`** is mandatory, with its assumptions in plain language (section 3).
  `flag_rate` is recorded alongside it because precision at a 40% flag rate and precision at a
  0.5% flag rate describe two different products.
- **`baseline_comparison`** holds the candidates that lost. They are documented, never
  silently discarded (section 4).

**`feature_version` is the model's own input definition**, not necessarily the Phase 1
pipeline's. Tier-1 reads raw row columns the pipeline deliberately kept out of
`transactions.features`, so it mints its own hash via `Tier1InputSpec.to_feature_definition()`.
Whatever hash is recorded here must reconstruct exactly the inputs the model saw.
