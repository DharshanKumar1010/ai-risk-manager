# Model registry

`registry.json` is the **append-only** record of every model this project has trained.
Entries are added, never edited and never removed — a past prediction must remain
traceable to the exact model and feature definition that produced it.

Trained weights are build outputs and are gitignored (`*.pkl`, `*.pt`, `models/artifacts/`).
This registry is the part that is versioned.

Each entry, appended by the training script that produced it:

```json
{
  "model_id": "tier1-lightgbm-20260823T1412Z",
  "layer": "tier1_anomaly",
  "algorithm": "LightGBM",
  "trained_at": "2026-08-23T14:12:07Z",
  "training_window": { "start": "2019-01-01", "end": "2019-04-30" },
  "feature_version": "sha256:…",
  "hyperparameters": { "num_leaves": 31, "learning_rate": 0.05 },
  "random_seed": 42,
  "heldout_test": {
    "pr_auc": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "threshold": 0.0,
    "confusion_matrix": { "tn": 0, "fp": 0, "fn": 0, "tp": 0 }
  },
  "notes": ""
}
```

`feature_version` must match the hash emitted by the Phase 1 feature store for the inputs
this model was trained on. `heldout_test` figures come from the test split only — see
`.claude/skills/ml-evaluation-standards/SKILL.md`.
