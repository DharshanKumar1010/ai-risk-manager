---
name: ml-evaluation-standards
description: The RiskIQ model-evaluation bar. Invoke before reporting any model result and as the gate on Phases 1-6 and 10. Expands the evaluation non-negotiables in CLAUDE.md into checkable items.
---

# RiskIQ Model Evaluation Standards

The buildathon bar is a working detector with **measured precision/recall on a held-out
test set** and an **honest false-positive cost**. A result that does not meet the standards
below is not reportable — not to the user, not in a README, not in the pitch video.

## 1. Splitting

- [ ] Train/val/test are split **chronologically**: first 70% train, next 15% val, last 15%
      test. Never `train_test_split(..., shuffle=True)` on transaction data.
- [ ] Split boundaries are asserted by an automated test, not promised in a docstring.
- [ ] No engineered feature reads a timestamp later than the transaction it describes.
      This is the leakage check — it is a test, not a code comment.
- [ ] Per-account aggregates (velocity, z-score vs. own history) are computed on a trailing
      window only. A group-wide `mean()` over the full dataset leaks the future into the past.
- [ ] Any threshold, hyperparameter, or feature selection is chosen on **train/val**.
      Touching test to pick anything makes it a validation set and invalidates the headline.

## 2. Metrics

- [ ] Headline metric is **PR-AUC**, not ROC-AUC. At a ~0.5% positive rate ROC-AUC looks
      impressive while the model is unusable; PR-AUC does not flatter class imbalance.
- [ ] Precision, recall, F1 and the **full confusion matrix** (TN/FP/FN/TP as raw counts)
      accompany every headline number.
- [ ] All headline numbers come from the **held-out test split only**. Train and validation
      numbers may be shown for over/underfitting diagnosis, always explicitly labelled.
- [ ] The positive-class base rate is stated alongside the metrics. Precision without a base
      rate is uninterpretable.
- [ ] The operating threshold is stated. "Precision 0.91" with no threshold is not a result.
- [ ] Accuracy is never the headline. At this class balance a model predicting all-negative
      scores >99% accuracy.

## 3. False-positive cost — mandatory

- [ ] Every reported result ships with a false-positive cost estimate. **A metrics section
      without one is incomplete for this project.**
- [ ] The cost model states its assumptions in plain language, in the code and the README.
- [ ] Cost is presented as an **estimate with stated assumptions**, never as ground truth.
- [ ] Where a threshold is recommended, the recommendation is justified by cost, not by F1
      alone, and a ±50% sensitivity analysis shows how it shifts under different assumptions.

## 4. Honesty

- [ ] **Never present one example as proof.** A single caught fraud is an anecdote; show the
      confusion matrix.
- [ ] Every model has an explicit **"What this does NOT catch"** section in its own README.
      Write it from the false negatives actually observed, not from imagination.
- [ ] Baselines that lost are documented as comparisons, not silently discarded.
- [ ] If a result looks too good, treat it as a leakage bug until proven otherwise and say so.
      PR-AUC above ~0.95 on fraud data is a leak signal, not a triumph.
- [ ] Live-demo output and held-out evaluation numbers are never shown together without a
      label distinguishing them.

## 5. Reproducibility

- [ ] Every training script sets **and logs** a random seed.
- [ ] Model metadata is appended to `models/registry.json` — algorithm, training window,
      feature_version, hyperparameters, held-out PR-AUC. **Append, never overwrite.**
- [ ] The `feature_version` hash recorded with a model matches the feature store that
      produced its inputs, so any past prediction can be traced to an exact feature definition.

## Reporting format

Every model result is reported in this shape:

```
### <model> — held-out test (n=<count>, positives=<count>, base rate=<pct>)
Threshold: <value>  (chosen on validation by <criterion>)

PR-AUC     0.XXX
Precision  0.XXX      Recall  0.XXX      F1  0.XXX

Confusion matrix        Predicted
                     neg        pos
Actual  neg        <TN>       <FP>
        pos        <FN>       <TP>

False-positive cost estimate: <value> per <unit>
  Assumptions: <explicit list>

What this does NOT catch:
  - <observed failure mode from actual FNs>
```

A phase gated on these standards does not pass while any checklist item is unmet.
