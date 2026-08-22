"""Tier-1: real-time per-transaction anomaly scoring.

Phase 2 implements this layer: an Isolation Forest unsupervised baseline against a
LightGBM supervised classifier, compared on held-out PR-AUC, exposing

    score(transaction: TransactionFeatures) -> Tier1Result

Tier-1 must be scoreable at ingestion time using per-transaction features only, with no
dependency on Tier-2 or Tier-3, and a p95 latency under 50ms.
"""
