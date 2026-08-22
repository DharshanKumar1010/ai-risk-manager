"""Causal cost layer: estimates the financial cost of each block/allow decision.

Phase 6 implements this layer as a treatment-effect problem — treatment is "flag or block
this transaction", outcome is net financial impact (lost legitimate revenue if blocked,
fraud loss if allowed through) — using a DR-Learner style meta-learner, exposing

    estimate_cost(transaction, decision) -> CostEstimate

Every figure this layer produces is an estimate built on stated assumptions, and must be
presented as one. The false-positive cost it reports is the track's explicit, named bar.
"""
