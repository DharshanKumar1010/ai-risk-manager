# RiskIQ — Executive Summary

Razorpay AI Buildathon 2026, Track 2: AI Risk Manager. One page. Full detail, every
number's citation, and the phase-by-phase obstacle log live in `BUILD_LOG.md`.

## What this is

A three-layer fraud, chargeback and abuse-ring decisioning system: a per-transaction
anomaly score, a causal cost layer that turns that score into a block/allow decision
by its *dollar* consequence rather than its raw probability, and a transaction-network
graph that finds coordinated abuse rings a per-transaction model cannot see by
construction. A behavioral sequence model and a meta-learner fusing all three signals
were also built and measured — and retired, because the held-out numbers said so.

## What was tried, what worked, what didn't

**Worked, and shipped.** Tier-1's LightGBM anomaly score reaches 0.5276 PR-AUC
(15.2x its no-skill floor) on IEEE-CIS held-out test, at ~6ms p95 latency. Tier-3's
Louvain ring detector finds abuse rings at 0.6465 ring-level PR-AUC (6.0x lift). The
causal cost layer — built after discovering the brief's original ask (IPW on historical
treatment decisions) was not executable on this data, because every historical
transaction in both corpora was *allowed*, with no control arm to recover a treatment
effect from — collapses provably onto a cost-weighted plug-in over Tier-1's calibrated
probability, and that plug-in ranks 22.41% cheaper than probability-only ranking at a
matched review budget.

**Tried, and didn't.** The meta-learner — Tier-1 + Tier-2 + Tier-3 fused by XGBoost,
the architecture's fourth layer — loses to Tier-1 alone on the held-out ranking metric
(0.4954 vs 0.5276, CI excludes zero on the negative side) and was retired rather than
shipped. A second cost-aware policy that *trains* on cost instead of just
*thresholding* by it reached 19.79% cheaper than probability ranking — statistically
tied with the shipped 22.41% policy — meaning cost-sensitive training bought nothing
over cost-sensitive thresholding on this data, exactly what the layer's own algebra
predicted before the run. And PaySim's Tier-1 result (PR-AUC 0.9999) tripped this
project's own automated leak-suspicion wire; investigated rather than reported, it
turned out to be a property of the simulator (its fraud agent transfers the exact full
account balance, 97.5% of the time) rather than a leak — and is quarantined from every
headline as a result.

## Why the honest negatives are evidence of rigor

A fraud-detection submission that reports only wins is not more trustworthy than one
that reports wins and losses — it is less legible about which parts of it to believe.
The meta-learner tie, Tier-3's provisional-not-final headline, and PaySim's quarantined
figures are not omissions patched over after the fact; they are measured results,
reported at the same standard as every figure that flattered the project, because the
project's own evaluation standards required exactly that. That standard also caught
real bugs: a single Phase 9.5 audit session — six independent review passes across
security, ML evaluation, two bug hunts, dead code, and cross-phase consistency — found
and fixed 2 new security findings (a fail-open environment default that silently
disabled every secret-placeholder guard; an unrate-limited token-minting endpoint), one
piece of documentation drift in the Tier-3 headline across two earlier phases, and one
test that had been silently checking nothing since a dependency upgrade changed what it
was inspecting. All three were confirmed by direct reproduction before being called
real, not assumed from the finding's description — and all three are fixed, not just
logged.

## The sentence that should stick

**Cost-aware ranking pays in proportion to how heterogeneous the loss is.** At a flat
$500 chargeback fee, almost every missed fraud costs about the same, and there is
little for a cost model to exploit; at $15, transaction amount varies enough that
ranking by expected cost rather than raw probability catches 22.41% less in expected
loss at the same review budget. That is not a fixed advantage this system claims —
it is a conditional one, quantified under both regimes, because quoting only the
flattering number is exactly what this project's own standards forbid.
