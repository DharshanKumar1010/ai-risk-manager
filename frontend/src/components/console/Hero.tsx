import { StatTile } from '@/components/ui/StatTile'
import { METRICS } from '@/data/metrics.generated'
import { formatPercent } from '@/lib/format'

/**
 * The showcase header. Every number here is the same held-out figure `MetricsPanel` reports in
 * full below it, not a rounder or friendlier restatement -- see ml-evaluation-standards item
 * 4.6 on why a headline number and its full provenance must never drift apart. `cost_delta_pct`
 * is already a percentage (not a [0,1] fraction), which is why it is divided by 100 before
 * `formatPercent` -- the same conversion `CostComparisonChart` applies to the same field.
 */
export function Hero() {
  const { tier1, causal_cost: causalCost } = METRICS
  const costDelta = causalCost?.cost_reduction_vs_baseline ?? null

  return (
    <section aria-labelledby="hero-heading" className="space-y-4 border-b border-border pb-8">
      <div>
        <h1
          id="hero-heading"
          className="font-mono text-2xl font-semibold tracking-tight text-text"
        >
          RiskIQ
        </h1>
        <p className="mt-2 max-w-2xl font-sans text-sm text-text-muted">
          Real-time fraud, chargeback and abuse-ring detection: a three-layer decisioning
          system — anomaly score, abuse-ring graph, and a causal cost layer — not a single
          classifier. The figures below are
          measured on the held-out test split -- see "Held-out evaluation" further down for the
          full confusion matrix and false-positive cost behind each one.
        </p>
      </div>

      {(tier1 !== null || costDelta !== null) && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {tier1 !== null && (
            <StatTile
              label="Tier-1 PR-AUC"
              value={tier1.pr_auc.toFixed(4)}
              detail={`held-out test · n=${tier1.rows.toLocaleString()}`}
              tone="accent"
            />
          )}
          {costDelta !== null && (
            <StatTile
              label="Cost saved vs probability-only"
              value={formatPercent(Math.abs(costDelta.cost_delta_pct) / 100)}
              detail={`at ${formatPercent(costDelta.flag_rate ?? 0)} flag rate`}
              tone="accent"
            />
          )}
          {tier1 !== null && (
            <StatTile
              label="Fraud base rate"
              value={formatPercent(tier1.base_rate)}
              detail={`${tier1.positives.toLocaleString()} of ${tier1.rows.toLocaleString()} rows`}
            />
          )}
        </div>
      )}
    </section>
  )
}
