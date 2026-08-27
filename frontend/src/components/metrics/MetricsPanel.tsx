import { CalibrationChart } from '@/components/metrics/CalibrationChart'
import { ConfusionMatrix } from '@/components/metrics/ConfusionMatrix'
import { CostComparisonChart } from '@/components/metrics/CostComparisonChart'
import { PrCurveChart } from '@/components/metrics/PrCurveChart'
import { ProvenanceChip } from '@/components/metrics/ProvenanceChip'
import { SensitivityTable } from '@/components/metrics/SensitivityTable'
import { METRICS } from '@/data/metrics.generated'
import { formatPercent } from '@/lib/format'

/**
 * Phase 8 step 5: PR curve, confusion matrix, calibration curve, and the cost-sensitivity
 * chart, all pulling from the real held-out evaluation and labelled as such -- never mixed
 * with the live-demo decision table without the label that distinguishes the two. Every
 * number in this panel comes from `frontend/src/data/metrics.generated.ts`, a build-time
 * export from `models/registry.json` (see `backend/app/ml/export_metrics.py` for why it is
 * not a live endpoint).
 */
export function MetricsPanel() {
  const { tier1, causal_cost: causalCost, meta_learner: metaLearner, tier3 } = METRICS

  return (
    <section aria-labelledby="metrics-heading" className="space-y-8">
      <div>
        <h2 id="metrics-heading" className="font-sans text-sm font-semibold text-text">
          Held-out evaluation
        </h2>
        <p className="mt-1 font-sans text-xs text-text-faint">
          Every figure below is computed on the held-out test split, never on live-demo
          traffic. Generated {new Date(METRICS.generated_at).toLocaleString()} from registry{' '}
          {METRICS.registry_sha256}.
        </p>
      </div>

      {tier1 !== null && (
        <div className="space-y-3">
          <h3 className="font-sans text-xs font-semibold uppercase tracking-wide text-text-faint">
            Tier-1 — precision / recall
          </h3>
          <ProvenanceChip
            modelId={tier1.model_id}
            rows={tier1.rows}
            positives={tier1.positives}
            baseRate={tier1.base_rate}
          />
          <div className="grid gap-4 md:grid-cols-2">
            <Panel>
              <ConfusionMatrix
                matrix={tier1.confusion_matrix}
                flagRate={tier1.false_positive_cost.flag_rate}
              />
            </Panel>
            <Panel>
              {tier1.pr_curve !== undefined ? (
                <PrCurveChart
                  curve={tier1.pr_curve}
                  operatingPoint={{ precision: tier1.precision, recall: tier1.recall }}
                />
              ) : (
                <p className="text-sm text-text-faint">PR curve not exported.</p>
              )}
            </Panel>
          </div>
          <p className="font-mono text-xs text-text-muted">
            PR-AUC {tier1.pr_auc.toFixed(4)} (95% CI [{tier1.pr_auc_ci95[0].toFixed(4)},{' '}
            {tier1.pr_auc_ci95[1].toFixed(4)}]) — {tier1.pr_auc_no_skill_floor.toFixed(4)}{' '}
            no-skill floor
          </p>
          <FalsePositiveCostNote cost={tier1.false_positive_cost} />
        </div>
      )}

      {tier3 !== null && (
        <div className="space-y-3">
          <h3 className="font-sans text-xs font-semibold uppercase tracking-wide text-text-faint">
            Tier-3 Ring Detection (held-out test)
          </h3>
          <p className="font-sans text-xs text-text-faint">
            Measured at the <strong>ring</strong> level, not the transaction level Tier-1's
            {tier1 !== null ? ` ${tier1.pr_auc.toFixed(4)}` : ' PR-AUC'} above is measured at.
            IEEE-CIS accounts barely recur, so a per-transaction ring score abstains on most
            test rows and is separately below its own no-skill floor. The two PR-AUCs answer
            different questions; shown together only so the ring graph elsewhere on this page
            has a measured number attached to it.
          </p>
          <ProvenanceChip
            modelId={tier3.model_id}
            rows={tier3.rows}
            positives={tier3.positives}
            baseRate={tier3.base_rate}
          />
          <Panel>
            <ConfusionMatrix
              matrix={tier3.confusion_matrix}
              flagRate={tier3.false_positive_cost.flag_rate}
            />
          </Panel>
          <p className="font-mono text-xs text-text-muted">
            Ring PR-AUC {tier3.pr_auc.toFixed(4)} (95% CI [{tier3.pr_auc_ci95[0].toFixed(4)},{' '}
            {tier3.pr_auc_ci95[1].toFixed(4)}]) — {tier3.pr_auc_no_skill_floor.toFixed(4)}{' '}
            no-skill floor
          </p>
        </div>
      )}

      {causalCost !== null && (
        <div className="space-y-3">
          <h3 className="font-sans text-xs font-semibold uppercase tracking-wide text-text-faint">
            Cost-aware ranking (Phase 6)
          </h3>
          <ProvenanceChip
            modelId={causalCost.model_id}
            rows={causalCost.rows}
            positives={causalCost.positives}
            baseRate={causalCost.base_rate}
          />
          <div className="grid gap-4 md:grid-cols-2">
            <Panel>
              {causalCost.policies !== null && (
                <CostComparisonChart
                  defaultPolicies={causalCost.policies}
                  defaultDelta={causalCost.cost_reduction_vs_baseline}
                  cnpPolicies={causalCost.cnp_regime ?? []}
                  cnpDelta={causalCost.cnp_regime_deltas?.find((d) => d.policy === 'plug_in')}
                />
              )}
            </Panel>
            <Panel>
              {causalCost.calibration !== null ? (
                <CalibrationChart calibration={causalCost.calibration} />
              ) : (
                <p className="text-sm text-text-faint">Calibration data unavailable.</p>
              )}
            </Panel>
          </div>
          {causalCost.calibration !== null && (
            <p className="font-mono text-xs text-text-muted">
              ECE {(causalCost.calibration.expected_calibration_error * 100).toFixed(2)}% ·
              Brier {causalCost.calibration.brier_score.toFixed(4)} — {causalCost.calibration.note}
            </p>
          )}
          {causalCost.sensitivity !== null && (
            <Panel>
              <SensitivityTable rows={causalCost.sensitivity} />
            </Panel>
          )}
          {causalCost.ope_validation_caveat !== null && (
            <p className="rounded-console border border-border bg-surface-raised p-3 font-sans text-xs text-text-muted">
              {causalCost.ope_validation_caveat}
            </p>
          )}
        </div>
      )}

      {metaLearner !== null && (
        <div className="rounded-console border border-signal-degraded/30 bg-signal-degraded-bg p-4">
          <h3 className="font-sans text-xs font-semibold uppercase tracking-wide text-signal-degraded">
            Retired: meta-learner fusion
          </h3>
          <p className="mt-1 font-mono text-xs text-text-muted">
            {metaLearner.model_id} — PR-AUC {metaLearner.pr_auc.toFixed(4)} vs Tier-1 alone.
            {metaLearner.delta_vs_tier1_alone !== null && (
              <>
                {' '}
                Delta {formatPercent(metaLearner.delta_vs_tier1_alone.delta)}, CI [
                {formatPercent(metaLearner.delta_vs_tier1_alone.interval[0])},{' '}
                {formatPercent(metaLearner.delta_vs_tier1_alone.interval[1])}] —{' '}
                {metaLearner.delta_vs_tier1_alone.verdict}.
              </>
            )}
          </p>
          <p className="mt-1 font-sans text-xs text-text-muted">
            Measured losing to Tier-1 alone and not shipped in the decision path. Kept here
            because a phase this project's own standards hold itself to says baselines that
            lost are documented, not discarded.
          </p>
        </div>
      )}
    </section>
  )
}

function Panel({ children }: { children: React.ReactNode }) {
  return <div className="rounded-console border border-border bg-surface p-4">{children}</div>
}

/**
 * Deliberately does not state the per-unit review cost or chargeback fee -- only the totals
 * and the generic methodological caveats. Combined with a live threshold those two numbers
 * hand a visitor the exact decision boundary (`p <= r/(A+f+r)`); see types/metrics.ts's module
 * comment and export_metrics.py's `_scrub_false_positive_cost` for why neither ships here.
 */
function FalsePositiveCostNote({
  cost,
}: {
  cost: NonNullable<typeof METRICS.tier1>['false_positive_cost']
}) {
  return (
    <div className="space-y-1 font-sans text-xs text-text-faint">
      <p>
        Total estimated cost ${cost.total_cost.toLocaleString()} on this split (
        {cost.false_positives.toLocaleString()} false positives, {cost.false_negatives.toLocaleString()}{' '}
        false negatives) — an estimate under stated assumptions rather than ground truth.
      </p>
      {cost.assumptions.length > 0 && (
        <ul className="list-disc space-y-0.5 pl-4">
          {cost.assumptions.map((text) => (
            <li key={text}>{text}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
