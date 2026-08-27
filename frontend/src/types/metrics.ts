/**
 * The shape `backend/app/ml/export_metrics.py` must produce. Every field
 * ml-evaluation-standards requires alongside a headline number -- n, positives, base rate,
 * the raw confusion matrix, the FP-cost estimate and its assumptions -- is REQUIRED here, not
 * optional. That is the enforcement mechanism this whole export design rests on: a future edit
 * to the Python script that stops emitting one of these fields fails `tsc -b --noEmit`, which
 * `npm run lint` already runs in CI, rather than silently shipping a number with no label.
 *
 * **No operating threshold or per-unit cost figure appears anywhere in this file, on purpose.**
 * This module is inlined into a public, unauthenticated JS bundle -- unlike every API response,
 * which `schemas.py` already withholds `risk_band`/probability/cost fields from for the same
 * reason (an evasion oracle: the served policy is `allow` iff `p <= r/(A+f+r)`, so the exact
 * threshold plus both unit costs hands any visitor the live decision boundary with zero
 * probing). `export_metrics.py`'s `_scrub_false_positive_cost`/`_scrub_operating_point`/
 * `_strip_thresholds` strip it before this file is ever generated; re-adding a `threshold`,
 * `threshold_criterion`, `review_cost_per_false_positive` or
 * `flat_chargeback_fee_per_false_negative` field here would silently reopen a hole those
 * functions exist to close, so don't -- what ships instead is the cost *totals*, which satisfy
 * the same labelling obligation without exposing the boundary.
 */

export interface ConfusionMatrix {
  tn: number
  fp: number
  fn: number
  tp: number
}

export interface FalsePositiveCost {
  flag_rate: number
  false_positives: number
  false_negatives: number
  false_positive_cost: number
  false_negative_cost: number
  total_cost: number
  /** Generic methodological caveats only -- the two sentences stating the literal per-unit
   * dollar figures are stripped server-side, see the module comment above. */
  assumptions: string[]
  [key: string]: unknown
}

export interface PrCurvePoint {
  precision: number
  recall: number
}

export interface PrCurve {
  points: PrCurvePoint[]
  base_rate: number
  n: number
  positives: number
}

/** The one thing every headline metric block shares -- the labelling obligation itself. */
export interface HeldoutProvenance {
  rows: number
  positives: number
  base_rate: number
  confusion_matrix: ConfusionMatrix
  false_positive_cost: FalsePositiveCost
}

export interface Tier1Metrics extends HeldoutProvenance {
  model_id: string
  feature_version: string | null
  pr_auc: number
  pr_auc_ci95: [number, number]
  pr_auc_no_skill_floor: number
  precision: number
  recall: number
  f1: number
  capacity_constrained_operating_point: Record<string, unknown> | null
  /** Absent until export_metrics.py is run without --no-pr-curve -- present in every export
   * the frontend actually ships, since only --check (a CI-only diff mode) omits it. */
  pr_curve?: PrCurve
}

export interface CostDelta {
  policy: string
  baseline: string
  /** Present on the headline cost_reduction_vs_baseline; absent on cnp_regime_deltas, whose
   * matched flag rate is fixed and stated once at the enclosing table's level instead. */
  flag_rate?: number
  cost_delta_per_1000: number
  cost_delta_ci95: [number, number]
  cost_delta_pct: number
  value_recall_delta?: number
  value_recall_delta_ci95?: [number, number]
  verdict: string
}

export interface CalibrationBin {
  bin_lower: number
  bin_upper: number
  mean_predicted: number
  observed_frequency: number
  count: number
}

export interface Calibration {
  expected_calibration_error: number
  brier_score: number
  fitted_on: string
  /** Plain-language caveat on what a calibration error means for this layer's cost figures --
   * carried as a required field rather than left implicit, per the same labelling obligation
   * that makes false_positive_cost.assumptions required. */
  note: string
  bins: CalibrationBin[]
}

export interface SensitivityRow {
  varied: string
  factor: number
  total_cost: number
  false_positives: number
  false_negatives: number
  flag_rate: number
}

export interface RegimePolicyPoint {
  policy: string
  flag_rate: number
  precision: number
  recall: number
  recall_by_value: number
  caught_fraud_value: number
  missed_fraud_value: number
  confusion_matrix: ConfusionMatrix
  cost_per_1000: number
}

export interface CausalCostMetrics extends HeldoutProvenance {
  model_id: string
  cost_reduction_vs_baseline: CostDelta
  /** The three ranking strategies at the shipped operating point, default cost regime
   * ($3 review / $15 chargeback). Same shape as cnp_regime, so one chart component renders
   * both regimes. */
  policies: RegimePolicyPoint[] | null
  calibration: Calibration | null
  sensitivity: SensitivityRow[] | null
  cnp_regime: RegimePolicyPoint[] | null
  cnp_regime_deltas: CostDelta[] | null
  ope_validation_caveat: string | null
  notes: string[]
}

export interface MetaLearnerMetrics {
  model_id: string
  pr_auc: number
  pr_auc_ci95: [number, number]
  delta_vs_tier1_alone: { delta: number; interval: [number, number]; verdict: string } | null
  notes: string[]
}

/**
 * Tier-3's held-out result, at the ring level -- `unit_of_analysis` is a literal `'ring'`, not
 * a free-form string, precisely so a reader (or a component) cannot place this PR-AUC beside
 * Tier-1's transaction-level one without the type itself saying the units differ. IEEE-CIS's
 * per-transaction ring score is separately measured, elsewhere, as below its own no-skill
 * floor -- this block is the number the ring graph in the console actually earns.
 */
export interface Tier3Metrics extends HeldoutProvenance {
  model_id: string
  unit_of_analysis: 'ring'
  pr_auc: number
  pr_auc_ci95: [number, number]
  pr_auc_no_skill_floor: number
}

export interface GeneratedMetrics {
  generated_at: string
  registry_sha256: string
  tier1: Tier1Metrics | null
  causal_cost: CausalCostMetrics | null
  meta_learner: MetaLearnerMetrics | null
  tier3: Tier3Metrics | null
}
