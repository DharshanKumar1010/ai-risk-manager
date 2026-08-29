import { AttributionPanel } from '@/components/console/ExplainModal'
import { DecisionBadge, DegradedBadge } from '@/components/ui/Badge'
import { Dialog } from '@/components/ui/Dialog'
import { useAuth } from '@/hooks/useAuth'
import { useExplanation } from '@/hooks/useExplanation'
import { formatAmount, formatDateTime } from '@/lib/format'
import type { ScoreResponse } from '@/types/api'

interface ScoreExplainModalProps {
  result: ScoreResponse | null
  accountId: string
  amount: string
  onClose: () => void
}

/**
 * The "Why?" view for the scoring widget's own result -- always openable once there is a
 * `result`, unlike the old design this replaced, which disabled the button whenever no
 * analyst token was available (the common case: demo_mode 403s analyst-token minting in
 * production, so this button was permanently disabled on the deployed instance). It never
 * needs `GET /audit/{transaction_id}` to build an `AuditEntryResponse` first -- `ScoreResponse`
 * already carries `audit_id`, so this fetches `GET /audit/entry/{audit_id}/explain` directly.
 *
 * The summary section (decision, transaction, account, amount, model version) renders
 * unconditionally from data this widget already has for real -- the merchant's own request and
 * response -- never from a fabricated `AuditEntryResponse`. Only the attribution panel below it
 * depends on an analyst token, and gracefully explains the restriction when one isn't
 * available, via the same `AttributionPanel` `ExplainModal` uses for its row-click view.
 */
export function ScoreExplainModal({ result, accountId, amount, onClose }: ScoreExplainModalProps) {
  const { analystToken } = useAuth()
  const { status, explanation } = useExplanation(result?.audit_id ?? null, analystToken)

  return (
    <Dialog
      open={result !== null}
      onClose={onClose}
      title={result ? `Decision — ${result.transaction_id}` : 'Decision'}
    >
      {result === null ? null : (
        <div className="space-y-4 font-sans text-sm">
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2">
            <dt className="text-text-faint">Decision</dt>
            <dd className="flex items-center gap-2">
              <DecisionBadge decision={result.decision} />
              {result.degraded && <DegradedBadge />}
            </dd>

            <dt className="text-text-faint">Transaction ID</dt>
            <dd className="font-mono text-xs">{result.transaction_id}</dd>

            <dt className="text-text-faint">Account</dt>
            <dd className="font-mono text-xs">{accountId}</dd>

            <dt className="text-text-faint">Amount</dt>
            <dd className="font-mono text-xs tabular-nums">{formatAmount(amount)}</dd>

            <dt className="text-text-faint">Scored at</dt>
            <dd className="font-mono text-xs">{formatDateTime(result.decided_at)}</dd>

            <dt className="text-text-faint">Model version</dt>
            <dd className="font-mono text-xs">{result.model_version}</dd>
          </dl>

          <p className="text-text-muted">
            This transaction was scored by a four-layer system: Tier-1 anomaly detection →
            Tier-2 chargeback model → Tier-3 abuse-ring graph → Tier-4 causal cost ranker.
          </p>

          <AttributionPanel status={status} explanation={explanation} />
        </div>
      )}
    </Dialog>
  )
}
