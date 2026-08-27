import { useId, useState, type FormEvent } from 'react'

import { ExplainModal } from '@/components/console/ExplainModal'
import { DecisionBadge, DegradedBadge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { useAuth } from '@/hooks/useAuth'
import { ApiError, getAuditTrail, postScore } from '@/lib/api'
import type { AuditEntryResponse, ScoreResponse } from '@/types/api'

/**
 * A minimal, fixed transaction shape (IEEE-CIS's own smallest valid raw-column set -- the same
 * one backend/tests/conftest.py's `score_payload` fixture uses) so this widget can stay a
 * two-field form (account, amount) rather than 90 raw columns worth of inputs. `amount` is the
 * only field that varies per submission and is the one the fraud-detection literature and this
 * project's own cost model actually key decisions on.
 */
const DEMO_RAW_COLUMNS = { ProductCD: 'W', card1: 13926, card4: 'visa' } as const

function randomTransactionId(): string {
  return `demo-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

/**
 * The two-token showcase: score as a merchant (account_id + score:write, decision only), then
 * see why as an analyst (a completely separate token, explain:read + analyst). This is not two
 * views of one call -- it is two callers, because that is the access-control boundary
 * app/api/score.py and app/api/audit.py actually enforce. See useAuth's module docstring for
 * why neither token is persisted.
 *
 * The "why" step re-fetches the real audit row via `GET /audit/{transaction_id}` rather than
 * reusing anything from `ScoreResponse` -- the merchant response carries no
 * `model_versions`/`feature_version` to build one from, and inventing those fields here would
 * put fabricated data next to a real decision. `ExplainModal` is the same component
 * `DecisionTable` opens on a row click; this widget does not duplicate it.
 */
export function ScoringWidget() {
  const { mintMerchantToken, analystToken } = useAuth()
  const accountFieldId = useId()
  const amountFieldId = useId()

  const [accountId, setAccountId] = useState('acct-demo')
  const [amount, setAmount] = useState('150.00')
  const [status, setStatus] = useState<'idle' | 'scoring' | 'scored' | 'error'>('idle')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ScoreResponse | null>(null)
  const [explainEntry, setExplainEntry] = useState<AuditEntryResponse | null>(null)
  const [explainStatus, setExplainStatus] = useState<'idle' | 'loading'>('idle')

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    setStatus('scoring')
    setError(null)
    setResult(null)
    try {
      const merchantToken = await mintMerchantToken(accountId)
      const response = await postScore(
        {
          transaction_id: randomTransactionId(),
          account_id: accountId,
          event_time: new Date().toISOString(),
          amount,
          raw_columns: DEMO_RAW_COLUMNS,
        },
        merchantToken,
      )
      setResult(response)
      setStatus('scored')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not score this transaction.')
      setStatus('error')
    }
  }

  async function handleExplain(): Promise<void> {
    if (result === null || analystToken === null) return
    setExplainStatus('loading')
    try {
      const trail = await getAuditTrail(result.transaction_id, analystToken)
      const entry =
        trail.entries.find((candidate) => candidate.audit_id === result.audit_id) ??
        trail.entries[0] ??
        null
      setExplainEntry(entry)
    } finally {
      setExplainStatus('idle')
    }
  }

  return (
    <section aria-labelledby="scoring-widget-heading" className="space-y-3">
      <div>
        <h2 id="scoring-widget-heading" className="font-sans text-sm font-semibold text-text">
          Score a transaction
        </h2>
        <p className="font-sans text-xs text-text-faint">
          Submitted with a merchant token: account-scoped, decision-only. It cannot see why.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label htmlFor={accountFieldId} className="font-sans text-xs text-text-faint">
            account
          </label>
          <input
            id={accountFieldId}
            value={accountId}
            onChange={(event) => setAccountId(event.target.value)}
            required
            className="rounded-console border border-border bg-surface px-2 py-1.5 font-mono text-sm text-text"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor={amountFieldId} className="font-sans text-xs text-text-faint">
            amount (USD)
          </label>
          <input
            id={amountFieldId}
            value={amount}
            onChange={(event) => setAmount(event.target.value)}
            inputMode="decimal"
            pattern="[0-9]*\.?[0-9]*"
            required
            className="w-32 rounded-console border border-border bg-surface px-2 py-1.5 font-mono text-sm text-text"
          />
        </div>
        <Button type="submit" disabled={status === 'scoring'}>
          {status === 'scoring' ? 'Scoring…' : 'Score transaction'}
        </Button>
      </form>

      {status === 'error' && (
        <p className="font-sans text-sm text-signal-review">
          {error ?? 'Could not score this transaction.'}
        </p>
      )}

      {result && (
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-3 rounded-console border border-border bg-surface p-3">
            <DecisionBadge decision={result.decision} />
            {result.degraded && <DegradedBadge />}
            <span className="font-mono text-xs text-text-faint">{result.transaction_id}</span>
            <Button
              variant="ghost"
              onClick={() => void handleExplain()}
              disabled={explainStatus === 'loading' || analystToken === null}
            >
              Why? (analyst view)
            </Button>
          </div>
          <p className="font-sans text-xs text-text-faint">
            A live demo decision on this transaction, not a held-out evaluation result.
          </p>
        </div>
      )}

      <ExplainModal entry={explainEntry} onClose={() => setExplainEntry(null)} />
    </section>
  )
}
