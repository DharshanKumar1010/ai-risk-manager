import { useEffect, useState } from 'react'

import { Badge, DecisionBadge } from '@/components/ui/Badge'
import { Dialog } from '@/components/ui/Dialog'
import { useAuth } from '@/hooks/useAuth'
import { ApiError, getExplanation } from '@/lib/api'
import { formatPercent } from '@/lib/format'
import type { AuditEntryResponse, ExplanationResponse } from '@/types/api'

interface ExplainModalProps {
  entry: AuditEntryResponse | null
  onClose: () => void
}

/**
 * The drill-down Phase 8 step 3 asks for: attribution, contributing tiers, model versions,
 * feature version, degraded reason, and the audit record's own provenance. What it does NOT
 * show, because the backend never sends it to any caller: the cost estimate. `ExplanationResponse`
 * has no `cost_estimate` field -- `DecisionCost` is a hard line, not a trim, per Phase 6/7's
 * carried gate. There is no "causal cost estimate" panel here to omit; it was never a field
 * to begin with.
 *
 * Requires `explain:read` + `analyst` -- the analyst token `useAuth` mints automatically. If
 * this were ever opened with a merchant token it would 403, which is rendered as a plain
 * access message rather than a generic error, because that refusal is the point being shown.
 */
export function ExplainModal({ entry, onClose }: ExplainModalProps) {
  const { analystToken } = useAuth()
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null)
  const [status, setStatus] = useState<
    'idle' | 'loading' | 'ready' | 'forbidden' | 'no-token' | 'error'
  >('idle')

  useEffect(() => {
    if (entry === null) {
      setExplanation(null)
      setStatus('idle')
      return
    }
    if (analystToken === null) {
      // demo_mode restricts analyst-token minting to 403 in production -- distinct from
      // 'forbidden' (a merchant token was tried and refused): here there is no token to try.
      setExplanation(null)
      setStatus('no-token')
      return
    }
    let cancelled = false
    setStatus('loading')
    getExplanation(entry.audit_id, analystToken)
      .then((response) => {
        if (cancelled) return
        setExplanation(response)
        setStatus('ready')
      })
      .catch((err: unknown) => {
        if (cancelled) return
        if (err instanceof ApiError && err.status === 403) {
          setStatus('forbidden')
          return
        }
        setStatus('error')
      })
    return () => {
      cancelled = true
    }
  }, [entry, analystToken])

  return (
    <Dialog
      open={entry !== null}
      onClose={onClose}
      title={entry ? `Decision — ${entry.transaction_id}` : 'Decision'}
    >
      {entry === null ? null : (
        <div className="space-y-4 font-sans text-sm">
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2">
            <dt className="text-text-faint">Decision</dt>
            <dd>
              <DecisionBadge decision={entry.decision} />
            </dd>

            <dt className="text-text-faint">Decided at</dt>
            <dd className="font-mono text-xs">{entry.decided_at}</dd>

            <dt className="text-text-faint">Feature version</dt>
            <dd className="font-mono text-xs">{entry.feature_version}</dd>

            <dt className="text-text-faint">Model layers</dt>
            <dd className="flex flex-wrap gap-1">
              {Object.entries(entry.model_versions).map(([layer, versionId]) => (
                <Badge key={layer} tone="neutral" className="normal-case">
                  {layer}: {versionId}
                </Badge>
              ))}
            </dd>

            {entry.degraded && (
              <>
                <dt className="text-text-faint">Degraded</dt>
                <dd className="text-signal-degraded">
                  {entry.degraded_reason ?? 'reason not recorded'}
                </dd>
              </>
            )}
          </dl>

          <div className="border-t border-border pt-4">
            <h3 className="mb-2 font-sans text-xs font-semibold uppercase tracking-wide text-text-faint">
              Why (analyst view)
            </h3>
            {status === 'loading' && (
              <p className="text-text-faint">Loading attribution…</p>
            )}
            {status === 'forbidden' && (
              <p className="text-signal-review">
                This view requires an analyst token. A merchant integration cannot see why a
                decision was made — that is the access-control boundary this endpoint enforces,
                not a bug in this demo.
              </p>
            )}
            {status === 'error' && (
              <p className="text-text-faint">Attribution is unavailable right now.</p>
            )}
            {status === 'no-token' && (
              <p className="text-text-faint">
                Attribution requires reviewer access, which this demo instance does not grant
                automatically — not available here.
              </p>
            )}
            {status === 'ready' && explanation !== null && (
              <div className="space-y-3">
                <div>
                  <span className="text-text-faint">Calibrated probability: </span>
                  <span className="font-mono tabular-nums">
                    {formatPercent(explanation.risk_probability)}
                  </span>
                </div>
                <FeatureBars features={explanation.top_features} />
              </div>
            )}
          </div>
        </div>
      )}
    </Dialog>
  )
}

function FeatureBars({ features }: { features: ExplanationResponse['top_features'] }) {
  if (features.length === 0) {
    return <p className="text-text-faint">No attribution recorded for this decision.</p>
  }
  const maxMagnitude = Math.max(...features.map((f) => Math.abs(f.contribution)), 1e-9)
  return (
    <ul className="space-y-1.5">
      {features.map((feature) => {
        const width = (Math.abs(feature.contribution) / maxMagnitude) * 100
        const positive = feature.contribution >= 0
        return (
          <li key={feature.feature} className="flex items-center gap-2">
            <span className="w-40 shrink-0 truncate font-mono text-xs text-text-muted">
              {feature.feature}
            </span>
            <span className="flex h-4 flex-1 items-center">
              <span
                className={
                  positive
                    ? 'h-2 rounded-console bg-signal-review'
                    : 'h-2 rounded-console bg-accent'
                }
                style={{ width: `${width}%` }}
              />
            </span>
            <span className="w-16 shrink-0 text-right font-mono text-xs tabular-nums">
              {feature.contribution.toFixed(3)}
            </span>
          </li>
        )
      })}
    </ul>
  )
}
