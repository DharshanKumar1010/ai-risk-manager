import { useEffect, useState } from 'react'

import { ApiError, getExplanation } from '@/lib/api'
import type { ExplanationResponse } from '@/types/api'

export type ExplanationStatus = 'idle' | 'loading' | 'ready' | 'forbidden' | 'no-token' | 'error'

/**
 * `GET /audit/entry/{auditId}/explain`, shared by every view that opens an analyst attribution
 * panel -- `ExplainModal` (row click on a real audit entry) and `ScoreExplainModal` (the
 * scoring widget's own result, which already carries `audit_id` on `ScoreResponse` with no
 * extra fetch needed to get one). `auditId === null` means "nothing to explain yet" (idle, no
 * fetch); `analystToken === null` means "we already know this can't be fetched" (`no-token`),
 * distinct from `forbidden`, which is a token that was tried and refused with 403 -- demo_mode
 * restricts analyst-token minting in production, so both are real, reachable states.
 */
export function useExplanation(
  auditId: number | null,
  analystToken: string | null,
): { status: ExplanationStatus; explanation: ExplanationResponse | null } {
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null)
  const [status, setStatus] = useState<ExplanationStatus>('idle')

  useEffect(() => {
    if (auditId === null) {
      setExplanation(null)
      setStatus('idle')
      return
    }
    if (analystToken === null) {
      setExplanation(null)
      setStatus('no-token')
      return
    }
    let cancelled = false
    setStatus('loading')
    getExplanation(auditId, analystToken)
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
  }, [auditId, analystToken])

  return { status, explanation }
}
