import { useCallback, useEffect, useState } from 'react'

import { ApiError, getAuditFeed } from '@/lib/api'
import type { AuditEntryResponse } from '@/types/api'

export interface AuditFeedState {
  entries: AuditEntryResponse[]
  status: 'idle' | 'loading' | 'ready' | 'error'
  error: string | null
  /** Re-fetch GET /audit. Called on a manual refresh, and by the live feed hook (see
   * useLiveFeed's `onDecision`) so a newly scored decision appears here without this hook
   * needing to reshape a `FeedEvent` into an `AuditEntryResponse` -- the two carry different
   * fields by design; see types/api.ts's `FeedEvent` comment. */
  refresh: () => void
}

/** GET /audit, refetchable. Requires an analyst token -- the decision table is an analyst
 * view; see hooks/useAuth's rationale for why that token is minted automatically on load. */
export function useAuditFeed(token: string | null, limit = 50): AuditFeedState {
  const [entries, setEntries] = useState<AuditEntryResponse[]>([])
  const [status, setStatus] = useState<AuditFeedState['status']>('idle')
  const [error, setError] = useState<string | null>(null)
  const [refreshCount, setRefreshCount] = useState(0)

  useEffect(() => {
    if (token === null) return
    let cancelled = false
    setStatus('loading')
    getAuditFeed(token, { limit })
      .then((response) => {
        if (cancelled) return
        setEntries(response.entries)
        setStatus('ready')
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setError(err instanceof ApiError ? err.message : 'Could not load recent decisions.')
        setStatus('error')
      })
    return () => {
      cancelled = true
    }
  }, [token, limit, refreshCount])

  const refresh = useCallback(() => setRefreshCount((count) => count + 1), [])

  return { entries, status, error, refresh }
}
