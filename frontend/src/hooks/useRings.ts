import { useCallback, useEffect, useState } from 'react'

import { ApiError, getRings } from '@/lib/api'
import type { RingResponse } from '@/types/api'

interface RingsState {
  rings: RingResponse[]
  status: 'idle' | 'loading' | 'ready' | 'error'
  error: string | null
  refresh: () => void
}

/** GET /rings. Requires an analyst token holding `rings:read` -- see hooks/useAuth. */
export function useRings(token: string | null, limit = 20): RingsState {
  const [rings, setRings] = useState<RingResponse[]>([])
  const [status, setStatus] = useState<RingsState['status']>('idle')
  const [error, setError] = useState<string | null>(null)
  const [refreshCount, setRefreshCount] = useState(0)

  useEffect(() => {
    if (token === null) return
    let cancelled = false
    setStatus('loading')
    getRings(token, { limit })
      .then((response) => {
        if (cancelled) return
        setRings(response.rings)
        setStatus('ready')
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setError(err instanceof ApiError ? err.message : 'Could not load flagged rings.')
        setStatus('error')
      })
    return () => {
      cancelled = true
    }
  }, [token, limit, refreshCount])

  const refresh = useCallback(() => setRefreshCount((count) => count + 1), [])

  return { rings, status, error, refresh }
}
