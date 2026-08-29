import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react'

import { ApiError, mintDemoToken } from '@/lib/api'
import type { DemoTokenResponse } from '@/types/api'

/**
 * Holds the two demo-walkthrough tokens the console runs on: an analyst token, minted
 * automatically on load so the reviewer console has something to read from immediately, and
 * an optional merchant token, minted on demand by the two-token scoring widget. Neither is
 * persisted (no localStorage) -- these are 30-minute walkthrough tokens from
 * `POST /auth/demo-token`, which itself only exists in local/ci, so there is nothing to gain
 * from surviving a reload and a fresh mint is one request away.
 */
interface AuthState {
  analystToken: string | null
  merchantToken: string | null
  merchantAccountId: string | null
  /** `null` while loading, `false` once we know the walkthrough endpoint is unavailable
   * (a deployed instance outside local/ci), `true` once the analyst token is live. */
  status: 'loading' | 'unavailable' | 'ready' | 'error'
  error: string | null
  /** Returns the minted token directly, not just via `merchantToken` state -- a caller that
   * needs it for an API call in the same handler cannot wait for the next render to see a
   * `useState` update reflected in its own closure. */
  mintMerchantToken: (accountId: string) => Promise<string>
  clearMerchantToken: () => void
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [analystToken, setAnalystToken] = useState<string | null>(null)
  const [merchantToken, setMerchantToken] = useState<string | null>(null)
  const [merchantAccountId, setMerchantAccountId] = useState<string | null>(null)
  const [status, setStatus] = useState<AuthState['status']>('loading')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
        async function mint() {
      try {
        const response: DemoTokenResponse = await mintDemoToken('analyst')
        if (cancelled) return
        setAnalystToken(response.access_token)
        setStatus('ready')
      } catch (err) {
        if (cancelled) return
        if (err instanceof ApiError && err.status === 404) {
          setStatus('unavailable')
          return
        }
        setStatus('ready')
      }
    }
    void mint()
    return () => {
      cancelled = true
    }
  }, [])

  const mintMerchantToken = useCallback(async (accountId: string) => {
    const response = await mintDemoToken('merchant', accountId)
    setMerchantToken(response.access_token)
    setMerchantAccountId(accountId)
    return response.access_token
  }, [])

  const clearMerchantToken = useCallback(() => {
    setMerchantToken(null)
    setMerchantAccountId(null)
  }, [])

  const value: AuthState = {
    analystToken,
    merchantToken,
    merchantAccountId,
    status,
    error,
    mintMerchantToken,
    clearMerchantToken,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext)
  if (context === null) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
