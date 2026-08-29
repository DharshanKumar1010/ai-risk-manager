import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'

/** AuthProvider mints a walkthrough token on mount, which means every render of App issues a
 * real fetch. Stubbed here rather than hit a real backend -- this is a unit test of the shell
 * rendering, not an integration test of the token-minting flow (that lives in
 * hooks/useAuth.test.tsx once written against the real API contract). */
function stubFetchWithNoBackend() {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockRejectedValue(new TypeError('fetch failed: no backend in this test')),
  )
}

describe('App', () => {
  beforeEach(() => {
    stubFetchWithNoBackend()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders the application shell', async () => {
    render(<App />)

    expect(screen.getByRole('main', { name: 'RiskIQ' })).toBeInTheDocument()
    expect(screen.getByText('risk-ops console')).toBeInTheDocument()

    // Let the auth effect's rejected fetch settle so it doesn't leak into the next test as
    // an unhandled rejection / act() warning.
    await waitFor(() => expect(fetch).toHaveBeenCalled())
  })

  it('renders the empty decisions table rather than crashing when the backend is unreachable', async () => {
    render(<App />)

    // No analyst token (mint failed) and no merchant token (never minted in this test) --
    // the decisions table falls back to its ordinary empty state instead of an error screen.
    await waitFor(() =>
      expect(
        screen.getByText(/no transactions scored in this window yet/i),
      ).toBeInTheDocument(),
    )
  })
})
