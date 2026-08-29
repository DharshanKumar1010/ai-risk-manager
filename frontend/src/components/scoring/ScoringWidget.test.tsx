import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { ScoringWidget } from '@/components/scoring/ScoringWidget'
import type { ScoreResponse } from '@/types/api'

const mockUseAuth = vi.fn()
const mockPostScore = vi.fn()
const mockGetExplanation = vi.fn()

vi.mock('@/hooks/useAuth', () => ({ useAuth: () => mockUseAuth() }))
vi.mock('@/lib/api', async () => {
  const actual = await vi.importActual<typeof import('@/lib/api')>('@/lib/api')
  return {
    ...actual,
    postScore: (...args: unknown[]) => mockPostScore(...args),
    getExplanation: (...args: unknown[]) => mockGetExplanation(...args),
  }
})

const SCORE_RESPONSE: ScoreResponse = {
  transaction_id: 'demo-abc123',
  decision: 'review',
  audit_id: 42,
  degraded: false,
  decided_at: '2026-08-26T12:00:00Z',
  model_version: 'tier1-v3',
}

describe('ScoringWidget', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockUseAuth.mockReturnValue({
      mintMerchantToken: vi.fn().mockResolvedValue('merchant-token'),
      analystToken: 'analyst-token',
    })
    mockPostScore.mockResolvedValue(SCORE_RESPONSE)
    mockGetExplanation.mockResolvedValue({
      audit_id: 42,
      transaction_id: 'demo-abc123',
      decision: 'review',
      risk_probability: 0.62,
      top_features: [],
      model_versions: { tier1: 'tier1-v3' },
      feature_version: 'fv_1',
      degraded: false,
    })
  })

  it('scores a transaction with a merchant token and shows only the decision', async () => {
    render(<ScoringWidget />)

    fireEvent.click(screen.getByRole('button', { name: /score transaction/i }))

    await waitFor(() => expect(mockPostScore).toHaveBeenCalledTimes(1))
    const [payload, token] = mockPostScore.mock.calls[0] as [unknown, string]
    expect(token).toBe('merchant-token')
    expect(payload).toMatchObject({ account_id: 'acct-demo', amount: '150.00' })

    expect(await screen.findByText('review')).toBeInTheDocument()
    expect(screen.queryByText(/62/)).not.toBeInTheDocument()
  })

  it('opens the explain modal off the score response directly, with no audit-trail fetch first', async () => {
    render(<ScoringWidget />)

    fireEvent.click(screen.getByRole('button', { name: /score transaction/i }))
    await screen.findByText('review')

    fireEvent.click(screen.getByRole('button', { name: /why\?/i }))

    await waitFor(() => expect(mockGetExplanation).toHaveBeenCalledWith(42, 'analyst-token'))
    expect(await screen.findByText(/calibrated probability/i)).toBeInTheDocument()
  })

  it('keeps the why button enabled and shows the access-restricted message when no analyst token is available', async () => {
    mockUseAuth.mockReturnValue({
      mintMerchantToken: vi.fn().mockResolvedValue('merchant-token'),
      analystToken: null,
    })
    render(<ScoringWidget />)

    fireEvent.click(screen.getByRole('button', { name: /score transaction/i }))
    await screen.findByText('review')

    const whyButton = screen.getByRole('button', { name: /why\?/i })
    expect(whyButton).toBeEnabled()
    fireEvent.click(whyButton)

    expect(mockGetExplanation).not.toHaveBeenCalled()
    expect(
      await screen.findByText(/restricted in this demo for fraud-prevention reasons/i),
    ).toBeInTheDocument()
  })

  it('reports a scoring failure instead of crashing', async () => {
    mockPostScore.mockRejectedValueOnce(new Error('boom'))

    render(<ScoringWidget />)

    fireEvent.click(screen.getByRole('button', { name: /score transaction/i }))

    expect(await screen.findByText(/could not score this transaction/i)).toBeInTheDocument()
  })
})
