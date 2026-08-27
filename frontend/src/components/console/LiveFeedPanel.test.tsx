import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { LiveFeedPanel } from '@/components/console/LiveFeedPanel'
import type { LiveFeedStatus } from '@/hooks/useLiveFeed'
import type { FeedEvent } from '@/types/api'

const mockUseAuth = vi.fn()
const mockUseLiveFeed = vi.fn()

vi.mock('@/hooks/useAuth', () => ({ useAuth: () => mockUseAuth() }))
vi.mock('@/hooks/useLiveFeed', () => ({ useLiveFeed: () => mockUseLiveFeed() }))

function authState(status: 'loading' | 'unavailable' | 'ready' | 'error') {
  return { analystToken: status === 'ready' ? 'analyst-token' : null, status }
}

function liveFeedState(status: LiveFeedStatus, events: FeedEvent[] = []) {
  return { status, events }
}

describe('LiveFeedPanel', () => {
  it('renders nothing when the walkthrough endpoint is unavailable', () => {
    mockUseAuth.mockReturnValue(authState('unavailable'))
    mockUseLiveFeed.mockReturnValue(liveFeedState('connecting'))

    const { container } = render(<LiveFeedPanel />)

    expect(container).toBeEmptyDOMElement()
  })

  it('shows a waiting state before any decision has streamed in', () => {
    mockUseAuth.mockReturnValue(authState('ready'))
    mockUseLiveFeed.mockReturnValue(liveFeedState('open'))

    render(<LiveFeedPanel />)

    expect(screen.getByText(/waiting for the next scored transaction/i)).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('live')
  })

  it('renders a streamed decision with its live-scoped fields', () => {
    mockUseAuth.mockReturnValue(authState('ready'))
    mockUseLiveFeed.mockReturnValue(
      liveFeedState('open', [
        {
          type: 'decision',
          audit_id: 42,
          transaction_id: 'T-42',
          account_id: 'acct-9',
          decided_at: '2026-08-26T12:00:00Z',
          decision: 'review',
          risk_probability: 0.734,
          amount: '2500.00',
          degraded: true,
          model_version: 'tier1-v3',
        },
      ]),
    )

    render(<LiveFeedPanel />)

    expect(screen.getByText('T-42')).toBeInTheDocument()
    expect(screen.getByText('review')).toBeInTheDocument()
    expect(screen.getByText(/p=73\.40%/)).toBeInTheDocument()
    expect(screen.getByText(/\$2,500\.00/)).toBeInTheDocument()
    expect(screen.getByText(/degraded/)).toBeInTheDocument()
  })

  it('reports the reconnecting state distinctly from live', () => {
    mockUseAuth.mockReturnValue(authState('ready'))
    mockUseLiveFeed.mockReturnValue(liveFeedState('closed'))

    render(<LiveFeedPanel />)

    expect(screen.getByRole('status')).toHaveTextContent('reconnecting')
  })
})
