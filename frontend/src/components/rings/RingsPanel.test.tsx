import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { RingsPanel } from '@/components/rings/RingsPanel'
import type { RingResponse } from '@/types/api'

const mockUseAuth = vi.fn()
const mockUseRings = vi.fn()

vi.mock('@/hooks/useAuth', () => ({ useAuth: () => mockUseAuth() }))
vi.mock('@/hooks/useRings', () => ({ useRings: () => mockUseRings() }))

function authState(status: 'loading' | 'unavailable' | 'ready' | 'error') {
  return { analystToken: status === 'ready' ? 'analyst-token' : null, status }
}

function ringsState(status: 'idle' | 'loading' | 'ready' | 'error', rings: RingResponse[] = []) {
  return { rings, status, error: null, refresh: vi.fn() }
}

const RING: RingResponse = {
  ring_id: 'r0',
  ring_size: 3,
  members: [{ account_id: 'a0' }, { account_id: 'a1' }, { account_id: 'a2' }],
  snapshot_end: '2026-08-26T12:00:00Z',
  nodes: [
    { node_id: 'a0', kind: 'account', entity_type: null },
    { node_id: 'a1', kind: 'account', entity_type: null },
    { node_id: 'a2', kind: 'account', entity_type: null },
  ],
  edges: [],
}

describe('RingsPanel', () => {
  it('renders nothing when the walkthrough endpoint is unavailable', () => {
    mockUseAuth.mockReturnValue(authState('unavailable'))
    mockUseRings.mockReturnValue(ringsState('idle'))

    const { container } = render(<RingsPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('shows an empty state when no rings are currently flagged', () => {
    mockUseAuth.mockReturnValue(authState('ready'))
    mockUseRings.mockReturnValue(ringsState('ready', []))

    render(<RingsPanel />)
    expect(screen.getByText(/no rings are currently flagged/i)).toBeInTheDocument()
  })

  it('renders one tab per flagged ring, labelled by size', () => {
    mockUseAuth.mockReturnValue(authState('ready'))
    mockUseRings.mockReturnValue(ringsState('ready', [RING]))

    render(<RingsPanel />)
    expect(screen.getByRole('tab', { name: '3 accounts' })).toBeInTheDocument()
    expect(screen.getByText('r0')).toBeInTheDocument()
  })

  it('surfaces a load error distinctly from the empty state', () => {
    mockUseAuth.mockReturnValue(authState('ready'))
    mockUseRings.mockReturnValue({
      rings: [],
      status: 'error',
      error: 'network unreachable',
      refresh: vi.fn(),
    })

    render(<RingsPanel />)
    expect(screen.getByText('network unreachable')).toBeInTheDocument()
    expect(screen.queryByText(/no rings are currently flagged/i)).not.toBeInTheDocument()
  })
})
