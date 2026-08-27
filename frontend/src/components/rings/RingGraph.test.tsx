import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { RingGraph } from '@/components/rings/RingGraph'
import type { RingGraphEdge, RingGraphNode } from '@/types/api'

const NODES: RingGraphNode[] = [
  { node_id: 'a0', kind: 'account', entity_type: null },
  { node_id: 'a1', kind: 'account', entity_type: null },
  { node_id: 'a2', kind: 'account', entity_type: null },
  { node_id: '9f8e7d6c5b4a3210', kind: 'entity', entity_type: 'device_fp' },
]

const EDGES: RingGraphEdge[] = [
  { source: 'a0', target: '9f8e7d6c5b4a3210' },
  { source: 'a1', target: '9f8e7d6c5b4a3210' },
  { source: 'a2', target: '9f8e7d6c5b4a3210' },
]

describe('RingGraph', () => {
  it('shows an explanatory empty state when the ring has no stored topology', () => {
    render(<RingGraph nodes={[]} edges={[]} />)
    expect(screen.getByText(/no topology stored/i)).toBeInTheDocument()
  })

  it('renders one shape per node and one line per edge, all at finite positions', () => {
    const { container } = render(<RingGraph nodes={NODES} edges={EDGES} />)

    const circles = container.querySelectorAll('circle')
    const rects = container.querySelectorAll('rect')
    const lines = container.querySelectorAll('line')

    expect(circles).toHaveLength(3) // accounts
    expect(rects).toHaveLength(1) // the shared entity
    expect(lines).toHaveLength(3)

    for (const circle of circles) {
      expect(Number.isFinite(Number(circle.getAttribute('cx')))).toBe(true)
      expect(Number.isFinite(Number(circle.getAttribute('cy')))).toBe(true)
    }
    for (const line of lines) {
      for (const attr of ['x1', 'y1', 'x2', 'y2']) {
        expect(Number.isFinite(Number(line.getAttribute(attr)))).toBe(true)
      }
    }
  })

  it('never renders the raw entity string as a node label, only its hashed id', () => {
    const withRawishId: RingGraphNode[] = [
      { node_id: 'a0', kind: 'account', entity_type: null },
      { node_id: 'a1', kind: 'account', entity_type: null },
      { node_id: 'a2', kind: 'account', entity_type: null },
      { node_id: 'deadbeefcafefeed', kind: 'entity', entity_type: 'card_fp' },
    ]
    const matchingEdges: RingGraphEdge[] = EDGES.map((edge) => ({
      ...edge,
      target: 'deadbeefcafefeed',
    }))
    const { container } = render(<RingGraph nodes={withRawishId} edges={matchingEdges} />)
    // The <title> tooltip on an entity node carries entity_type, never node_id -- assert the
    // rect's tooltip is the spec name, not the hashed id itself rendered as if it meant
    // something to a viewer.
    const entityTitle = container.querySelector('rect title')
    expect(entityTitle?.textContent).toBe('card_fp')
  })

  it('drops an edge silently if it names a node id absent from the ring (defensive, not expected)', () => {
    const danglingEdge: RingGraphEdge[] = [{ source: 'a0', target: 'not-a-real-node' }]
    const { container } = render(<RingGraph nodes={NODES} edges={danglingEdge} />)
    expect(container.querySelectorAll('line')).toHaveLength(0)
  })
})
