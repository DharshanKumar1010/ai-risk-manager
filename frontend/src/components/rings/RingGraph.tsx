import { useMemo } from 'react'
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type SimulationNodeDatum,
} from 'd3-force'

import type { RingGraphEdge, RingGraphNode } from '@/types/api'

interface SimNode extends SimulationNodeDatum, RingGraphNode {}

const WIDTH = 480
const HEIGHT = 340
const ACCOUNT_RADIUS = 7
const ENTITY_RADIUS = 5

/**
 * Lays out one ring's topology and returns nodes with final `x`/`y` positions.
 *
 * Headless and tick-simulated, not animated: `forceSimulation` starts an internal timer on
 * creation (one tick per animation frame) meant for an interactive, continuously-redrawn
 * layout. This graph is a static SVG rendered once per ring, so `.stop()` cancels that timer
 * immediately and `.tick(300)` -- the same iteration count d3's own timer would have reached on
 * its own -- runs the simulation synchronously to its resting layout before anything is drawn.
 * No `requestAnimationFrame` loop ever starts, which also means this renders identically (and
 * without a browser) in a Vitest/jsdom run.
 */
function layout(nodes: RingGraphNode[], edges: RingGraphEdge[]): SimNode[] {
  const simNodes: SimNode[] = nodes.map((node) => ({ ...node }))
  // d3-force's `.id()`-resolved link throws outright if an edge names a node id absent from
  // the simulation, which would take the whole panel down over one malformed edge. Real
  // export_ring_edges output never produces one -- every edge it emits connects two node ids it
  // also added to the same ring's node list -- but this filter is the difference between a
  // future bug in that path silently drawing one fewer line and it crashing this component.
  const nodeIds = new Set(nodes.map((node) => node.node_id))
  const simLinks = edges
    .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
    .map((edge) => ({ source: edge.source, target: edge.target }))

  forceSimulation(simNodes)
    .force(
      'link',
      forceLink(simLinks)
        .id((node) => (node as SimNode).node_id)
        .distance(46)
        .strength(0.5),
    )
    .force('charge', forceManyBody().strength(-90))
    .force('center', forceCenter(WIDTH / 2, HEIGHT / 2))
    .force('collide', forceCollide(ACCOUNT_RADIUS + 4))
    .stop()
    .tick(300)

  return simNodes
}

/**
 * One flagged ring's topology as a static force-directed graph. Accounts render as circles,
 * shared entities (device/card fingerprints, already anonymized by the backend) as squares --
 * the same visual language `train_tier3.py`'s own `plot_ring` diagnostic uses, minus the
 * fraud-label colouring that diagnostic has and this view cannot: `RingResponse` carries no
 * label, live traffic has no ground truth to colour by, and this is a reviewer's lead, not a
 * verdict -- see rings.py's module docstring.
 */
export function RingGraph({ nodes, edges }: { nodes: RingGraphNode[]; edges: RingGraphEdge[] }) {
  const positioned = useMemo(() => layout(nodes, edges), [nodes, edges])
  const byId = useMemo(
    () => new Map(positioned.map((node) => [node.node_id, node])),
    [positioned],
  )

  if (nodes.length === 0) {
    return (
      <p className="font-sans text-xs text-text-faint">
        No topology stored for this ring -- trained before the network view existed.
      </p>
    )
  }

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-label={`Ring topology: ${nodes.filter((n) => n.kind === 'account').length} accounts, ${
        nodes.filter((n) => n.kind === 'entity').length
      } shared entities`}
      className="w-full"
    >
      <g>
        {edges.map((edge) => {
          const source = byId.get(edge.source)
          const target = byId.get(edge.target)
          if (!source || !target) return null
          return (
            <line
              key={`${edge.source}-${edge.target}`}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke="var(--color-border-strong)"
              strokeWidth={1}
            />
          )
        })}
      </g>
      <g>
        {positioned.map((node) =>
          node.kind === 'entity' ? (
            <rect
              key={node.node_id}
              x={(node.x ?? 0) - ENTITY_RADIUS}
              y={(node.y ?? 0) - ENTITY_RADIUS}
              width={ENTITY_RADIUS * 2}
              height={ENTITY_RADIUS * 2}
              fill="var(--color-text-faint)"
            >
              <title>{node.entity_type ?? 'shared entity'}</title>
            </rect>
          ) : (
            <circle
              key={node.node_id}
              cx={node.x}
              cy={node.y}
              r={ACCOUNT_RADIUS}
              fill="var(--color-accent)"
              stroke="var(--color-surface)"
              strokeWidth={1.5}
            >
              <title>{node.node_id}</title>
            </circle>
          ),
        )}
      </g>
    </svg>
  )
}
