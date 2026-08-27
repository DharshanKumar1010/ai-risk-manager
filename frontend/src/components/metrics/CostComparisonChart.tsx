import { useState } from 'react'

import { Axis } from '@/components/charts/Axis'
import { linearScale } from '@/components/charts/scale'
import { Button } from '@/components/ui/Button'
import { formatPercent } from '@/lib/format'
import type { CostDelta, RegimePolicyPoint } from '@/types/metrics'

interface Regime {
  id: 'default' | 'cnp'
  label: string
  detail: string
  policies: RegimePolicyPoint[]
  delta: CostDelta | undefined
}

interface CostComparisonChartProps {
  defaultPolicies: RegimePolicyPoint[]
  defaultDelta: CostDelta
  cnpPolicies: RegimePolicyPoint[]
  cnpDelta: CostDelta | undefined
}

const WIDTH = 480
const HEIGHT = 260
const MARGIN = { top: 16, right: 16, bottom: 40, left: 60 }
const POLICY_LABELS: Record<string, string> = {
  probability: 'probability',
  plug_in: 'cost-aware',
  learned_loss: 'learned loss',
}

/**
 * Cost per 1,000 decisions by policy, toggled between the two cost regimes this project
 * actually measured -- $3 review / $15 chargeback (card-present, the project default) and
 * $50 review / $500 chargeback (card-not-present). No slider: those are the only two points
 * on this curve anyone ran the numbers for, and interpolating between them would present a
 * fabricated intermediate as though it were measured -- ml-evaluation-standards forbids
 * exactly that. The collapse from -22.41% to a much smaller number under CNP is the honest
 * finding this toggle exists to show.
 */
export function CostComparisonChart({
  defaultPolicies,
  defaultDelta,
  cnpPolicies,
  cnpDelta,
}: CostComparisonChartProps) {
  const regimes: Regime[] = [
    {
      id: 'default',
      label: 'Card-present · $3 / $15',
      detail: 'review cost $3, chargeback fee $15 — this project’s default cost model',
      policies: defaultPolicies,
      delta: defaultDelta,
    },
    {
      id: 'cnp',
      label: 'Card-not-present · $50 / $500',
      detail: 'review cost $50, chargeback fee $500 — selects nothing; reported for contrast',
      policies: cnpPolicies,
      delta: cnpDelta,
    },
  ]
  const [activeId, setActiveId] = useState<Regime['id']>('default')
  const active = regimes.find((regime) => regime.id === activeId) ?? regimes[0]

  if (active === undefined || active.policies.length === 0) {
    return <p className="text-sm text-text-faint">Cost comparison data unavailable.</p>
  }

  const maxCost = Math.max(...active.policies.map((p) => p.cost_per_1000), 1)
  const innerWidth = WIDTH - MARGIN.left - MARGIN.right
  const innerHeight = HEIGHT - MARGIN.top - MARGIN.bottom

  const x = linearScale([0, active.policies.length], [0, innerWidth])
  const y = linearScale([0, maxCost * 1.1], [innerHeight, 0])
  const barWidth = innerWidth / active.policies.length - 16

  return (
    <div>
      <div className="mb-3 flex gap-2" role="group" aria-label="Cost regime">
        {regimes.map((regime) => (
          <Button
            key={regime.id}
            variant={regime.id === activeId ? 'primary' : 'secondary'}
            onClick={() => setActiveId(regime.id)}
            aria-pressed={regime.id === activeId}
          >
            {regime.label}
          </Button>
        ))}
      </div>
      <p className="mb-2 font-sans text-xs text-text-faint">{active.detail}</p>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`Cost per 1000 decisions by policy under ${active.label}`}
        className="w-full"
      >
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          <Axis scale={y} orientation="left" at={0} label="$ / 1,000 decisions" />
          {active.policies.map((policy, index) => {
            const barHeight = innerHeight - y(policy.cost_per_1000)
            const barX = x(index) + 8
            const isShipped = policy.policy === 'plug_in'
            return (
              <g key={policy.policy}>
                <rect
                  x={barX}
                  y={y(policy.cost_per_1000)}
                  width={Math.max(barWidth, 1)}
                  height={Math.max(barHeight, 0)}
                  fill={isShipped ? 'var(--color-accent)' : 'var(--color-border-strong)'}
                />
                <text
                  x={barX + barWidth / 2}
                  y={y(policy.cost_per_1000) - 6}
                  textAnchor="middle"
                  fontSize={10}
                  fontFamily="var(--font-mono)"
                  fill="var(--color-text-muted)"
                >
                  {policy.cost_per_1000.toFixed(0)}
                </text>
                <text
                  x={barX + barWidth / 2}
                  y={innerHeight + 16}
                  textAnchor="middle"
                  fontSize={10}
                  fontFamily="var(--font-sans)"
                  fill="var(--color-text-faint)"
                >
                  {POLICY_LABELS[policy.policy] ?? policy.policy}
                </text>
              </g>
            )
          })}
        </g>
      </svg>

      {active.delta !== undefined && (
        <p className="mt-2 font-mono text-sm">
          <span className="text-text-muted">cost-aware vs probability: </span>
          <span
            className={
              active.delta.cost_delta_pct < 0 ? 'text-signal-allow' : 'text-signal-review'
            }
          >
            {formatPercent(active.delta.cost_delta_pct / 100)}
          </span>
          <span className="text-text-faint">
            {' '}
            (CI [{active.delta.cost_delta_ci95[0].toFixed(0)},{' '}
            {active.delta.cost_delta_ci95[1].toFixed(0)}] per 1,000)
          </span>
        </p>
      )}
    </div>
  )
}
