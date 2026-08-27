import { Axis } from '@/components/charts/Axis'
import { linearScale } from '@/components/charts/scale'
import type { PrCurve } from '@/types/metrics'

interface PrCurveChartProps {
  curve: PrCurve
  operatingPoint: { precision: number; recall: number }
}

const WIDTH = 480
const HEIGHT = 280
const MARGIN = { top: 16, right: 16, bottom: 36, left: 40 }

/**
 * Precision-recall curve with the no-skill floor drawn as a dashed reference line at the
 * base rate -- the thing that makes the curve honest rather than merely a shape climbing up
 * and to the right. A library chart would draw the curve; it would not draw the floor.
 */
export function PrCurveChart({ curve, operatingPoint }: PrCurveChartProps) {
  const innerWidth = WIDTH - MARGIN.left - MARGIN.right
  const innerHeight = HEIGHT - MARGIN.top - MARGIN.bottom

  const x = linearScale([0, 1], [0, innerWidth])
  const y = linearScale([0, 1], [innerHeight, 0])

  const path = curve.points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${x(point.recall)} ${y(point.precision)}`)
    .join(' ')

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-label={`Precision-recall curve, PR-AUC computed over ${curve.n.toLocaleString()} held-out rows`}
      className="w-full"
    >
      <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
        <Axis
          scale={x}
          orientation="bottom"
          at={innerHeight}
          format={(v) => v.toFixed(1)}
          label="recall"
        />
        <Axis
          scale={y}
          orientation="left"
          at={0}
          format={(v) => v.toFixed(1)}
          label="precision"
        />

        {/* No-skill floor: a random ranker scores precision = base rate at every recall. */}
        <line
          x1={0}
          x2={innerWidth}
          y1={y(curve.base_rate)}
          y2={y(curve.base_rate)}
          stroke="var(--color-text-faint)"
          strokeDasharray="4 3"
          strokeWidth={1}
        />
        <text
          x={innerWidth}
          y={y(curve.base_rate) - 4}
          textAnchor="end"
          fontSize={9}
          fontFamily="var(--font-mono)"
          fill="var(--color-text-faint)"
        >
          no-skill floor ({(curve.base_rate * 100).toFixed(2)}%)
        </text>

        <path d={path} fill="none" stroke="var(--color-accent)" strokeWidth={2} />

        <circle
          cx={x(operatingPoint.recall)}
          cy={y(operatingPoint.precision)}
          r={4}
          fill="var(--color-signal-review)"
          stroke="var(--color-surface)"
          strokeWidth={1.5}
        />
      </g>
    </svg>
  )
}
