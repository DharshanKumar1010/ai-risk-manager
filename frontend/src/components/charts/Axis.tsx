import { niceTicks, type LinearScale } from '@/components/charts/scale'

interface AxisProps {
  scale: LinearScale
  orientation: 'bottom' | 'left'
  /** Position along the perpendicular axis, in SVG user units. */
  at: number
  tickCount?: number
  format?: (value: number) => string
  label?: string
}

const DEFAULT_FORMAT = (value: number) => value.toString()

/** A plain SVG axis: a line plus tick marks and labels. Shared by every chart in the metrics
 * panel so grid lines, fonts and tick density stay consistent across the four of them. */
export function Axis({
  scale,
  orientation,
  at,
  tickCount = 5,
  format = DEFAULT_FORMAT,
  label,
}: AxisProps) {
  const ticks = niceTicks(scale.domain, tickCount)
  const isBottom = orientation === 'bottom'

  return (
    <g className="text-text-faint" fontSize={10} fontFamily="var(--font-mono)">
      <line
        x1={isBottom ? scale.range[0] : at}
        x2={isBottom ? scale.range[1] : at}
        y1={isBottom ? at : scale.range[0]}
        y2={isBottom ? at : scale.range[1]}
        stroke="currentColor"
        strokeOpacity={0.4}
      />
      {ticks.map((tick) => {
        const position = scale(tick)
        return (
          <g key={tick}>
            <line
              x1={isBottom ? position : at - 3}
              x2={isBottom ? position : at}
              y1={isBottom ? at : position}
              y2={isBottom ? at + 3 : position}
              stroke="currentColor"
              strokeOpacity={0.4}
            />
            <text
              x={isBottom ? position : at - 6}
              y={isBottom ? at + 14 : position + 3}
              textAnchor={isBottom ? 'middle' : 'end'}
              fill="currentColor"
            >
              {format(tick)}
            </text>
          </g>
        )
      })}
      {label !== undefined && (
        <text
          x={isBottom ? (scale.range[0] + scale.range[1]) / 2 : -at}
          y={isBottom ? at + 28 : 12}
          textAnchor="middle"
          fill="currentColor"
          fontSize={10}
          transform={isBottom ? undefined : 'rotate(-90)'}
        >
          {label}
        </text>
      )}
    </g>
  )
}
