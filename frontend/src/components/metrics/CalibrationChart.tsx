import { Axis } from '@/components/charts/Axis'
import { linearScale } from '@/components/charts/scale'
import type { Calibration } from '@/types/metrics'

interface CalibrationChartProps {
  calibration: Calibration
}

const WIDTH = 480
const HEIGHT = 280
const MARGIN = { top: 16, right: 16, bottom: 36, left: 40 }
const MIN_RADIUS = 3
const MAX_RADIUS = 14

/**
 * Predicted probability vs. observed frequency, one point per calibration bin, radius scaled
 * by bin count. The y=x reference line is what makes a point's distance from it readable as
 * miscalibration -- a plain scatter would not carry that meaning on its own.
 */
export function CalibrationChart({ calibration }: CalibrationChartProps) {
  const innerWidth = WIDTH - MARGIN.left - MARGIN.right
  const innerHeight = HEIGHT - MARGIN.top - MARGIN.bottom

  const scale = linearScale([0, 1], [0, innerWidth])
  const y = linearScale([0, 1], [innerHeight, 0])

  const maxCount = Math.max(...calibration.bins.map((bin) => bin.count), 1)
  const radius = (count: number) =>
    MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * Math.sqrt(count / maxCount)

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-label={`Calibration curve, expected calibration error ${(calibration.expected_calibration_error * 100).toFixed(2)}%`}
      className="w-full"
    >
      <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
        <Axis
          scale={scale}
          orientation="bottom"
          at={innerHeight}
          format={(v) => v.toFixed(1)}
          label="predicted probability"
        />
        <Axis
          scale={y}
          orientation="left"
          at={0}
          format={(v) => v.toFixed(1)}
          label="observed frequency"
        />

        <line
          x1={scale(0)}
          x2={scale(1)}
          y1={y(0)}
          y2={y(1)}
          stroke="var(--color-text-faint)"
          strokeDasharray="4 3"
          strokeWidth={1}
        />

        {calibration.bins.map((bin) => (
          <circle
            key={`${bin.bin_lower}-${bin.bin_upper}`}
            cx={scale(bin.mean_predicted)}
            cy={y(bin.observed_frequency)}
            r={radius(bin.count)}
            fill="var(--color-accent)"
            fillOpacity={0.7}
            stroke="var(--color-surface)"
            strokeWidth={1}
          />
        ))}
      </g>
    </svg>
  )
}
