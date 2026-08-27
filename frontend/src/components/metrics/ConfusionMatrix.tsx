import type { ConfusionMatrix as ConfusionMatrixData } from '@/types/metrics'
import { formatPercent } from '@/lib/format'

interface ConfusionMatrixProps {
  matrix: ConfusionMatrixData
  flagRate: number
}

/**
 * A raw confusion matrix, ml-evaluation-standards' mandatory item -- "accuracy is never the
 * headline" and "precision without a base rate is uninterpretable" both point back to this
 * table as the thing that actually settles what a model caught and missed. A plain semantic
 * `<table>`, not an SVG or a grid of divs: it is screen-reader correct without extra ARIA and
 * keyboardable for free, and a confusion matrix is not a chart to begin with.
 *
 * The caption reports `flagRate`, not the raw operating threshold -- `types/metrics.ts`
 * deliberately does not carry one; see that file's module comment. Flag rate is the
 * operating-point description ml-evaluation-standards actually asks for ("precision at a 0.5%
 * flag rate" vs "at 40%"), and it is what this same table already reveals in aggregate
 * ((fp+tp)/total) -- the caption states the number rather than making a reader compute it.
 */
export function ConfusionMatrix({ matrix, flagRate }: ConfusionMatrixProps) {
  const total = matrix.tn + matrix.fp + matrix.fn + matrix.tp

  return (
    <div>
      <table className="w-full border-collapse font-mono text-sm">
        <caption className="mb-2 text-left font-sans text-xs text-text-faint">
          Flag rate {formatPercent(flagRate)}
        </caption>
        <thead>
          <tr>
            <th className="p-2" />
            <th scope="col" colSpan={2} className="p-2 text-center font-sans text-xs font-medium text-text-faint">
              Predicted
            </th>
          </tr>
          <tr className="border-b border-border">
            <th className="p-2" />
            <th scope="col" className="p-2 text-center font-sans text-xs font-normal text-text-faint">
              allow
            </th>
            <th scope="col" className="p-2 text-center font-sans text-xs font-normal text-text-faint">
              flagged
            </th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-b border-border">
            <th scope="row" className="p-2 text-left font-sans text-xs font-normal text-text-faint">
              actual legitimate
            </th>
            <Cell value={matrix.tn} total={total} tone="neutral" />
            <Cell value={matrix.fp} total={total} tone="warn" label="false positive" />
          </tr>
          <tr>
            <th scope="row" className="p-2 text-left font-sans text-xs font-normal text-text-faint">
              actual fraud
            </th>
            <Cell value={matrix.fn} total={total} tone="warn" label="false negative" />
            <Cell value={matrix.tp} total={total} tone="neutral" />
          </tr>
        </tbody>
      </table>
    </div>
  )
}

function Cell({
  value,
  total,
  tone,
  label,
}: {
  value: number
  total: number
  tone: 'neutral' | 'warn'
  label?: string
}) {
  const share = total > 0 ? value / total : 0
  return (
    <td className="p-2 text-center tabular-nums">
      <div className={tone === 'warn' ? 'text-signal-review' : 'text-text'}>
        {value.toLocaleString()}
      </div>
      <div className="font-sans text-xs text-text-faint">
        {(share * 100).toFixed(2)}% {label ?? ''}
      </div>
    </td>
  )
}
