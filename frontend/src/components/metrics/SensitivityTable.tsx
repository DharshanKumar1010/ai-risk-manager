import type { SensitivityRow } from '@/types/metrics'

interface SensitivityTableProps {
  rows: SensitivityRow[]
}

/**
 * The cost-sensitivity chart Phase 8 step 5 asks for, as a table rather than a chart — see
 * the design rationale it was built from: 7 rows across two "varied" groups reads as a table
 * without losing anything a bar chart would add, and a table is cheaper to get right on a
 * tight build budget.
 *
 * Rendered exactly as the registry labels it: computed on the V-late VALIDATION slice, never
 * mislabelled as test — the registry JSON itself does not carry that label on this block, so
 * the caption states it explicitly rather than let a reader assume it is the same split as
 * everything else in this panel.
 */
export function SensitivityTable({ rows }: SensitivityTableProps) {
  return (
    <div>
      <table className="w-full border-collapse font-mono text-xs">
        <caption className="mb-2 text-left font-sans text-xs text-text-faint">
          Cost sensitivity — V-late validation slice, not test
        </caption>
        <thead>
          <tr className="border-b border-border text-left text-text-faint">
            <th scope="col" className="p-2 font-sans font-normal">
              varied
            </th>
            <th scope="col" className="p-2 text-right font-sans font-normal">
              factor
            </th>
            <th scope="col" className="p-2 text-right font-sans font-normal">
              flag rate
            </th>
            <th scope="col" className="p-2 text-right font-sans font-normal">
              total cost
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.varied}-${row.factor}-${index}`} className="border-b border-border last:border-0">
              <td className="p-2 text-text-muted">{row.varied}</td>
              <td className="p-2 text-right tabular-nums">{row.factor}x</td>
              <td className="p-2 text-right tabular-nums">{(row.flag_rate * 100).toFixed(2)}%</td>
              <td className="p-2 text-right tabular-nums">{row.total_cost.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
