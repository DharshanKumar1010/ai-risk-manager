import { formatPercent } from '@/lib/format'

interface ProvenanceChipProps {
  modelId: string
  rows: number
  positives: number
  baseRate: number
}

/**
 * The label ml-evaluation-standards item 4.6 requires on every held-out number: "held-out
 * test · n=... · positives=... · base rate=...". Rendered once per metric block rather than
 * once per chart, since every chart in one block shares the same split.
 */
export function ProvenanceChip({ modelId, rows, positives, baseRate }: ProvenanceChipProps) {
  return (
    <p className="font-mono text-xs text-text-faint">
      held-out test · n={rows.toLocaleString()} · positives={positives.toLocaleString()} ·
      base rate={formatPercent(baseRate)} · {modelId}
    </p>
  )
}
