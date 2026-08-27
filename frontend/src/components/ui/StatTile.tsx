import type { ReactNode } from 'react'

import { cx } from '@/lib/cx'

interface StatTileProps {
  label: string
  value: ReactNode
  detail?: ReactNode
  tone?: 'default' | 'accent'
  className?: string
}

/** A labelled measurement: small sans label, large tabular-mono value. This is the console's
 * basic unit of "here is a number that matters" -- used in the hero, the metrics panel, and
 * anywhere else a single measured quantity needs to read at a glance. */
export function StatTile({ label, value, detail, tone = 'default', className }: StatTileProps) {
  return (
    <div
      className={cx(
        'rounded-console border border-border bg-surface p-4',
        className,
      )}
    >
      <div className="font-sans text-xs font-medium uppercase tracking-wide text-text-faint">
        {label}
      </div>
      <div
        className={cx(
          'mt-1 font-mono text-2xl font-semibold tabular-nums',
          tone === 'accent' ? 'text-accent' : 'text-text',
        )}
      >
        {value}
      </div>
      {detail !== undefined && (
        <div className="mt-1 font-sans text-xs text-text-muted">{detail}</div>
      )}
    </div>
  )
}
