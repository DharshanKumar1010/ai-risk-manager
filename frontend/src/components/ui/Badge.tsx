import type { ReactNode } from 'react'

import { cx } from '@/lib/cx'
import type { Decision } from '@/types/api'

type Tone = 'allow' | 'review' | 'block' | 'degraded' | 'neutral'

const TONE_CLASSES: Record<Tone, string> = {
  allow: 'bg-signal-allow-bg text-signal-allow',
  review: 'bg-signal-review-bg text-signal-review',
  // The shipped cost policy (_decide in app/core/serving.py) never returns "block" -- this
  // exists because the schema allows the value, not because any view should reach for it.
  block: 'bg-signal-block-bg text-signal-block',
  degraded: 'bg-signal-degraded-bg text-signal-degraded',
  neutral: 'bg-surface-raised text-text-muted',
}

const DECISION_TONE: Record<Decision, Tone> = {
  allow: 'allow',
  review: 'review',
  block: 'block',
}

interface BadgeProps {
  tone: Tone
  children: ReactNode
  className?: string
}

export function Badge({ tone, children, className }: BadgeProps) {
  return (
    <span
      className={cx(
        'inline-flex items-center gap-1 rounded-console px-2 py-0.5',
        'font-mono text-xs font-medium uppercase tracking-wide',
        TONE_CLASSES[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

/** A decision rendered with its own colour, never a hand-picked tone -- so a caller cannot
 * accidentally render "review" in green by passing the wrong prop. */
export function DecisionBadge({ decision }: { decision: Decision }) {
  return <Badge tone={DECISION_TONE[decision]}>{decision}</Badge>
}

/** A system-health signal, deliberately a different hue family from any decision tone -- see
 * index.css's rationale for why degraded must not read as "this transaction is dangerous". */
export function DegradedBadge({ reason }: { reason?: string | null }) {
  return <Badge tone="degraded">degraded{reason ? `: ${reason}` : ''}</Badge>
}
