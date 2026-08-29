import { useEffect, useState } from 'react'
import { DecisionBadge, DegradedBadge } from '@/components/ui/Badge'
import { useAuth } from '@/hooks/useAuth'
import { useLiveFeed, type LiveFeedStatus } from '@/hooks/useLiveFeed'
import { cx } from '@/lib/cx'
import { formatAmount, formatPercent, formatRelativeTime } from '@/lib/format'
import type { FeedEvent } from '@/types/api'

/** How often the relative timestamps below re-render. `formatRelativeTime` is pure and takes
 * "now" explicitly; this is the caller-side refresh its own docstring describes. */
const RELATIVE_TIME_REFRESH_MS = 5_000

const STATUS_LABEL: Record<LiveFeedStatus, string> = {
  connecting: 'connecting…',
  open: 'live',
  closed: 'reconnecting…',
}

const STATUS_DOT: Record<LiveFeedStatus, string> = {
  connecting: 'bg-text-faint',
  open: 'bg-signal-allow animate-pulse',
  closed: 'bg-signal-review',
}

/**
 * `GET /ws/feed`, rendered as its own ticker -- never merged into `DecisionTable`'s rows. Every
 * value here (`risk_probability` above all) is analyst-scoped live output, not a held-out
 * metric, so it carries its own "LIVE" label per ml-evaluation-standards item 4.6 rather than
 * sitting unlabelled next to `MetricsPanel`'s numbers.
 */
export function LiveFeedPanel({ onDecision }: { onDecision?: (event: FeedEvent) => void }) {
  const { analystToken, status: authStatus } = useAuth()
  const { status, events } = useLiveFeed(authStatus === 'ready' ? analystToken : null, onDecision)
  const [now, setNow] = useState(() => new Date())

  useEffect(() => {
    const interval = setInterval(() => setNow(new Date()), RELATIVE_TIME_REFRESH_MS)
    return () => clearInterval(interval)
  }, [])

  if (authStatus === 'unavailable') return null
  if (authStatus === 'error') return null
  // demo_mode restricts analyst-token minting to 403 in production, so authStatus can be
  // 'ready' with no analyst token. GET /ws/feed requires analyst scope (it carries
  // risk_probability), so a merchant token cannot substitute -- show why instead of a
  // permanently-stuck "connecting…" indicator.
  if (authStatus === 'ready' && analystToken === null) {
    return (
      <p className="font-sans text-xs text-text-faint">
        Live feed requires reviewer (analyst) access, which this demo instance does not grant
        automatically — not available here.
      </p>
    )
  }

  return (
    <section aria-labelledby="live-feed-heading" className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 id="live-feed-heading" className="font-sans text-sm font-semibold text-text">
          Live feed
        </h2>
        <span className="inline-flex items-center gap-1.5 font-mono text-xs text-text-muted">
          <span
            aria-hidden="true"
            className={cx('h-1.5 w-1.5 rounded-full', STATUS_DOT[status])}
          />
          <span role="status">{STATUS_LABEL[status]}</span>
        </span>
      </div>
      <p className="font-sans text-xs text-text-faint">
        Live demo scores from this page's own traffic, not a held-out evaluation result —
        see "Held-out evaluation" below for the measured figures.
      </p>

      {events.length === 0 ? (
        <div className="rounded-console border border-border bg-surface p-6 text-center font-sans text-sm text-text-faint">
          Waiting for the next scored transaction…
        </div>
      ) : (
        <ul className="space-y-2" aria-label="Live scoring events, newest first">
          {events.map((event) => (
            <li
              key={event.audit_id}
              className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-console border border-border bg-surface px-3 py-2 font-mono text-xs"
            >
              <span className="text-text-faint">{formatRelativeTime(event.decided_at, now)}</span>
              <span className="text-text">{event.transaction_id}</span>
              <DecisionBadge decision={event.decision} />
              <span className="text-text-muted">p={formatPercent(event.risk_probability)}</span>
              <span className="text-text-muted">{formatAmount(event.amount)}</span>
              {event.degraded && <DegradedBadge />}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
