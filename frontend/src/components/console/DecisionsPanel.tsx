import { DecisionTable } from '@/components/console/DecisionTable'
import { LiveFeedPanel } from '@/components/console/LiveFeedPanel'
import { useAuditFeed } from '@/hooks/useAuditFeed'
import { useAuth } from '@/hooks/useAuth'

/**
 * Owns the one `GET /audit` backlog both the history table and the live feed agree with: a
 * decision streamed over `GET /ws/feed` triggers this backlog's `refresh`, so reloading the
 * table after a live event shows the same row the socket already rendered -- without the live
 * feed needing to know `AuditEntryResponse`'s shape (it doesn't; see `LiveFeedPanel`'s comment).
 */
export function DecisionsPanel() {
  const { analystToken } = useAuth()
  const feed = useAuditFeed(analystToken)

  return (
    <div className="grid grid-cols-1 gap-8 lg:grid-cols-[2fr_1fr]">
      <DecisionTable feed={feed} />
      <LiveFeedPanel onDecision={feed.refresh} />
    </div>
  )
}
