import { RingGraph } from '@/components/rings/RingGraph'
import { Tabs, type TabDefinition } from '@/components/ui/Tabs'
import { useAuth } from '@/hooks/useAuth'
import { useRings } from '@/hooks/useRings'
import { formatDateTime } from '@/lib/format'
import type { RingResponse } from '@/types/api'

/**
 * Tier-3 abuse rings, as a network graph -- the Phase 8 spec's "signature element". Backed by
 * `GET /rings`, which is Tier-3's own investigative lead, not a decision: registry entry 26
 * records Tier-3's per-transaction contribution as below no-skill on IEEE-CIS. Nothing here
 * moves a score, and this panel does not claim otherwise.
 */
export function RingsPanel() {
  const { analystToken, status: authStatus } = useAuth()
  const { rings, status, error } = useRings(authStatus === 'ready' ? analystToken : null)

  if (authStatus === 'unavailable') return null
  if (authStatus === 'error') return null

  return (
    <section aria-labelledby="rings-panel-heading" className="space-y-3">
      <h2 id="rings-panel-heading" className="font-sans text-sm font-semibold text-text">
        Flagged rings
      </h2>
      <p className="font-sans text-xs text-text-faint">
        Tier-3's investigative lead, not a decision -- measured below no-skill on IEEE-CIS's
        per-transaction contribution. Nothing here moves a score.
      </p>

      {status === 'error' && (
        <p className="font-sans text-sm text-signal-review">
          {error ?? 'Could not load flagged rings.'}
        </p>
      )}

      {status === 'loading' && rings.length === 0 ? (
        <EmptyPanel>Loading flagged rings…</EmptyPanel>
      ) : rings.length === 0 && status === 'ready' ? (
        <EmptyPanel>No rings are currently flagged.</EmptyPanel>
      ) : rings.length > 0 ? (
        <RingTabs rings={rings} />
      ) : null}
    </section>
  )
}

function RingTabs({ rings }: { rings: RingResponse[] }) {
  const tabs: TabDefinition[] = rings.map((ring) => ({
    id: ring.ring_id,
    label: `${ring.ring_size} accounts`,
    content: <RingDetail ring={ring} />,
  }))
  return <Tabs tabs={tabs} />
}

function RingDetail({ ring }: { ring: RingResponse }) {
  return (
    <div className="space-y-3 rounded-console border border-border bg-surface p-4">
      <dl className="flex flex-wrap gap-x-6 gap-y-1 font-mono text-xs text-text-muted">
        <div>
          <dt className="inline text-text-faint">ring </dt>
          <dd className="inline">{ring.ring_id}</dd>
        </div>
        <div>
          <dt className="inline text-text-faint">size </dt>
          <dd className="inline">{ring.ring_size}</dd>
        </div>
        {ring.snapshot_end && (
          <div>
            <dt className="inline text-text-faint">snapshot </dt>
            <dd className="inline">{formatDateTime(ring.snapshot_end)}</dd>
          </div>
        )}
      </dl>
      <RingGraph nodes={ring.nodes} edges={ring.edges} />
    </div>
  )
}

function EmptyPanel({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-console border border-border bg-surface p-8 text-center font-sans text-sm text-text-faint">
      {children}
    </div>
  )
}
