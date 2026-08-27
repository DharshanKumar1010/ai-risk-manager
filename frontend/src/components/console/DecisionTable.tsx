import { useState } from 'react'

import { DataTable, type DataTableColumn } from '@/components/ui/DataTable'
import { DecisionBadge, DegradedBadge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { ExplainModal } from '@/components/console/ExplainModal'
import type { AuditFeedState } from '@/hooks/useAuditFeed'
import { useAuth } from '@/hooks/useAuth'
import { formatTime } from '@/lib/format'
import type { AuditEntryResponse } from '@/types/api'

const COLUMNS: DataTableColumn<AuditEntryResponse>[] = [
  {
    key: 'decided_at',
    header: 'Time',
    render: (row) => <span className="font-mono text-xs">{formatTime(row.decided_at)}</span>,
  },
  {
    key: 'transaction_id',
    header: 'Transaction',
    render: (row) => <span className="font-mono text-xs">{row.transaction_id}</span>,
  },
  {
    key: 'account_id',
    header: 'Account',
    render: (row) => <span className="font-mono text-xs">{row.account_id}</span>,
  },
  {
    key: 'decision',
    header: 'Decision',
    render: (row) => <DecisionBadge decision={row.decision} />,
  },
  {
    key: 'degraded',
    header: 'Status',
    render: (row) => (row.degraded ? <DegradedBadge reason={row.degraded_reason} /> : null),
  },
]

/**
 * The console's central view: the most recent recorded decisions, sourced from `GET /audit`.
 * Deliberately not `GET /transactions` -- `POST /score` does not persist the transaction it
 * scores (see BUILD_LOG.md's Phase 7 known gaps), so a live-scored row would never appear in
 * a transactions-backed table. `audit_log` is written on every scoring call regardless.
 *
 * "This service has never blocked a transaction, only reviewed or allowed one" is a true
 * statement about the shipped cost policy, not a limitation of this view -- see Badge.tsx.
 *
 * `feed` is owned by the caller (see `DecisionsPanel`), not by this component -- the live feed
 * hook needs the same `refresh` this table's button uses, so one `useAuditFeed` call is shared
 * rather than each panel fetching its own copy of the backlog.
 */
export function DecisionTable({ feed }: { feed: AuditFeedState }) {
  const { status: authStatus } = useAuth()
  const { entries, status, error, refresh } = feed
  const [selected, setSelected] = useState<AuditEntryResponse | null>(null)

  if (authStatus === 'unavailable') {
    return (
      <EmptyPanel>
        The walkthrough token endpoint is not available on this instance. This console is
        running against a deployed backend outside local/ci — see the README for how to run it
        with a live walkthrough session.
      </EmptyPanel>
    )
  }

  if (authStatus === 'error') {
    return <EmptyPanel>Could not start a reviewer session. Try reloading.</EmptyPanel>
  }

  return (
    <section aria-labelledby="decision-table-heading" className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 id="decision-table-heading" className="font-sans text-sm font-semibold text-text">
          Recent decisions
        </h2>
        <Button variant="ghost" onClick={refresh} disabled={status === 'loading'}>
          Refresh
        </Button>
      </div>

      {status === 'error' && (
        <p className="font-sans text-sm text-signal-review">
          {error ?? 'Could not load recent decisions.'}
        </p>
      )}

      {(status === 'loading' && entries.length === 0) || authStatus === 'loading' ? (
        <EmptyPanel>Loading recent decisions…</EmptyPanel>
      ) : (
        <DataTable
          columns={COLUMNS}
          rows={entries}
          rowKey={(row) => String(row.audit_id)}
          onRowClick={setSelected}
          caption="Recent scoring decisions"
          emptyState="No transactions scored in this window yet."
        />
      )}

      <ExplainModal entry={selected} onClose={() => setSelected(null)} />
    </section>
  )
}

function EmptyPanel({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-console border border-border bg-surface p-8 text-center font-sans text-sm text-text-faint">
      {children}
    </div>
  )
}
