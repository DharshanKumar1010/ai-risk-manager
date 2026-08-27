import type { ReactNode } from 'react'

import { cx } from '@/lib/cx'

export interface DataTableColumn<T> {
  key: string
  header: string
  render: (row: T) => ReactNode
  /** Numeric/monospace columns align right and use tabular figures. */
  align?: 'left' | 'right'
}

interface DataTableProps<T> {
  columns: DataTableColumn<T>[]
  rows: T[]
  rowKey: (row: T) => string
  onRowClick?: (row: T) => void
  emptyState: ReactNode
  caption: string
}

/** A plain, semantic `<table>` -- the confusion matrix's own reasoning in the metrics panel
 * applies here too: a real table is screen-reader-correct and keyboardable in a way no
 * grid-of-divs styled to look like one can match without reimplementing the same semantics. */
export function DataTable<T>({
  columns,
  rows,
  rowKey,
  onRowClick,
  emptyState,
  caption,
}: DataTableProps<T>) {
  if (rows.length === 0) {
    return (
      <div className="rounded-console border border-border bg-surface p-8 text-center font-sans text-sm text-text-faint">
        {emptyState}
      </div>
    )
  }

  return (
    <div className="overflow-x-auto rounded-console border border-border">
      <table className="w-full border-collapse font-sans text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-border bg-surface-raised">
            {columns.map((column) => (
              <th
                key={column.key}
                scope="col"
                className={cx(
                  'px-3 py-2 font-medium text-text-faint',
                  column.align === 'right' ? 'text-right' : 'text-left',
                )}
              >
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const key = rowKey(row)
            const clickable = onRowClick !== undefined
            return (
              <tr
                key={key}
                onClick={clickable ? () => onRowClick(row) : undefined}
                tabIndex={clickable ? 0 : undefined}
                onKeyDown={
                  clickable
                    ? (event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault()
                          onRowClick(row)
                        }
                      }
                    : undefined
                }
                className={cx(
                  'border-b border-border last:border-0',
                  clickable && 'cursor-pointer hover:bg-surface-raised',
                )}
              >
                {columns.map((column) => (
                  <td
                    key={column.key}
                    className={cx(
                      'px-3 py-2 text-text',
                      column.align === 'right' ? 'text-right' : 'text-left',
                    )}
                  >
                    {column.render(row)}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
