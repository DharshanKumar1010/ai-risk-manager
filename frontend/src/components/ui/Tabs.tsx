import { useId, useState, type ReactNode } from 'react'

import { cx } from '@/lib/cx'

export interface TabDefinition {
  id: string
  label: string
  content: ReactNode
}

interface TabsProps {
  tabs: TabDefinition[]
  defaultTabId?: string
}

/** A minimal, keyboard-navigable tab set: arrow keys move focus and selection together
 * (WAI-ARIA's "automatic activation" pattern), Home/End jump to the first/last tab. */
export function Tabs({ tabs, defaultTabId }: TabsProps) {
  const [activeId, setActiveId] = useState(defaultTabId ?? tabs[0]?.id ?? '')
  const baseId = useId()

  const activeIndex = tabs.findIndex((tab) => tab.id === activeId)

  function handleKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (tabs.length === 0) return
    let nextIndex: number | null = null
    if (event.key === 'ArrowRight') nextIndex = (activeIndex + 1) % tabs.length
    else if (event.key === 'ArrowLeft') nextIndex = (activeIndex - 1 + tabs.length) % tabs.length
    else if (event.key === 'Home') nextIndex = 0
    else if (event.key === 'End') nextIndex = tabs.length - 1
    if (nextIndex === null) return
    event.preventDefault()
    const next = tabs[nextIndex]
    if (next === undefined) return
    setActiveId(next.id)
    document.getElementById(`${baseId}-tab-${next.id}`)?.focus()
  }

  const active = tabs.find((tab) => tab.id === activeId)

  return (
    <div>
      <div
        role="tablist"
        aria-label="Sections"
        onKeyDown={handleKeyDown}
        className="flex gap-1 border-b border-border"
      >
        {tabs.map((tab) => {
          const selected = tab.id === activeId
          return (
            <button
              key={tab.id}
              id={`${baseId}-tab-${tab.id}`}
              role="tab"
              type="button"
              aria-selected={selected}
              aria-controls={`${baseId}-panel-${tab.id}`}
              tabIndex={selected ? 0 : -1}
              onClick={() => setActiveId(tab.id)}
              className={cx(
                'border-b-2 px-3 py-2 font-sans text-sm font-medium transition-colors',
                selected
                  ? 'border-accent text-text'
                  : 'border-transparent text-text-faint hover:text-text-muted',
              )}
            >
              {tab.label}
            </button>
          )
        })}
      </div>
      {active !== undefined && (
        <div
          id={`${baseId}-panel-${active.id}`}
          role="tabpanel"
          aria-labelledby={`${baseId}-tab-${active.id}`}
          tabIndex={0}
          className="pt-4"
        >
          {active.content}
        </div>
      )}
    </div>
  )
}
