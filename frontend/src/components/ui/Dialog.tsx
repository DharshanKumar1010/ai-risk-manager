import { useEffect, useRef, type ReactNode } from 'react'

import { cx } from '@/lib/cx'

interface DialogProps {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
  className?: string
}

/**
 * The native `<dialog>` element via `showModal()` -- focus trapping, Escape-to-close, the
 * `::backdrop` pseudo-element and top-layer stacking all come from the browser rather than
 * from hand-rolled JS, at zero runtime dependencies. jsdom does not implement any of this
 * (see setupTests.ts's shim comment), so the shim covers open/close state for tests; focus
 * trapping and the backdrop are verified in a real browser during the a11y pass, not here.
 *
 * `onClose` fires from the dialog's own native `close` event (Escape, or a form submit with
 * method="dialog") as well as from an explicit close button, so the two paths cannot drift.
 */
export function Dialog({ open, onClose, title, children, className }: DialogProps) {
  const ref = useRef<HTMLDialogElement>(null)

  useEffect(() => {
    const node = ref.current
    if (node === null) return
    if (open && !node.open) node.showModal()
    if (!open && node.open) node.close()
  }, [open])

  useEffect(() => {
    const node = ref.current
    if (node === null) return
    const handleClose = () => onClose()
    node.addEventListener('close', handleClose)
    return () => node.removeEventListener('close', handleClose)
  }, [onClose])

  return (
    <dialog
      ref={ref}
      aria-labelledby="dialog-title"
      className={cx(
        'w-full max-w-2xl rounded-console border border-border bg-surface p-0 text-text',
        'backdrop:bg-bg/70',
        className,
      )}
    >
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <h2 id="dialog-title" className="font-sans text-sm font-semibold">
          {title}
        </h2>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="rounded-console px-2 py-1 text-text-faint hover:bg-surface-raised hover:text-text"
        >
          &times;
        </button>
      </div>
      <div className="max-h-[70vh] overflow-y-auto p-4">{children}</div>
    </dialog>
  )
}
