import { cx } from '@/lib/cx'

/** A small spinning ring, CSS-only -- no image/SVG asset, matching this app's system-only
 * asset policy (see index.css's font-stack rationale). `animate-spin` respects the global
 * `prefers-reduced-motion` override in index.css. */
export function Spinner({ className }: { className?: string }) {
  return (
    <span
      role="status"
      aria-label="Loading"
      className={cx(
        'inline-block h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-2 border-current border-t-transparent opacity-70',
        className,
      )}
    />
  )
}
