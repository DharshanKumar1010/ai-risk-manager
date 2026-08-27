/** Formatting helpers. Every one of these takes the wire-format string the API actually sends
 * (see types/api.ts's note on Decimal and datetime serialisation) rather than a pre-parsed
 * number or Date, so a call site cannot silently pass a value that was never validated against
 * what the backend really returns. */

const CURRENCY = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

const PERCENT = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

const DATETIME = new Intl.DateTimeFormat('en-US', {
  dateStyle: 'medium',
  timeStyle: 'medium',
})

const TIME_ONLY = new Intl.DateTimeFormat('en-US', {
  timeStyle: 'medium',
})

/** Format a Decimal-as-string amount ("150.5000") as USD currency. */
export function formatAmount(amount: string): string {
  const value = Number.parseFloat(amount)
  return Number.isFinite(value) ? CURRENCY.format(value) : amount
}

/** Format a fraction in [0, 1] as a percentage, e.g. 0.2241 -> "22.41%". */
export function formatPercent(fraction: number): string {
  return PERCENT.format(fraction)
}

/** Format an ISO-8601 datetime string for display. */
export function formatDateTime(iso: string): string {
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? iso : DATETIME.format(date)
}

/** Time-only, for a dense feed where the date is implied by "today". */
export function formatTime(iso: string): string {
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? iso : TIME_ONLY.format(date)
}

/** Relative "3m ago" style label, refreshed by the caller on an interval -- this function
 * itself is pure and takes "now" explicitly so it stays trivially testable. */
export function formatRelativeTime(iso: string, now: Date = new Date()): string {
  const then = new Date(iso)
  if (Number.isNaN(then.getTime())) return iso
  const seconds = Math.max(0, Math.round((now.getTime() - then.getTime()) / 1000))
  if (seconds < 5) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.round(hours / 24)
  return `${days}d ago`
}
