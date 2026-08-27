import { describe, expect, it } from 'vitest'

import { formatRelativeTime } from '@/lib/format'

describe('formatRelativeTime', () => {
  const now = new Date('2026-08-27T12:00:00Z')

  it('reads a just-elapsed timestamp as "just now"', () => {
    expect(formatRelativeTime('2026-08-27T11:59:57Z', now)).toBe('just now')
  })

  it('formats seconds', () => {
    expect(formatRelativeTime('2026-08-27T11:59:30Z', now)).toBe('30s ago')
  })

  it('formats minutes', () => {
    expect(formatRelativeTime('2026-08-27T11:55:00Z', now)).toBe('5m ago')
  })

  it('formats hours', () => {
    expect(formatRelativeTime('2026-08-27T09:00:00Z', now)).toBe('3h ago')
  })

  it('formats days', () => {
    expect(formatRelativeTime('2026-08-25T12:00:00Z', now)).toBe('2d ago')
  })

  it('clamps a future timestamp to "just now" rather than a negative duration', () => {
    expect(formatRelativeTime('2026-08-27T12:00:05Z', now)).toBe('just now')
  })

  it('falls back to the raw string for an unparseable timestamp', () => {
    expect(formatRelativeTime('not-a-date', now)).toBe('not-a-date')
  })
})
