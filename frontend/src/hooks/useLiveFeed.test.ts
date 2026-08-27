import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useLiveFeed } from '@/hooks/useLiveFeed'
import type { FeedEvent } from '@/types/api'

vi.mock('@/lib/api', () => ({
  mintWsTicket: vi.fn().mockResolvedValue({ ticket: 'a-ticket', expires_in: 30 }),
  wsBaseUrl: () => 'ws://localhost:8000',
}))

/** A controllable stand-in for the browser's WebSocket. jsdom's real one tries an actual
 * network connection, which is exactly what this test must not depend on -- see setupTests.ts
 * for the project's established pattern of shimming a jsdom gap only where a real test needs
 * it, rather than in general. Each instance is captured in `instances` so a test can drive its
 * lifecycle (`emitOpen`, `emitMessage`, `emitClose`) directly. */
class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  url: string
  listeners: Record<string, ((event: unknown) => void)[]> = {}
  closed = false

  constructor(url: string) {
    this.url = url
    FakeWebSocket.instances.push(this)
  }

  addEventListener(type: string, handler: (event: unknown) => void): void {
    ;(this.listeners[type] ??= []).push(handler)
  }

  close(): void {
    this.closed = true
    this.emit('close', {})
  }

  emit(type: string, event: unknown): void {
    for (const handler of this.listeners[type] ?? []) handler(event)
  }

  emitOpen(): void {
    this.emit('open', {})
  }

  emitMessage(payload: unknown): void {
    this.emit('message', { data: JSON.stringify(payload) })
  }
}

function decisionEvent(overrides: Partial<FeedEvent> = {}): FeedEvent {
  return {
    type: 'decision',
    audit_id: 1,
    transaction_id: 'T-1',
    account_id: 'acct-1',
    decided_at: '2026-08-26T12:00:00Z',
    decision: 'allow',
    risk_probability: 0.12,
    amount: '150.00',
    degraded: false,
    model_version: 'tier1-v1',
    ...overrides,
  }
}

describe('useLiveFeed', () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
    vi.stubGlobal('WebSocket', FakeWebSocket)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('does nothing while there is no analyst token', () => {
    const { result } = renderHook(() => useLiveFeed(null))
    expect(FakeWebSocket.instances).toHaveLength(0)
    expect(result.current.status).toBe('connecting')
    expect(result.current.events).toEqual([])
  })

  it('connects using a freshly minted ticket and reports open', async () => {
    const { result } = renderHook(() => useLiveFeed('analyst-token'))

    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
    expect(FakeWebSocket.instances[0]?.url).toBe('ws://localhost:8000/ws/feed?ticket=a-ticket')

    act(() => FakeWebSocket.instances[0]?.emitOpen())
    expect(result.current.status).toBe('open')
  })

  it('collects decision events newest-first and ignores hello/ping frames', async () => {
    const { result } = renderHook(() => useLiveFeed('analyst-token'))
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
    const socket = FakeWebSocket.instances[0]
    if (!socket) throw new Error('expected a socket instance')

    act(() => {
      socket.emitOpen()
      socket.emitMessage({ type: 'hello', server_time: '2026-08-26T12:00:00Z' })
      socket.emitMessage({ type: 'ping' })
      socket.emitMessage(decisionEvent({ audit_id: 1 }))
      socket.emitMessage(decisionEvent({ audit_id: 2 }))
    })

    expect(result.current.events.map((event) => event.audit_id)).toEqual([2, 1])
  })

  it('calls onDecision for every decision event, using the latest callback', async () => {
    const onDecision = vi.fn()
    const { rerender } = renderHook(({ cb }) => useLiveFeed('analyst-token', cb), {
      initialProps: { cb: onDecision },
    })
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
    const socket = FakeWebSocket.instances[0]
    if (!socket) throw new Error('expected a socket instance')

    const secondCallback = vi.fn()
    rerender({ cb: secondCallback })

    act(() => socket.emitMessage(decisionEvent({ audit_id: 7 })))

    expect(onDecision).not.toHaveBeenCalled()
    expect(secondCallback).toHaveBeenCalledWith(decisionEvent({ audit_id: 7 }))
  })

  it('reconnects with backoff after the socket closes', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    renderHook(() => useLiveFeed('analyst-token'))
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))

    act(() => FakeWebSocket.instances[0]?.close())
    expect(FakeWebSocket.instances).toHaveLength(1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000)
    })
    expect(FakeWebSocket.instances).toHaveLength(2)
  })

  it('closes the socket and stops reconnecting on unmount', async () => {
    const { unmount } = renderHook(() => useLiveFeed('analyst-token'))
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
    const socket = FakeWebSocket.instances[0]
    if (!socket) throw new Error('expected a socket instance')

    unmount()

    expect(socket.closed).toBe(true)
  })
})
