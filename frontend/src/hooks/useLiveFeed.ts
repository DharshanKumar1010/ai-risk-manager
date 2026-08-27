import { useEffect, useRef, useState } from 'react'

import { mintWsTicket, wsBaseUrl } from '@/lib/api'
import type { FeedEvent, FeedMessage } from '@/types/api'

export type LiveFeedStatus = 'connecting' | 'open' | 'closed'

interface LiveFeedState {
  status: LiveFeedStatus
  /** Most recent events, newest first. Bounded -- this is a live ticker, not a store. */
  events: FeedEvent[]
}

const MAX_EVENTS = 20
const INITIAL_BACKOFF_MS = 1_000
const MAX_BACKOFF_MS = 30_000

/**
 * `GET /ws/feed` client: mints a fresh ticket, connects, and reconnects with exponential
 * backoff on any drop. Each attempt mints a new ticket rather than reusing one, since a ticket
 * is deliberately only 30 seconds old by the time it could be replayed -- see
 * backend/app/core/security.py's `mint_ws_ticket` docstring.
 *
 * `onDecision` fires for every decision event, in addition to it landing in `events` -- the
 * decision table uses it to refresh its own `GET /audit` backlog so the two views agree without
 * this hook needing to know anything about `AuditEntryResponse`'s shape (see types/api.ts's
 * `FeedEvent` comment for why the two are not merged into one list).
 */
export function useLiveFeed(
  analystToken: string | null,
  onDecision?: (event: FeedEvent) => void,
): LiveFeedState {
  const [status, setStatus] = useState<LiveFeedStatus>('connecting')
  const [events, setEvents] = useState<FeedEvent[]>([])
  const onDecisionRef = useRef(onDecision)
  useEffect(() => {
    onDecisionRef.current = onDecision
  })

  useEffect(() => {
    if (analystToken === null) return
    const token = analystToken

    let cancelled = false
    let socket: WebSocket | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let backoffMs = INITIAL_BACKOFF_MS

    async function connect(): Promise<void> {
      if (cancelled) return
      setStatus('connecting')
      try {
        const ticket = await mintWsTicket(token)
        if (cancelled) return
        socket = new WebSocket(`${wsBaseUrl()}/ws/feed?ticket=${ticket.ticket}`)
      } catch {
        scheduleReconnect()
        return
      }

      socket.addEventListener('open', () => {
        if (cancelled) return
        backoffMs = INITIAL_BACKOFF_MS
        setStatus('open')
      })

      socket.addEventListener('message', (raw) => {
        if (cancelled) return
        let message: FeedMessage
        try {
          message = JSON.parse(String(raw.data)) as FeedMessage
        } catch {
          return
        }
        if (message.type !== 'decision') return
        setEvents((current) => [message, ...current].slice(0, MAX_EVENTS))
        onDecisionRef.current?.(message)
      })

      socket.addEventListener('close', () => {
        if (cancelled) return
        setStatus('closed')
        scheduleReconnect()
      })

      socket.addEventListener('error', () => {
        socket?.close()
      })
    }

    function scheduleReconnect(): void {
      if (cancelled) return
      reconnectTimer = setTimeout(() => {
        void connect()
      }, backoffMs)
      backoffMs = Math.min(backoffMs * 2, MAX_BACKOFF_MS)
    }

    void connect()

    return () => {
      cancelled = true
      if (reconnectTimer !== null) clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [analystToken])

  return { status, events }
}
