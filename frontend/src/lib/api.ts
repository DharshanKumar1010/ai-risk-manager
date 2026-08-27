import type {
  AuditFeedResponse,
  AuditListResponse,
  DemoTokenRequestBody,
  DemoTokenResponse,
  ExplanationResponse,
  Persona,
  RingListResponse,
  ScoreRequestBody,
  ScoreResponse,
  TransactionListResponse,
  WsTicketResponse,
} from '@/types/api'

/**
 * Base URL for the backend. `VITE_API_BASE_URL` is set by docker-compose in development and
 * must be set at build time for a deployed frontend -- Vite inlines `import.meta.env.VITE_*`
 * values into the bundle, so there is no runtime configuration step to add later. Falls back
 * to same-origin (`''`), which only works when the frontend is served from the same host as
 * the API -- true for local `vite dev` proxied setups, false for the deployed Vercel/Render
 * split, where the env var is required.
 */
const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? ''

/**
 * The same host `API_BASE_URL` points at, as a `ws(s)://` origin for `GET /ws/feed`. Browsers
 * have no notion of an http-vs-ws base URL, so this exists purely to swap the scheme; an
 * absolute `API_BASE_URL` (the deployed and docker-compose cases) has its `http(s)` prefix
 * replaced, and an empty one (same-origin dev) falls back to `window.location`'s own origin.
 */
export function wsBaseUrl(): string {
  if (API_BASE_URL === '') {
    const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${scheme}//${window.location.host}`
  }
  return API_BASE_URL.replace(/^http/, 'ws')
}

/**
 * Thrown for any non-2xx response. Carries the parsed body when the backend sent one, so a
 * caller can read `detail` the way every route on this API shapes its errors -- see
 * backend/app/api/score.py's `unknown_raw_columns` shape for the one response that nests it.
 */
export class ApiError extends Error {
  readonly status: number
  readonly body: unknown

  constructor(status: number, body: unknown) {
    super(ApiError.messageFor(status, body))
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }

  private static messageFor(status: number, body: unknown): string {
    if (body && typeof body === 'object' && 'detail' in body) {
      const detail = (body as { detail: unknown }).detail
      if (typeof detail === 'string') return detail
    }
    return `Request failed with status ${status}`
  }
}

interface RequestOptions {
  token?: string
  signal?: AbortSignal
}

async function request<T>(
  method: 'GET' | 'POST',
  path: string,
  body: unknown,
  options: RequestOptions,
): Promise<T> {
  const headers: Record<string, string> = {}
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  if (options.token !== undefined) headers.Authorization = `Bearer ${options.token}`

  const init: RequestInit = {
    method,
    headers,
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
    ...(options.signal !== undefined ? { signal: options.signal } : {}),
  }
  const response = await fetch(`${API_BASE_URL}${path}`, init)

  if (!response.ok) {
    let parsed: unknown = null
    try {
      parsed = await response.json()
    } catch {
      // Body wasn't JSON (a bare 404 from an unrouted path, a proxy error page). The status
      // code alone still tells the caller what happened.
    }
    throw new ApiError(response.status, parsed)
  }

  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

/** POST /score. Requires a merchant-persona token holding `score:write`. */
export function postScore(body: ScoreRequestBody, token: string): Promise<ScoreResponse> {
  return request<ScoreResponse>('POST', '/score', body, { token })
}

/** GET /audit -- the decision table's and the live feed's backlog source. */
export function getAuditFeed(
  token: string,
  params: { limit?: number; offset?: number } = {},
): Promise<AuditFeedResponse> {
  const query = new URLSearchParams()
  if (params.limit !== undefined) query.set('limit', String(params.limit))
  if (params.offset !== undefined) query.set('offset', String(params.offset))
  const suffix = query.size > 0 ? `?${query.toString()}` : ''
  return request<AuditFeedResponse>('GET', `/audit${suffix}`, undefined, { token })
}

/** GET /audit/{transaction_id} */
export function getAuditTrail(
  transactionId: string,
  token: string,
): Promise<AuditListResponse> {
  return request<AuditListResponse>(
    'GET',
    `/audit/${encodeURIComponent(transactionId)}`,
    undefined,
    { token },
  )
}

/**
 * GET /audit/entry/{audit_id}/explain. Requires `explain:read` AND `analyst` -- a
 * merchant-persona token gets 403, which `ApiError` surfaces with that status intact so the
 * caller can render "the merchant view cannot see this" rather than a generic failure.
 */
export function getExplanation(auditId: number, token: string): Promise<ExplanationResponse> {
  return request<ExplanationResponse>('GET', `/audit/entry/${auditId}/explain`, undefined, {
    token,
  })
}

/** GET /transactions */
export function getTransactions(
  token: string,
  params: { limit?: number; offset?: number } = {},
): Promise<TransactionListResponse> {
  const query = new URLSearchParams()
  if (params.limit !== undefined) query.set('limit', String(params.limit))
  if (params.offset !== undefined) query.set('offset', String(params.offset))
  const suffix = query.size > 0 ? `?${query.toString()}` : ''
  return request<TransactionListResponse>('GET', `/transactions${suffix}`, undefined, {
    token,
  })
}

/** GET /rings. Requires `rings:read` AND `analyst`. */
export function getRings(
  token: string,
  params: { limit?: number; offset?: number; min_size?: number } = {},
): Promise<RingListResponse> {
  const query = new URLSearchParams()
  if (params.limit !== undefined) query.set('limit', String(params.limit))
  if (params.offset !== undefined) query.set('offset', String(params.offset))
  if (params.min_size !== undefined) query.set('min_size', String(params.min_size))
  const suffix = query.size > 0 ? `?${query.toString()}` : ''
  return request<RingListResponse>('GET', `/rings${suffix}`, undefined, { token })
}

/**
 * POST /auth/demo-token. Only routed in local/ci -- see backend/app/api/auth.py. A deployed
 * instance outside those environments returns 404 for this call, which callers should treat
 * as "the walkthrough token endpoint is unavailable here", not as a generic error.
 */
export function mintDemoToken(persona: Persona, accountId?: string): Promise<DemoTokenResponse> {
  const body: DemoTokenRequestBody =
    accountId !== undefined ? { persona, account_id: accountId } : { persona }
  return request<DemoTokenResponse>('POST', '/auth/demo-token', body, {})
}

/**
 * POST /auth/ws-ticket. Requires the same scopes `GET /ws/feed` itself requires (`analyst`,
 * `audit:read`, and `explain:read` -- the feed carries `risk_probability`, the same figure
 * `explain:read` gates elsewhere on this API, so a token missing it must not mint a ticket
 * either; Phase 9.5 corrected this comment, which had dropped `explain:read`) -- mint
 * immediately before each connection attempt, not once and cached: the ticket is deliberately
 * 30 seconds old by design, see backend/app/core/security.py.
 */
export function mintWsTicket(token: string): Promise<WsTicketResponse> {
  return request<WsTicketResponse>('POST', '/auth/ws-ticket', undefined, { token })
}
