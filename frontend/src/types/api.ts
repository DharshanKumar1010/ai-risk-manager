/**
 * Mirrors backend/app/api/schemas.py exactly -- field names, optionality and wire types.
 *
 * Two wire-format notes that are easy to get wrong once and never notice:
 *
 * `amount` fields are `string`, not `number`. Pydantic's `Decimal` serialises to a JSON
 * string ("150.5000"), not a JSON number -- confirmed against a live response, not assumed.
 * Parse with `parseFloat` only at render time (see lib/format.ts), never store as a number.
 *
 * Every `datetime` field is an ISO-8601 string with a `Z` suffix. Kept as `string` here too;
 * pages that need a `Date` construct one at the point of use rather than in the type.
 *
 * What is deliberately absent from this file, because the backend never sends it: any field
 * of `DecisionCost`, the operating threshold, or `risk_band` on any response but the audit
 * ones. `risk_probability` appears on exactly two types, `ExplanationResponse` and `FeedEvent`
 * -- both analyst-scoped, both documented at the type. Re-adding a probability or a cost field
 * here without the backend actually sending one just produces `undefined` at runtime; it does
 * not create a leak by itself, but it is a sign this file has drifted from the schema it
 * mirrors.
 */

/** `_decide` in app/core/serving.py can only ever return "allow" or "review" -- "block" is a
 * schema value the shipped cost policy never produces. Kept in the type because the schema
 * allows it and a future policy might, but no UI here should render a legend swatch for it.
 */
export type Decision = 'allow' | 'review' | 'block'

export interface ScoreResponse {
  transaction_id: string
  decision: Decision
  audit_id: number
  degraded: boolean
  decided_at: string
  model_version: string
}

export interface ScoreRequestBody {
  transaction_id: string
  account_id: string
  event_time: string
  amount: string
  raw_columns: Record<string, string | number | boolean | null>
}

export interface TransactionSummary {
  transaction_id: string
  account_id: string
  event_time: string
  amount: string
  transaction_type: string | null
}

export interface TransactionListResponse {
  transactions: TransactionSummary[]
  count: number
}

export interface AuditEntryResponse {
  audit_id: number
  transaction_id: string
  account_id: string
  decided_at: string
  decision: Decision
  model_versions: Record<string, string>
  feature_version: string
  degraded: boolean
  degraded_reason: string | null
}

export interface AuditListResponse {
  transaction_id: string
  entries: AuditEntryResponse[]
}

/** GET /audit -- the decision table's and the live feed's backlog source. */
export interface AuditFeedResponse {
  entries: AuditEntryResponse[]
  count: number
}

export interface FeatureContribution {
  feature: string
  contribution: number
}

/**
 * The one response on the whole API that carries `risk_probability`. Analyst-scoped
 * (`explain:read` AND `analyst`) -- see backend/app/api/audit.py. Never fetch this route
 * with a merchant-persona token; it will 403, which is the point.
 */
export interface ExplanationResponse {
  audit_id: number
  transaction_id: string
  decision: Decision
  risk_probability: number
  top_features: FeatureContribution[]
  model_versions: Record<string, string>
  feature_version: string
  degraded: boolean
}

export interface RingMember {
  account_id: string
}

/**
 * One node of a flagged ring's topology. An account node's `node_id` is the plain
 * `account_id` -- already on `RingResponse.members`. An entity node's `node_id` is a truncated
 * SHA-256 of the raw shared fingerprint (a card/device composite), minted server-side; this
 * type never carries the fingerprint itself.
 */
export interface RingGraphNode {
  node_id: string
  kind: 'account' | 'entity'
  entity_type: string | null
}

export interface RingGraphEdge {
  source: string
  target: string
}

export interface RingResponse {
  ring_id: string
  ring_size: number
  members: RingMember[]
  snapshot_end: string | null
  /** Phase 8 addition. Empty on a ring trained before the topology export existed. */
  nodes: RingGraphNode[]
  edges: RingGraphEdge[]
}

export interface RingListResponse {
  rings: RingResponse[]
  count: number
  model_version: string
}

export type Persona = 'merchant' | 'analyst'

export interface DemoTokenRequestBody {
  persona: Persona
  account_id?: string
}

export interface DemoTokenResponse {
  access_token: string
  token_type: 'bearer'
  persona: Persona
  scopes: string[]
  expires_in: number
}

/**
 * One decision, pushed over `GET /ws/feed` as it is made. Analyst-only, matching the socket's
 * own scope gate -- which is why this carries `risk_probability` and `amount` even though
 * `AuditEntryResponse` (the same decision, read back later from `GET /audit`) carries neither.
 * The two are not the same shape and this hook never pretends otherwise: a live event renders
 * in its own panel, distinctly labelled as live, rather than being coerced into a
 * `AuditEntryResponse` row with fabricated `model_versions`/`feature_version` fields the
 * backend never sent for it. See ml-evaluation-standards item 4.6 -- live output must never be
 * shown next to held-out numbers without a distinguishing label, and that applies here too.
 */
export interface FeedEvent {
  type: 'decision'
  audit_id: number
  transaction_id: string
  account_id: string
  decided_at: string
  decision: Decision
  risk_probability: number
  amount: string
  degraded: boolean
  model_version: string
}

/** Non-decision frames the socket also sends; a live-feed consumer must ignore both. */
export interface FeedHelloMessage {
  type: 'hello'
  server_time: string
}

export interface FeedPingMessage {
  type: 'ping'
}

export type FeedMessage = FeedEvent | FeedHelloMessage | FeedPingMessage

export interface WsTicketResponse {
  ticket: string
  expires_in: number
}
