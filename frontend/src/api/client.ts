const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

// API key handling: the backend runs open (no key needed) unless its API_KEYS setting is
// configured, in which case every data endpoint wants an X-API-Key header. The key lives in
// localStorage so the access gate only has to ask once per browser.
const API_KEY_STORAGE = 'rag_assistant_api_key'

export function getStoredApiKey(): string | null {
  try {
    return localStorage.getItem(API_KEY_STORAGE)
  } catch {
    return null
  }
}

export function storeApiKey(key: string | null): void {
  try {
    if (key) localStorage.setItem(API_KEY_STORAGE, key)
    else localStorage.removeItem(API_KEY_STORAGE)
  } catch {
    // Storage unavailable (private mode etc.) -- the key just won't persist across reloads.
  }
}

function authHeaders(): Record<string, string> {
  const key = getStoredApiKey()
  return key ? { 'X-API-Key': key } : {}
}

export type AuthStatus = 'open' | 'authorized' | 'unauthorized'

export async function checkAuth(): Promise<AuthStatus> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/auth/check`, { headers: authHeaders() })
    if (response.status === 401) return 'unauthorized'
    if (!response.ok) return 'open'
    const body = await response.json()
    return body.auth_required ? 'authorized' : 'open'
  } catch {
    // Backend unreachable -- the health banner covers that; don't also block on auth.
    return 'open'
  }
}

export interface RetrievalCounts {
  vector: number
  bm25: number
  web: number
}

export interface NodeLatency {
  node: string
  latency_ms: number
}

export interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
}

export interface ResearchSummary {
  route: string | null
  condensed_question?: string | null
  sub_queries: string[]
  retrieval_counts: RetrievalCounts
  fused_document_count: number
  confidence_score: number | null
  correction_attempted: boolean
  node_latencies_ms: NodeLatency[]
  total_latency_ms: number
}

export interface ResearchResponse {
  question: string
  report: string
  answer?: string | null
  route: string | null
  confidence_score: number | null
  summary?: ResearchSummary | null
}

export interface StreamEvent {
  type: 'progress' | 'done' | 'error'
  node?: string | null
  message?: string | null
  report?: string | null
  answer?: string | null
  route?: string | null
  confidence_score?: number | null
  detail?: string | null
  summary?: ResearchSummary | null
  conversation_id?: string | null
}

export interface ConversationSummaryInfo {
  id: string
  title: string
  created_at: number
  updated_at: number
  message_count: number
}

export interface ConversationMessage {
  role: 'user' | 'assistant'
  content: string
  report?: string | null
  summary?: ResearchSummary | null
  created_at: number
}

export interface ConversationDetail {
  id: string
  title: string
  created_at: number
  updated_at: number
  messages: ConversationMessage[]
}

export interface IngestResponse {
  task_id: string
  filename: string
  original_filename: string
  size_bytes: number
  status: 'queued'
  message: string
}

export type IngestStage = 'queued' | 'parsing' | 'indexing' | 'indexed' | 'failed'

export interface IngestTaskStatus {
  task_id: string
  filename: string
  original_filename: string
  stage: IngestStage
  message: string
  error?: string | null
  indexed_chunks?: number | null
}

export class ResearchApiError extends Error {}

// The backend caps each history turn at 8000 chars and only reads the most recent turns, so
// trim client-side rather than let a long answer fail request validation.
const MAX_HISTORY_TURN_CHARS = 8000

function trimHistory(history: ChatTurn[]): ChatTurn[] {
  return history
    .filter((turn) => turn.content.trim().length > 0)
    .slice(-20)
    .map((turn) => ({ ...turn, content: turn.content.slice(0, MAX_HISTORY_TURN_CHARS) }))
}

export interface StreamResearchOptions {
  /** Continue a server-persisted conversation; the server loads its own transcript. */
  conversationId?: string | null
  /** Stateless fallback history, used only when no conversationId is given. */
  history?: ChatTurn[]
  signal?: AbortSignal
}

export async function streamResearch(
  question: string,
  onEvent: (event: StreamEvent) => void,
  options: StreamResearchOptions = {},
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/research/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({
      question,
      conversation_id: options.conversationId ?? null,
      history: trimHistory(options.history ?? []),
    }),
    signal: options.signal,
  })

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new ResearchApiError(body?.detail ?? `Request failed with status ${response.status}`)
  }

  const reader = response.body?.getReader()
  if (!reader) {
    throw new ResearchApiError('Streaming is not supported by this browser.')
  }

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      const line = frame.split('\n').find((l) => l.startsWith('data: '))
      if (!line) continue
      onEvent(JSON.parse(line.slice('data: '.length)))
    }
  }
}

export async function research(question: string, history: ChatTurn[] = []): Promise<ResearchResponse> {
  const response = await fetch(`${API_BASE_URL}/research`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ question, history: trimHistory(history) }),
  })

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new ResearchApiError(body?.detail ?? `Request failed with status ${response.status}`)
  }

  return response.json()
}

export async function ingestFile(file: File): Promise<IngestResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/api/v1/ingest`, {
    method: 'POST',
    headers: authHeaders(),
    body: formData,
  })

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new ResearchApiError(body?.detail ?? `Upload failed with status ${response.status}`)
  }

  return response.json()
}

export async function listConversations(): Promise<ConversationSummaryInfo[]> {
  const response = await fetch(`${API_BASE_URL}/api/v1/conversations`, { headers: authHeaders() })
  if (!response.ok) {
    throw new ResearchApiError(`Listing conversations failed with status ${response.status}`)
  }
  return response.json()
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  const response = await fetch(`${API_BASE_URL}/api/v1/conversations/${id}`, { headers: authHeaders() })
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new ResearchApiError(body?.detail ?? `Loading conversation failed with status ${response.status}`)
  }
  return response.json()
}

export async function deleteConversation(id: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/conversations/${id}`, { method: 'DELETE', headers: authHeaders() })
  if (!response.ok && response.status !== 404) {
    throw new ResearchApiError(`Deleting conversation failed with status ${response.status}`)
  }
}

export async function ingestUrl(url: string): Promise<IngestResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/ingest/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ url }),
  })

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new ResearchApiError(body?.detail ?? `URL ingestion failed with status ${response.status}`)
  }

  return response.json()
}

export async function getIngestStatus(taskId: string): Promise<IngestTaskStatus> {
  const response = await fetch(`${API_BASE_URL}/api/v1/ingest/${taskId}`, { headers: authHeaders() })

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new ResearchApiError(body?.detail ?? `Status check failed with status ${response.status}`)
  }

  return response.json()
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`)
    return response.ok
  } catch {
    return false
  }
}
