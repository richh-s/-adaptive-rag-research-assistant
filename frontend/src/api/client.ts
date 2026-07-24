const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

export interface RetrievalCounts {
  vector: number
  bm25: number
  web: number
}

export interface NodeLatency {
  node: string
  latency_ms: number
}

export interface ResearchSummary {
  route: string | null
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
  route: string | null
  confidence_score: number | null
  summary?: ResearchSummary | null
}

export interface StreamEvent {
  type: 'progress' | 'done' | 'error'
  node?: string | null
  message?: string | null
  report?: string | null
  route?: string | null
  confidence_score?: number | null
  detail?: string | null
  summary?: ResearchSummary | null
}

export interface IngestResponse {
  filename: string
  original_filename: string
  size_bytes: number
  status: 'queued'
  message: string
}

export class ResearchApiError extends Error {}

export async function streamResearch(
  question: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/research/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
    signal,
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

export async function research(question: string): Promise<ResearchResponse> {
  const response = await fetch(`${API_BASE_URL}/research`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
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
    body: formData,
  })

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new ResearchApiError(body?.detail ?? `Upload failed with status ${response.status}`)
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
