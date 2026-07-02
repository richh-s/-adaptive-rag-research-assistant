const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

export interface ResearchResponse {
  question: string
  report: string
  route: string | null
  confidence_score: number | null
}

export class ResearchApiError extends Error {}

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

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`)
    return response.ok
  } catch {
    return false
  }
}
