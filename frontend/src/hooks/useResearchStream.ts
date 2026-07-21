import { useState } from 'react'
import { streamResearch, ResearchApiError, type ResearchResponse, type StreamEvent } from '../api/client'

export interface NodeVisit {
  node: string
  message: string
  seq: number
}

export interface UseResearchStreamResult {
  loading: boolean
  error: string | null
  result: ResearchResponse | null
  visits: NodeVisit[]
  submit: (question: string) => Promise<void>
}

export function useResearchStream(): UseResearchStreamResult {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ResearchResponse | null>(null)
  const [visits, setVisits] = useState<NodeVisit[]>([])

  async function submit(question: string) {
    const trimmed = question.trim()
    if (!trimmed || loading) return

    setLoading(true)
    setError(null)
    setResult(null)
    setVisits([])

    let seq = 0
    try {
      await streamResearch(trimmed, (event: StreamEvent) => {
        if (event.type === 'progress') {
          if (event.node) {
            seq += 1
            const visit: NodeVisit = { node: event.node, message: event.message ?? event.node, seq }
            setVisits((prev) => [...prev, visit])
          }
        } else if (event.type === 'done') {
          setResult({
            question: trimmed,
            report: event.report ?? '',
            route: event.route ?? null,
            confidence_score: event.confidence_score ?? null,
            summary: event.summary ?? null,
          })
        } else if (event.type === 'error') {
          setError(event.detail ?? 'The research request failed.')
        }
      })
    } catch (err) {
      setError(err instanceof ResearchApiError ? err.message : 'Could not reach the research API.')
    } finally {
      setLoading(false)
    }
  }

  return { loading, error, result, visits, submit }
}
