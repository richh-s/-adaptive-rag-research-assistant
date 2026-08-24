import { useState } from 'react'
import {
  getConversation,
  streamResearch,
  ResearchApiError,
  type ResearchResponse,
  type StreamEvent,
} from '../api/client'

export interface NodeVisit {
  node: string
  message: string
  seq: number
}

export interface CompletedTurn {
  id: number
  question: string
  result: ResearchResponse
}

export interface UseResearchStreamResult {
  loading: boolean
  error: string | null
  /** Finished question/answer pairs, oldest first — the conversation so far. */
  turns: CompletedTurn[]
  /** Server id of the current conversation; every exchange is persisted under it. */
  conversationId: string | null
  /** The question currently being researched, shown while its stream is in flight. */
  pendingQuestion: string | null
  visits: NodeVisit[]
  submit: (question: string) => Promise<void>
  /** Load a persisted conversation from the server and continue it. */
  openConversation: (id: string) => Promise<void>
  /** Start a fresh conversation. The previous one stays on the server. */
  reset: () => void
}

export function useResearchStream(): UseResearchStreamResult {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [turns, setTurns] = useState<CompletedTurn[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null)
  const [visits, setVisits] = useState<NodeVisit[]>([])

  async function submit(question: string) {
    const trimmed = question.trim()
    if (!trimmed || loading) return

    setLoading(true)
    setError(null)
    setPendingQuestion(trimmed)
    setVisits([])

    let seq = 0
    try {
      // The server owns the transcript: the first turn creates a persisted conversation
      // (its id arrives on the "done" frame), and follow-ups just send that id back.
      await streamResearch(
        trimmed,
        (event: StreamEvent) => {
          if (event.type === 'progress') {
            if (event.node) {
              seq += 1
              const visit: NodeVisit = { node: event.node, message: event.message ?? event.node, seq }
              setVisits((prev) => [...prev, visit])
            }
          } else if (event.type === 'done') {
            const result: ResearchResponse = {
              question: trimmed,
              report: event.report ?? '',
              answer: event.answer ?? null,
              route: event.route ?? null,
              confidence_score: event.confidence_score ?? null,
              summary: event.summary ?? null,
            }
            setTurns((prev) => [...prev, { id: Date.now(), question: trimmed, result }])
            if (event.conversation_id) setConversationId(event.conversation_id)
          } else if (event.type === 'error') {
            setError(event.detail ?? 'The research request failed.')
          }
        },
        { conversationId },
      )
    } catch (err) {
      setError(err instanceof ResearchApiError ? err.message : 'Could not reach the research API.')
    } finally {
      setLoading(false)
      setPendingQuestion(null)
    }
  }

  async function openConversation(id: string) {
    if (loading) return
    setError(null)
    try {
      const detail = await getConversation(id)
      const loaded: CompletedTurn[] = []
      let question: string | null = null
      for (const message of detail.messages) {
        if (message.role === 'user') {
          question = message.content
        } else if (question !== null) {
          loaded.push({
            id: message.created_at,
            question,
            result: {
              question,
              report: message.report ?? message.content,
              answer: message.content,
              route: message.summary?.route ?? null,
              confidence_score: message.summary?.confidence_score ?? null,
              summary: message.summary ?? null,
            },
          })
          question = null
        }
      }
      setTurns(loaded)
      setConversationId(detail.id)
      setPendingQuestion(null)
      setVisits([])
    } catch (err) {
      setError(err instanceof ResearchApiError ? err.message : 'Could not load the conversation.')
    }
  }

  function reset() {
    if (loading) return
    setTurns([])
    setConversationId(null)
    setError(null)
    setVisits([])
    setPendingQuestion(null)
  }

  return {
    loading,
    error,
    turns,
    conversationId,
    pendingQuestion,
    visits,
    submit,
    openConversation,
    reset,
  }
}
