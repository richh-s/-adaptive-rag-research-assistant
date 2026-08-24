import { useRef, useState } from 'react'
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
  /** The answer text accumulated so far from token frames, while streaming. */
  streamingAnswer: string
  visits: NodeVisit[]
  submit: (question: string) => Promise<void>
  /** Abort the in-flight request. The turn is discarded client-side. */
  stop: () => void
  /** Re-submit the question whose request just failed (null if none). */
  retry: (() => Promise<void>) | null
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
  const [streamingAnswer, setStreamingAnswer] = useState('')
  const [visits, setVisits] = useState<NodeVisit[]>([])
  const [failedQuestion, setFailedQuestion] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  async function submit(question: string) {
    const trimmed = question.trim()
    if (!trimmed || loading) return

    const controller = new AbortController()
    abortRef.current = controller

    setLoading(true)
    setError(null)
    setFailedQuestion(null)
    setPendingQuestion(trimmed)
    setStreamingAnswer('')
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
          } else if (event.type === 'token') {
            if (event.token) setStreamingAnswer((prev) => prev + event.token)
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
            setFailedQuestion(trimmed)
          }
        },
        { conversationId, signal: controller.signal },
      )
    } catch (err) {
      if (controller.signal.aborted) {
        // User pressed Stop -- not an error; just drop the in-flight turn quietly.
      } else if (err instanceof ResearchApiError) {
        setError(err.message)
        setFailedQuestion(trimmed)
      } else {
        setError('Could not reach the research API.')
        setFailedQuestion(trimmed)
      }
    } finally {
      abortRef.current = null
      setLoading(false)
      setPendingQuestion(null)
      setStreamingAnswer('')
    }
  }

  function stop() {
    abortRef.current?.abort()
  }

  async function openConversation(id: string) {
    if (loading) return
    setError(null)
    setFailedQuestion(null)
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
    setFailedQuestion(null)
    setVisits([])
    setPendingQuestion(null)
    setStreamingAnswer('')
  }

  return {
    loading,
    error,
    turns,
    conversationId,
    pendingQuestion,
    streamingAnswer,
    visits,
    submit,
    stop,
    retry: failedQuestion ? () => submit(failedQuestion) : null,
    openConversation,
    reset,
  }
}
