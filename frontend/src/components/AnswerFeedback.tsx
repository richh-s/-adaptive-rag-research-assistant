import { useState } from 'react'
import { submitFeedback, type FeedbackRating, type ResearchResponse } from '../api/client'
import './AnswerFeedback.css'

interface AnswerFeedbackProps {
  question: string
  result: ResearchResponse
}

/**
 * Thumbs up/down on one answer.
 *
 * Optimistic and unblocking on purpose: the rating is recorded in local state immediately and
 * the request is fire-and-forget. A rating is a side channel — the user already has their
 * answer, and making them wait on (or see an error from) a feedback call would be a worse
 * experience than silently failing to record one opinion.
 *
 * A downvote reveals an optional note field, because "this was wrong" is far more actionable
 * with a sentence attached, and the downvoted questions are what a stale eval dataset is
 * missing. The note is never demanded — asking for one before accepting the rating would cost
 * most of the ratings.
 */
export function AnswerFeedback({ question, result }: AnswerFeedbackProps) {
  const [rating, setRating] = useState<FeedbackRating | null>(null)
  const [note, setNote] = useState('')
  const [noteSent, setNoteSent] = useState(false)

  function send(next: FeedbackRating, withNote?: string) {
    void submitFeedback({
      question,
      rating: next,
      conversation_id: result.conversation_id ?? null,
      note: withNote?.trim() ? withNote.trim() : null,
      route: result.route ?? null,
      confidence_score: result.confidence_score ?? null,
    })
  }

  function choose(next: FeedbackRating) {
    if (rating === next) return
    setRating(next)
    send(next)
  }

  return (
    <div className="answer-feedback">
      <span className="answer-feedback-label" id={`feedback-label-${result.conversation_id ?? 'x'}`}>
        Was this answer helpful?
      </span>
      <div className="answer-feedback-buttons" role="group" aria-label="Was this answer helpful?">
        <button
          type="button"
          className={`answer-feedback-button${rating === 'up' ? ' is-chosen' : ''}`}
          aria-pressed={rating === 'up'}
          onClick={() => choose('up')}
        >
          <span aria-hidden="true">👍</span> Yes
        </button>
        <button
          type="button"
          className={`answer-feedback-button${rating === 'down' ? ' is-chosen' : ''}`}
          aria-pressed={rating === 'down'}
          onClick={() => choose('down')}
        >
          <span aria-hidden="true">👎</span> No
        </button>
      </div>

      {rating !== null && <span className="answer-feedback-thanks">Thanks — recorded.</span>}

      {rating === 'down' && !noteSent && (
        <form
          className="answer-feedback-note"
          onSubmit={(event) => {
            event.preventDefault()
            send('down', note)
            setNoteSent(true)
          }}
        >
          <label className="visually-hidden" htmlFor="feedback-note">
            What was wrong with this answer?
          </label>
          <input
            id="feedback-note"
            type="text"
            value={note}
            onChange={(event) => setNote(event.target.value)}
            placeholder="What was wrong? (optional)"
            maxLength={2000}
          />
          <button type="submit" disabled={!note.trim()}>
            Send
          </button>
        </form>
      )}

      {noteSent && <span className="answer-feedback-thanks">Note sent.</span>}
    </div>
  )
}
