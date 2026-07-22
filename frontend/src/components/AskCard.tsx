import type { FormEvent } from 'react'
import { EXAMPLE_QUESTIONS } from '../constants/exampleQuestions'
import './AskCard.css'

interface AskCardProps {
  question: string
  onQuestionChange: (question: string) => void
  onSubmit: (e: FormEvent) => void
  loading: boolean
}

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M4 12L20 4L13 20L11 13L4 12Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  )
}

export function AskCard({ question, onQuestionChange, onSubmit, loading }: AskCardProps) {
  return (
    <div className="ask-card">
      <form onSubmit={onSubmit} className="ask-form">
        <textarea
          value={question}
          onChange={(e) => onQuestionChange(e.target.value)}
          placeholder="Ask a research question…"
          rows={3}
        />
        <div className="ask-form-footer">
          <span className="ask-hint">Routes automatically to local docs, web search, or both</span>
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? (
              'Researching…'
            ) : (
              <>
                Ask <SendIcon />
              </>
            )}
          </button>
        </div>
      </form>

      <div className="examples">
        <span className="examples-label">Try an example</span>
        <div className="example-chips">
          {EXAMPLE_QUESTIONS.map((example) => (
            <button
              key={example.label}
              type="button"
              className="example-chip"
              onClick={() => onQuestionChange(example.question)}
              disabled={loading}
            >
              {example.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
