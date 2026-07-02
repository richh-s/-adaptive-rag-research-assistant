import type { FormEvent } from 'react'
import { EXAMPLE_QUESTIONS } from '../constants/exampleQuestions'
import './AskCard.css'

interface AskCardProps {
  question: string
  onQuestionChange: (question: string) => void
  onSubmit: (e: FormEvent) => void
  loading: boolean
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
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? 'Researching…' : 'Ask'}
        </button>
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
