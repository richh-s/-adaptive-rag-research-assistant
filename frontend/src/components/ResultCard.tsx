import ReactMarkdown from 'react-markdown'
import type { ResearchResponse } from '../api/client'
import './ResultCard.css'

interface ResultCardProps {
  result: ResearchResponse
}

export function ResultCard({ result }: ResultCardProps) {
  return (
    <div className="result">
      <div className="meta">
        {result.route && <span className="badge">route: {result.route}</span>}
        {result.confidence_score !== null && (
          <span className="badge">confidence: {result.confidence_score.toFixed(2)}</span>
        )}
      </div>
      <ReactMarkdown>{result.report}</ReactMarkdown>
    </div>
  )
}
