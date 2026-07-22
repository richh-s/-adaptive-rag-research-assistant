import ReactMarkdown from 'react-markdown'
import type { ResearchResponse } from '../api/client'
import './ResultCard.css'

interface ResultCardProps {
  result: ResearchResponse
}

export function ResultCard({ result }: ResultCardProps) {
  return (
    <div className="result">
      <ReactMarkdown>{result.report}</ReactMarkdown>
    </div>
  )
}
