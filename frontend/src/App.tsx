import { useState, type FormEvent } from 'react'
import { streamResearch, ResearchApiError, type ResearchResponse } from './api/client'
import { useHealthStatus } from './hooks/useHealthStatus'
import { Header } from './components/Header'
import { AskCard } from './components/AskCard'
import { ResultCard } from './components/ResultCard'
import { ErrorBanner } from './components/ErrorBanner'
import { ProgressIndicator } from './components/ProgressIndicator'
import './App.css'

function App() {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ResearchResponse | null>(null)
  const [progressMessages, setProgressMessages] = useState<string[]>([])
  const backendUp = useHealthStatus()

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!question.trim() || loading) return

    const submittedQuestion = question.trim()
    setLoading(true)
    setError(null)
    setResult(null)
    setProgressMessages([])
    try {
      await streamResearch(submittedQuestion, (event) => {
        if (event.type === 'progress') {
          if (event.message) setProgressMessages((prev) => [...prev, event.message as string])
        } else if (event.type === 'done') {
          setResult({
            question: submittedQuestion,
            report: event.report ?? '',
            route: event.route ?? null,
            confidence_score: event.confidence_score ?? null,
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

  return (
    <div className="page">
      <Header backendUp={backendUp} />
      <AskCard
        question={question}
        onQuestionChange={setQuestion}
        onSubmit={handleSubmit}
        loading={loading}
      />
      {loading && <ProgressIndicator messages={progressMessages} />}
      {error && <ErrorBanner message={error} />}
      {result && <ResultCard result={result} />}
    </div>
  )
}

export default App
