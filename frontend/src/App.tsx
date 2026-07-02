import { useState, type FormEvent } from 'react'
import { research, ResearchApiError, type ResearchResponse } from './api/client'
import { useHealthStatus } from './hooks/useHealthStatus'
import { Header } from './components/Header'
import { AskCard } from './components/AskCard'
import { ResultCard } from './components/ResultCard'
import { ErrorBanner } from './components/ErrorBanner'
import './App.css'

function App() {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ResearchResponse | null>(null)
  const backendUp = useHealthStatus()

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!question.trim() || loading) return

    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(await research(question.trim()))
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
      {error && <ErrorBanner message={error} />}
      {result && <ResultCard result={result} />}
    </div>
  )
}

export default App
