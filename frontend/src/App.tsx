import { useState, type FormEvent } from 'react'
import { useHealthStatus } from './hooks/useHealthStatus'
import { useResearchStream } from './hooks/useResearchStream'
import { Header } from './components/Header'
import { AskCard } from './components/AskCard'
import { ResultCard } from './components/ResultCard'
import { ResearchSummaryPanel } from './components/ResearchSummaryPanel'
import { ErrorBanner } from './components/ErrorBanner'
import { GraphVisualization } from './components/GraphVisualization'
import './App.css'

function App() {
  const [question, setQuestion] = useState('')
  const backendUp = useHealthStatus()
  const { loading, error, result, visits, submit } = useResearchStream()

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    void submit(question)
  }

  return (
    <div className="page">
      <Header backendUp={backendUp} />
      <div className="page-content">
        <AskCard
          question={question}
          onQuestionChange={setQuestion}
          onSubmit={handleSubmit}
          loading={loading}
        />
        {loading && <GraphVisualization visits={visits} loading={loading} />}
        {error && <ErrorBanner message={error} />}
        {result && <ResultCard result={result} />}
        {result?.summary && <ResearchSummaryPanel summary={result.summary} />}
      </div>
    </div>
  )
}

export default App
