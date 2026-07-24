import { useState, type FormEvent } from 'react'
import { useHealthStatus } from './hooks/useHealthStatus'
import { useResearchStream } from './hooks/useResearchStream'
import { Header } from './components/Header'
import { AskCard } from './components/AskCard'
import { ResultCard } from './components/ResultCard'
import { ResearchSummaryPanel } from './components/ResearchSummaryPanel'
import { ErrorBanner } from './components/ErrorBanner'
import { GraphVisualization } from './components/GraphVisualization'
import { CorpusManager } from './components/CorpusManager'
import './App.css'

function App() {
  const [question, setQuestion] = useState('')
  const [corpusManagerOpen, setCorpusManagerOpen] = useState(false)
  const backendUp = useHealthStatus()
  const { loading, error, result, visits, submit } = useResearchStream()

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    void submit(question)
  }

  return (
    <div className="page">
      <Header backendUp={backendUp} onManageCorpus={() => setCorpusManagerOpen(true)} />
      <CorpusManager open={corpusManagerOpen} onClose={() => setCorpusManagerOpen(false)} />
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
        {result?.summary && (
          <details className="tech-details">
            <summary>Show technical details</summary>
            <ResearchSummaryPanel summary={result.summary} />
          </details>
        )}
      </div>
    </div>
  )
}

export default App
