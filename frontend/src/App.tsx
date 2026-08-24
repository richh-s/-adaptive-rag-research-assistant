import { useEffect, useRef, useState, type FormEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import { checkAuth, type AuthStatus } from './api/client'
import { useHealthStatus } from './hooks/useHealthStatus'
import { useResearchStream } from './hooks/useResearchStream'
import { AccessGate } from './components/AccessGate'
import { Header } from './components/Header'
import { AskCard } from './components/AskCard'
import { ResultCard } from './components/ResultCard'
import { ResearchSummaryPanel } from './components/ResearchSummaryPanel'
import { ErrorBanner } from './components/ErrorBanner'
import { GraphVisualization } from './components/GraphVisualization'
import { CorpusManager } from './components/CorpusManager'
import { ConversationsPanel } from './components/ConversationsPanel'
import './App.css'

function App() {
  const [question, setQuestion] = useState('')
  const [corpusManagerOpen, setCorpusManagerOpen] = useState(false)
  const [conversationsOpen, setConversationsOpen] = useState(false)
  const [authStatus, setAuthStatus] = useState<AuthStatus | 'checking'>('checking')
  const backendUp = useHealthStatus()

  useEffect(() => {
    let cancelled = false
    void checkAuth().then((status) => {
      if (!cancelled) setAuthStatus(status)
    })
    return () => {
      cancelled = true
    }
  }, [])
  const {
    loading,
    error,
    turns,
    conversationId,
    pendingQuestion,
    streamingAnswer,
    visits,
    submit,
    stop,
    retry,
    openConversation,
    reset,
  } = useResearchStream()
  const threadEndRef = useRef<HTMLDivElement>(null)

  // Keep the latest activity in view: scroll when a turn completes, a question starts,
  // or the answer begins streaming in (once -- not on every token).
  const streamingStarted = streamingAnswer.length > 0
  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns.length, pendingQuestion, streamingStarted])

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    void submit(question)
    setQuestion('')
  }

  const hasConversation = turns.length > 0 || pendingQuestion !== null

  if (authStatus === 'checking') {
    return <div className="page" />
  }
  if (authStatus === 'unauthorized') {
    return (
      <div className="page">
        <AccessGate onAuthorized={() => setAuthStatus('authorized')} />
      </div>
    )
  }

  return (
    <div className="page">
      <Header
        backendUp={backendUp}
        onManageCorpus={() => setCorpusManagerOpen(true)}
        onShowConversations={() => setConversationsOpen(true)}
      />
      <CorpusManager open={corpusManagerOpen} onClose={() => setCorpusManagerOpen(false)} />
      <ConversationsPanel
        open={conversationsOpen}
        onClose={() => setConversationsOpen(false)}
        onOpenConversation={(id) => void openConversation(id)}
        activeConversationId={conversationId}
        onDeleted={(id) => {
          if (id === conversationId) reset()
        }}
      />
      <div className="page-content">
        {hasConversation && (
          <div className="chat-thread">
            {turns.map((turn) => (
              <div key={turn.id} className="chat-turn">
                <div className="chat-question">{turn.question}</div>
                <ResultCard result={turn.result} />
                {turn.result.summary && (
                  <details className="tech-details">
                    <summary>Show technical details</summary>
                    <ResearchSummaryPanel summary={turn.result.summary} />
                  </details>
                )}
              </div>
            ))}
            {pendingQuestion !== null && (
              <div className="chat-turn">
                <div className="chat-question">{pendingQuestion}</div>
                {loading && !streamingAnswer && (
                  <GraphVisualization visits={visits} loading={loading} />
                )}
                {streamingAnswer && (
                  <div className="result streaming-answer" aria-live="polite">
                    <ReactMarkdown>{streamingAnswer}</ReactMarkdown>
                  </div>
                )}
                {loading && (
                  <button type="button" className="stop-button" onClick={stop}>
                    ■ Stop
                  </button>
                )}
              </div>
            )}
            <div ref={threadEndRef} aria-hidden="true" />
          </div>
        )}
        {error && <ErrorBanner message={error} onRetry={retry ?? undefined} />}
        <AskCard
          question={question}
          onQuestionChange={setQuestion}
          onSubmit={handleSubmit}
          loading={loading}
          followUp={turns.length > 0}
        />
        {turns.length > 0 && !loading && (
          <button type="button" className="new-conversation" onClick={reset}>
            Start a new conversation
          </button>
        )}
        {turns.length > 0 && !loading && (
          <span className="saved-note">
            Saved automatically — find it under “Conversations” in the header.
          </span>
        )}
      </div>
    </div>
  )
}

export default App
