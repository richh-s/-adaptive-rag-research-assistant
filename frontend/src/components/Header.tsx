import './Header.css'

interface HeaderProps {
  backendUp: boolean | null
}

export function Header({ backendUp }: HeaderProps) {
  const statusClass = backendUp ? 'up' : backendUp === false ? 'down' : ''
  const statusLabel = backendUp === null ? 'checking API…' : backendUp ? 'API connected' : 'API unreachable'

  return (
    <header>
      <h1>Adaptive RAG Research Assistant</h1>
      <p className="subtitle">
        Ask a question. It autonomously routes to local retrieval, web search, or both, fuses and
        grades the results, and synthesizes a cited answer.
      </p>
      <span className={`status ${statusClass}`}>{statusLabel}</span>
    </header>
  )
}
