import './Header.css'

interface HeaderProps {
  backendUp: boolean | null
}

function LogoMark() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="5" cy="6" r="2.4" fill="currentColor" />
      <circle cx="19" cy="6" r="2.4" fill="currentColor" />
      <circle cx="12" cy="18" r="2.6" fill="currentColor" />
      <path
        d="M6.8 7.6L11 16M17.2 7.6L13 16M7 6H17"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  )
}

export function Header({ backendUp }: HeaderProps) {
  const statusClass = backendUp ? 'up' : backendUp === false ? 'down' : ''
  const statusLabel = backendUp === null ? 'Checking API…' : backendUp ? 'API connected' : 'API unreachable'

  return (
    <div className="site-header">
      <nav className="nav-bar">
        <div className="nav-brand">
          <span className="nav-logo">
            <LogoMark />
          </span>
          <span className="nav-name">Adaptive RAG</span>
        </div>
        <span className={`status ${statusClass}`}>{statusLabel}</span>
      </nav>

      <div className="hero">
        <span className="eyebrow">Self-routing · multi-source · cited</span>
        <h1>Ask a research question</h1>
        <p className="subtitle">
          It decides whether to search a local knowledge base, the web, or both — fuses and grades
          the results, then synthesizes a cited, transparent answer.
        </p>
      </div>
    </div>
  )
}
