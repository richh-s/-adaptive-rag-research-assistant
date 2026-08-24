import { useState, type FormEvent } from 'react'
import { checkAuth, storeApiKey } from '../api/client'
import './AccessGate.css'

interface AccessGateProps {
  /** Called once the entered key has been verified against the backend. */
  onAuthorized: () => void
}

/** Shown instead of the app when the backend requires an API key and none (or a stale one)
 * is stored. Verifies the key against /api/v1/auth/check before letting the user through. */
export function AccessGate({ onAuthorized }: AccessGateProps) {
  const [key, setKey] = useState('')
  const [checking, setChecking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const trimmed = key.trim()
    if (!trimmed || checking) return
    setChecking(true)
    setError(null)
    storeApiKey(trimmed)
    const status = await checkAuth()
    setChecking(false)
    if (status === 'unauthorized') {
      storeApiKey(null)
      setError('That key was not accepted — check it and try again.')
      return
    }
    onAuthorized()
  }

  return (
    <div className="access-gate">
      <div className="access-card">
        <h2>Access key required</h2>
        <p>
          This deployment is protected. Enter the API key you were given to use the assistant —
          it's remembered on this browser.
        </p>
        <form onSubmit={handleSubmit}>
          <input
            type="password"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder="API key"
            aria-label="API key"
            autoFocus
          />
          <button type="submit" disabled={checking || !key.trim()}>
            {checking ? 'Checking…' : 'Unlock'}
          </button>
        </form>
        {error && <p className="access-error">{error}</p>}
      </div>
    </div>
  )
}
