import { useEffect, useState, type FormEvent } from 'react'
import { checkAuth, storeApiKey } from '../api/client'
import { fetchAuthConfig, startSignIn, type AuthConfig } from '../api/sso'
import './AccessGate.css'

interface AccessGateProps {
  /** Called once the entered key has been verified against the backend. */
  onAuthorized: () => void
}

/** Shown instead of the app when the backend requires credentials and none (or stale ones)
 * are held. Offers single sign-on when the deployment has an identity provider configured,
 * and an API key field when it accepts keys -- both, when it accepts both. */
export function AccessGate({ onAuthorized }: AccessGateProps) {
  const [key, setKey] = useState('')
  const [checking, setChecking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [config, setConfig] = useState<AuthConfig | null>(null)

  useEffect(() => {
    let cancelled = false
    void fetchAuthConfig().then((loaded) => {
      if (!cancelled) setConfig(loaded)
    })
    return () => {
      cancelled = true
    }
  }, [])

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

  async function handleSignIn() {
    if (!config?.oidc) return
    setError(null)
    try {
      await startSignIn(config.oidc)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign-in could not be started.')
    }
  }

  const sso = config?.oidc ?? null
  // Until the config arrives, and for a backend too old to serve it, show the key field.
  const keys = config === null || config.api_keys || !sso

  return (
    <div className="access-gate">
      <div className="access-card">
        <h2>{sso && !keys ? 'Sign in required' : 'Access required'}</h2>
        <p>
          This deployment is protected.{' '}
          {sso ? 'Sign in with your company account' : 'Enter the API key you were given'}
          {sso && keys ? ', or enter an API key' : ''} to use the assistant.
        </p>
        {sso && (
          <button type="button" className="access-sso" onClick={() => void handleSignIn()}>
            Sign in with SSO
          </button>
        )}
        {keys && (
          <form onSubmit={handleSubmit}>
            <input
              type="password"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              placeholder="API key"
              aria-label="API key"
              autoFocus={!sso}
            />
            <button type="submit" disabled={checking || !key.trim()}>
              {checking ? 'Checking…' : 'Unlock'}
            </button>
          </form>
        )}
        {error && <p className="access-error">{error}</p>}
      </div>
    </div>
  )
}
