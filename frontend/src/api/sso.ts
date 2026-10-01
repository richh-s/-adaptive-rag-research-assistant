// Single sign-on for the web UI: OpenID Connect authorization code flow with PKCE.
//
// A browser app cannot keep a client secret, so it is a *public* client and proves it started
// the login it is finishing with PKCE instead: a random verifier kept in this tab, and its
// SHA-256 sent to the identity provider up front. No library -- the flow is three HTTP
// exchanges, and a dependency that handles tokens is a dependency worth not having.
//
// The access token is kept in sessionStorage, not localStorage: it dies with the tab and is not
// shared across every tab of the origin, which bounds what an XSS bug could take. There is no
// refresh token either. When the token expires the API answers 401 and the access gate asks
// the user to sign in again, which the identity provider's own session usually makes a single
// redirect with no password prompt.

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

const TOKEN_STORAGE = 'rag_assistant_sso_token'
const PENDING_STORAGE = 'rag_assistant_sso_pending'
// Treated as expired a little early, so a request never leaves with a token that lapses in
// flight.
const EXPIRY_SKEW_MS = 30_000

export interface OidcClientConfig {
  issuer: string
  client_id: string
  scopes: string
  audience: string
}

export interface AuthConfig {
  auth_required: boolean
  api_keys: boolean
  oidc: OidcClientConfig | null
}

interface StoredToken {
  access_token: string
  expires_at: number
}

interface PendingLogin {
  state: string
  verifier: string
  redirect_uri: string
  token_endpoint: string
  client_id: string
}

export async function fetchAuthConfig(): Promise<AuthConfig | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/config`)
    if (!response.ok) return null
    return (await response.json()) as AuthConfig
  } catch {
    return null
  }
}

function storage(): Storage | null {
  try {
    return sessionStorage
  } catch {
    return null
  }
}

export function getAccessToken(now: number = Date.now()): string | null {
  const raw = storage()?.getItem(TOKEN_STORAGE)
  if (!raw) return null
  try {
    const token = JSON.parse(raw) as StoredToken
    if (token.expires_at - EXPIRY_SKEW_MS <= now) {
      storage()?.removeItem(TOKEN_STORAGE)
      return null
    }
    return token.access_token
  } catch {
    storage()?.removeItem(TOKEN_STORAGE)
    return null
  }
}

export function signOut(): void {
  storage()?.removeItem(TOKEN_STORAGE)
}

function base64Url(bytes: Uint8Array): string {
  let binary = ''
  for (const byte of bytes) binary += String.fromCharCode(byte)
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

function randomString(byteLength = 32): string {
  const bytes = new Uint8Array(byteLength)
  crypto.getRandomValues(bytes)
  return base64Url(bytes)
}

export async function pkceChallenge(verifier: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier))
  return base64Url(new Uint8Array(digest))
}

async function discover(issuer: string): Promise<{ authorization_endpoint: string; token_endpoint: string }> {
  const response = await fetch(`${issuer.replace(/\/$/, '')}/.well-known/openid-configuration`)
  if (!response.ok) throw new Error(`Could not reach the identity provider (${response.status})`)
  return response.json()
}

/** Redirects the browser to the identity provider. Resolves only if the redirect fails. */
export async function startSignIn(config: OidcClientConfig): Promise<void> {
  const endpoints = await discover(config.issuer)
  const verifier = randomString()
  const state = randomString(16)
  const redirectUri = `${window.location.origin}${window.location.pathname}`
  const pending: PendingLogin = {
    state,
    verifier,
    redirect_uri: redirectUri,
    token_endpoint: endpoints.token_endpoint,
    client_id: config.client_id,
  }
  storage()?.setItem(PENDING_STORAGE, JSON.stringify(pending))

  const params = new URLSearchParams({
    response_type: 'code',
    client_id: config.client_id,
    redirect_uri: redirectUri,
    scope: config.scopes,
    state,
    code_challenge: await pkceChallenge(verifier),
    code_challenge_method: 'S256',
  })
  // Auth0 issues an API-scoped access token only when asked for the audience; providers that
  // do not know the parameter ignore it.
  if (config.audience) params.set('audience', config.audience)
  window.location.assign(`${endpoints.authorization_endpoint}?${params.toString()}`)
}

/** Finishes a sign-in if this page load is the identity provider's redirect back. Returns
 * true when a token was obtained. Safe to call on every load: without a pending login and a
 * matching `state` it does nothing, which is also what stops a crafted link from planting
 * someone else's authorization code in this tab. */
export async function completeSignIn(location: Location = window.location): Promise<boolean> {
  const params = new URLSearchParams(location.search)
  const code = params.get('code')
  const returnedState = params.get('state')
  const rawPending = storage()?.getItem(PENDING_STORAGE)
  if (!code || !returnedState || !rawPending) return false
  storage()?.removeItem(PENDING_STORAGE)
  const pending = JSON.parse(rawPending) as PendingLogin
  // The code and state are removed from the address bar either way, so a reload or a copied
  // URL never replays them.
  window.history.replaceState(null, '', pending.redirect_uri)
  if (pending.state !== returnedState) return false

  const response = await fetch(pending.token_endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code,
      redirect_uri: pending.redirect_uri,
      client_id: pending.client_id,
      code_verifier: pending.verifier,
    }),
  })
  if (!response.ok) return false
  const body = (await response.json()) as { access_token?: string; expires_in?: number }
  if (!body.access_token) return false
  const token: StoredToken = {
    access_token: body.access_token,
    expires_at: Date.now() + (body.expires_in ?? 3600) * 1000,
  }
  storage()?.setItem(TOKEN_STORAGE, JSON.stringify(token))
  return true
}
