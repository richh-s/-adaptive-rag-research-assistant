import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { completeSignIn, getAccessToken, pkceChallenge, signOut } from './sso'

const PENDING = 'rag_assistant_sso_pending'
const TOKEN = 'rag_assistant_sso_token'

function locationWith(search: string): Location {
  return { search } as Location
}

describe('sso', () => {
  beforeEach(() => {
    sessionStorage.clear()
    vi.spyOn(window.history, 'replaceState').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('computes the PKCE challenge from RFC 7636 appendix B', async () => {
    expect(await pkceChallenge('dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk')).toBe(
      'E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM',
    )
  })

  it('drops a token that has expired, or is about to', () => {
    sessionStorage.setItem(TOKEN, JSON.stringify({ access_token: 'old', expires_at: 1_000 }))
    expect(getAccessToken(2_000)).toBeNull()
    expect(sessionStorage.getItem(TOKEN)).toBeNull()

    sessionStorage.setItem(TOKEN, JSON.stringify({ access_token: 'fresh', expires_at: 100_000 }))
    expect(getAccessToken(10_000)).toBe('fresh')
    signOut()
    expect(getAccessToken(10_000)).toBeNull()
  })

  it('ignores a redirect it did not start', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    expect(await completeSignIn(locationWith('?code=abc&state=xyz'))).toBe(false)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('refuses a state that does not match the login this tab started', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    sessionStorage.setItem(
      PENDING,
      JSON.stringify({
        state: 'expected',
        verifier: 'v',
        redirect_uri: 'https://app.example.com/',
        token_endpoint: 'https://idp.example.com/token',
        client_id: 'spa',
      }),
    )

    expect(await completeSignIn(locationWith('?code=abc&state=forged'))).toBe(false)
    expect(fetchMock).not.toHaveBeenCalled()
    // The pending login is consumed either way, so it cannot be replayed.
    expect(sessionStorage.getItem(PENDING)).toBeNull()
  })

  it('exchanges the code with the verifier and stores the access token', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ access_token: 'at-123', expires_in: 600 }),
    })
    vi.stubGlobal('fetch', fetchMock)
    sessionStorage.setItem(
      PENDING,
      JSON.stringify({
        state: 's1',
        verifier: 'the-verifier',
        redirect_uri: 'https://app.example.com/',
        token_endpoint: 'https://idp.example.com/token',
        client_id: 'spa',
      }),
    )

    expect(await completeSignIn(locationWith('?code=abc&state=s1'))).toBe(true)

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('https://idp.example.com/token')
    const body = new URLSearchParams(init.body as URLSearchParams)
    expect(body.get('code_verifier')).toBe('the-verifier')
    expect(body.get('grant_type')).toBe('authorization_code')
    expect(body.get('client_id')).toBe('spa')
    expect(getAccessToken()).toBe('at-123')
    expect(window.history.replaceState).toHaveBeenCalledWith(null, '', 'https://app.example.com/')
  })
})
