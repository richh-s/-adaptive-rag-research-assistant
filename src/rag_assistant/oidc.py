"""Single sign-on: verifying OpenID Connect access tokens from a company identity provider.

API keys identify a *tenant*. A company deploying this needs to know which *person* is asking
-- to scope their conversations, to apply document permissions, to answer "who asked that"
after the fact -- and it needs that identity to come from the directory it already runs
(Okta, Entra ID, Google Workspace, Auth0, Keycloak) rather than from a key file someone has to
keep in sync by hand.

The API is an OAuth resource server: the browser (or a machine client) obtains a token from
the identity provider and presents it as `Authorization: Bearer <jwt>`; this module verifies
it locally and maps its claims onto a `Principal`.

Verification is deliberately strict, because every relaxation here is a known way JWT
validation goes wrong in practice:

* **Signature** against the issuer's published keys (JWKS), fetched over HTTPS and cached.
  Only the configured asymmetric algorithms are accepted -- never `none`, never HS256, since a
  symmetric algorithm would let anyone holding the *public* key forge tokens.
* **Issuer** must match exactly, and **audience** must include this API. Without the audience
  check, a token the same provider minted for an unrelated application is honoured here;
  config.py refuses to start with an issuer and no audience for that reason.
* **Expiry** is required, not merely checked when present.
* **Tenant** comes from a configured claim. If that claim is configured and absent, the token
  is rejected rather than defaulted: silently filing a user under the default tenant would
  show them that tenant's documents.

Unknown signing keys trigger one JWKS refresh (providers rotate keys), rate-limited so a
stream of tokens with random `kid`s cannot turn every request into an outbound fetch.
"""

import hashlib
import json
import logging
import threading
import time
from functools import lru_cache

import httpx
import jwt

from rag_assistant.auth import READ, WRITE, Principal
from rag_assistant.config import get_settings
from rag_assistant.ingestion.ownership import safe_owner_dirname

logger = logging.getLogger(__name__)

_JWKS_TTL_SECONDS = 600.0
# The fastest an unknown `kid` may force a refetch.
_JWKS_MIN_REFRESH_SECONDS = 60.0
_HTTP_TIMEOUT_SECONDS = 5.0


class TokenRejected(Exception):
    """A token that failed verification. The message is for the audit log only; clients get
    a generic 401 so a probing caller learns nothing about which check failed."""


def looks_like_jwt(value: str) -> bool:
    parts = value.split(".")
    return len(parts) == 3 and all(parts[:2]) and len(value) > 40


class _JwksCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._keys: dict[str, jwt.PyJWK] = {}
        self._fetched_at = 0.0
        self._url: str | None = None

    def reset(self) -> None:
        with self._lock:
            self._keys = {}
            self._fetched_at = 0.0
            self._url = None

    def _fetch(self, url: str) -> None:
        response = httpx.get(url, timeout=_HTTP_TIMEOUT_SECONDS, follow_redirects=False)
        response.raise_for_status()
        keys: dict[str, jwt.PyJWK] = {}
        for raw in response.json().get("keys", []):
            # Signing keys only; an encryption key in the set is not a key tokens are signed
            # with, and a key whose algorithm PyJWT cannot load is skipped rather than fatal.
            if raw.get("use", "sig") != "sig":
                continue
            try:
                keys[raw.get("kid", "")] = jwt.PyJWK(raw)
            except Exception:
                logger.warning("Skipping unusable JWKS key kid=%r", raw.get("kid"))
        self._keys = keys
        self._fetched_at = time.monotonic()
        self._url = url

    def key_for(self, kid: str) -> jwt.PyJWK:
        url = jwks_url()
        with self._lock:
            now = time.monotonic()
            stale = self._url != url or now - self._fetched_at > _JWKS_TTL_SECONDS
            if stale:
                self._fetch(url)
            if kid not in self._keys and now - self._fetched_at > _JWKS_MIN_REFRESH_SECONDS:
                # The provider may have rotated keys since the last fetch.
                self._fetch(url)
            key = self._keys.get(kid)
        if key is None:
            raise TokenRejected(f"unknown signing key {kid!r}")
        return key


_jwks = _JwksCache()


@lru_cache(maxsize=4)
def _discover(issuer: str) -> dict:
    url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    response = httpx.get(url, timeout=_HTTP_TIMEOUT_SECONDS, follow_redirects=False)
    response.raise_for_status()
    document = response.json()
    # A discovery document naming a different issuer is either misconfiguration or a
    # substituted document, and trusting its JWKS URL would mean trusting its keys.
    if document.get("issuer", "").rstrip("/") != issuer.rstrip("/"):
        raise TokenRejected(
            f"discovery document issuer {document.get('issuer')!r} does not match {issuer!r}"
        )
    return document


def jwks_url() -> str:
    settings = get_settings()
    if settings.oidc_jwks_url:
        return settings.oidc_jwks_url
    return _discover(settings.oidc_issuer)["jwks_uri"]


def reset_oidc_cache() -> None:
    _jwks.reset()
    _discover.cache_clear()


def _claim(claims: dict, path: str):
    """A claim by name, or by dotted path for providers that nest them (Keycloak puts roles
    under `realm_access.roles`)."""
    value = claims
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _as_set(value) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        # `scope` is space-delimited by spec; some providers also emit comma-joined groups.
        return frozenset(part for part in value.replace(",", " ").split() if part)
    if isinstance(value, list | tuple | set | frozenset):
        return frozenset(str(v) for v in value if str(v).strip())
    return frozenset()


def verify_token(token: str) -> Principal:
    """Verifies `token` and maps its claims onto a Principal. Raises TokenRejected."""
    settings = get_settings()
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise TokenRejected(f"malformed token header: {exc}") from exc

    algorithms = sorted(settings.csv("oidc_algorithms"))
    if header.get("alg") not in algorithms:
        raise TokenRejected(f"algorithm {header.get('alg')!r} is not accepted")

    try:
        key = _jwks.key_for(header.get("kid", ""))
    except TokenRejected:
        raise
    except Exception as exc:
        # The identity provider being unreachable is not the caller's fault, but it is still
        # not a verified token. Logged loudly: every SSO user is locked out while it lasts.
        logger.error("Could not fetch the identity provider's signing keys", exc_info=True)
        raise TokenRejected(f"signing keys unavailable: {exc}") from exc

    try:
        claims = jwt.decode(
            token,
            key=key.key,
            algorithms=algorithms,
            audience=settings.oidc_audience,
            issuer=settings.oidc_issuer,
            leeway=settings.oidc_leeway_seconds,
            options={"require": ["exp", "iss", "aud"]},
        )
    except jwt.PyJWTError as exc:
        raise TokenRejected(f"invalid token: {exc}") from exc

    subject = _claim(claims, settings.oidc_user_claim)
    if not subject:
        raise TokenRejected(f"token has no {settings.oidc_user_claim!r} claim")

    if settings.oidc_tenant_claim:
        tenant = _claim(claims, settings.oidc_tenant_claim)
        if not tenant:
            raise TokenRejected(f"token has no {settings.oidc_tenant_claim!r} claim")
        owner = safe_owner_dirname(str(tenant))
    else:
        owner = safe_owner_dirname(settings.oidc_default_tenant)

    groups = _as_set(_claim(claims, settings.oidc_groups_claim))
    token_scopes = _as_set(claims.get("scope")) | _as_set(claims.get("scp"))
    admin = bool(groups & settings.csv("oidc_admin_groups"))
    can_write = (
        admin
        or bool(groups & settings.csv("oidc_write_groups"))
        or (bool(settings.oidc_write_scope) and settings.oidc_write_scope in token_scopes)
    )
    email = _claim(claims, settings.oidc_email_claim) if settings.oidc_email_claim else None

    return Principal(
        owner=owner,
        method="oidc",
        scopes=frozenset({READ, WRITE}) if can_write else frozenset({READ}),
        subject=str(subject),
        email=str(email) if email else None,
        groups=groups,
        bypass_acl=admin,
        # Stable per person across token refreshes, so rate-limit buckets and audit lines
        # follow the user rather than whichever token they happen to hold.
        fingerprint=hashlib.sha256(json.dumps([owner, str(subject)]).encode()).hexdigest()[:16],
    )


def public_client_config() -> dict | None:
    """What the web UI needs to start a sign-in: the issuer and its public client id. Never
    includes anything secret -- a browser-based client is public by definition, which is why
    it signs in with PKCE rather than a client secret."""
    settings = get_settings()
    if not (settings.oidc_issuer and settings.oidc_client_id):
        return None
    return {
        "issuer": settings.oidc_issuer,
        "client_id": settings.oidc_client_id,
        "scopes": settings.oidc_scopes,
        "audience": settings.oidc_audience,
    }
