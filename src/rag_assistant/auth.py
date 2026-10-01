"""API-key authentication, scopes, expiry and per-key limits.

Two ways to configure keys, and the simple one still works exactly as it did:

    API_KEYS=alice:sk-live-a1b2,bob:sk-live-c3d4

Every such key gets full scopes, no expiry and the global rate limit — fine for a demo, and
unchanged from before. What it cannot express is anything an actual deployment needs: a key
that only reads, a key that stops working in March, a key allowed more requests than the rest.
Those live in a JSON file named by `API_KEYS_FILE`:

    {"keys": [
      {"key": "sk-live-a1b2", "owner": "alice", "scopes": ["read", "write"],
       "expires_at": "2026-12-31T23:59:59Z", "rate_limit_rpm": 120},
      {"key": "sk-ro-c3d4", "owner": "reporting", "scopes": ["read"]}
    ]}

A file keeps secrets out of the process listing and lets a key be revoked or rotated by
editing one place, which an env var full of comma-separated secrets does not.

A key may also carry an identity -- `"user": "svc-reporting", "groups": ["finance"]` -- which
subjects it to document-level permissions (see ingestion/acl.py). A key with neither is a
tenant-wide key and reads every document its tenant owns, which is what every key could do
before permissions existed.

People sign in with single sign-on instead (see oidc.py). Both paths resolve to the same
`Principal`, so nothing downstream knows or cares which one a request used.

With neither set the API runs open -- every request is the "public" tenant -- which keeps
local development and the hosted demo frictionless. Setting either flips every data/LLM
endpoint to require `X-API-Key: <key>` (or `Authorization: Bearer <key>`).

The resolved owner rides a contextvar (mirroring tracing.py's trace_id) so anything downstream
-- conversation persistence, corpus scoping, rate-limit keying -- can read it without threading
a parameter through every call.
"""

import hashlib
import hmac
import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from rag_assistant.config import get_settings

logger = logging.getLogger(__name__)

PUBLIC_OWNER = "public"

READ = "read"
WRITE = "write"
# Operating the deployment rather than using it: rebuilding the index, switching generations,
# running connectors on demand. Deliberately not in ALL_SCOPES -- a key file entry has to ask
# for it by name, and a key from the plain API_KEYS variable never has it.
ADMIN = "admin"
ALL_SCOPES = frozenset({READ, WRITE})

owner_var: ContextVar[str] = ContextVar("api_owner", default=PUBLIC_OWNER)
# The authenticated key for the current request, for scope checks and audit logging. None
# when auth is disabled or the request authenticated with an SSO token.
api_key_var: ContextVar["ApiKey | None"] = ContextVar("api_key", default=None)
# Who is asking, however they authenticated. None when auth is disabled.
principal_var: ContextVar["Principal | None"] = ContextVar("principal", default=None)


@dataclass(frozen=True)
class ApiKey:
    key: str
    owner: str
    scopes: frozenset[str] = field(default=ALL_SCOPES)
    expires_at: datetime | None = None
    rate_limit_rpm: int | None = None
    # Optional identity. A key naming a user or groups is subject to document ACLs; one
    # naming neither reads its whole tenant (see the module docstring).
    user: str | None = None
    groups: frozenset[str] = field(default_factory=frozenset)

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        # `_parse_expiry` normalises what it reads, but a record built directly in code would
        # otherwise carry a naive datetime here -- and comparing naive to aware raises
        # TypeError, turning an expiry check into a 500 on every authenticated request.
        expires_at = (
            self.expires_at if self.expires_at.tzinfo else self.expires_at.replace(tzinfo=UTC)
        )
        return (now or datetime.now(UTC)) >= expires_at

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    @property
    def fingerprint(self) -> str:
        """Short stable hash, for logs and rate-limit buckets. Never the key itself: an audit
        trail that records secrets is a secret store nobody is guarding."""
        return hashlib.sha256(self.key.encode()).hexdigest()[:16]

    def principal(self) -> "Principal":
        return Principal(
            owner=self.owner,
            method="api_key",
            scopes=self.scopes,
            subject=self.user,
            groups=self.groups,
            bypass_acl=self.user is None and not self.groups,
            fingerprint=self.fingerprint,
            rate_limit_rpm=self.rate_limit_rpm,
        )


@dataclass(frozen=True)
class Principal:
    """An authenticated caller: which tenant they act in, who they are inside it, and what
    they may do.

    `principals()` is what document ACLs are checked against -- namespaced strings for the
    caller's user id, email, email domain and groups. It returns None for a caller that
    bypasses ACLs (a tenant-wide API key, an SSO member of an admin group), which every
    retrieval function reads as "no document-level restriction".
    """

    owner: str
    method: str
    scopes: frozenset[str] = field(default=ALL_SCOPES)
    subject: str | None = None
    email: str | None = None
    groups: frozenset[str] = field(default_factory=frozenset)
    bypass_acl: bool = False
    fingerprint: str = ""
    rate_limit_rpm: int | None = None

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def principals(self) -> frozenset[str] | None:
        if self.bypass_acl:
            return None
        from rag_assistant.ingestion.acl import DOMAIN_PREFIX, GROUP_PREFIX, USER_PREFIX

        tokens: set[str] = set()
        if self.subject:
            tokens.add(USER_PREFIX + self.subject)
        if self.email:
            email = self.email.strip().lower()
            tokens.add(USER_PREFIX + email)
            if "@" in email:
                tokens.add(DOMAIN_PREFIX + email.rsplit("@", 1)[1])
        tokens.update(GROUP_PREFIX + group for group in self.groups)
        return frozenset(tokens)

    @property
    def conversation_owner(self) -> str:
        """The key conversations and feedback are stored under.

        Per user when the caller has a user identity, per tenant otherwise. Once documents
        have per-user permissions, a transcript is as sensitive as the documents it quotes:
        a tenant-wide conversation list would show one user's answers -- drawn from documents
        only they may read -- to everyone else in the tenant. The tenant stays the prefix, so
        erasing a tenant still reaches every one of its users' conversations.
        """
        if self.subject:
            return f"{self.owner}{CONVERSATION_OWNER_SEPARATOR}{self.subject}"
        return self.owner


# Separates tenant from user in a conversation owner key. Owner labels are filesystem-safe
# by construction and can never contain it.
CONVERSATION_OWNER_SEPARATOR = "::"


def get_owner() -> str:
    return owner_var.get()


def get_api_key() -> "ApiKey | None":
    return api_key_var.get()


def get_principal() -> "Principal | None":
    return principal_var.get()


def get_principals() -> frozenset[str] | None:
    """The current caller's ACL principals, for retrieval.

    With auth disabled there is no identity at all, so the caller holds no principals: an
    open deployment serves every *unrestricted* document and no restricted one. Permissions
    are only as good as the identity behind them, and anonymous access cannot satisfy one.
    """
    principal = principal_var.get()
    if principal is None:
        return frozenset()
    return principal.principals()


def get_conversation_owner() -> str:
    principal = principal_var.get()
    return principal.conversation_owner if principal else owner_var.get()


def parse_api_keys(raw: str) -> dict[str, str]:
    """Parses API_KEYS into {key: owner_label}. Empty/blank input means auth is disabled.

    Kept as-is for the simple format; `load_api_keys` is the richer entry point.
    """
    keys: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            label, key = entry.split(":", 1)
            label, key = label.strip(), key.strip()
        else:
            label, key = "", entry
        if not key:
            continue
        keys[key] = label or f"key-{hashlib.sha256(key.encode()).hexdigest()[:8]}"
    return keys


def _parse_expiry(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # A naive timestamp is read as UTC rather than as local time: a key file is deployment
    # configuration that moves between machines, and "expires at 6pm" meaning something
    # different per host is a bug waiting for a timezone change.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _load_from_file(path_str: str) -> list[ApiKey]:
    path = Path(path_str)
    records: list[ApiKey] = []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        # Refusing to start would be defensible, but this runs on every settings refresh and
        # a malformed file would then take the service down rather than one credential.
        # Loading nothing means the file's keys stop working, which is loud enough.
        logger.error(
            "Could not read API_KEYS_FILE at %s; its keys will not work", path_str, exc_info=True
        )
        return records

    for entry in payload.get("keys", []):
        key = (entry.get("key") or "").strip()
        if not key:
            continue
        scopes = entry.get("scopes")
        user = (entry.get("user") or "").strip() or None
        groups = entry.get("groups") or []
        records.append(
            ApiKey(
                key=key,
                owner=(entry.get("owner") or "").strip()
                or f"key-{hashlib.sha256(key.encode()).hexdigest()[:8]}",
                scopes=frozenset(scopes) if scopes else ALL_SCOPES,
                expires_at=_parse_expiry(entry.get("expires_at")),
                rate_limit_rpm=entry.get("rate_limit_rpm"),
                user=user,
                groups=frozenset(str(g).strip() for g in groups if str(g).strip()),
            )
        )
    return records


@lru_cache
def _load_api_keys_cached(api_keys: str, keys_file: str, _file_mtime: float) -> tuple[ApiKey, ...]:
    records = [ApiKey(key=key, owner=owner) for key, owner in parse_api_keys(api_keys).items()]
    if keys_file:
        records.extend(_load_from_file(keys_file))
    return tuple(records)


def load_api_keys() -> tuple[ApiKey, ...]:
    """Every configured key, from both sources.

    Cached on its *inputs* rather than as a bare memo, which matters twice. It cannot go stale
    relative to settings -- a caller that changes API_KEYS and clears the settings cache gets
    fresh keys without having to know this cache exists. And because the file's mtime is part
    of the key, editing the key file revokes or rotates a credential on the next request
    instead of on the next restart, which is the difference between revocation being an
    operation and being an outage.
    """
    settings = get_settings()
    keys_file = str(settings.api_keys_file) if settings.api_keys_file else ""
    mtime = 0.0
    if keys_file:
        try:
            mtime = settings.api_keys_file.stat().st_mtime
        except OSError:
            # A missing file is reported by _load_from_file; a constant mtime here just means
            # the miss isn't re-read on every single request.
            mtime = 0.0
    return _load_api_keys_cached(settings.api_keys, keys_file, mtime)


def reset_api_key_cache() -> None:
    _load_api_keys_cached.cache_clear()


def auth_enabled() -> bool:
    return bool(load_api_keys()) or bool(get_settings().oidc_issuer)


def resolve_key(presented_key: str | None) -> ApiKey | None:
    """Maps a presented key to its record, or None if rejected.

    Comparison is constant-time and, critically, every configured key is compared even after a
    match. Returning early on the first hit leaks — through timing — how far down the list a
    key sits, which is a slow but real oracle for enumerating valid prefixes.

    Expired keys are rejected here rather than filtered at load time, so a key that expires
    while the process is running stops working without needing a restart.
    """
    configured = load_api_keys()
    if not configured:
        return None
    if not presented_key:
        return None

    matched: ApiKey | None = None
    for record in configured:
        if hmac.compare_digest(presented_key, record.key) and matched is None:
            matched = record
    if matched is None:
        return None
    if matched.is_expired():
        logger.warning(
            "rejected an expired API key",
            extra={"node": matched.owner, "route": matched.fingerprint},
        )
        return None
    return matched


def resolve_owner(presented_key: str | None) -> str | None:
    """Owner label for a presented credential, or None if rejected. With auth disabled every
    request (keyed or not) resolves to the public tenant."""
    if not auth_enabled():
        return PUBLIC_OWNER
    principal = resolve_principal(presented_key)
    return principal.owner if principal else None


class CredentialRejected(Exception):
    """A presented credential that is recognisably an SSO token but failed verification.
    Carries the reason for the audit log; the client only ever sees a generic 401."""


def resolve_principal(presented: str | None) -> Principal | None:
    """The caller behind a presented credential: an API key first, then an SSO token.

    API keys are compared first and always, so a deployment that turns on SSO keeps every
    existing key working. A value that matches no key is tried as a token only when it looks
    like one and SSO is configured; anything else is simply rejected. Raises
    `CredentialRejected` for a token that failed verification, so the audit log can say why.
    """
    if not presented:
        return None
    record = resolve_key(presented)
    if record is not None:
        return record.principal()
    settings = get_settings()
    if settings.oidc_issuer:
        from rag_assistant import oidc

        if oidc.looks_like_jwt(presented):
            try:
                return oidc.verify_token(presented)
            except oidc.TokenRejected as exc:
                raise CredentialRejected(str(exc)) from exc
    return None


def extract_key(headers: dict[bytes, bytes]) -> str | None:
    """Pulls the API key from raw ASGI headers: X-Api-Key first, Bearer token second."""
    api_key = headers.get(b"x-api-key")
    if api_key:
        return api_key.decode("latin-1").strip()
    authorization = headers.get(b"authorization")
    if authorization:
        value = authorization.decode("latin-1").strip()
        if value.lower().startswith("bearer "):
            return value[7:].strip()
    return None


# Anything that mutates the corpus or stored conversations needs `write`; reading and asking
# questions needs `read`. Listed most-specific-first and matched in order.
_WRITE_RULES: tuple[tuple[str, str], ...] = (
    ("POST", "/api/v1/ingest"),
    ("DELETE", "/api/v1/conversations"),
    # Removing indexed documents is a write, per document and per tenant alike. The tenant
    # rule was missing: `DELETE /api/v1/tenant/data` matched neither prefix above, fell
    # through to the READ default, and let a key issued read-only erase every document,
    # conversation and feedback row its tenant owned -- while the endpoint's own docstring
    # said a read-only key would get a 403. The default is the safe direction for a *read*
    # endpoint added without thought, and the reason each write endpoint has to be named
    # here; this is what happens when one is not.
    ("DELETE", "/api/v1/sources"),
    ("DELETE", "/api/v1/tenant/data"),
    # Changing who may read a document is a write to that document.
    ("PUT", "/api/v1/sources"),
    ("POST", "/api/v1/connectors"),
)
# Every method: reading index-generation state is as much an operator question as changing it.
_ADMIN_PREFIXES: tuple[str, ...] = ("/api/v1/admin",)


def required_scope(method: str, path: str) -> str:
    """The scope a request needs. Defaults to `read`, so a new endpoint is readable by every
    valid key rather than silently unreachable -- and a new *write* endpoint must be added
    here deliberately."""
    if path.startswith(_ADMIN_PREFIXES):
        return ADMIN
    for rule_method, prefix in _WRITE_RULES:
        if method == rule_method and path.startswith(prefix):
            return WRITE
    return READ


def rate_limit_for_identity(identity: str) -> int | None:
    """Per-key requests-per-minute override for a rate-limit bucket identity, or None.

    slowapi hands the limit provider the *bucket key*, which is a fingerprint rather than the
    secret, so the lookup goes through fingerprints too -- the limiter never sees a raw key.
    """
    if not identity.startswith("key:"):
        return None
    fingerprint = identity[4:]
    for record in load_api_keys():
        if record.fingerprint == fingerprint and record.rate_limit_rpm:
            return record.rate_limit_rpm
    return None


def audit(event: str, *, path: str, method: str, outcome: str) -> None:
    """One structured line per authentication decision.

    Records the owner and key fingerprint, never the key. Without this there is no answer to
    "which credential did that", which is the first question asked after a leak and the one
    a static env var full of keys has never been able to answer.
    """
    principal = get_principal()
    if principal is None:
        who = f"owner={PUBLIC_OWNER} key=none"
    elif principal.method == "api_key":
        who = f"owner={principal.owner} key={principal.fingerprint}"
    else:
        who = f"owner={principal.owner} user={principal.subject} via={principal.method}"
    logger.info(
        event,
        extra={"route": path, "node": f"{method} {who} outcome={outcome}"},
    )
