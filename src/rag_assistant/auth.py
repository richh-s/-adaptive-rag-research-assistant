"""API-key authentication.

Keys are configured via the API_KEYS setting as a comma-separated list; each entry is either
`label:key` (label becomes the tenant/owner id) or a bare `key` (owner derived from the key's
hash). With API_KEYS unset the API runs open -- every request is the "public" tenant -- which
keeps local development and the hosted demo frictionless; setting even one key flips every
data/LLM endpoint to require `X-API-Key: <key>` (or `Authorization: Bearer <key>`).

The resolved owner rides a contextvar (mirroring tracing.py's trace_id) so anything downstream
-- conversation persistence, rate-limit keying -- can read it without threading a parameter
through every call.
"""

import hashlib
import hmac
from contextvars import ContextVar

from rag_assistant.config import get_settings

PUBLIC_OWNER = "public"

owner_var: ContextVar[str] = ContextVar("api_owner", default=PUBLIC_OWNER)


def get_owner() -> str:
    return owner_var.get()


def parse_api_keys(raw: str) -> dict[str, str]:
    """Parses API_KEYS into {key: owner_label}. Empty/blank input means auth is disabled."""
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


def auth_enabled() -> bool:
    return bool(parse_api_keys(get_settings().api_keys))


def resolve_owner(presented_key: str | None) -> str | None:
    """Maps a presented key to its owner label, or None if rejected. With auth disabled every
    request (keyed or not) resolves to the public tenant. Comparison is constant-time --
    string equality on secrets leaks timing."""
    configured = parse_api_keys(get_settings().api_keys)
    if not configured:
        return PUBLIC_OWNER
    if not presented_key:
        return None
    for key, owner in configured.items():
        if hmac.compare_digest(presented_key, key):
            return owner
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
