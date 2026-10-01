"""Document-level permissions: which people inside a tenant may retrieve a document.

Tenancy answers "which organisation owns this"; it cannot say "only finance may read the
board pack". Every user of a tenant used to see every document the tenant indexed, which is
fine for a team wiki and wrong for almost everything else a company would put in front of a
RAG system. This module adds the second boundary.

The model is deliberately small:

* A document either has **no ACL** -- visible to everyone in its tenant, exactly as before --
  or an ACL listing the *principals* allowed to read it.
* A principal is a namespaced string: ``user:<id>``, ``group:<name>`` or ``domain:<name>``.
  The caller's own principals come from their identity (see auth.py): their user id, the
  groups their SSO token carries, and their email domain.
* A caller may read a restricted document when the two sets intersect. There is no deny rule
  and no inheritance; a source system with richer semantics is flattened into this shape by
  its connector, and flattened *conservatively* -- see ``NOBODY``.

Where an ACL lives, and why there: in a sidecar file next to the document,

    data/corpus/_t/acme/board-pack.pdf
    data/corpus/_t/acme/board-pack.pdf.acl.json   {"users": [...], "groups": [...]}

for the same reason ownership is encoded in the path (see ownership.py). The manifest can be
deleted, reset by a fresh deploy or rebuilt from scratch by a re-index, and a permission that
only lived there would silently revert to "visible to the whole tenant" -- the wrong
direction to fail in. A sidecar travels with the file through backups, restores and
re-indexes, and a document whose sidecar exists but cannot be parsed is treated as readable
by nobody rather than by everybody.

At index time the ACL is copied onto every chunk (``acl_restricted`` plus an ``acl`` list), so
retrieval filters on it inside the vector search itself -- post-filtering would silently
shrink k, the same argument vector_store.py makes for the tenant filter.
"""

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

ACL_SUFFIX = ".acl.json"

USER_PREFIX = "user:"
GROUP_PREFIX = "group:"
DOMAIN_PREFIX = "domain:"

# A principal no identity can ever hold. An ACL of just this makes a document readable only
# by callers who bypass ACLs entirely (tenant admins), which is the fail-closed outcome for a
# sidecar that cannot be read or a source-system permission that cannot be represented.
NOBODY = "nobody:"

# Chunk metadata keys. `acl` is an array, which Chroma filters with `$contains` and Postgres
# with `?|`; `acl_restricted` is what lets an unrestricted chunk skip the array test, and it
# is compared with `$ne: True` so chunks indexed before ACLs existed -- which lack the key
# entirely -- stay visible to their tenant rather than disappearing.
META_RESTRICTED = "acl_restricted"
META_ACL = "acl"


@dataclass(frozen=True)
class DocumentAcl:
    """Who may read one document. Empty means "everyone in the tenant"."""

    users: frozenset[str] = field(default_factory=frozenset)
    groups: frozenset[str] = field(default_factory=frozenset)
    domains: frozenset[str] = field(default_factory=frozenset)
    # Set when the source said "restricted" but named nobody representable -- a sidecar that
    # would not parse, or a source-system rule this model cannot express. Kept distinct from
    # an empty ACL because the two mean opposite things.
    deny_all: bool = False

    @property
    def restricted(self) -> bool:
        return self.deny_all or bool(self.users or self.groups or self.domains)

    def principals(self) -> list[str]:
        """The namespaced principal strings stored on chunks. Sorted so the fingerprint and
        the stored metadata are stable across runs."""
        if self.deny_all:
            return [NOBODY]
        tokens = (
            [USER_PREFIX + u for u in self.users]
            + [GROUP_PREFIX + g for g in self.groups]
            + [DOMAIN_PREFIX + d for d in self.domains]
        )
        return sorted(tokens) or [NOBODY]

    def fingerprint(self) -> str:
        """Recorded in the manifest, so a permission change is detected without re-reading
        the vector store -- and applied without re-embedding (see build_index)."""
        if not self.restricted:
            return "open"
        return hashlib.sha256("\n".join(self.principals()).encode()).hexdigest()[:16]

    def chunk_metadata(self) -> dict:
        # Always a non-empty list: Chroma rejects empty arrays, and an unrestricted chunk's
        # list is never consulted because `acl_restricted` short-circuits the filter.
        return {META_RESTRICTED: self.restricted, META_ACL: self.principals()}

    def to_json(self) -> dict:
        payload: dict = {
            "users": sorted(self.users),
            "groups": sorted(self.groups),
        }
        if self.domains:
            payload["domains"] = sorted(self.domains)
        if self.deny_all:
            payload["deny_all"] = True
        return payload

    @classmethod
    def from_json(cls, payload: dict) -> "DocumentAcl":
        return cls(
            users=_clean(payload.get("users")),
            groups=_clean(payload.get("groups")),
            domains=_clean(payload.get("domains")),
            deny_all=bool(payload.get("deny_all", False)),
        )


OPEN = DocumentAcl()


def _clean(values: Iterable[str] | None) -> frozenset[str]:
    if not values:
        return frozenset()
    if isinstance(values, str):
        values = values.split(",")
    return frozenset(v.strip() for v in values if isinstance(v, str) and v.strip())


def parse_acl_fields(users: str | None, groups: str | None) -> DocumentAcl:
    """ACL from the comma-separated form fields the upload endpoint accepts."""
    return DocumentAcl(users=_clean(users), groups=_clean(groups))


def sidecar_path(document_path: Path) -> Path:
    return document_path.with_name(document_path.name + ACL_SUFFIX)


def is_sidecar(path: Path) -> bool:
    return path.name.endswith(ACL_SUFFIX)


def read_acl(document_path: Path) -> DocumentAcl:
    """The ACL for one corpus file. No sidecar means unrestricted; an unreadable one means
    readable by nobody, never by everybody."""
    path = sidecar_path(document_path)
    if not path.exists():
        return OPEN
    try:
        return DocumentAcl.from_json(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        logger.error(
            "Unreadable ACL sidecar %s; treating %s as restricted to nobody",
            path,
            document_path.name,
            exc_info=True,
        )
        return DocumentAcl(deny_all=True)


def write_acl(document_path: Path, acl: DocumentAcl) -> None:
    """Writes (or, for an unrestricted ACL, removes) the sidecar. Atomic rename, for the same
    reason the manifest uses one: a half-written sidecar is read as deny-all."""
    path = sidecar_path(document_path)
    if not acl.restricted:
        path.unlink(missing_ok=True)
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(acl.to_json(), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def principals_can_read(metadata: dict, principals: frozenset[str] | None) -> bool:
    """The Python form of the ACL predicate, for the in-memory keyword index and for anything
    filtering manifest entries. `principals=None` means the caller bypasses ACLs (a tenant
    admin, or an internal caller such as the eval harness)."""
    if principals is None:
        return True
    if not metadata.get(META_RESTRICTED):
        return True
    allowed = metadata.get(META_ACL) or []
    return any(token in principals for token in allowed)


def entry_readable(entry: dict, principals: frozenset[str] | None) -> bool:
    """`principals_can_read` for a manifest entry, which records the ACL as its JSON form."""
    if principals is None:
        return True
    raw = entry.get("acl")
    if not raw:
        return True
    return principals_can_read(DocumentAcl.from_json(raw).chunk_metadata(), principals)
