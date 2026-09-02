"""Corpus ownership: which tenant a document belongs to, and who is allowed to retrieve it.

Tenancy used to stop at conversations -- transcripts were scoped per API key, but the
knowledge base was one shared Chroma collection, so a document uploaded by one tenant was
retrievable by every other one. For a single-user demo that's invisible; for a multi-user
deployment it's a data-isolation bug, not a missing feature.

Ownership is encoded in the corpus *layout* rather than in a sidecar table, because the
layout is the thing that survives. A manifest can be deleted, reset by a fresh deploy, or
fall out of sync with the files on disk, and every one of those failures defaults documents
to visible-to-everyone -- the wrong direction to fail in. A path cannot drift from itself:

    data/corpus/anthropic.md            -> owner "public"  (baseline corpus, everyone sees it)
    data/corpus/_t/alice/report.md      -> owner "alice"   (only alice retrieves it)

Flat files stay public, which is what makes this backward compatible: the corpus baked into
the Docker image needs no migration and keeps its existing `source` keys, so citations,
the eval dataset's `expected_sources`, and any manifest written before tenancy all still
line up.
"""

import re
from pathlib import Path

from rag_assistant.auth import PUBLIC_OWNER

# Tenant files live under one reserved directory rather than directly under the corpus root,
# so a tenant label can never collide with (or shadow) a baseline corpus filename, and
# `load_documents` can tell "owned subtree" from "someone dropped a folder in the corpus"
# without guessing.
TENANT_DIR = "_t"

_UNSAFE_OWNER_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_owner_dirname(owner: str) -> str:
    """Filesystem-safe directory name for a tenant.

    Owner labels come from the operator-configured API_KEYS setting rather than from request
    input, so this is defence in depth rather than the primary control -- but a label
    containing `../` would otherwise write outside the corpus, and that is too cheap to
    prevent to leave to trust in a config file.
    """
    cleaned = _UNSAFE_OWNER_CHARS_RE.sub("_", owner).strip("_")
    return cleaned or PUBLIC_OWNER


def owner_corpus_dir(corpus_dir: Path, owner: str) -> Path:
    """Where a tenant's uploads are written. The public tenant keeps writing flat into the
    corpus root, so an open demo's on-disk layout is byte-for-byte what it was before
    tenancy existed."""
    if owner == PUBLIC_OWNER:
        return corpus_dir
    return corpus_dir / TENANT_DIR / safe_owner_dirname(owner)


def owner_of_relative_path(relative_path: Path) -> str:
    """The owner implied by a corpus-relative path -- the inverse of `owner_corpus_dir`."""
    parts = relative_path.parts
    if len(parts) >= 3 and parts[0] == TENANT_DIR:
        return parts[1]
    return PUBLIC_OWNER


def visible_owners(owner: str) -> list[str]:
    """The owner values a tenant may retrieve: their own documents plus the shared baseline
    corpus. Returned as a list because it is passed straight into Chroma's `$in` filter."""
    if owner == PUBLIC_OWNER:
        return [PUBLIC_OWNER]
    return [owner, PUBLIC_OWNER]


def display_source(source: str) -> str:
    """The filename to show a user in a citation.

    Sources are corpus-relative paths so they stay unique across tenants, but `_t/alice/
    q3-report.md` is noise in a citation list -- the reader knows which tenant they are.
    """
    return Path(source).name
