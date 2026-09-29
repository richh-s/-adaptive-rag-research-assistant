"""A mounted directory -- an SMB/NFS share, a synced folder, a docs checkout -- as a source.

    {"name": "handbook", "type": "filesystem", "owner": "acme", "path": "/mnt/handbook",
     "include": ["**/*.pdf", "**/*.md"], "default_acl": {"groups": ["staff"]}}

Operating-system permissions are not read: they are not portable across the systems a share
is mounted from, and mapping POSIX modes or NTFS ACLs onto directory groups is guesswork.
So `default_acl` is required, and a file may narrow it with the same `<file>.acl.json`
sidecar the corpus itself uses (see ingestion/acl.py).
"""

import fnmatch
from collections.abc import Iterator
from pathlib import Path

from rag_assistant.connectors.base import (
    ConnectorConfig,
    ConnectorConfigError,
    RemoteDocument,
    supported_suffix,
)
from rag_assistant.ingestion.acl import ACL_SUFFIX, read_acl, sidecar_path


class FilesystemConnector:
    def __init__(self, config: ConnectorConfig, validate_only: bool = False, transport=None):
        self._config = config
        raw_path = config.options.get("path")
        if not raw_path:
            raise ConnectorConfigError(f"connector {config.name!r} needs a 'path'")
        if config.default_acl is None:
            raise ConnectorConfigError(
                f"connector {config.name!r}: filesystem connectors need a default_acl -- "
                'OS permissions are not read. Use "tenant" or {"groups": [...]}.'
            )
        self._root = Path(raw_path)
        self._include = list(config.options.get("include") or ["**/*"])

    def _matches(self, relative: str) -> bool:
        return any(
            fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch("/" + relative, pattern)
            for pattern in self._include
        )

    def list_documents(self) -> Iterator[RemoteDocument]:
        if not self._root.is_dir():
            # Raised rather than listing nothing: an unmounted share must not read as "every
            # document was deleted".
            raise FileNotFoundError(f"{self._root} is not a directory (is the share mounted?)")
        root = self._root.resolve()
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.endswith(ACL_SUFFIX) or path.name.startswith("."):
                continue
            # A symlink pointing outside the share would otherwise pull arbitrary files from
            # the host into a tenant's corpus.
            try:
                path.resolve().relative_to(root)
            except ValueError:
                continue
            relative = path.relative_to(root).as_posix()
            if not supported_suffix(path.suffix) or not self._matches(relative):
                continue
            stat = path.stat()
            acl = read_acl(path) if sidecar_path(path).exists() else self._config.default_acl
            yield RemoteDocument(
                id=relative,
                title=path.name,
                suffix=path.suffix.lower(),
                version=f"{stat.st_mtime_ns}:{stat.st_size}",
                acl=acl,
                url=None,
            )

    def fetch(self, document: RemoteDocument) -> bytes:
        return (self._root / document.id).read_bytes()
