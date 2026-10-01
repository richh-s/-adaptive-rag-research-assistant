"""The connector contract, and how connectors are configured.

A connector answers two questions about an external system: *what documents are there, at
what version, readable by whom* (`list_documents`), and *what are this document's bytes*
(`fetch`). Everything else -- deciding what changed, writing files, deleting what vanished,
guarding against a listing that came back suspiciously empty, re-indexing -- is the sync
engine's job (see sync.py), written once rather than once per source system.

Configuration lives in the JSON file named by CONNECTORS_FILE:

    {"connectors": [
      {"name": "eng-wiki", "type": "confluence", "owner": "acme",
       "interval_minutes": 60,
       "base_url": "https://acme.atlassian.net/wiki", "space_key": "ENG",
       "email_env": "CONFLUENCE_EMAIL", "token_env": "CONFLUENCE_API_TOKEN",
       "default_acl": {"groups": ["engineering"]}},
      {"name": "policies", "type": "google_drive", "owner": "acme",
       "folder_id": "1AbC...", "credentials_file_env": "DRIVE_SERVICE_ACCOUNT_FILE"},
      {"name": "handbook", "type": "filesystem", "owner": "public",
       "path": "/mnt/handbook", "default_acl": "tenant"}
    ]}

Secrets are never in the file. Each connector names the *environment variables* that hold
them (`token_env`, `credentials_file_env`), so the file can live in version control and a
credential is rotated where every other secret in the deployment is.

`default_acl` is the permission for documents the source system does not restrict itself.
It is required wherever the connector cannot see the source's full permission model --
Confluence page restrictions say nothing about who may view the *space*, and a file share's
OS permissions are not portable -- because the silent alternative is to publish every such
document to the whole tenant. `"tenant"` says that is intended; a `{"users": [...],
"groups": [...]}` object narrows it.
"""

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rag_assistant.config import get_settings
from rag_assistant.ingestion.acl import OPEN, DocumentAcl
from rag_assistant.ingestion.loaders import SUPPORTED_SUFFIXES
from rag_assistant.ingestion.ownership import safe_owner_dirname

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


class ConnectorConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RemoteDocument:
    """One document as the source system describes it, before any bytes are fetched.

    `version` is whatever the source uses to say "this changed" -- a Confluence version
    number, a Drive revision, a file's mtime and size. The sync engine fetches only when it
    differs from the last sync's, which is what makes a sync of an unchanged 10,000-page space
    a listing rather than 10,000 downloads.
    """

    id: str
    title: str
    suffix: str
    version: str
    acl: DocumentAcl = OPEN
    url: str | None = None


class Connector(Protocol):
    def list_documents(self) -> Iterator[RemoteDocument]: ...

    def fetch(self, document: RemoteDocument) -> bytes: ...


@dataclass(frozen=True)
class ConnectorConfig:
    name: str
    type: str
    owner: str
    interval_minutes: int = 60
    default_acl: DocumentAcl | None = None
    options: dict = field(default_factory=dict)

    def secret(self, key: str, required: bool = True) -> str:
        """The value of the environment variable the config names under `key`."""
        import os

        variable = self.options.get(key)
        if not variable:
            if required:
                raise ConnectorConfigError(f"connector {self.name!r} needs {key!r}")
            return ""
        value = os.environ.get(variable, "")
        if required and not value:
            raise ConnectorConfigError(
                f"connector {self.name!r}: environment variable {variable!r} is not set"
            )
        return value


def _parse_default_acl(raw, name: str) -> DocumentAcl | None:
    if raw is None:
        return None
    if raw == "tenant":
        return OPEN
    if isinstance(raw, dict):
        acl = DocumentAcl.from_json(raw)
        if not acl.restricted:
            raise ConnectorConfigError(
                f"connector {name!r}: an empty default_acl object would publish to the whole "
                'tenant; say "tenant" if that is intended'
            )
        return acl
    raise ConnectorConfigError(f'connector {name!r}: default_acl must be "tenant" or an object')


def parse_config(raw: dict) -> ConnectorConfig:
    name = str(raw.get("name", ""))
    if not _NAME_RE.match(name):
        raise ConnectorConfigError(
            f"connector name {name!r} must be lowercase letters, digits, '-' or '_'"
        )
    kind = str(raw.get("type", ""))
    owner = str(raw.get("owner", "")).strip()
    if not owner:
        raise ConnectorConfigError(f"connector {name!r} needs an owner (the tenant it syncs into)")
    interval = int(raw.get("interval_minutes", 60))
    if interval < 1:
        raise ConnectorConfigError(f"connector {name!r}: interval_minutes must be at least 1")
    reserved = {"name", "type", "owner", "interval_minutes", "default_acl"}
    return ConnectorConfig(
        name=name,
        type=kind,
        owner=safe_owner_dirname(owner),
        interval_minutes=interval,
        default_acl=_parse_default_acl(raw.get("default_acl"), name),
        options={k: v for k, v in raw.items() if k not in reserved},
    )


def load_connector_configs(path: Path | None = None) -> list[ConnectorConfig]:
    """Every configured connector, validated. A malformed file raises rather than loading
    the parts that parse: a connector that silently vanished from the list would stop syncing,
    and its documents would stay in the corpus, frozen at whatever they were."""
    path = path or get_settings().connectors_file
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    configs = [parse_config(raw) for raw in payload.get("connectors", [])]
    names = [c.name for c in configs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ConnectorConfigError(f"duplicate connector names: {sorted(duplicates)}")
    for config in configs:
        # Constructed once here so a missing option or unknown type fails at load, in front
        # of the operator, rather than at 3am inside the scheduler.
        build_connector(config, validate_only=True)
    return configs


def build_connector(config: ConnectorConfig, validate_only: bool = False, transport=None):
    from rag_assistant.connectors.confluence import ConfluenceConnector
    from rag_assistant.connectors.filesystem import FilesystemConnector
    from rag_assistant.connectors.google_drive import GoogleDriveConnector

    types = {
        "confluence": ConfluenceConnector,
        "google_drive": GoogleDriveConnector,
        "filesystem": FilesystemConnector,
    }
    if config.type not in types:
        raise ConnectorConfigError(
            f"connector {config.name!r}: unknown type {config.type!r} (known: {sorted(types)})"
        )
    return types[config.type](config, validate_only=validate_only, transport=transport)


def supported_suffix(suffix: str) -> bool:
    return suffix.lower() in SUPPORTED_SUFFIXES
