"""Confluence (Cloud or Data Center) pages as a source.

    {"name": "eng-wiki", "type": "confluence", "owner": "acme",
     "base_url": "https://acme.atlassian.net/wiki", "space_key": "ENG",
     "email_env": "CONFLUENCE_EMAIL", "token_env": "CONFLUENCE_API_TOKEN",
     "default_acl": {"groups": ["engineering"]}}

Cloud authenticates with an account email and API token (basic auth); Data Center with a
personal access token alone (omit `email_env`, and it is sent as a bearer token).

**Permissions.** Confluence has two layers, and the REST API exposes only one of them to an
ordinary token. *Page restrictions* are read here, for the page and for every ancestor,
because a restriction on a parent page applies to all of its children. *Space permissions* --
who may view the space at all -- need admin rights to read, so they are not; `default_acl`
states them instead and is required, since the alternative is publishing every unrestricted
page in the space to the whole tenant.

A page may be restricted at several levels at once, and Confluence requires a reader to pass
*all* of them. This module's ACL model is "any of these principals", which cannot express
"in group A *and* in group B". It is therefore flattened conservatively: the principals named
at every restricted level (their intersection). Someone Confluence would admit through two
different groups may be refused here; nobody Confluence would refuse is admitted. When the
intersection is empty, the page is readable by nobody but tenant admins.

Restricted users are identified by the email Confluence exposes for them when it does, and by
Atlassian account id otherwise. Group restrictions map to `group:<name>`, which matches SSO
group claims when the directory syncs group names into Confluence (SCIM does).
"""

import base64
import html
from collections.abc import Iterator
from urllib.parse import urljoin

from rag_assistant.connectors.base import ConnectorConfig, ConnectorConfigError, RemoteDocument
from rag_assistant.connectors.http import make_client, request
from rag_assistant.ingestion.acl import (
    GROUP_PREFIX,
    NOBODY,
    USER_PREFIX,
    DocumentAcl,
)

_PAGE_SIZE = 50


class ConfluenceConnector:
    def __init__(self, config: ConnectorConfig, validate_only: bool = False, transport=None):
        self._config = config
        options = config.options
        self._base = str(options.get("base_url", "")).rstrip("/")
        self._space = str(options.get("space_key", ""))
        if not self._base.startswith("https://") and not options.get("allow_http"):
            raise ConnectorConfigError(
                f"connector {config.name!r}: base_url must be https:// -- it carries a credential"
            )
        if not self._space:
            raise ConnectorConfigError(f"connector {config.name!r} needs a 'space_key'")
        if config.default_acl is None:
            raise ConnectorConfigError(
                f"connector {config.name!r}: Confluence space permissions cannot be read with "
                "an ordinary token, so default_acl is required -- the groups that may view "
                f'space {self._space!r}, or "tenant" if the whole tenant may.'
            )
        self._restriction_cache: dict[str, frozenset[str] | None] = {}
        if validate_only:
            return
        token = config.secret("token_env")
        email = config.secret("email_env", required=False)
        if email:
            encoded = base64.b64encode(f"{email}:{token}".encode()).decode()
            auth_header = f"Basic {encoded}"
        else:
            auth_header = f"Bearer {token}"
        self._client = make_client(
            transport=transport,
            headers={"Authorization": auth_header, "Accept": "application/json"},
        )

    def _get(self, path: str, **params) -> dict:
        return request(
            self._client, "GET", urljoin(self._base + "/", path.lstrip("/")), params=params
        ).json()

    def list_documents(self) -> Iterator[RemoteDocument]:
        start = 0
        while True:
            page = self._get(
                "rest/api/content",
                spaceKey=self._space,
                type="page",
                status="current",
                expand="version,ancestors",
                limit=_PAGE_SIZE,
                start=start,
            )
            results = page.get("results", [])
            for item in results:
                yield self._describe(item)
            if not page.get("_links", {}).get("next") or not results:
                return
            start += len(results)

    def _describe(self, item: dict) -> RemoteDocument:
        page_id = str(item["id"])
        levels = [page_id] + [str(a["id"]) for a in item.get("ancestors", [])]
        return RemoteDocument(
            id=page_id,
            title=f"{item.get('title') or page_id}.html",
            suffix=".html",
            version=str(item.get("version", {}).get("number", "")),
            acl=self._acl_for(levels),
            url=self._base + item.get("_links", {}).get("webui", ""),
        )

    def _restriction(self, content_id: str) -> frozenset[str] | None:
        """The principals a read restriction on one page names, or None if it has none."""
        if content_id not in self._restriction_cache:
            data = self._get(
                f"rest/api/content/{content_id}/restriction/byOperation/read",
                expand="restrictions.user,restrictions.group",
            )
            restrictions = data.get("restrictions", {})
            users = restrictions.get("user", {}).get("results", []) or []
            groups = restrictions.get("group", {}).get("results", []) or []
            tokens = {
                USER_PREFIX + (u.get("email") or u.get("accountId") or u.get("username") or "")
                for u in users
            } | {GROUP_PREFIX + (g.get("name") or "") for g in groups}
            tokens.discard(USER_PREFIX)
            tokens.discard(GROUP_PREFIX)
            self._restriction_cache[content_id] = frozenset(tokens) if (users or groups) else None
        return self._restriction_cache[content_id]

    def _acl_for(self, levels: list[str]) -> DocumentAcl:
        restricted = [r for r in (self._restriction(level) for level in levels) if r is not None]
        if not restricted:
            return self._config.default_acl
        allowed = frozenset.intersection(*restricted)
        # The space-level default still applies beneath page restrictions: a reader must be
        # able to view the space *and* pass the restriction.
        default = self._config.default_acl
        if default.restricted:
            allowed = allowed & frozenset(default.principals())
        if not allowed or allowed == {NOBODY}:
            return DocumentAcl(deny_all=True)
        return _acl_from_tokens(allowed)

    def fetch(self, document: RemoteDocument) -> bytes:
        data = self._get(f"rest/api/content/{document.id}", expand="body.export_view")
        body = data.get("body", {}).get("export_view", {}).get("value", "")
        title = html.escape(data.get("title") or document.title)
        return f"<html><head><title>{title}</title></head><body>{body}</body></html>".encode()


def _acl_from_tokens(tokens: frozenset[str]) -> DocumentAcl:
    from rag_assistant.ingestion.acl import DOMAIN_PREFIX

    return DocumentAcl(
        users=frozenset(t[len(USER_PREFIX) :] for t in tokens if t.startswith(USER_PREFIX)),
        groups=frozenset(t[len(GROUP_PREFIX) :] for t in tokens if t.startswith(GROUP_PREFIX)),
        domains=frozenset(t[len(DOMAIN_PREFIX) :] for t in tokens if t.startswith(DOMAIN_PREFIX)),
    )
