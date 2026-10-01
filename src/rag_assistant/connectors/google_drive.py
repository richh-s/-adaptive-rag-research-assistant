"""Google Drive folders (My Drive or shared drives) as a source.

    {"name": "policies", "type": "google_drive", "owner": "acme",
     "folder_id": "1AbCdEf...", "credentials_file_env": "DRIVE_SERVICE_ACCOUNT_FILE",
     "impersonate": "rag-reader@acme.com"}

Authenticates as a service account: share the folder with the service account's email, or
use domain-wide delegation and name the user to act as in `impersonate`. The token exchange
is the standard signed-JWT grant, done here with PyJWT rather than by adding Google's client
library for one HTTP call.

The folder is walked recursively. Google Docs are exported as HTML and Slides as plain text;
other files are downloaded as-is when their type is one the corpus can parse. Sheets and
anything else are skipped rather than indexed badly.

**Permissions.** Drive exposes a file's complete permission list, so unlike Confluence no
`default_acl` is needed: each `user` and `group` permission maps to `user:<email>` /
`group:<email>`, a `domain` permission to `domain:<name>` (which SSO users hold for their own
email domain), and `anyone` to visible-to-the-tenant. A file whose permissions cannot be read
-- the service account lacks the right on a shared drive -- is indexed as readable by nobody
but tenant admins, never as readable by everybody.
"""

import json
import time
from collections.abc import Iterator
from pathlib import Path

import jwt

from rag_assistant.connectors.base import (
    ConnectorConfig,
    ConnectorConfigError,
    RemoteDocument,
    supported_suffix,
)
from rag_assistant.connectors.http import make_client, request
from rag_assistant.ingestion.acl import OPEN, DocumentAcl

_API = "https://www.googleapis.com/drive/v3"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
_FOLDER = "application/vnd.google-apps.folder"
# Google-native formats and what each is exported as.
_EXPORTS = {
    "application/vnd.google-apps.document": ("text/html", ".html"),
    "application/vnd.google-apps.presentation": ("text/plain", ".txt"),
}
_FILE_FIELDS = (
    "id,name,mimeType,modifiedTime,version,md5Checksum,webViewLink,"
    "permissions(type,role,emailAddress,domain)"
)


class GoogleDriveConnector:
    def __init__(self, config: ConnectorConfig, validate_only: bool = False, transport=None):
        self._config = config
        self._folder = str(config.options.get("folder_id", ""))
        if not self._folder:
            raise ConnectorConfigError(f"connector {config.name!r} needs a 'folder_id'")
        if not config.options.get("credentials_file_env"):
            raise ConnectorConfigError(
                f"connector {config.name!r} needs 'credentials_file_env' naming the variable "
                "that holds the service-account key file's path"
            )
        self._token: str | None = None
        self._token_expires = 0.0
        # id -> export MIME type, for the Google-native documents seen while listing.
        self._native: dict[str, str] = {}
        if validate_only:
            return
        key_path = config.secret("credentials_file_env")
        self._credentials = json.loads(Path(key_path).read_text(encoding="utf-8"))
        self._client = make_client(transport=transport)

    # ---- auth ----

    def _access_token(self) -> str:
        if self._token and time.time() < self._token_expires - 60:
            return self._token
        now = int(time.time())
        claims = {
            "iss": self._credentials["client_email"],
            "scope": _SCOPE,
            "aud": _TOKEN_URL,
            "iat": now,
            "exp": now + 3600,
        }
        if self._config.options.get("impersonate"):
            claims["sub"] = self._config.options["impersonate"]
        assertion = jwt.encode(
            claims,
            self._credentials["private_key"],
            algorithm="RS256",
            headers={"kid": self._credentials.get("private_key_id")},
        )
        response = request(
            self._client,
            "POST",
            _TOKEN_URL,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": assertion,
            },
        ).json()
        self._token = response["access_token"]
        self._token_expires = time.time() + float(response.get("expires_in", 3600))
        return self._token

    def _get(self, url: str, **params):
        return request(
            self._client,
            "GET",
            url,
            params=params,
            headers={"Authorization": f"Bearer {self._access_token()}"},
        )

    # ---- listing ----

    def list_documents(self) -> Iterator[RemoteDocument]:
        pending = [self._folder]
        seen: set[str] = set()
        while pending:
            folder = pending.pop()
            if folder in seen:
                continue
            seen.add(folder)
            page_token = None
            while True:
                params = {
                    "q": f"'{folder}' in parents and trashed = false",
                    "fields": f"nextPageToken,files({_FILE_FIELDS})",
                    "pageSize": 100,
                    "supportsAllDrives": "true",
                    "includeItemsFromAllDrives": "true",
                }
                if page_token:
                    params["pageToken"] = page_token
                page = self._get(f"{_API}/files", **params).json()
                for item in page.get("files", []):
                    if item.get("mimeType") == _FOLDER:
                        pending.append(item["id"])
                        continue
                    document = self._describe(item)
                    if document is not None:
                        yield document
                page_token = page.get("nextPageToken")
                if not page_token:
                    break

    def _describe(self, item: dict) -> RemoteDocument | None:
        mime = item.get("mimeType", "")
        name = item.get("name") or item["id"]
        if mime in _EXPORTS:
            suffix = _EXPORTS[mime][1]
            self._native[item["id"]] = _EXPORTS[mime][0]
        else:
            suffix = Path(name).suffix.lower()
            if not supported_suffix(suffix):
                return None
        version = str(item.get("version") or item.get("md5Checksum") or item.get("modifiedTime"))
        return RemoteDocument(
            id=item["id"],
            title=name if name.lower().endswith(suffix) else f"{name}{suffix}",
            suffix=suffix,
            version=version,
            acl=self._acl_for(item),
            url=item.get("webViewLink"),
        )

    def _acl_for(self, item: dict) -> DocumentAcl:
        permissions = item.get("permissions")
        if permissions is None:
            try:
                permissions = (
                    self._get(
                        f"{_API}/files/{item['id']}/permissions",
                        fields="permissions(type,role,emailAddress,domain)",
                        supportsAllDrives="true",
                    )
                    .json()
                    .get("permissions")
                )
            except Exception:
                permissions = None
        if permissions is None:
            return DocumentAcl(deny_all=True)
        return acl_from_permissions(permissions)

    def fetch(self, document: RemoteDocument) -> bytes:
        # Native Google formats have no bytes of their own and must be exported; which ids are
        # native was recorded while listing, since a Doc exported as .html and an uploaded
        # .html file are otherwise indistinguishable here.
        export_mime = self._native.get(document.id)
        if export_mime:
            return self._get(f"{_API}/files/{document.id}/export", mimeType=export_mime).content
        return self._get(
            f"{_API}/files/{document.id}", alt="media", supportsAllDrives="true"
        ).content


def acl_from_permissions(permissions: list[dict]) -> DocumentAcl:
    """Drive permissions as an ACL. `anyone` means anyone in the tenant may read it -- the
    document is already public to the internet, so narrowing it here would protect nothing."""
    users, groups, domains = set(), set(), set()
    for permission in permissions:
        kind = permission.get("type")
        if kind == "anyone":
            return OPEN
        if kind == "user" and permission.get("emailAddress"):
            users.add(permission["emailAddress"].lower())
        elif kind == "group" and permission.get("emailAddress"):
            groups.add(permission["emailAddress"].lower())
        elif kind == "domain" and permission.get("domain"):
            domains.add(permission["domain"].lower())
    if not (users or groups or domains):
        return DocumentAcl(deny_all=True)
    return DocumentAcl(users=frozenset(users), groups=frozenset(groups), domains=frozenset(domains))
