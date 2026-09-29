"""Source connectors: mirroring external systems into a tenant's corpus.

The filesystem connector is exercised end to end -- sync, then real indexing, then real
retrieval -- because the claim is that a synced document behaves exactly like an uploaded one,
deletion and permissions included. Confluence and Google Drive are exercised against their
REST shapes through httpx's mock transport: what matters there is that listings paginate,
that permissions are translated conservatively, and that the engine's safety rules hold
whatever the source returns.
"""

import json

import httpx
import pytest

from rag_assistant.config import get_settings
from rag_assistant.connectors.base import (
    ConnectorConfigError,
    RemoteDocument,
    load_connector_configs,
    parse_config,
)
from rag_assistant.connectors.google_drive import acl_from_permissions
from rag_assistant.connectors.sync import (
    SyncInProgress,
    _sync_lock,
    connector_dir,
    connector_status,
    load_state,
    sync_connector,
)
from rag_assistant.ingestion.acl import NOBODY, DocumentAcl, read_acl, sidecar_path, write_acl
from rag_assistant.ingestion.manifest import load_manifest
from rag_assistant.retrieval.vector_store import get_retriever, reset_store_cache
from tests.conftest import FakeHashingEmbeddings


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "index"))
    get_settings.cache_clear()
    embeddings = FakeHashingEmbeddings()
    monkeypatch.setattr(
        "rag_assistant.retrieval.vector_store.index_embeddings", lambda persist_dir: embeddings
    )
    reset_store_cache()
    yield tmp_path
    reset_store_cache()


@pytest.fixture
def share(env):
    root = env / "share"
    root.mkdir()
    (root / "policy.md").write_text("The travel policy caps hotel spend at 200 dollars.")
    (root / "salaries.md").write_text("The salary bands for engineering are confidential.")
    write_acl(root / "salaries.md", DocumentAcl(groups=frozenset({"hr"})))
    (root / "notes.xlsx").write_bytes(b"not a supported type")
    return root


def _fs_config(root, **overrides):
    raw = {
        "name": "handbook",
        "type": "filesystem",
        "owner": "acme",
        "path": str(root),
        "default_acl": "tenant",
    }
    raw.update(overrides)
    return parse_config(raw)


def _indexed_titles():
    manifest = load_manifest(get_settings().chroma_persist_dir)
    return {source.rsplit("/", 1)[-1].rsplit("_", 1)[0] for source in manifest}


# ---- configuration ----


def test_configs_load_and_validate(tmp_path):
    path = tmp_path / "connectors.json"
    path.write_text(
        json.dumps(
            {
                "connectors": [
                    {
                        "name": "wiki",
                        "type": "confluence",
                        "owner": "Acme Corp",
                        "base_url": "https://acme.atlassian.net/wiki",
                        "space_key": "ENG",
                        "token_env": "CONFLUENCE_TOKEN",
                        "default_acl": {"groups": ["engineering"]},
                    }
                ]
            }
        )
    )
    [config] = load_connector_configs(path)
    assert config.owner == "Acme_Corp"
    assert config.default_acl == DocumentAcl(groups=frozenset({"engineering"}))


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"name": "Bad Name", "type": "filesystem", "owner": "a"}, "lowercase"),
        ({"name": "x", "type": "filesystem"}, "owner"),
        ({"name": "x", "type": "sharepoint", "owner": "a"}, "unknown type"),
        ({"name": "x", "type": "filesystem", "owner": "a", "path": "/tmp"}, "default_acl"),
        (
            {
                "name": "x",
                "type": "confluence",
                "owner": "a",
                "base_url": "https://c.example.com",
                "space_key": "S",
                "token_env": "T",
            },
            "default_acl",
        ),
        (
            {
                "name": "x",
                "type": "confluence",
                "owner": "a",
                "base_url": "http://c.example.com",
                "space_key": "S",
                "default_acl": "tenant",
            },
            "https",
        ),
        (
            {"name": "x", "type": "filesystem", "owner": "a", "path": "/t", "default_acl": {}},
            "tenant",
        ),
    ],
)
def test_misconfiguration_fails_at_load_not_at_sync_time(tmp_path, raw, message):
    """Most of these would otherwise publish documents to the whole tenant, or send a
    credential in clear text -- the kind of mistake that has to fail in front of the operator."""
    from rag_assistant.connectors.base import build_connector

    with pytest.raises(ConnectorConfigError, match=message):
        build_connector(parse_config(raw), validate_only=True)


def test_duplicate_connector_names_are_refused(tmp_path):
    path = tmp_path / "connectors.json"
    entry = {"name": "x", "type": "filesystem", "owner": "a", "path": "/t", "default_acl": "tenant"}
    path.write_text(json.dumps({"connectors": [entry, entry]}))
    with pytest.raises(ConnectorConfigError, match="duplicate"):
        load_connector_configs(path)


# ---- the sync engine, end to end ----


def test_a_sync_mirrors_the_source_and_indexes_it(share):
    result = sync_connector(_fs_config(share))

    assert result.outcome == "ok"
    assert (result.added, result.listed) == (2, 2)  # the .xlsx is skipped, not failed
    assert _indexed_titles() == {"policy", "salaries"}
    docs = get_retriever(k=10, owner="acme", principals=frozenset({"group:engineering"})).invoke(
        "travel hotel salary bands"
    )
    assert {d.metadata["source"].rsplit("/", 1)[-1].rsplit("_", 1)[0] for d in docs} == {"policy"}


def test_an_unchanged_source_fetches_nothing(share, monkeypatch):
    config = _fs_config(share)
    sync_connector(config)
    from rag_assistant.connectors import filesystem

    monkeypatch.setattr(
        filesystem.FilesystemConnector,
        "fetch",
        lambda self, doc: pytest.fail("fetched an unchanged document"),
    )

    result = sync_connector(config)

    assert (result.unchanged, result.added, result.updated) == (2, 0, 0)


def test_edits_and_deletions_upstream_reach_the_index(share):
    config = _fs_config(share)
    sync_connector(config)
    (share / "policy.md").write_text("The travel policy now caps hotel spend at 250 dollars.")
    (share / "salaries.md").unlink()

    result = sync_connector(config)

    assert (result.updated, result.deleted) == (1, 1)
    assert _indexed_titles() == {"policy"}
    synced = list(connector_dir(config).glob("salaries*"))
    assert synced == []


def test_a_permission_change_upstream_is_applied(share):
    config = _fs_config(share)
    sync_connector(config)

    write_acl(share / "salaries.md", DocumentAcl(groups=frozenset({"hr", "engineering"})))
    result = sync_connector(config)

    assert result.permissions_changed == 1
    docs = get_retriever(k=10, owner="acme", principals=frozenset({"group:engineering"})).invoke(
        "salary bands"
    )
    assert any("salaries" in d.metadata["source"] for d in docs)


def test_the_deletion_guard_refuses_a_listing_that_lost_most_of_the_source(share):
    """An expired credential or an unmounted share makes a source *look* empty. Faithful
    deletion sync would erase the corpus; the guard refuses and changes nothing."""
    config = _fs_config(share)
    sync_connector(config)
    for path in list(share.iterdir()):
        path.unlink()

    result = sync_connector(config)

    assert result.outcome == "refused"
    assert "--force" in result.error
    assert _indexed_titles() == {"policy", "salaries"}

    forced = sync_connector(config, force=True)
    assert forced.outcome == "ok" and forced.deleted == 2
    assert _indexed_titles() == set()


def test_the_guard_needs_several_deletions_before_the_fraction_applies(env):
    """A three-document source losing two of them is a normal Tuesday, not an outage."""
    root = env / "small"
    root.mkdir()
    for name in ("a", "b", "c"):
        (root / f"{name}.md").write_text(f"Document {name} about quarterly planning.")
    config = _fs_config(root)
    sync_connector(config)
    (root / "a.md").unlink()
    (root / "b.md").unlink()

    result = sync_connector(config)

    assert result.outcome == "ok" and result.deleted == 2


def test_a_failed_listing_changes_nothing(share):
    config = _fs_config(share)
    sync_connector(config)
    share.rename(share.with_name("unmounted"))

    result = sync_connector(config)

    assert result.outcome == "error"
    assert "not a directory" in result.error
    assert _indexed_titles() == {"policy", "salaries"}


def test_a_document_that_fails_to_fetch_keeps_its_previous_copy(share, monkeypatch):
    config = _fs_config(share)
    sync_connector(config)
    (share / "policy.md").write_text("A new version that cannot be fetched.")
    from rag_assistant.connectors import filesystem

    original = filesystem.FilesystemConnector.fetch

    def _flaky(self, document):
        if document.id == "policy.md":
            raise OSError("read timed out")
        return original(self, document)

    monkeypatch.setattr(filesystem.FilesystemConnector, "fetch", _flaky)

    result = sync_connector(config)

    assert result.outcome == "partial"
    assert result.failed and "policy.md" in result.failed[0]
    assert _indexed_titles() == {"policy", "salaries"}


def test_symlinks_out_of_the_share_are_not_followed(share, env):
    outside = env / "private.md"
    outside.write_text("Host secrets that must not be indexed.")
    (share / "link.md").symlink_to(outside)

    result = sync_connector(_fs_config(share))

    assert result.listed == 2


def test_status_records_the_last_run_and_when_the_next_is_due(share):
    config = _fs_config(share)
    assert connector_status(config)["due"] is True

    sync_connector(config)
    status = connector_status(config)

    assert status["documents"] == 2
    assert status["last_run"]["outcome"] == "ok"
    assert status["due"] is False
    assert load_state(connector_dir(config))["items"]


def test_two_syncs_of_one_connector_cannot_overlap(share):
    config = _fs_config(share)
    with _sync_lock(connector_dir(config)):
        with pytest.raises(SyncInProgress):
            sync_connector(config)


# ---- Confluence ----


def _confluence_transport(pages, restrictions, bodies, calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        path = request.url.path
        if path.endswith("/rest/api/content") and request.url.params.get("spaceKey"):
            start = int(request.url.params.get("start", 0))
            chunk = pages[start : start + 2]
            links = {"next": "/rest/api/content?start=x"} if start + 2 < len(pages) else {}
            return httpx.Response(200, json={"results": chunk, "_links": links})
        if "/restriction/byOperation/read" in path:
            page_id = path.split("/content/")[1].split("/")[0]
            users, groups = restrictions.get(page_id, ([], []))
            return httpx.Response(
                200,
                json={
                    "restrictions": {
                        "user": {"results": [{"accountId": u} for u in users]},
                        "group": {"results": [{"name": g} for g in groups]},
                    }
                },
            )
        page_id = path.rsplit("/", 1)[1]
        return httpx.Response(
            200,
            json={"title": f"Page {page_id}", "body": {"export_view": {"value": bodies[page_id]}}},
        )

    return httpx.MockTransport(handler)


def _confluence_config(monkeypatch, default_acl=None):
    monkeypatch.setenv("CONF_TOKEN", "t0ken")
    monkeypatch.setenv("CONF_EMAIL", "bot@acme.com")
    return parse_config(
        {
            "name": "wiki",
            "type": "confluence",
            "owner": "acme",
            "base_url": "https://acme.atlassian.net/wiki",
            "space_key": "ENG",
            "token_env": "CONF_TOKEN",
            "email_env": "CONF_EMAIL",
            "default_acl": default_acl or "tenant",
        }
    )


def _page(page_id, version=1, ancestors=()):
    return {
        "id": page_id,
        "title": f"Page {page_id}",
        "version": {"number": version},
        "ancestors": [{"id": a} for a in ancestors],
        "_links": {"webui": f"/spaces/ENG/pages/{page_id}"},
    }


def test_confluence_pages_paginate_and_carry_page_and_ancestor_restrictions(env, monkeypatch):
    pages = [_page("1"), _page("2", ancestors=["1"]), _page("3"), _page("4", ancestors=["3"])]
    restrictions = {
        # Page 3 is restricted to two groups and its child 4 to one of them and a person:
        # Confluence requires passing both levels, so only the shared group survives.
        "3": ([], ["finance", "leadership"]),
        "4": (["acc-dana"], ["finance"]),
    }
    bodies = {p["id"]: f"<p>Body of page {p['id']} about revenue.</p>" for p in pages}
    calls = []
    from rag_assistant.connectors.confluence import ConfluenceConnector

    connector = ConfluenceConnector(
        _confluence_config(monkeypatch),
        transport=_confluence_transport(pages, restrictions, bodies, calls),
    )

    listed = {d.id: d for d in connector.list_documents()}

    assert set(listed) == {"1", "2", "3", "4"}
    assert not listed["1"].acl.restricted
    assert not listed["2"].acl.restricted
    assert listed["3"].acl == DocumentAcl(groups=frozenset({"finance", "leadership"}))
    assert listed["4"].acl == DocumentAcl(groups=frozenset({"finance"}))
    assert all(c.headers["authorization"].startswith("Basic ") for c in calls)
    # Each page's restrictions are fetched once, however many descendants ask about it.
    restriction_calls = [c for c in calls if "restriction" in c.url.path]
    assert len(restriction_calls) == 4


def test_restriction_levels_with_nothing_in_common_admit_nobody(env, monkeypatch):
    from rag_assistant.connectors.confluence import ConfluenceConnector

    pages = [_page("1"), _page("2", ancestors=["1"])]
    restrictions = {"1": ([], ["legal"]), "2": ([], ["finance"])}
    connector = ConfluenceConnector(
        _confluence_config(monkeypatch),
        transport=_confluence_transport(pages, restrictions, {}, []),
    )

    listed = {d.id: d for d in connector.list_documents()}

    assert listed["2"].acl.deny_all
    assert listed["2"].acl.principals() == [NOBODY]


def test_the_space_default_narrows_unrestricted_and_restricted_pages(env, monkeypatch):
    from rag_assistant.connectors.confluence import ConfluenceConnector

    pages = [_page("1"), _page("2")]
    restrictions = {"2": ([], ["finance", "sales"])}
    connector = ConfluenceConnector(
        _confluence_config(monkeypatch, default_acl={"groups": ["finance", "engineering"]}),
        transport=_confluence_transport(pages, restrictions, {}, []),
    )

    listed = {d.id: d for d in connector.list_documents()}

    assert listed["1"].acl == DocumentAcl(groups=frozenset({"finance", "engineering"}))
    assert listed["2"].acl == DocumentAcl(groups=frozenset({"finance"}))


def test_a_confluence_sync_indexes_pages_as_html(env, monkeypatch):
    pages = [_page("1"), _page("2")]
    bodies = {"1": "<p>Onboarding checklist.</p>", "2": "<p>Incident runbook.</p>"}
    transport = _confluence_transport(pages, {}, bodies, [])

    result = sync_connector(_confluence_config(monkeypatch), transport=transport)

    assert result.outcome == "ok" and result.added == 2
    files = sorted(p.name for p in connector_dir(_confluence_config(monkeypatch)).glob("*.html"))
    assert files[0].startswith("Page_1_") and files[0].endswith(".html")
    assert (
        "<title>Page 1</title>"
        in (connector_dir(_confluence_config(monkeypatch)) / files[0]).read_text()
    )


# ---- Google Drive ----


def test_drive_permissions_translate_conservatively():
    assert not acl_from_permissions([{"type": "anyone", "role": "reader"}]).restricted
    acl = acl_from_permissions(
        [
            {"type": "user", "emailAddress": "Dana@Acme.com"},
            {"type": "group", "emailAddress": "finance@acme.com"},
            {"type": "domain", "domain": "acme.com"},
        ]
    )
    assert acl == DocumentAcl(
        users=frozenset({"dana@acme.com"}),
        groups=frozenset({"finance@acme.com"}),
        domains=frozenset({"acme.com"}),
    )
    # Permissions present but none representable: nobody, never everybody.
    assert acl_from_permissions([{"type": "user"}]).deny_all


@pytest.fixture
def service_account(tmp_path, monkeypatch):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    path = tmp_path / "sa.json"
    path.write_text(
        json.dumps(
            {
                "client_email": "rag@proj.iam.gserviceaccount.com",
                "private_key": pem,
                "private_key_id": "k1",
            }
        )
    )
    monkeypatch.setenv("DRIVE_SA", str(path))
    return key


def test_drive_walks_folders_exports_docs_and_reads_permissions(env, service_account):
    import jwt as pyjwt

    from rag_assistant.connectors.google_drive import GoogleDriveConnector

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        url = str(request.url)
        if url.startswith("https://oauth2.googleapis.com/token"):
            form = dict(x.split("=", 1) for x in request.content.decode().split("&"))
            claims = pyjwt.decode(
                form["assertion"],
                service_account.public_key(),
                algorithms=["RS256"],
                audience="https://oauth2.googleapis.com/token",
            )
            assert claims["iss"] == "rag@proj.iam.gserviceaccount.com"
            return httpx.Response(200, json={"access_token": "ya29", "expires_in": 3600})
        assert request.headers["authorization"] == "Bearer ya29"
        path = request.url.path
        if path.endswith("/files") and "'root-folder' in parents" in request.url.params["q"]:
            return httpx.Response(
                200,
                json={
                    "files": [
                        {
                            "id": "sub",
                            "name": "Sub",
                            "mimeType": "application/vnd.google-apps.folder",
                        },
                        {
                            "id": "doc1",
                            "name": "Policy",
                            "mimeType": "application/vnd.google-apps.document",
                            "version": "7",
                            "permissions": [{"type": "domain", "domain": "acme.com"}],
                        },
                        {
                            "id": "sheet",
                            "name": "Budget",
                            "mimeType": "application/vnd.google-apps.spreadsheet",
                        },
                    ]
                },
            )
        if path.endswith("/files") and "'sub' in parents" in request.url.params["q"]:
            return httpx.Response(
                200,
                json={
                    "files": [
                        {
                            "id": "pdf1",
                            "name": "Report.pdf",
                            "mimeType": "application/pdf",
                            "version": "3",
                        },
                    ]
                },
            )
        if path.endswith("/files/pdf1/permissions"):
            return httpx.Response(403, json={"error": "insufficient permissions"})
        if path.endswith("/files/doc1/export"):
            assert request.url.params["mimeType"] == "text/html"
            return httpx.Response(200, content=b"<html><body>Policy text</body></html>")
        raise AssertionError(f"unexpected request {url}")

    config = parse_config(
        {
            "name": "drive",
            "type": "google_drive",
            "owner": "acme",
            "folder_id": "root-folder",
            "credentials_file_env": "DRIVE_SA",
        }
    )
    connector = GoogleDriveConnector(config, transport=httpx.MockTransport(handler))

    listed = {d.id: d for d in connector.list_documents()}

    assert set(listed) == {"doc1", "pdf1"}  # the spreadsheet is skipped, the folder walked
    assert listed["doc1"].suffix == ".html"
    assert listed["doc1"].acl == DocumentAcl(domains=frozenset({"acme.com"}))
    # Permissions the service account could not read fail closed.
    assert listed["pdf1"].acl.deny_all
    assert connector.fetch(listed["doc1"]).startswith(b"<html>")
    token_requests = [c for c in calls if "oauth2" in str(c.url)]
    assert len(token_requests) == 1


def test_a_synced_sidecar_matches_what_the_source_said(share):
    config = _fs_config(share)
    sync_connector(config)
    synced = next(connector_dir(config).glob("salaries_*.md"))

    assert sidecar_path(synced).exists()
    assert read_acl(synced) == DocumentAcl(groups=frozenset({"hr"}))


def test_remote_document_filenames_are_safe_and_stable():
    from rag_assistant.connectors.sync import local_filename

    document = RemoteDocument(id="42", title="../../etc/passwd.md", suffix=".md", version="1")
    name = local_filename(document)
    assert "/" not in name and name.endswith(".md")
    assert name == local_filename(document)


# ---- HTTP retries ----


def test_rate_limited_requests_are_retried_honouring_retry_after(monkeypatch):
    from rag_assistant.connectors import http as connector_http

    sleeps = []
    monkeypatch.setattr(connector_http.time, "sleep", sleeps.append)
    responses = iter(
        [
            httpx.Response(429, headers={"retry-after": "7"}),
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    client = connector_http.make_client(
        transport=httpx.MockTransport(lambda request: next(responses))
    )

    response = connector_http.request(client, "GET", "https://api.example.com/x")

    assert response.json() == {"ok": True}
    assert sleeps == [7.0, 2.0]


def test_retries_are_bounded_and_client_errors_are_not_retried(monkeypatch):
    from rag_assistant.connectors import http as connector_http

    monkeypatch.setattr(connector_http.time, "sleep", lambda seconds: None)
    calls = []

    def _always(status):
        def handler(request):
            calls.append(status)
            return httpx.Response(status)

        return handler

    with pytest.raises(httpx.HTTPStatusError):
        connector_http.request(
            connector_http.make_client(transport=httpx.MockTransport(_always(500))),
            "GET",
            "https://api.example.com/x",
        )
    assert len(calls) == 4

    calls.clear()
    with pytest.raises(httpx.HTTPStatusError):
        connector_http.request(
            connector_http.make_client(transport=httpx.MockTransport(_always(401))),
            "GET",
            "https://api.example.com/x",
        )
    assert len(calls) == 1
