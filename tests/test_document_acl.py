"""Document-level permissions inside a tenant.

The claim under test is an access-control one, so the important tests go end to end: index
real files with real sidecars into a real Chroma collection and a real keyword index, then
retrieve as different people and assert who sees what. A unit test of the predicate alone
would pass just as happily if no retrieval path ever called it.
"""

import json

import pytest
from fastapi.testclient import TestClient

from rag_assistant import api
from rag_assistant.auth import Principal
from rag_assistant.ingestion.acl import (
    NOBODY,
    DocumentAcl,
    entry_readable,
    principals_can_read,
    read_acl,
    sidecar_path,
    write_acl,
)
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.ingestion.manifest import load_manifest
from rag_assistant.ingestion.ownership import TENANT_DIR
from rag_assistant.retrieval.bm25_store import bm25_search, invalidate_bm25_index
from rag_assistant.retrieval.vector_store import get_retriever
from tests.conftest import FakeHashingEmbeddings

FINANCE = frozenset({"user:dana", "group:finance"})
ENGINEER = frozenset({"user:erin", "group:engineering"})


@pytest.fixture
def acme_corpus(tmp_path):
    """One tenant with an open document, a finance-only document and a document whose
    sidecar is corrupt."""
    corpus = tmp_path / "corpus"
    acme = corpus / TENANT_DIR / "acme"
    acme.mkdir(parents=True)
    (acme / "handbook.md").write_text("The quarterly revenue handbook explains expense policy.")
    board = acme / "board.md"
    board.write_text("The quarterly revenue forecast shows layoffs in the widgets division.")
    write_acl(board, DocumentAcl(groups=frozenset({"finance"})))
    broken = acme / "broken.md"
    broken.write_text("The quarterly revenue audit notes are pending review.")
    sidecar_path(broken).write_text("{not json")
    return corpus


def _index(corpus, persist_dir, embeddings):
    result = build_index(source_dir=corpus, persist_dir=persist_dir, embeddings=embeddings)
    invalidate_bm25_index(persist_dir)
    return result


def _vector_sources(persist_dir, embeddings, principals):
    docs = get_retriever(
        k=10,
        embeddings=embeddings,
        persist_dir=persist_dir,
        owner="acme",
        principals=principals,
    ).invoke("quarterly revenue")
    return {d.metadata["source"].rsplit("/", 1)[-1] for d in docs}


def _keyword_sources(persist_dir, principals):
    hits = bm25_search(
        "quarterly revenue", k=10, persist_dir=persist_dir, owner="acme", principals=principals
    )
    return {h.source_id.rsplit("/", 1)[-1] for h in hits}


# ---- the model ----


def test_no_sidecar_means_the_whole_tenant(tmp_path):
    document = tmp_path / "open.md"
    document.write_text("x")
    assert read_acl(document) == DocumentAcl()
    assert not read_acl(document).restricted


def test_an_unreadable_sidecar_fails_closed(tmp_path):
    """A corrupt sidecar must never read as "no restriction" -- that would publish a
    document someone deliberately restricted."""
    document = tmp_path / "secret.md"
    document.write_text("x")
    sidecar_path(document).write_text("{not json")

    acl = read_acl(document)

    assert acl.restricted and acl.deny_all
    assert acl.principals() == [NOBODY]


def test_writing_an_open_acl_removes_the_sidecar(tmp_path):
    document = tmp_path / "doc.md"
    document.write_text("x")
    write_acl(document, DocumentAcl(groups=frozenset({"finance"})))
    assert sidecar_path(document).exists()

    write_acl(document, DocumentAcl())

    assert not sidecar_path(document).exists()


def test_the_fingerprint_changes_with_the_principals_and_nothing_else():
    a = DocumentAcl(groups=frozenset({"finance"}), users=frozenset({"dana"}))
    b = DocumentAcl(users=frozenset({"dana"}), groups=frozenset({"finance"}))
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != DocumentAcl(groups=frozenset({"legal"})).fingerprint()
    assert DocumentAcl().fingerprint() == "open"


def test_the_predicate_admits_any_shared_principal():
    restricted = DocumentAcl(groups=frozenset({"finance"})).chunk_metadata()
    assert principals_can_read(restricted, FINANCE)
    assert not principals_can_read(restricted, ENGINEER)
    assert not principals_can_read(restricted, frozenset())
    # None is a caller that bypasses ACLs -- a tenant admin or an internal tool.
    assert principals_can_read(restricted, None)
    # Chunks indexed before ACLs existed carry no ACL keys and stay visible.
    assert principals_can_read({"owner": "acme"}, ENGINEER)


def test_principals_come_from_user_email_domain_and_groups():
    principal = Principal(
        owner="acme",
        method="oidc",
        subject="00u1",
        email="Dana@Acme.com",
        groups=frozenset({"finance"}),
    )
    assert principal.principals() == {
        "user:00u1",
        "user:dana@acme.com",
        "domain:acme.com",
        "group:finance",
    }
    assert Principal(owner="acme", method="oidc", bypass_acl=True).principals() is None


# ---- retrieval ----


def test_vector_retrieval_hides_documents_the_caller_may_not_read(acme_corpus, tmp_path):
    embeddings = FakeHashingEmbeddings()
    persist_dir = tmp_path / "chroma"
    _index(acme_corpus, persist_dir, embeddings)

    assert _vector_sources(persist_dir, embeddings, ENGINEER) == {"handbook.md"}
    assert _vector_sources(persist_dir, embeddings, FINANCE) == {"handbook.md", "board.md"}
    # The corrupt sidecar denies everyone except callers who bypass ACLs.
    assert _vector_sources(persist_dir, embeddings, None) == {
        "handbook.md",
        "board.md",
        "broken.md",
    }


def test_keyword_retrieval_applies_the_same_permissions(acme_corpus, tmp_path):
    persist_dir = tmp_path / "chroma"
    _index(acme_corpus, persist_dir, FakeHashingEmbeddings())

    assert _keyword_sources(persist_dir, ENGINEER) == {"handbook.md"}
    assert _keyword_sources(persist_dir, FINANCE) == {"handbook.md", "board.md"}


def test_filtering_happens_inside_the_search_so_k_is_not_shrunk(acme_corpus, tmp_path):
    """Asking for one document must return one the caller can read, not zero because the
    single best match happened to be restricted."""
    embeddings = FakeHashingEmbeddings()
    persist_dir = tmp_path / "chroma"
    _index(acme_corpus, persist_dir, embeddings)

    docs = get_retriever(
        k=1, embeddings=embeddings, persist_dir=persist_dir, owner="acme", principals=ENGINEER
    ).invoke("quarterly revenue forecast layoffs widgets")

    assert [d.metadata["source"].rsplit("/", 1)[-1] for d in docs] == ["handbook.md"]


def test_a_caller_with_no_principals_sees_only_unrestricted_documents(acme_corpus, tmp_path):
    embeddings = FakeHashingEmbeddings()
    persist_dir = tmp_path / "chroma"
    _index(acme_corpus, persist_dir, embeddings)

    assert _vector_sources(persist_dir, embeddings, frozenset()) == {"handbook.md"}
    assert _keyword_sources(persist_dir, frozenset()) == {"handbook.md"}


class _CountingEmbeddings(FakeHashingEmbeddings):
    def __init__(self):
        super().__init__()
        self.documents_embedded = 0

    def embed_documents(self, texts):
        self.documents_embedded += len(texts)
        return super().embed_documents(texts)


def test_a_permission_change_is_applied_without_re_embedding(acme_corpus, tmp_path):
    """Sharing a document with one more group is the most common change a synced source
    makes. Paying an embedding call per chunk for it would make keeping permissions current
    too expensive to do."""
    embeddings = _CountingEmbeddings()
    persist_dir = tmp_path / "chroma"
    _index(acme_corpus, persist_dir, embeddings)
    assert _vector_sources(persist_dir, embeddings, ENGINEER) == {"handbook.md"}
    embedded_before = embeddings.documents_embedded

    board = acme_corpus / TENANT_DIR / "acme" / "board.md"
    write_acl(board, DocumentAcl(groups=frozenset({"finance", "engineering"})))
    result = build_index(source_dir=acme_corpus, persist_dir=persist_dir, embeddings=embeddings)

    assert result.acl_updates == 1
    assert result.changed_files == 0
    assert embeddings.documents_embedded == embedded_before
    assert _vector_sources(persist_dir, embeddings, ENGINEER) == {"handbook.md", "board.md"}
    # The cached keyword index picked the change up too, without being rebuilt.
    assert _keyword_sources(persist_dir, ENGINEER) == {"handbook.md", "board.md"}


def test_revoking_access_takes_effect_on_the_next_ingest(acme_corpus, tmp_path):
    embeddings = FakeHashingEmbeddings()
    persist_dir = tmp_path / "chroma"
    _index(acme_corpus, persist_dir, embeddings)
    handbook = acme_corpus / TENANT_DIR / "acme" / "handbook.md"

    write_acl(handbook, DocumentAcl(users=frozenset({"dana"})))
    build_index(source_dir=acme_corpus, persist_dir=persist_dir, embeddings=embeddings)

    assert "handbook.md" not in _vector_sources(persist_dir, embeddings, ENGINEER)
    assert "handbook.md" not in _keyword_sources(persist_dir, ENGINEER)
    assert "handbook.md" in _vector_sources(persist_dir, embeddings, FINANCE)


def test_the_manifest_records_the_acl_for_listings(acme_corpus, tmp_path):
    persist_dir = tmp_path / "chroma"
    _index(acme_corpus, persist_dir, FakeHashingEmbeddings())
    manifest = load_manifest(persist_dir)
    board = manifest[f"{TENANT_DIR}/acme/board.md"]
    handbook = manifest[f"{TENANT_DIR}/acme/handbook.md"]

    assert board["acl"] == {"groups": ["finance"], "users": []}
    assert "acl" not in handbook
    assert entry_readable(board, FINANCE) and not entry_readable(board, ENGINEER)
    assert entry_readable(handbook, ENGINEER)


def test_the_router_is_not_told_about_documents_the_caller_cannot_read(
    acme_corpus, tmp_path, monkeypatch
):
    """A filename is information. Describing the corpus to the router with a restricted
    document in it would leak exactly what retrieval withholds."""
    persist_dir = tmp_path / "chroma"
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(persist_dir))
    _index(acme_corpus, persist_dir, FakeHashingEmbeddings())
    from rag_assistant.graph.nodes.router import _describe_local_corpus

    assert "board" not in _describe_local_corpus("acme", ENGINEER)
    assert "board" in _describe_local_corpus("acme", FINANCE)


# ---- the API ----


@pytest.fixture
def api_env(tmp_path, monkeypatch):
    """Three identities in one tenant: an admin key with no identity (reads everything), and
    two people with API keys that carry a user and groups."""
    keys = tmp_path / "keys.json"
    keys.write_text(
        json.dumps(
            {
                "keys": [
                    {"key": "k-admin", "owner": "acme"},
                    {"key": "k-dana", "owner": "acme", "user": "dana", "groups": ["finance"]},
                    {"key": "k-erin", "owner": "acme", "user": "erin", "groups": ["engineering"]},
                ]
            }
        )
    )
    monkeypatch.setenv("API_KEYS_FILE", str(keys))
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RATE_LIMIT_RPM", "1000")
    monkeypatch.setenv("RATE_LIMIT_RPM_GLOBAL", "1000")
    embeddings = FakeHashingEmbeddings()
    monkeypatch.setattr(
        "rag_assistant.retrieval.vector_store.index_embeddings", lambda persist_dir: embeddings
    )
    from rag_assistant.retrieval.vector_store import reset_store_cache

    reset_store_cache()
    yield TestClient(api.app)
    reset_store_cache()


def _upload(client, key, name, text, **fields):
    response = client.post(
        "/api/v1/ingest",
        headers={"X-API-Key": key},
        files={"file": (name, text.encode(), "text/markdown")},
        data=fields,
    )
    assert response.status_code == 202, response.text
    return response.json()


def _sources(client, key):
    response = client.get("/api/v1/sources", headers={"X-API-Key": key})
    assert response.status_code == 200
    return {s["display_name"].rsplit("_", 1)[0]: s for s in response.json()}


def test_an_upload_can_be_restricted_to_a_group(api_env):
    _upload(api_env, "k-dana", "forecast.md", "Revenue forecast.", allowed_groups="finance")
    _upload(api_env, "k-dana", "handbook.md", "Expense policy handbook.")

    assert set(_sources(api_env, "k-dana")) == {"forecast", "handbook"}
    assert set(_sources(api_env, "k-erin")) == {"handbook"}
    assert _sources(api_env, "k-dana")["forecast"]["restricted"] is True
    # A key with no identity is tenant-wide, as every key was before permissions existed.
    assert set(_sources(api_env, "k-admin")) == {"forecast", "handbook"}


def test_research_carries_the_callers_principals_into_the_graph(api_env, monkeypatch):
    captured = {}

    async def _fake(state, config=None):
        captured.update(state)
        return {"research_report": "ok", "route": "vector", "confidence_score": 0.9}

    monkeypatch.setattr(api._graph, "ainvoke", _fake)

    api_env.post(
        "/api/v1/research",
        headers={"X-API-Key": "k-erin"},
        json={"question": "What is the expense policy?", "save": False},
    )
    assert captured["principals"] == ["group:engineering", "user:erin"]

    api_env.post(
        "/api/v1/research",
        headers={"X-API-Key": "k-admin"},
        json={"question": "What is the expense policy?", "save": False},
    )
    assert captured["principals"] is None


def test_a_document_the_caller_cannot_read_cannot_be_deleted_either(api_env):
    _upload(api_env, "k-dana", "forecast.md", "Revenue forecast.", allowed_groups="finance")
    source = _sources(api_env, "k-dana")["forecast"]["source"]

    as_erin = api_env.delete(f"/api/v1/sources/{source}", headers={"X-API-Key": "k-erin"})
    assert as_erin.status_code == 404

    as_dana = api_env.delete(f"/api/v1/sources/{source}", headers={"X-API-Key": "k-dana"})
    assert as_dana.status_code == 200


def test_permissions_can_be_changed_in_place(api_env):
    _upload(api_env, "k-dana", "forecast.md", "Revenue forecast.", allowed_groups="finance")
    source = _sources(api_env, "k-dana")["forecast"]["source"]

    response = api_env.put(
        f"/api/v1/sources/{source}/acl",
        headers={"X-API-Key": "k-dana"},
        json={"groups": ["finance", "engineering"]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["restricted"] is True
    assert "forecast" in _sources(api_env, "k-erin")


def test_an_acl_change_that_would_lock_out_its_author_is_refused(api_env):
    _upload(api_env, "k-dana", "forecast.md", "Revenue forecast.", allowed_groups="finance")
    source = _sources(api_env, "k-dana")["forecast"]["source"]

    response = api_env.put(
        f"/api/v1/sources/{source}/acl",
        headers={"X-API-Key": "k-dana"},
        json={"groups": ["legal"]},
    )

    assert response.status_code == 400
    assert "forecast" in _sources(api_env, "k-dana")


def test_nobody_can_change_permissions_on_a_document_they_cannot_read(api_env):
    _upload(api_env, "k-dana", "forecast.md", "Revenue forecast.", allowed_groups="finance")
    source = _sources(api_env, "k-dana")["forecast"]["source"]

    response = api_env.put(
        f"/api/v1/sources/{source}/acl",
        headers={"X-API-Key": "k-erin"},
        json={"groups": ["engineering"]},
    )

    assert response.status_code == 404


def test_conversations_are_private_to_each_user_in_a_tenant(api_env, monkeypatch):
    """Once documents have per-user permissions, a transcript quoting one is as sensitive as
    the document. A tenant-wide conversation list would hand Dana's answers to Erin."""

    async def _fake(state, config=None):
        return {"research_report": "r", "final_answer": "a", "route": "vector"}

    monkeypatch.setattr(api._graph, "ainvoke", _fake)
    created = api_env.post(
        "/api/v1/research",
        headers={"X-API-Key": "k-dana"},
        json={"question": "What is the revenue forecast?"},
    ).json()["conversation_id"]

    dana = api_env.get("/api/v1/conversations", headers={"X-API-Key": "k-dana"}).json()
    erin = api_env.get("/api/v1/conversations", headers={"X-API-Key": "k-erin"}).json()

    assert [c["id"] for c in dana] == [created]
    assert erin == []
    assert (
        api_env.get(f"/api/v1/conversations/{created}", headers={"X-API-Key": "k-erin"}).status_code
        == 404
    )


def test_erasing_a_tenant_reaches_every_users_conversations(api_env, monkeypatch):
    async def _fake(state, config=None):
        return {"research_report": "r", "final_answer": "a", "route": "vector"}

    monkeypatch.setattr(api._graph, "ainvoke", _fake)
    for key in ("k-dana", "k-erin"):
        api_env.post(
            "/api/v1/research", headers={"X-API-Key": key}, json={"question": "Any question?"}
        )

    purged = api_env.delete("/api/v1/tenant/data", headers={"X-API-Key": "k-admin"}).json()

    assert purged["conversations_removed"] == 2


def test_another_tenants_ingest_task_is_not_visible(api_env, tmp_path, monkeypatch):
    keys = json.loads((tmp_path / "keys.json").read_text())
    keys["keys"].append({"key": "k-other", "owner": "globex"})
    (tmp_path / "keys.json").write_text(json.dumps(keys))
    monkeypatch.setattr(api, "_run_ingest_in_background", lambda *a, **k: None)

    task = _upload(api_env, "k-dana", "forecast.md", "Revenue forecast.")["task_id"]

    assert api_env.get(f"/api/v1/ingest/{task}", headers={"X-API-Key": "k-dana"}).status_code == 200
    assert (
        api_env.get(f"/api/v1/ingest/{task}", headers={"X-API-Key": "k-other"}).status_code == 404
    )
