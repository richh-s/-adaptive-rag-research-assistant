"""Tests for corpus-level tenant isolation.

The claim under test is a data-isolation one, so these go end to end where it matters: index
two tenants' documents into one real Chroma collection and one real BM25 index, then assert
that each tenant's retrieval cannot see the other's. Unit-testing the filter helper alone
would pass just as happily if the filter were never wired into a retrieval path.
"""

import pytest
from fastapi.testclient import TestClient

from rag_assistant import api
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.ingestion.loaders import load_documents
from rag_assistant.ingestion.ownership import (
    TENANT_DIR,
    display_source,
    owner_corpus_dir,
    owner_of_relative_path,
    safe_owner_dirname,
    visible_owners,
)
from rag_assistant.retrieval.bm25_store import bm25_search, invalidate_bm25_index
from rag_assistant.retrieval.vector_store import get_retriever


@pytest.fixture
def tenant_corpus(tmp_path):
    """A corpus with one shared baseline file and one private file per tenant."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "baseline.md").write_text(
        "Anthropic is an AI safety company known for Constitutional AI research."
    )

    alice_dir = corpus / TENANT_DIR / "alice"
    alice_dir.mkdir(parents=True)
    (alice_dir / "alice_secret.md").write_text(
        "Project Zephyr is Alice's confidential quarterly revenue plan for widgets."
    )

    bob_dir = corpus / TENANT_DIR / "bob"
    bob_dir.mkdir(parents=True)
    (bob_dir / "bob_secret.md").write_text(
        "Project Mistral is Bob's confidential quarterly revenue plan for gadgets."
    )
    return corpus


# ---- path <-> owner mapping ----


def test_flat_files_are_public_and_tenant_subtrees_are_owned():
    from pathlib import Path

    assert owner_of_relative_path(Path("anthropic.md")) == "public"
    assert owner_of_relative_path(Path(f"{TENANT_DIR}/alice/report.md")) == "alice"


def test_public_uploads_stay_flat_so_the_open_demo_layout_is_unchanged(tmp_path):
    assert owner_corpus_dir(tmp_path, "public") == tmp_path
    assert owner_corpus_dir(tmp_path, "alice") == tmp_path / TENANT_DIR / "alice"


def test_owner_directory_names_cannot_escape_the_corpus():
    """Owner labels come from operator config rather than request input, but a label
    containing a traversal would write outside the corpus root."""
    assert safe_owner_dirname("../../etc") == "etc"
    assert "/" not in safe_owner_dirname("a/b")
    assert safe_owner_dirname("") == "public"


def test_a_tenant_sees_their_own_documents_and_the_public_corpus():
    assert visible_owners("alice") == ["alice", "public"]
    assert visible_owners("public") == ["public"]


def test_citations_show_the_filename_not_the_tenant_path():
    assert display_source(f"{TENANT_DIR}/alice/q3-report.md") == "q3-report.md"
    assert display_source("anthropic.md") == "anthropic.md"


# ---- loading ----


def test_loader_tags_every_document_with_its_owner(tenant_corpus):
    documents = load_documents(tenant_corpus)

    owners = {d.metadata["source"]: d.metadata["owner"] for d in documents}
    assert owners["baseline.md"] == "public"
    assert owners[f"{TENANT_DIR}/alice/alice_secret.md"] == "alice"
    assert owners[f"{TENANT_DIR}/bob/bob_secret.md"] == "bob"


def test_source_keys_stay_unique_across_tenants_with_identical_filenames(tmp_path):
    """Two tenants uploading `report.md` must not collide -- a shared key would have one
    tenant's manifest entry and chunk IDs overwrite the other's, deleting their chunks."""
    corpus = tmp_path / "corpus"
    for owner in ("alice", "bob"):
        directory = corpus / TENANT_DIR / owner
        directory.mkdir(parents=True)
        (directory / "report.md").write_text(f"{owner} private content here.")

    sources = {d.metadata["source"] for d in load_documents(corpus)}

    assert sources == {
        f"{TENANT_DIR}/alice/report.md",
        f"{TENANT_DIR}/bob/report.md",
    }


# ---- retrieval isolation ----


def test_vector_retrieval_is_scoped_to_the_tenant(tenant_corpus, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"
    build_index(source_dir=tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)

    alice_docs = get_retriever(
        k=10, embeddings=fake_embeddings, persist_dir=persist_dir, owner="alice"
    ).invoke("confidential quarterly revenue plan")
    sources = {d.metadata["source"] for d in alice_docs}

    assert f"{TENANT_DIR}/bob/bob_secret.md" not in sources
    assert f"{TENANT_DIR}/alice/alice_secret.md" in sources
    # The shared baseline corpus stays visible to everyone.
    assert any(d.metadata["owner"] == "public" for d in alice_docs)


def test_public_tenant_cannot_see_any_tenant_documents(tenant_corpus, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"
    build_index(source_dir=tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)

    docs = get_retriever(
        k=10, embeddings=fake_embeddings, persist_dir=persist_dir, owner="public"
    ).invoke("confidential quarterly revenue plan")

    assert {d.metadata["owner"] for d in docs} == {"public"}


def test_bm25_retrieval_is_scoped_to_the_tenant(tenant_corpus, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"
    build_index(source_dir=tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)
    invalidate_bm25_index(persist_dir)

    alice_hits = bm25_search(
        "confidential quarterly revenue", k=10, persist_dir=persist_dir, owner="alice"
    )
    bob_hits = bm25_search(
        "confidential quarterly revenue", k=10, persist_dir=persist_dir, owner="bob"
    )

    assert any("alice_secret" in h.source_id for h in alice_hits)
    assert not any("bob_secret" in h.source_id for h in alice_hits)
    assert any("bob_secret" in h.source_id for h in bob_hits)
    assert not any("alice_secret" in h.source_id for h in bob_hits)


def test_bm25_filters_candidates_before_the_top_k_cut(tenant_corpus, fake_embeddings, tmp_path):
    """Post-filtering would silently shrink k, so a tenant whose top hits belong to someone
    else would get fewer documents with no indication why."""
    persist_dir = tmp_path / "chroma"
    build_index(source_dir=tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)
    invalidate_bm25_index(persist_dir)

    hits = bm25_search("Project revenue plan", k=1, persist_dir=persist_dir, owner="alice")

    assert len(hits) == 1
    assert "bob" not in hits[0].source_id


# ---- router corpus description ----


def test_router_corpus_description_lists_only_what_the_tenant_can_see(
    tenant_corpus, fake_embeddings, tmp_path, monkeypatch
):
    """Listing another tenant's filenames would leak them through the router prompt even
    though retrieval filters them out."""
    persist_dir = tmp_path / "chroma"
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(persist_dir))
    build_index(source_dir=tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)

    from rag_assistant.graph.nodes.router import _describe_local_corpus

    alice_view = _describe_local_corpus("alice")

    assert "alice secret" in alice_view
    assert "bob" not in alice_view
    assert "baseline" in alice_view


# ---- API ----


def test_uploads_land_in_the_uploading_tenants_subtree(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEYS", "alice:secret-a")
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setenv("RATE_LIMIT_RPM", "1000")
    monkeypatch.setenv("RATE_LIMIT_RPM_GLOBAL", "1000")
    monkeypatch.setattr(api, "_run_ingest_in_background", lambda *args, **kwargs: None)
    client = TestClient(api.app)

    response = client.post(
        "/api/v1/ingest",
        headers={"X-API-Key": "secret-a"},
        files={"file": ("notes.md", b"Alice private notes about widgets.", "text/markdown")},
    )

    assert response.status_code == 202
    written = list((tmp_path / "corpus" / TENANT_DIR / "alice").glob("*.md"))
    assert len(written) == 1
    assert written[0].read_text() == "Alice private notes about widgets."


def test_research_passes_the_authenticated_owner_into_the_graph(monkeypatch):
    """The isolation only holds if the API actually tells the graph who is asking."""
    monkeypatch.setenv("API_KEYS", "alice:secret-a")
    monkeypatch.setenv("RATE_LIMIT_RPM", "1000")
    monkeypatch.setenv("RATE_LIMIT_RPM_GLOBAL", "1000")
    captured = {}

    def _fake_invoke(state, config=None):
        captured.update(state)
        return {"research_report": "ok", "route": "vector", "confidence_score": 0.9}

    monkeypatch.setattr(api._graph, "invoke", _fake_invoke)
    client = TestClient(api.app)

    response = client.post(
        "/api/v1/research",
        headers={"X-API-Key": "secret-a"},
        json={"question": "What is in my documents?", "save": False},
    )

    assert response.status_code == 200
    assert captured["owner"] == "alice"
