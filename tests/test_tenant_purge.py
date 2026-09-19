"""Tests for tenant data erasure.

Retention bounds how long data lives. This is the different question -- "delete my data now"
-- and the failure mode worth testing is partial erasure: a purge that clears conversations
while leaving the tenant's documents indexed and retrievable has not erased anything, and
reports success while doing it.
"""

import pytest
from fastapi.testclient import TestClient

from rag_assistant import api, auth, tenancy
from rag_assistant.conversations import store as conversations_store
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.ingestion.manifest import load_manifest
from rag_assistant.ingestion.ownership import TENANT_DIR
from rag_assistant.retrieval.bm25_store import bm25_search, invalidate_bm25_index
from rag_assistant.retrieval.parent_store import get_parents
from rag_assistant.retrieval.vector_store import get_retriever


@pytest.fixture
def two_tenants(tmp_path, monkeypatch, fake_embeddings):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "shared.md").write_text("Anthropic is an AI safety company.")
    for name in ("alice", "bob"):
        d = corpus / TENANT_DIR / name
        d.mkdir(parents=True)
        (d / f"{name}_secret.md").write_text(
            f"Project {name.title()} is {name}'s confidential quarterly revenue plan."
        )
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    persist_dir = tmp_path / "idx"
    build_index(source_dir=corpus, persist_dir=persist_dir, embeddings=fake_embeddings)
    invalidate_bm25_index(persist_dir)
    return corpus, persist_dir


def test_purge_removes_every_store_the_tenant_touched(two_tenants, fake_embeddings, monkeypatch):
    corpus, persist_dir = two_tenants
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(persist_dir))
    from rag_assistant.config import get_settings

    get_settings.cache_clear()

    alice_sources = [s for s in load_manifest(persist_dir) if "alice" in s]
    conv = conversations_store.create_conversation("alice chat", owner="alice")
    conversations_store.record_feedback(
        conversation_id=conv.id, question="q", rating="up", owner="alice"
    )

    result = tenancy.purge_tenant("alice", persist_dir=persist_dir)

    assert result.sources == len(alice_sources)
    assert result.chunks > 0
    assert result.conversations == 1
    assert result.feedback == 1
    # Manifest
    assert not any("alice" in s for s in load_manifest(persist_dir))
    # Vector store, through a real retrieval as the tenant themselves
    hits = get_retriever(
        k=10, embeddings=fake_embeddings, persist_dir=persist_dir, owner="alice"
    ).invoke("confidential quarterly revenue plan")
    assert not any("alice" in d.metadata["source"] for d in hits)
    # Keyword index -- a separate store that would otherwise keep serving the purged text
    keyword_hits = bm25_search(
        "confidential quarterly revenue", k=10, persist_dir=persist_dir, owner="alice"
    )
    assert not any("alice" in d.metadata.get("source", "") for d in keyword_hits)
    # Corpus files
    assert not (corpus / TENANT_DIR / "alice").exists()


def test_purging_one_tenant_leaves_the_other_intact(two_tenants, fake_embeddings):
    corpus, persist_dir = two_tenants

    tenancy.purge_tenant("alice", persist_dir=persist_dir)

    bob_hits = get_retriever(
        k=10, embeddings=fake_embeddings, persist_dir=persist_dir, owner="bob"
    ).invoke("confidential quarterly revenue plan")

    assert any("bob" in d.metadata["source"] for d in bob_hits)
    assert (corpus / TENANT_DIR / "bob").exists()
    assert any("shared" in s for s in load_manifest(persist_dir))


def test_purging_the_public_tenant_does_not_delete_the_shared_corpus(two_tenants, fake_embeddings):
    """The public tenant's 'directory' is the corpus root, which holds every other tenant's
    subtree. Removing it would erase the corpus rather than one tenant's slice."""
    corpus, persist_dir = two_tenants

    tenancy.purge_tenant("public", persist_dir=persist_dir)

    # The public tenant's indexed documents go, as for any tenant...
    assert not any(entry.get("owner") == "public" for entry in load_manifest(persist_dir).values())
    # ...but the directory tree itself, and every other tenant inside it, survives.
    assert corpus.exists()
    assert (corpus / TENANT_DIR / "alice").exists()
    assert (corpus / TENANT_DIR / "bob").exists()


def test_parent_sections_go_too(two_tenants, fake_embeddings):
    """Parents are keyed by source, not by chunk id, so they are easy to leave behind -- and
    a leftover section body is the full text of a purged document."""
    corpus, persist_dir = two_tenants
    manifest = load_manifest(persist_dir)
    alice_source = next(s for s in manifest if "alice" in s)
    docs = get_retriever(
        k=10, embeddings=fake_embeddings, persist_dir=persist_dir, owner="alice"
    ).invoke("Project Alice confidential")
    # Only alice's own documents. The shared public document comes back on this query too,
    # and its parent section must *survive* -- collecting it here would assert the opposite
    # of the intended behaviour and fail for the right reason by accident.
    alice_parent_ids = [
        d.metadata["parent_id"]
        for d in docs
        if d.metadata.get("parent_id") and "alice" in d.metadata.get("source", "")
    ]
    assert alice_parent_ids, "the fixture should produce at least one parent for alice"

    tenancy.purge_tenant("alice", persist_dir=persist_dir)

    assert alice_source not in load_manifest(persist_dir)
    assert get_parents(persist_dir, alice_parent_ids) == {}


def test_purge_is_idempotent(two_tenants):
    corpus, persist_dir = two_tenants

    first = tenancy.purge_tenant("alice", persist_dir=persist_dir)
    second = tenancy.purge_tenant("alice", persist_dir=persist_dir)

    assert first.sources > 0
    assert second.sources == 0
    assert second.chunks == 0


def test_the_endpoint_purges_only_the_calling_tenant(two_tenants, monkeypatch):
    corpus, persist_dir = two_tenants
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(persist_dir))
    from rag_assistant.config import get_settings

    get_settings.cache_clear()
    client = TestClient(api.app)
    token = auth.owner_var.set("alice")
    try:
        response = client.delete("/api/v1/tenant/data")
    finally:
        auth.owner_var.reset(token)

    assert response.status_code == 200
    body = response.json()
    assert body["owner"] == "alice"
    assert body["sources_removed"] >= 1
    assert body["corpus_files_removed"] is True
    assert (corpus / TENANT_DIR / "bob").exists()
