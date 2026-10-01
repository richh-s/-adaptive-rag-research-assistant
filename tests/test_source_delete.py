"""Removing one document rather than everything a tenant owns.

Erasure existed only per tenant, which is the right unit for "delete my account" and the
wrong one for every other reason a document has to go: a takedown request, a retention date,
a file whose parse was bad enough that re-ingesting would reproduce the problem.
"""

import pytest
from fastapi.testclient import TestClient

from rag_assistant import auth, tenancy
from rag_assistant.api import app
from rag_assistant.config import get_settings
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.ingestion.manifest import load_manifest
from rag_assistant.retrieval.bm25_store import bm25_search
from rag_assistant.retrieval.vector_store import get_retriever, reset_store_cache


@pytest.fixture
def indexed(sample_corpus_dir, fake_embeddings, tmp_path, monkeypatch):
    """A two-document corpus, indexed, with the app pointed at it."""
    # A third document, because BM25's IDF is zero for a term appearing in one of two
    # documents -- the keyword assertion below would then be testing arithmetic, not deletion.
    (sample_corpus_dir / "cohere.md").write_text(
        "Cohere builds enterprise language models and embedding APIs for retrieval."
    )
    persist_dir = tmp_path / "chroma"
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(persist_dir))
    monkeypatch.setenv("CORPUS_DIR", str(sample_corpus_dir))
    get_settings.cache_clear()
    reset_store_cache()
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    return persist_dir, sample_corpus_dir


def test_deleting_a_source_removes_it_from_every_store(indexed, fake_embeddings):
    persist_dir, corpus_dir = indexed

    result = tenancy.purge_source("anthropic.md", owner="public", persist_dir=persist_dir)

    assert result.chunks > 0
    assert result.file_removed is True
    assert "anthropic.md" not in load_manifest(persist_dir)
    assert not (corpus_dir / "anthropic.md").exists()
    remaining = get_retriever(k=20, embeddings=fake_embeddings, persist_dir=persist_dir).invoke(
        "Anthropic Constitutional AI"
    )
    assert all(d.metadata["source"] != "anthropic.md" for d in remaining)


def test_the_keyword_index_stops_serving_the_deleted_text(indexed):
    """BM25 is an in-memory index built from the chunks that just disappeared. Without
    invalidation it keeps returning the deleted document's text until the process restarts --
    the same failure `purge_tenant` already guards against."""
    persist_dir, _ = indexed
    assert bm25_search("Constitutional", k=10, persist_dir=persist_dir, owner="public")

    tenancy.purge_source("anthropic.md", owner="public", persist_dir=persist_dir)

    hits = bm25_search("Constitutional", k=10, persist_dir=persist_dir, owner="public")
    assert all(d.source_id != "anthropic.md" for d in hits)


def test_the_other_documents_are_untouched(indexed):
    persist_dir, corpus_dir = indexed

    tenancy.purge_source("anthropic.md", owner="public", persist_dir=persist_dir)

    assert "mistral.md" in load_manifest(persist_dir)
    assert (corpus_dir / "mistral.md").exists()


def test_deleting_someone_elses_source_reports_it_as_absent(indexed):
    """Not "forbidden": that would confirm the file exists to anyone willing to guess names,
    which is the leak the retrieval filter exists to prevent."""
    persist_dir, _ = indexed

    with pytest.raises(tenancy.SourceNotFound):
        tenancy.purge_source("anthropic.md", owner="alice", persist_dir=persist_dir)

    assert "anthropic.md" in load_manifest(persist_dir)


def test_deleting_an_unknown_source_raises_rather_than_silently_succeeding(indexed):
    persist_dir, _ = indexed
    with pytest.raises(tenancy.SourceNotFound):
        tenancy.purge_source("nope.md", owner="public", persist_dir=persist_dir)


def test_a_source_resolving_outside_the_corpus_is_refused(indexed, monkeypatch):
    """The manifest key is derived from an uploaded filename. Confirming it resolves inside
    the corpus costs one comparison against the cost of a traversal deleting something else.
    """
    persist_dir, corpus_dir = indexed
    outside = corpus_dir.parent / "not_corpus.md"
    outside.write_text("should survive")
    manifest = load_manifest(persist_dir)
    manifest["../not_corpus.md"] = {"owner": "public", "chunk_ids": []}
    from rag_assistant.ingestion.manifest import save_manifest

    save_manifest(persist_dir, manifest)

    result = tenancy.purge_source("../not_corpus.md", owner="public", persist_dir=persist_dir)

    assert result.file_removed is False
    assert outside.exists()


def test_the_endpoint_deletes_and_reports_what_it_removed(indexed):
    response = TestClient(app).delete("/api/v1/sources/anthropic.md")

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "anthropic.md"
    assert body["chunks_removed"] > 0
    assert body["file_removed"] is True


def test_the_endpoint_handles_tenant_paths_with_slashes(indexed, monkeypatch, fake_embeddings):
    """A source key carries slashes for tenant-owned files, and the un-suffixed path
    converter would stop at the first one and 404 every tenant document."""
    persist_dir, corpus_dir = indexed
    tenant_dir = corpus_dir / "_t" / "alice"
    tenant_dir.mkdir(parents=True)
    (tenant_dir / "private.md").write_text("Alice's private notes about retrieval.")
    reset_store_cache()
    build_index(source_dir=corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    monkeypatch.setattr(auth, "get_owner", lambda: "alice")

    response = TestClient(app).delete("/api/v1/sources/_t/alice/private.md")

    assert response.status_code == 200
    assert response.json()["source"] == "_t/alice/private.md"


def test_the_endpoint_404s_for_an_unknown_source(indexed):
    response = TestClient(app).delete("/api/v1/sources/nope.md")

    assert response.status_code == 404


def test_deleting_documents_requires_a_write_scope():
    assert auth.required_scope("DELETE", "/api/v1/sources/a.md") == auth.WRITE


def test_purging_a_tenant_requires_a_write_scope():
    """Regression: this fell through to the READ default, so a key issued read-only could
    erase every document, conversation and feedback row its tenant owned -- while the
    endpoint's docstring promised a read-only key a 403."""
    assert auth.required_scope("DELETE", "/api/v1/tenant/data") == auth.WRITE
