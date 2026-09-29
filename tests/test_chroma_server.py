"""Chroma in *server* mode, against a real Chroma server.

`CHROMA_SERVER_HOST` is the documented way off the single-worker ceiling: embedded Chroma
locks its SQLite file to one process, so a second replica needs the index to live somewhere
both can reach. Until now that switch was covered only at the construction boundary -- a test
asserting that an `HttpClient` gets built with the configured host. That checks the wiring and
nothing about whether the resulting store actually stores, filters or retrieves, which is the
only reason the switch exists.

Skips without RAG_TEST_CHROMA_HOST, which is how a developer machine and a fork stay green;
CI sets it against a service container (see .github/workflows/ci.yml).
"""

import os
import uuid

import pytest

from rag_assistant.auth import PUBLIC_OWNER
from rag_assistant.config import get_settings
from rag_assistant.retrieval import vector_store
from rag_assistant.retrieval.vector_store import get_retriever, get_vector_store
from rag_assistant.schemas.api import RetrievalFilters

CHROMA_HOST = os.environ.get("RAG_TEST_CHROMA_HOST")
CHROMA_PORT = os.environ.get("RAG_TEST_CHROMA_PORT", "8000")

pytestmark = pytest.mark.skipif(not CHROMA_HOST, reason="RAG_TEST_CHROMA_HOST is not set")


@pytest.fixture
def server_mode(monkeypatch):
    """Points the app at the real server, with the store cache dropped so it takes effect."""
    monkeypatch.setenv("CHROMA_SERVER_HOST", CHROMA_HOST)
    monkeypatch.setenv("CHROMA_SERVER_PORT", CHROMA_PORT)
    monkeypatch.setenv("VECTOR_BACKEND", "chroma")
    get_settings.cache_clear()
    vector_store.reset_store_cache()
    yield
    vector_store.reset_store_cache()
    get_settings.cache_clear()


@pytest.fixture
def seeded(server_mode, fake_embeddings):
    """A handful of documents owned by two different tenants.

    Ids are unique per run because a server, unlike a tmp_path, is shared state that outlives
    the test -- the collection is the same one every run.
    """
    run = uuid.uuid4().hex[:8]
    store = get_vector_store(embeddings=fake_embeddings)
    ids = [f"{run}-{i}" for i in range(3)]
    store.add_texts(
        texts=[
            "Anthropic builds the Claude model family and researches Constitutional AI.",
            "Mistral AI is a French company building open-weight models such as Mixtral.",
            "A private note about quarterly revenue that belongs to one tenant only.",
        ],
        metadatas=[
            {"owner": PUBLIC_OWNER, "source": f"{run}-anthropic.md", "ingested_at": 1000.0},
            {"owner": PUBLIC_OWNER, "source": f"{run}-mistral.md", "ingested_at": 2000.0},
            {"owner": f"tenant-{run}", "source": f"{run}-private.md", "ingested_at": 3000.0},
        ],
        ids=ids,
    )
    yield run, store, ids
    try:
        store.delete(ids=ids)
    except Exception:  # pragma: no cover - cleanup must not mask a real failure
        pass


def test_the_store_talks_to_the_server_rather_than_a_local_directory(server_mode, fake_embeddings):
    """The point of the switch: no SQLite file, so no single-process lock."""
    store = get_vector_store(embeddings=fake_embeddings)

    assert store._collection.count() >= 0  # a real round trip to the server
    # `is_persistent` is the discriminator that matters: False means no local SQLite file,
    # which is the whole reason server mode exists. The client class name is the same either
    # way, so asserting on it would pass against an embedded store.
    client_settings = store._client.get_settings()
    assert client_settings.is_persistent is False
    assert "fastapi" in client_settings.chroma_api_impl.lower()


def test_documents_written_in_server_mode_are_retrievable(seeded, fake_embeddings):
    run, _store, _ids = seeded

    retriever = get_retriever(k=3, embeddings=fake_embeddings, owner=PUBLIC_OWNER)
    hits = retriever.invoke("Which company builds Claude?")

    sources = {doc.metadata["source"] for doc in hits}
    assert f"{run}-anthropic.md" in sources


def test_tenant_scoping_is_enforced_by_the_server(seeded, fake_embeddings):
    """The owner predicate has to be applied by Chroma during search, not after: post-filtering
    silently shrinks k. On the embedded backend that is covered elsewhere; this is the same
    contract over HTTP, where the `where` clause is serialised and evaluated remotely."""
    run, _store, _ids = seeded

    hits = get_retriever(k=5, embeddings=fake_embeddings, owner=PUBLIC_OWNER).invoke(
        "quarterly revenue"
    )

    assert all(doc.metadata["owner"] == PUBLIC_OWNER for doc in hits)
    assert f"{run}-private.md" not in {doc.metadata["source"] for doc in hits}


def test_the_owner_sees_their_own_documents(seeded, fake_embeddings):
    run, _store, _ids = seeded

    hits = get_retriever(k=5, embeddings=fake_embeddings, owner=f"tenant-{run}").invoke(
        "quarterly revenue"
    )

    assert f"{run}-private.md" in {doc.metadata["source"] for doc in hits}


def test_metadata_filters_survive_the_wire(seeded, fake_embeddings):
    """A multi-clause `$and` is built locally and evaluated on the server; a shape Chroma
    rejects is a 4xx from the server rather than a local ValueError, so it needs checking
    here and not only against the embedded client."""
    run, _store, _ids = seeded

    filters = RetrievalFilters(sources=[f"{run}-mistral.md"])
    hits = get_retriever(
        k=5, embeddings=fake_embeddings, owner=PUBLIC_OWNER, filters=filters
    ).invoke("open-weight models")

    assert hits, "a source filter that matches an indexed document returned nothing"
    assert {doc.metadata["source"] for doc in hits} == {f"{run}-mistral.md"}


def test_the_collection_uses_cosine_distance(server_mode, fake_embeddings):
    """Chroma defaults to squared L2. Getting this wrong ranks by the wrong metric and is
    invisible -- the server has its own copy of the collection metadata, so the embedded
    test cannot speak for it."""
    store = get_vector_store(embeddings=fake_embeddings)

    metadata = store._collection.metadata or {}
    assert metadata.get("hnsw:space") == "cosine"


def test_server_mode_and_embedded_mode_can_coexist_in_one_process(
    server_mode, fake_embeddings, tmp_path
):
    """The cache key separates them; a collision would hand a caller the wrong index."""
    remote = get_vector_store(embeddings=fake_embeddings)
    local = get_vector_store(embeddings=fake_embeddings, persist_dir=tmp_path / "embedded")

    assert remote is not local
    assert remote._client.get_settings().is_persistent is False
    assert local._client.get_settings().is_persistent is True
