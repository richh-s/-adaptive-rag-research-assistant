"""Index generations: rebuilding the index beside the serving one and switching to it.

The properties under test are the ones that make a re-embed safe to run against a live
deployment: serving never sees a half-built index, the new generation converges on whatever
the corpus holds by the time it is activated (including changes made during the build),
queries are always embedded with the model that built the generation they read, and rolling
back is a pointer flip rather than a rebuild.

Two fake embedding models with *different dimensions* stand in for "the old model" and "the
new model", so a query embedded with the wrong one fails loudly instead of returning
plausible neighbours -- which is exactly the failure a real model swap would hide.
"""

import pytest

from rag_assistant.config import get_settings
from rag_assistant.ingestion import generations, reindex
from rag_assistant.ingestion.acl import DocumentAcl, write_acl
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.ingestion.index_metadata import load_index_metadata
from rag_assistant.ingestion.manifest import load_manifest
from rag_assistant.ingestion.ownership import TENANT_DIR
from rag_assistant.retrieval.bm25_store import bm25_search
from rag_assistant.retrieval.vector_store import (
    TenantCollections,
    get_retriever,
    get_vector_store,
    reset_store_cache,
)
from tests.conftest import FakeHashingEmbeddings

OLD_MODEL = "models/gemini-embedding-001"
NEW_MODEL = "openai/text-embedding-3-large"


class _NamedEmbeddings(FakeHashingEmbeddings):
    def __init__(self, name: str, dim: int):
        super().__init__(dim=dim)
        self.name = name
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return super().embed_documents(texts)


@pytest.fixture
def models(monkeypatch):
    """Resolves recorded model names to fake models, everywhere a model is resolved."""
    registry = {
        OLD_MODEL: _NamedEmbeddings(OLD_MODEL, 32),
        NEW_MODEL: _NamedEmbeddings(NEW_MODEL, 48),
    }

    def _resolve(model_name=None):
        return registry[model_name or get_settings().embedding_model_name]

    monkeypatch.setattr("rag_assistant.retrieval.vector_store.get_embeddings_model", _resolve)
    monkeypatch.setattr("rag_assistant.ingestion.reindex.get_embeddings_model", _resolve)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    return registry


@pytest.fixture
def deployment(tmp_path, monkeypatch, models):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "anthropic.md").write_text("Anthropic builds the Claude model family.")
    (corpus / "mistral.md").write_text("Mistral AI is a French company building Mixtral.")
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("INDEX_POINTER_POLL_SECONDS", "0")
    get_settings.cache_clear()
    reset_store_cache()
    build_index()
    yield corpus
    reset_store_cache()


def _sources(query: str = "company model family", owner: str = "public", k: int = 10):
    docs = get_retriever(k=k, owner=owner).invoke(query)
    return {d.metadata["source"] for d in docs}


def test_without_a_pointer_the_legacy_index_serves(deployment):
    assert generations.active_generation() == generations.LEGACY
    assert generations.active_index_dir() == get_settings().chroma_persist_dir
    assert _sources() == {"anthropic.md", "mistral.md"}


def test_a_generation_directory_maps_back_to_its_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path))
    get_settings.cache_clear()
    directory = generations.generation_dir("g20260101000000abcdef")
    assert generations.generation_of_dir(directory) == "g20260101000000abcdef"
    # Every other directory is the legacy generation, which is what tests and older callers
    # passing their own persist_dir have always meant.
    assert generations.generation_of_dir(tmp_path / "elsewhere") == generations.LEGACY


def test_generation_ids_are_validated_before_they_touch_a_path_or_schema():
    for bad in ("../etc", "g1", "G20260101000000", "g2026; DROP SCHEMA public"):
        with pytest.raises(ValueError):
            generations.generation_dir(bad)


def test_building_a_generation_does_not_change_what_serves(deployment, models):
    result = reindex.build_generation(embedding_model=NEW_MODEL)

    assert generations.active_generation() == generations.LEGACY
    assert result.mode == "re-embed"
    assert result.sources == 2
    new_dir = generations.generation_dir(result.generation)
    assert load_index_metadata(new_dir).embedding_model == NEW_MODEL
    # Serving still embeds with the model the legacy index recorded.
    assert _sources() == {"anthropic.md", "mistral.md"}


def test_a_re_embed_reuses_the_indexed_text_and_recomputes_only_vectors(deployment, models):
    """No re-parse, no vision, no description calls: the text is byte-identical, so only the
    vectors are recomputed -- once per chunk, not twice."""
    legacy_chunks = sum(
        len(e["chunk_ids"]) for e in load_manifest(get_settings().chroma_persist_dir).values()
    )

    result = reindex.build_generation(embedding_model=NEW_MODEL)

    assert result.chunks == legacy_chunks
    assert models[NEW_MODEL].calls == legacy_chunks
    assert result.catch_up.parsed_files == 0


def test_activation_switches_serving_to_the_new_model(deployment, models):
    result = reindex.build_generation(embedding_model=NEW_MODEL)

    reindex.activate(result.generation, settle=False)

    assert generations.active_generation() == result.generation
    store = get_vector_store()
    assert store.embeddings is models[NEW_MODEL]
    assert _sources() == {"anthropic.md", "mistral.md"}


def test_changes_made_during_the_build_reach_the_new_generation(deployment, models):
    """Ingestion keeps writing to the serving generation while a build runs. The catch-up
    pass must bring every addition, deletion and permission change across."""
    result = reindex.build_generation(embedding_model=NEW_MODEL)
    (deployment / "cohere.md").write_text("Cohere builds enterprise language models.")
    (deployment / "mistral.md").unlink()
    build_index()  # the live deployment ingests into the serving (legacy) generation

    reindex.activate(result.generation, settle=False)

    manifest = load_manifest(generations.active_index_dir())
    assert set(manifest) == {"anthropic.md", "cohere.md"}
    assert _sources() == {"anthropic.md", "cohere.md"}


def test_rollback_restores_the_previous_generation_and_catches_it_up(deployment, models):
    result = reindex.build_generation(embedding_model=NEW_MODEL)
    reindex.activate(result.generation, settle=False)
    (deployment / "cohere.md").write_text("Cohere builds enterprise language models.")
    build_index()

    pointer = reindex.rollback()

    assert pointer.generation == generations.LEGACY
    assert get_vector_store().embeddings is models[OLD_MODEL]
    # A document ingested while the new generation served is not lost by rolling back.
    assert "cohere.md" in _sources()


def test_garbage_collection_keeps_the_serving_and_rollback_generations(deployment, models):
    first = reindex.build_generation(embedding_model=NEW_MODEL).generation
    reindex.activate(first, settle=False)
    second = reindex.build_generation(embedding_model=NEW_MODEL).generation
    unused = reindex.build_generation(embedding_model=NEW_MODEL).generation
    reindex.activate(second, settle=False)

    deleted = reindex.garbage_collect()

    assert deleted == [unused]
    assert not generations.generation_dir(unused).exists()
    assert generations.generation_dir(first).exists()
    assert set(generations.list_generations()) == {generations.LEGACY, first, second}


def test_a_generation_that_was_never_built_cannot_be_activated(deployment):
    with pytest.raises(reindex.ReindexError):
        reindex.activate("g20260101000000abcdef", settle=False)


def test_permissions_are_carried_into_the_new_generation(deployment, models):
    restricted = deployment / "mistral.md"
    write_acl(restricted, DocumentAcl(groups=frozenset({"finance"})))
    build_index()

    result = reindex.build_generation(embedding_model=NEW_MODEL)
    reindex.activate(result.generation, settle=False)

    docs = get_retriever(k=10, principals=frozenset({"group:engineering"})).invoke("company")
    assert {d.metadata["source"] for d in docs} == {"anthropic.md"}
    hits = bm25_search("Mistral", k=10, principals=frozenset({"group:engineering"}))
    assert hits == []


def test_a_corpus_build_re_parses_every_file(deployment, models):
    result = reindex.build_generation(embedding_model=NEW_MODEL, from_corpus=True)

    assert result.mode == "corpus"
    assert result.catch_up.parsed_files == 2
    assert result.sources == 2


def test_readiness_reports_a_configured_model_that_differs_as_pending(deployment, monkeypatch):
    from rag_assistant import readiness

    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    get_settings.cache_clear()

    ok, note = readiness.check_embeddings()

    assert ok
    assert OLD_MODEL in note


# ---- strict tenant isolation ----


@pytest.fixture
def tenant_deployment(tmp_path, monkeypatch, models):
    corpus = tmp_path / "corpus"
    for owner, text in (
        ("alice", "Project Zephyr is Alice's confidential revenue plan."),
        ("bob", "Project Mistral is Bob's confidential revenue plan."),
    ):
        directory = corpus / TENANT_DIR / owner
        directory.mkdir(parents=True)
        (directory / f"{owner}_secret.md").write_text(text)
    (corpus / "baseline.md").write_text("Anthropic publishes confidential revenue research.")
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("INDEX_POINTER_POLL_SECONDS", "0")
    get_settings.cache_clear()
    reset_store_cache()
    yield corpus
    reset_store_cache()


def _owners_seen(owner):
    docs = get_retriever(k=10, owner=owner).invoke("confidential revenue plan")
    return {d.metadata["owner"] for d in docs}


def test_strict_isolation_gives_each_tenant_its_own_collection(tenant_deployment, monkeypatch):
    monkeypatch.setenv("TENANT_ISOLATION", "strict")
    get_settings.cache_clear()
    build_index()

    store = get_vector_store()
    assert isinstance(store, TenantCollections)
    assert store.collection_for("alice")._collection.count() == 1
    assert store.collection_for("bob")._collection.count() == 1
    assert store.collection_for("public")._collection.count() == 1
    assert _owners_seen("alice") == {"alice", "public"}
    assert _owners_seen("bob") == {"bob", "public"}
    assert _owners_seen("public") == {"public"}


def test_under_strict_isolation_another_tenants_vectors_are_not_searched_at_all(
    tenant_deployment, monkeypatch
):
    """The property that makes it a second boundary: even with the where-clause removed, a
    tenant's search cannot reach another tenant's collection."""
    monkeypatch.setenv("TENANT_ISOLATION", "strict")
    get_settings.cache_clear()
    build_index()
    monkeypatch.setattr(
        "rag_assistant.retrieval.vector_store.build_where_clause",
        lambda owner, filters=None, principals=None: {"source": {"$ne": ""}},
    )

    assert "bob" not in _owners_seen("alice")


def test_strict_isolation_supports_deletes_the_keyword_index_and_erasure(
    tenant_deployment, monkeypatch
):
    from rag_assistant import tenancy

    monkeypatch.setenv("TENANT_ISOLATION", "strict")
    get_settings.cache_clear()
    build_index()

    # Terms unique to one document each: a term in every document has zero IDF under BM25.
    hits = bm25_search("Zephyr Mistral Anthropic", k=10, owner="alice")
    assert {h.metadata["owner"] for h in hits} == {"alice", "public"}

    tenancy.purge_tenant("alice")
    store = get_vector_store()
    assert store.collection_for("alice")._collection.count() == 0
    assert store.collection_for("bob")._collection.count() == 1


def test_changing_the_setting_never_hides_an_existing_shared_index(tenant_deployment, monkeypatch):
    """An index written as one shared collection is read as one until a new generation
    replaces it -- flipping TENANT_ISOLATION alone must not make every tenant's documents
    vanish."""
    build_index()  # recorded as the shared layout
    monkeypatch.setenv("TENANT_ISOLATION", "strict")
    get_settings.cache_clear()
    reset_store_cache()

    assert not isinstance(get_vector_store(), TenantCollections)
    assert _owners_seen("alice") == {"alice", "public"}


def test_moving_to_strict_isolation_is_a_rebuild_and_a_pointer_flip(tenant_deployment, monkeypatch):
    build_index()
    monkeypatch.setenv("TENANT_ISOLATION", "strict")
    get_settings.cache_clear()

    result = reindex.build_generation(embedding_model=OLD_MODEL)
    reindex.activate(result.generation, settle=False)

    store = get_vector_store()
    assert isinstance(store, TenantCollections)
    assert store.collection_for("alice")._collection.count() == 1
    assert _owners_seen("alice") == {"alice", "public"}
    assert _owners_seen("bob") == {"bob", "public"}
