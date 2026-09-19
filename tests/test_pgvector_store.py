"""Tests for the pgvector vector-store backend.

Skipped unless a Postgres with the `vector` extension is reachable at
`RAG_TEST_DATABASE_URL`, so a default checkout and CI without a database service stay green.
Run one locally with:

    initdb -D /tmp/ragpg/data -U postgres --auth=trust
    pg_ctl -D /tmp/ragpg/data -l /tmp/ragpg/log \
        -o "-p 55432 -c unix_socket_directories= -c listen_addresses=127.0.0.1" start
    RAG_TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/postgres \
        uv run pytest tests/test_pgvector_store.py

These assert the *same* behaviours the Chroma tests do -- retrieval, idempotent re-ingest,
tenant isolation, metadata filtering -- because the two backends are interchangeable only if
they actually behave the same. A backend that merely stores vectors without matching the
tenancy semantics would be a data-isolation bug that flipping one config value silently
introduces, which is precisely the kind of thing a unit test of the filter helper alone
would never catch.
"""

import hashlib
import os

import pytest
from langchain_core.embeddings import Embeddings

from rag_assistant.config import get_settings
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.ingestion.ownership import TENANT_DIR
from rag_assistant.retrieval.vector_store import (
    get_retriever,
    get_vector_store,
    reset_store_cache,
)

DATABASE_URL = os.environ.get("RAG_TEST_DATABASE_URL", "")


def _pgvector_reachable() -> bool:
    if not DATABASE_URL:
        return False
    try:
        import psycopg

        with psycopg.connect(DATABASE_URL, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _pgvector_reachable(),
    reason="No Postgres with pgvector at RAG_TEST_DATABASE_URL",
)


@pytest.fixture(autouse=True)
def pgvector_backend(monkeypatch):
    """Points the store at Postgres and hands every test an empty schema.

    The tables are dropped rather than truncated so each test re-runs the migration from
    scratch: the embedding column's width is fixed on first write, so a truncated table
    would carry the previous test's dimension into the next one and quietly make the
    self-configuration test pass for the wrong reason.

    Every index table goes, not just the chunks. Dropping `corpus_chunks` while leaving
    `corpus_manifest` reproduces exactly the divergence this backend exists to prevent: the
    manifest reports the corpus as already indexed, ingestion skips every file, and the test
    retrieves from an empty collection. It cost a debugging round the first time.
    """
    import psycopg

    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    get_settings.cache_clear()
    reset_store_cache()

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for table in (
                "corpus_chunks",
                "corpus_manifest",
                "corpus_parents",
                "corpus_index_state",
                "pgvector_schema_migrations",
            ):
                cur.execute(f"DROP TABLE IF EXISTS {table}")
            conn.commit()
    yield
    reset_store_cache()


@pytest.fixture
def tenant_corpus(tmp_path):
    """One shared public file plus one private file per tenant -- the same shape
    tests/test_tenancy.py builds for the Chroma path."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "baseline.md").write_text(
        "Anthropic is an AI safety company known for Constitutional AI research."
    )
    alice = corpus / TENANT_DIR / "alice"
    alice.mkdir(parents=True)
    (alice / "alice_secret.md").write_text(
        "Project Zephyr is Alice's confidential quarterly revenue plan for widgets."
    )
    bob = corpus / TENANT_DIR / "bob"
    bob.mkdir(parents=True)
    (bob / "bob_secret.md").write_text(
        "Project Mistral is Bob's confidential quarterly revenue plan for gadgets."
    )
    return corpus


def _sql_write(query, params=()):
    import psycopg

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            conn.commit()


def _sql(query, params=()):
    import psycopg

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()


# ---- parity with the Chroma tests ----


def test_build_index_and_retrieve_relevant_chunk(sample_corpus_dir, fake_embeddings, tmp_path):
    result = build_index(
        source_dir=sample_corpus_dir, persist_dir=tmp_path / "idx", embeddings=fake_embeddings
    )
    assert result.indexed_chunks > 0
    assert result.changed_files == 2

    results = get_retriever(k=1, embeddings=fake_embeddings, persist_dir=tmp_path / "idx").invoke(
        "Who founded Anthropic and what model do they build?"
    )

    assert len(results) == 1
    assert results[0].metadata["source"] == "anthropic.md"


def test_build_index_is_idempotent_on_rerun(sample_corpus_dir, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "idx"
    first = build_index(
        source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings
    )
    second = build_index(
        source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings
    )

    assert second.changed_files == 0
    assert second.skipped_files == 2
    # The row count, not just the reported counters: an upsert that re-inserted under new
    # ids would still report "skipped" while doubling the table.
    assert _sql("SELECT COUNT(*) FROM corpus_chunks")[0][0] == first.indexed_chunks


def test_reindexing_a_changed_file_replaces_rather_than_duplicates(
    sample_corpus_dir, fake_embeddings, tmp_path
):
    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    (sample_corpus_dir / "anthropic.md").write_text(
        "Anthropic builds Claude and now also publishes interpretability research."
    )
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)

    rows = _sql("SELECT content FROM corpus_chunks WHERE source = 'anthropic.md'")
    assert len(rows) == 1
    assert "interpretability" in rows[0][0]


# ---- tenant isolation ----


def test_a_tenant_cannot_retrieve_another_tenants_document(
    tenant_corpus, fake_embeddings, tmp_path
):
    persist_dir = tmp_path / "idx"
    build_index(source_dir=tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)

    alice_hits = get_retriever(
        k=10, embeddings=fake_embeddings, persist_dir=persist_dir, owner="alice"
    ).invoke("confidential quarterly revenue plan")
    sources = {d.metadata["source"] for d in alice_hits}

    assert any("alice_secret" in s for s in sources)
    assert not any("bob_secret" in s for s in sources)


def test_the_public_tenant_sees_only_public_documents(tenant_corpus, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "idx"
    build_index(source_dir=tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)

    hits = get_retriever(k=10, embeddings=fake_embeddings, persist_dir=persist_dir).invoke(
        "confidential quarterly revenue plan"
    )

    assert {d.metadata["source"] for d in hits} == {"baseline.md"}


def test_tenant_filtering_does_not_shrink_k(tenant_corpus, fake_embeddings, tmp_path):
    """The reason the predicate is in the SQL rather than applied to the result set.

    Post-filtering would take the global top-k and then drop the rows belonging to other
    tenants, so a tenant whose nearest neighbours are someone else's documents gets fewer
    rows -- sometimes none -- with no error. The graph reads that as "the corpus has
    nothing" and falls back to web search, which is a silently wrong answer rather than a
    visible failure.
    """
    persist_dir = tmp_path / "idx"
    build_index(source_dir=tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)

    visible = _sql(
        "SELECT COUNT(*) FROM corpus_chunks WHERE owner = ANY(%s)", (["public", "alice"],)
    )[0][0]
    hits = get_retriever(
        k=visible, embeddings=fake_embeddings, persist_dir=persist_dir, owner="alice"
    ).invoke("gadgets widgets revenue")

    assert len(hits) == visible


# ---- metadata filters ----


def test_source_filter_narrows_retrieval(sample_corpus_dir, fake_embeddings, tmp_path):
    from rag_assistant.schemas.api import RetrievalFilters

    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)

    hits = get_retriever(
        k=10,
        embeddings=fake_embeddings,
        persist_dir=persist_dir,
        filters=RetrievalFilters(sources=["mistral.md"]),
    ).invoke("Anthropic Claude Constitutional AI")

    assert hits
    assert {d.metadata["source"] for d in hits} == {"mistral.md"}


def test_ingested_before_filter_excludes_everything_indexed_after_it(
    sample_corpus_dir, fake_embeddings, tmp_path
):
    from datetime import datetime, timedelta, timezone

    from rag_assistant.schemas.api import RetrievalFilters

    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)

    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    hits = get_retriever(
        k=10,
        embeddings=fake_embeddings,
        persist_dir=persist_dir,
        filters=RetrievalFilters(ingested_before=cutoff),
    ).invoke("Anthropic")

    assert hits == []


# ---- the surface the rest of the codebase reaches for ----


def test_get_returns_the_chunks_bm25_indexes_from(sample_corpus_dir, fake_embeddings, tmp_path):
    """BM25 builds its index from the vector store's stored chunks rather than re-reading the
    corpus, so `get` has to return the same shape on both backends or the keyword path
    silently indexes nothing."""
    persist_dir = tmp_path / "idx"
    result = build_index(
        source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings
    )
    store = get_vector_store(embeddings=fake_embeddings, persist_dir=persist_dir)

    stored = store.get(include=["documents", "metadatas"])

    assert len(stored["ids"]) == result.indexed_chunks
    assert len(stored["documents"]) == result.indexed_chunks
    assert all("source" in m for m in stored["metadatas"])


def test_bm25_search_works_against_the_pgvector_backed_chunks(fake_embeddings, tmp_path):
    """Four documents rather than two, matching tests/test_bm25_store.py.

    BM25's IDF term is `log((N - df + 0.5) / (df + 0.5))`, which is <= 0 for a term appearing
    in half or more of the corpus -- and `bm25_search` drops non-positive scores rather than
    padding results with noise. On a two-document corpus every distinguishing term sits at
    exactly df=1, N=2 and scores zero, so the search returns nothing for reasons that have
    nothing to do with which backend stored the chunks.
    """
    from rag_assistant.retrieval.bm25_store import bm25_search, invalidate_bm25_index

    corpus = tmp_path / "bm25corpus"
    corpus.mkdir()
    (corpus / "anthropic.md").write_text(
        "Anthropic was founded by Dario Amodei and builds Claude with Constitutional AI."
    )
    (corpus / "mistral.md").write_text(
        "Mistral AI is a French company in Paris building open-weight models like Mixtral."
    )
    (corpus / "openai.md").write_text(
        "OpenAI builds the GPT family of models and operates the ChatGPT product."
    )
    (corpus / "meta.md").write_text(
        "Meta AI releases the Llama family of open-weight models from its FAIR lab."
    )
    persist_dir = tmp_path / "idx"
    build_index(source_dir=corpus, persist_dir=persist_dir, embeddings=fake_embeddings)
    invalidate_bm25_index(persist_dir)

    hits = bm25_search("Mixtral Paris sovereignty", k=1, persist_dir=persist_dir)

    assert hits
    assert hits[0].metadata["source"] == "mistral.md"


def test_count_and_peek_serve_readiness_and_the_dimension_check(
    sample_corpus_dir, fake_embeddings, tmp_path
):
    from rag_assistant.ingestion.index_metadata import read_embedding_dimension

    persist_dir = tmp_path / "idx"
    result = build_index(
        source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings
    )
    store = get_vector_store(embeddings=fake_embeddings, persist_dir=persist_dir)

    assert store._collection.count() == result.indexed_chunks
    assert read_embedding_dimension(store) == fake_embeddings.dim


def test_delete_removes_only_the_named_chunks(sample_corpus_dir, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    store = get_vector_store(embeddings=fake_embeddings, persist_dir=persist_dir)
    ids = store.get()["ids"]

    store.delete(ids=ids[:1])

    assert set(store.get()["ids"]) == set(ids[1:])


def test_reset_collection_empties_the_table(sample_corpus_dir, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    store = get_vector_store(embeddings=fake_embeddings, persist_dir=persist_dir)

    store.reset_collection()

    assert store._collection.count() == 0


# ---- dimension self-configuration and embedding drift ----


def test_first_write_fixes_the_dimension_and_builds_the_hnsw_index(
    sample_corpus_dir, fake_embeddings, tmp_path
):
    build_index(
        source_dir=sample_corpus_dir, persist_dir=tmp_path / "idx", embeddings=fake_embeddings
    )

    typmod = _sql(
        "SELECT a.atttypmod FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
        "WHERE c.relname = 'corpus_chunks' AND a.attname = 'embedding'"
    )[0][0]
    indexes = [
        r[0]
        for r in _sql("SELECT indexname FROM pg_indexes WHERE tablename=%s", ("corpus_chunks",))
    ]

    assert typmod == fake_embeddings.dim
    assert "idx_corpus_chunks_embedding_hnsw" in indexes


def test_an_embedding_model_of_a_different_width_fails_loudly(
    sample_corpus_dir, fake_embeddings, tmp_path
):
    """The failure mode this backend is meant to remove.

    With the index built at one model's width, pointing queries at a model of a *different*
    width is caught by pgvector itself. The dangerous case elsewhere is a model of the *same*
    width, which returns plausible neighbours from a space the stored vectors don't occupy --
    that one is still the embedding-model check in index_metadata, not this.
    """
    from conftest import FakeHashingEmbeddings

    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    reset_store_cache()

    wider = FakeHashingEmbeddings(dim=fake_embeddings.dim * 2)
    (sample_corpus_dir / "anthropic.md").write_text("Anthropic rewrote this file entirely.")

    with pytest.raises(Exception) as excinfo:
        build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=wider)

    assert "dimension" in str(excinfo.value).lower()


# ---- backend parity ----


@pytest.fixture
def parity_corpus(tmp_path):
    """Six public documents with overlapping vocabulary, so a ranking comparison has enough
    room to disagree. Two visible documents would put the assertion within coin-flip distance
    of passing by accident."""
    corpus = tmp_path / "parity"
    corpus.mkdir()
    (corpus / "anthropic.md").write_text(
        "Anthropic was founded by Dario Amodei and builds the Claude model family, "
        "focused on Constitutional AI and interpretability research."
    )
    (corpus / "mistral.md").write_text(
        "Mistral AI is a French company founded in Paris building open-weight models "
        "like Mixtral, emphasizing European AI sovereignty."
    )
    (corpus / "openai.md").write_text(
        "OpenAI builds the GPT family and operates ChatGPT, with research spanning "
        "reinforcement learning and alignment."
    )
    (corpus / "deepmind.md").write_text(
        "Google DeepMind builds Gemini and publishes research on reinforcement learning, "
        "protein folding and AI safety."
    )
    (corpus / "meta.md").write_text(
        "Meta AI releases the Llama family of open-weight models from its FAIR lab, "
        "emphasizing open research."
    )
    (corpus / "cohere.md").write_text(
        "Cohere builds enterprise language models and embedding models for retrieval "
        "augmented generation."
    )
    return corpus


def test_ranking_matches_the_chroma_backend_on_the_same_corpus(
    parity_corpus, fake_embeddings, tmp_path, monkeypatch
):
    """The claim that makes this a *backend* rather than a second system.

    Both stores are handed the same corpus and the same deterministic embeddings, then asked
    the same questions, and must return the same documents in the same order -- order, not
    membership, because a backend that retrieves the right set in the wrong order quietly
    changes which document the synthesis prompt sees first.

    This does *not* pin the distance metric. `FakeHashingEmbeddings` returns unit vectors, and
    on unit vectors cosine and Euclidean rank identically, so this test passes just as happily
    against an L2 ordering (verified by mutating the operator). That is what
    `test_ranking_is_cosine_not_euclidean` below is for.
    """
    queries = [
        "Who founded Anthropic and what is Constitutional AI?",
        "open-weight models from a European company",
        "reinforcement learning and AI safety research",
        "embedding models for retrieval augmented generation",
    ]

    def _rank(backend: str, persist_dir):
        monkeypatch.setenv("VECTOR_BACKEND", backend)
        get_settings.cache_clear()
        reset_store_cache()
        build_index(source_dir=parity_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)
        return [
            [
                d.metadata["source"]
                for d in get_retriever(
                    k=4, embeddings=fake_embeddings, persist_dir=persist_dir
                ).invoke(q)
            ]
            for q in queries
        ]

    pg_ranking = _rank("pgvector", tmp_path / "pg")
    chroma_ranking = _rank("chroma", tmp_path / "chroma")

    assert pg_ranking == chroma_ranking
    # Guards the assertion above against passing because both backends returned nothing, or
    # returned so few documents that there was no order to disagree about.
    assert all(len(r) == 4 for r in pg_ranking)


class _UnnormalizedHashingEmbeddings(Embeddings):
    """`FakeHashingEmbeddings` without the normalization step.

    Magnitude therefore carries information, which is the only condition under which cosine
    and Euclidean distance disagree about ordering. The shared fixture cannot be used for the
    metric test precisely because it normalizes.
    """

    dim = 64

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in text.lower().split():
            vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        return vector

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def test_ranking_is_cosine_not_euclidean(tmp_path):
    """Pins the operator class, which nothing else here does.

    `aligned.md` repeats the query term and nothing else, so it points exactly along the
    query vector but three units out. `mixed.md` contains the query term once plus an
    unrelated term, so it sits closer in straight-line terms but off-axis. Cosine ranks
    `aligned` first (identical direction); Euclidean ranks `mixed` first (shorter distance).

    An HNSW index built with the wrong operator class does not error -- the `<=>` query
    simply stops using it and falls back to a sequential scan -- so the metric has to be
    asserted on results rather than inferred from the schema.
    """
    corpus = tmp_path / "metric"
    corpus.mkdir()
    (corpus / "aligned.md").write_text("alpha alpha alpha")
    (corpus / "mixed.md").write_text("alpha beta")

    embeddings = _UnnormalizedHashingEmbeddings()
    # The premise fails silently if the two terms collide into one bucket.
    assert embeddings._embed("alpha") != embeddings._embed("beta")

    build_index(source_dir=corpus, persist_dir=tmp_path / "idx", embeddings=embeddings)
    hits = get_retriever(k=2, embeddings=embeddings, persist_dir=tmp_path / "idx").invoke("alpha")

    assert [d.metadata["source"] for d in hits] == ["aligned.md", "mixed.md"]


# ---- shared index state (manifest, parents, cross-replica BM25) ----


def test_the_manifest_lives_in_postgres_not_on_local_disk(
    sample_corpus_dir, fake_embeddings, tmp_path
):
    """The whole point of moving it: a second replica reads the same manifest.

    Asserting the local file is *absent* rather than merely that the rows exist -- a manifest
    written to both places would pass a rows-exist check while still letting the two diverge.
    """
    from rag_assistant.ingestion.manifest import load_manifest, manifest_path

    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)

    assert not manifest_path(persist_dir).exists()
    assert _sql("SELECT COUNT(*) FROM corpus_manifest")[0][0] == 2
    # Read back through a persist_dir that has never been written to: this is what a replica
    # that did not perform the ingest sees.
    assert set(load_manifest(tmp_path / "never-written")) == {"anthropic.md", "mistral.md"}


def test_a_removed_source_disappears_from_the_shared_manifest(
    sample_corpus_dir, fake_embeddings, tmp_path
):
    """Callers encode removal by omitting the source from the dict they save, so the shared
    implementation has to delete rather than only upsert."""
    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    (sample_corpus_dir / "mistral.md").unlink()
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)

    sources = {r[0] for r in _sql("SELECT source FROM corpus_manifest")}

    assert sources == {"anthropic.md"}
    assert _sql("SELECT COUNT(*) FROM corpus_chunks WHERE source = 'mistral.md'")[0][0] == 0


def test_parent_sections_resolve_from_a_replica_that_did_not_ingest(
    sample_corpus_dir, fake_embeddings, tmp_path
):
    """Small-to-big degrades silently when parents are missing -- it falls back to the chunk
    and nothing errors -- so this asserts the sections are readable through a persist
    directory that never saw the ingest."""
    from rag_assistant.retrieval.parent_store import count_parents, get_parents

    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    parent_ids = [r[0] for r in _sql("SELECT parent_id FROM corpus_parents")]

    assert parent_ids
    assert not (persist_dir / "parents.db").exists()
    resolved = get_parents(tmp_path / "never-written", parent_ids)
    assert set(resolved) == set(parent_ids)
    assert all(v for v in resolved.values())
    assert count_parents(tmp_path / "never-written") == len(parent_ids)


def test_re_chunking_a_source_does_not_orphan_its_old_sections(
    sample_corpus_dir, fake_embeddings, tmp_path
):
    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    before = _sql("SELECT COUNT(*) FROM corpus_parents WHERE source = 'anthropic.md'")[0][0]
    (sample_corpus_dir / "anthropic.md").write_text("Anthropic builds Claude.")
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)

    after = _sql("SELECT COUNT(*) FROM corpus_parents WHERE source = 'anthropic.md'")[0][0]

    assert before >= 1
    assert after >= 1
    # Delete-then-insert, not upsert: the count reflects the new chunking, never the union.
    assert after <= before


def test_ingesting_bumps_the_shared_index_version(sample_corpus_dir, fake_embeddings, tmp_path):
    from rag_assistant.retrieval.pgvector_store import current_index_version

    persist_dir = tmp_path / "idx"
    start = current_index_version()
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    after_change = current_index_version()
    # A no-op re-ingest changes nothing, so it must not bump: a version that moves on every
    # run makes every replica rebuild its BM25 index on every poll forever.
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    after_noop = current_index_version()

    assert after_change > start
    assert after_noop == after_change


def test_a_replica_rebuilds_its_bm25_index_when_another_replica_ingests(
    fake_embeddings, tmp_path, monkeypatch
):
    """The staleness no local call can catch.

    Replica A ingests. Replica B holds a BM25 index built before that and never receives an
    invalidation, because invalidation is an in-process function call. Simulated here by
    building B's index, ingesting, and then reaching past B's own cache invalidation the way
    a separate process would -- B must still converge on the next poll.
    """
    from rag_assistant.retrieval import bm25_store

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("Anthropic builds Claude with Constitutional AI research.")
    (corpus / "b.md").write_text("Mistral builds Mixtral open-weight models in Paris.")
    (corpus / "c.md").write_text("OpenAI builds GPT models and operates ChatGPT.")
    persist_dir = tmp_path / "idx"
    build_index(source_dir=corpus, persist_dir=persist_dir, embeddings=fake_embeddings)

    # Replica B's warm index, and its view of the world.
    monkeypatch.setenv("BM25_VERSION_POLL_SECONDS", "0")
    get_settings.cache_clear()
    bm25_store.invalidate_bm25_index(persist_dir)
    replica_b = bm25_store.get_bm25_index(persist_dir)
    assert "cohere.md::0" not in replica_b.documents

    # Replica A ingests a new document. It would normally invalidate its own cache; this is
    # the one thing a different process cannot do for B.
    (corpus / "cohere.md").write_text("Cohere builds enterprise retrieval augmented generation.")
    build_index(source_dir=corpus, persist_dir=persist_dir, embeddings=fake_embeddings)
    bm25_store._index_cache[str(persist_dir)] = replica_b  # undo A's local invalidation

    refreshed = bm25_store.get_bm25_index(persist_dir)

    assert any(chunk_id.startswith("cohere.md") for chunk_id in refreshed.documents)
    assert refreshed.built_at_version > replica_b.built_at_version


def test_a_replica_does_not_rebuild_while_the_version_is_unchanged(
    sample_corpus_dir, fake_embeddings, tmp_path, monkeypatch
):
    """The other half: polling must be cheap and must not thrash. A rebuild re-reads every
    chunk, so a version check that reported "changed" every time would turn each query into a
    full index rebuild."""
    from rag_assistant.retrieval import bm25_store

    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    monkeypatch.setenv("BM25_VERSION_POLL_SECONDS", "0")
    get_settings.cache_clear()
    bm25_store.invalidate_bm25_index(persist_dir)

    first = bm25_store.get_bm25_index(persist_dir)
    again = bm25_store.get_bm25_index(persist_dir)

    assert again is first


# ---- backup and restore of Postgres-backed index state ----


def test_a_backup_taken_on_pgvector_actually_contains_the_index(
    sample_corpus_dir, fake_embeddings, tmp_path
):
    """The regression this exists to prevent.

    `create_backup` archives the persist directory and the corpus. With the index in
    Postgres, that directory holds no vectors, no manifest and no parent sections -- so the
    archive was well-formed, reported its source count correctly (the count is read through
    the live manifest, which *does* reach Postgres), and restored nothing. No error anywhere.
    """
    import json
    import tarfile

    from rag_assistant.backup import create_backup, read_backup_metadata

    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)

    archive = create_backup(
        output_dir=tmp_path / "backups", persist_dir=persist_dir, corpus_dir=sample_corpus_dir
    )

    metadata = read_backup_metadata(archive)
    assert set(metadata.postgres_tables) >= {"corpus_chunks", "corpus_manifest", "corpus_parents"}
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
        assert "postgres/corpus_chunks.jsonl" in names
        chunks = tar.extractfile("postgres/corpus_chunks.jsonl").read().decode()
    rows = [json.loads(line) for line in chunks.splitlines() if line.strip()]
    assert len(rows) == metadata.indexed_sources or len(rows) > 0
    # The embeddings themselves must be in there -- the whole reason to back up an index
    # rather than re-ingest is that re-embedding costs money.
    assert rows[0]["embedding"]


def test_restoring_brings_the_index_back(sample_corpus_dir, fake_embeddings, tmp_path):
    from rag_assistant.backup import create_backup, restore_backup

    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    archive = create_backup(
        output_dir=tmp_path / "backups", persist_dir=persist_dir, corpus_dir=sample_corpus_dir
    )
    before = _sql("SELECT COUNT(*) FROM corpus_chunks")[0][0]
    query = "Who founded Anthropic and what model do they build?"
    # Captured rather than hard-coded. What this test is entitled to assert is that a restore
    # is *faithful* -- that retrieval afterwards behaves exactly as before. Asserting a
    # specific document would additionally encode which one this fake embedding happens to
    # rank first, which is a fact about the fixture, not about the restore.
    ranking_before = [
        d.metadata["source"]
        for d in get_retriever(k=2, embeddings=fake_embeddings, persist_dir=persist_dir).invoke(
            query
        )
    ]

    # Simulate the loss.
    _sql_write("TRUNCATE corpus_chunks, corpus_manifest, corpus_parents")
    assert _sql("SELECT COUNT(*) FROM corpus_chunks")[0][0] == 0

    restore_backup(archive, persist_dir=persist_dir, corpus_dir=sample_corpus_dir)

    assert _sql("SELECT COUNT(*) FROM corpus_chunks")[0][0] == before
    assert _sql("SELECT COUNT(*) FROM corpus_manifest")[0][0] == 2
    # A row count proves the rows came back, not that the vectors are still usable as an
    # index -- an embedding mangled in the round trip restores as a perfectly countable row.
    ranking_after = [
        d.metadata["source"]
        for d in get_retriever(k=2, embeddings=fake_embeddings, persist_dir=persist_dir).invoke(
            query
        )
    ]
    assert ranking_after == ranking_before
    assert len(ranking_after) == 2


def test_restore_refuses_when_the_target_is_not_configured_for_the_archive(
    sample_corpus_dir, fake_embeddings, tmp_path, monkeypatch
):
    """Loading a pgvector archive into a Chroma deployment would put the index where nothing
    reads it -- a restore that reports success and serves nothing."""
    from rag_assistant.backup import create_backup, restore_backup

    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    archive = create_backup(
        output_dir=tmp_path / "backups", persist_dir=persist_dir, corpus_dir=sample_corpus_dir
    )

    monkeypatch.setenv("VECTOR_BACKEND", "chroma")
    get_settings.cache_clear()
    reset_store_cache()

    with pytest.raises(ValueError) as excinfo:
        restore_backup(archive, persist_dir=persist_dir, corpus_dir=sample_corpus_dir)

    assert "not configured to read" in str(excinfo.value)


def test_the_dump_is_internally_consistent(sample_corpus_dir, fake_embeddings, tmp_path):
    """Every chunk id the manifest names must be present in the chunk dump. Reading the
    tables at four different instants would let an ingest land between them and produce an
    archive describing chunks it does not contain."""
    import json
    import tarfile

    from rag_assistant.backup import create_backup

    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)
    archive = create_backup(
        output_dir=tmp_path / "backups", persist_dir=persist_dir, corpus_dir=sample_corpus_dir
    )

    with tarfile.open(archive, "r:gz") as tar:
        manifest_rows = [
            json.loads(line)
            for line in tar.extractfile("postgres/corpus_manifest.jsonl")
            .read()
            .decode()
            .splitlines()
            if line.strip()
        ]
        chunk_rows = [
            json.loads(line)
            for line in tar.extractfile("postgres/corpus_chunks.jsonl").read().decode().splitlines()
            if line.strip()
        ]

    archived_ids = {row["chunk_id"] for row in chunk_rows}
    for row in manifest_rows:
        entry = row["entry"] if isinstance(row["entry"], dict) else json.loads(row["entry"])
        assert set(entry["chunk_ids"]) <= archived_ids


def test_the_embedding_model_record_is_readable_from_a_replica_that_did_not_ingest(
    sample_corpus_dir, fake_embeddings, tmp_path
):
    """The guard this record exists to be, restored.

    `check_embedding_model` treats a missing record as "cannot verify" and reports ready --
    right for a fresh deployment, wrong for a replica that simply did not perform the ingest.
    While this lived on local disk, every such replica skipped the check entirely, which is
    the one defence against a same-width embedding model swap: pgvector rejects a different
    *width* at insert, but a same-width model returns plausible neighbours from a space the
    stored vectors do not occupy, with no error anywhere.
    """
    from rag_assistant.ingestion.index_metadata import check_embedding_model, load_index_metadata

    persist_dir = tmp_path / "idx"
    build_index(source_dir=sample_corpus_dir, persist_dir=persist_dir, embeddings=fake_embeddings)

    assert not (persist_dir / "index_metadata.json").exists()
    # Read through a persist directory that never saw the ingest -- i.e. another replica.
    elsewhere = tmp_path / "never-written"
    recorded = load_index_metadata(elsewhere)
    assert recorded is not None
    assert recorded.embedding_model == get_settings().gemini_embedding_model

    ok, error = check_embedding_model(elsewhere, "models/some-other-embedding-model")

    assert ok is False
    assert "different vector space" in error
