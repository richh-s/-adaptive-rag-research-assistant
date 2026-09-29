"""pgvector: document ACLs, row-level tenant security and index generations, for real.

Skipped unless a Postgres with pgvector is reachable at RAG_TEST_DATABASE_URL, like
tests/test_pgvector_store.py. The row-level-security tests matter most here, and they need a
real database twice over: a policy is enforced by Postgres, not by this codebase, and it is
*not* enforced for superusers -- which is what CI's service container connects as. So those
tests create an ordinary login role and connect as it; run as the superuser, they would pass
while proving nothing.
"""

import os
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest

from rag_assistant.config import get_settings
from rag_assistant.ingestion import generations, reindex
from rag_assistant.ingestion.acl import DocumentAcl, write_acl
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.ingestion.ownership import TENANT_DIR
from rag_assistant.retrieval.bm25_store import bm25_search
from rag_assistant.retrieval.vector_store import get_retriever, get_vector_store, reset_store_cache
from tests.conftest import FakeHashingEmbeddings

DATABASE_URL = os.environ.get("RAG_TEST_DATABASE_URL", "")


def _reachable() -> bool:
    if not DATABASE_URL:
        return False
    try:
        import psycopg

        with psycopg.connect(DATABASE_URL, connect_timeout=2) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _reachable(), reason="No Postgres with pgvector at RAG_TEST_DATABASE_URL"
)


def _sql(query, params=(), url=DATABASE_URL, fetch=True):
    import psycopg

    with psycopg.connect(url, autocommit=True) as conn:
        cursor = conn.execute(query, params)
        return cursor.fetchall() if fetch and cursor.description else None


def _reset_database():
    reset_store_cache()
    for (schema,) in _sql(
        "SELECT nspname FROM pg_namespace WHERE starts_with(nspname, 'rag_idx_')"
    ):
        _sql(f'DROP SCHEMA "{schema}" CASCADE', fetch=False)
    for table in (
        "corpus_chunks",
        "corpus_manifest",
        "corpus_parents",
        "corpus_index_state",
        "corpus_index_metadata",
        "rag_active_index",
        "pgvector_schema_migrations",
    ):
        _sql(f"DROP TABLE IF EXISTS public.{table} CASCADE", fetch=False)


@pytest.fixture
def pg(tmp_path, monkeypatch):
    _reset_database()
    corpus = tmp_path / "corpus"
    for owner in ("alice", "bob"):
        directory = corpus / TENANT_DIR / owner
        directory.mkdir(parents=True)
        (directory / f"{owner}_plan.md").write_text(
            f"Project {owner.title()} revenue plan for the widgets division."
        )
    (corpus / "baseline.md").write_text("Anthropic publishes revenue research on widgets.")
    board = corpus / TENANT_DIR / "alice" / "board.md"
    board.write_text("The board minutes discuss the widgets division revenue layoffs.")
    write_acl(board, DocumentAcl(groups=frozenset({"finance"})))

    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("INDEX_POINTER_POLL_SECONDS", "0")
    get_settings.cache_clear()
    embeddings = FakeHashingEmbeddings()
    monkeypatch.setattr(
        "rag_assistant.retrieval.vector_store.get_embeddings_model", lambda name=None: embeddings
    )
    monkeypatch.setattr(
        "rag_assistant.ingestion.reindex.get_embeddings_model", lambda name=None: embeddings
    )
    reset_store_cache()
    build_index()
    yield corpus
    _reset_database()


def _names(docs):
    return {d.metadata["source"].rsplit("/", 1)[-1] for d in docs}


# ---- document ACLs ----


def test_the_acl_predicate_is_applied_inside_the_vector_search(pg):
    query = "widgets division revenue"
    as_engineer = get_retriever(
        k=10, owner="alice", principals=frozenset({"group:engineering"})
    ).invoke(query)
    as_finance = get_retriever(k=10, owner="alice", principals=frozenset({"group:finance"})).invoke(
        query
    )

    assert _names(as_engineer) == {"alice_plan.md", "baseline.md"}
    assert _names(as_finance) == {"alice_plan.md", "baseline.md", "board.md"}


def test_the_acl_predicate_is_applied_to_postgres_keyword_search(pg, monkeypatch):
    monkeypatch.setenv("KEYWORD_BACKEND", "postgres")
    get_settings.cache_clear()

    hits = bm25_search("board minutes", k=10, owner="alice", principals=frozenset({"user:x"}))
    finance = bm25_search(
        "board minutes", k=10, owner="alice", principals=frozenset({"group:finance"})
    )

    assert hits == []
    assert {h.source_id.rsplit("/", 1)[-1] for h in finance} == {"board.md"}


def test_a_permission_change_updates_metadata_in_place(pg):
    board = pg / TENANT_DIR / "alice" / "board.md"
    write_acl(board, DocumentAcl(groups=frozenset({"finance", "engineering"})))

    result = build_index()

    assert result.acl_updates == 1 and result.changed_files == 0
    docs = get_retriever(k=10, owner="alice", principals=frozenset({"group:engineering"})).invoke(
        "board minutes"
    )
    assert "board.md" in _names(docs)


# ---- row-level security ----


@pytest.fixture
def app_role(pg):
    """An ordinary role, as a production deployment should connect: superusers bypass RLS."""
    role = f"rag_app_{uuid.uuid4().hex[:8]}"
    password = uuid.uuid4().hex
    _sql(f"CREATE ROLE {role} LOGIN PASSWORD '{password}'", fetch=False)
    _sql(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {role}",
        fetch=False,
    )
    # So the role can own tables it creates, as an application role running its own
    # migrations does. Postgres 15 no longer grants this on `public` by default.
    _sql(f"GRANT CREATE ON SCHEMA public TO {role}", fetch=False)
    _sql(
        f"GRANT CREATE ON DATABASE {urlsplit(DATABASE_URL).path.lstrip('/')} TO {role}", fetch=False
    )
    parts = urlsplit(DATABASE_URL)
    host = parts.hostname + (f":{parts.port}" if parts.port else "")
    url = urlunsplit((parts.scheme, f"{role}:{password}@{host}", parts.path, "", ""))
    yield url
    _sql(f"DROP OWNED BY {role}", fetch=False)
    _sql(f"DROP ROLE {role}", fetch=False)


def _as_role(url, visible_owners, query):
    import psycopg

    with psycopg.connect(url) as conn:
        if visible_owners is not None:
            conn.execute("SELECT set_config('rag.visible_owners', %s, false)", (visible_owners,))
        return conn.execute(query).fetchall()


def test_row_security_hides_other_tenants_rows_even_without_a_where_clause(app_role):
    """The whole point of the second layer: a query that forgot its tenant predicate."""
    rows = _as_role(app_role, "alice,public", "SELECT DISTINCT owner FROM corpus_chunks")
    assert {r[0] for r in rows} == {"alice", "public"}


def test_row_security_fails_closed_when_no_scope_was_set(app_role):
    assert _as_role(app_role, None, "SELECT COUNT(*) FROM corpus_chunks") == [(0,)]


def test_row_security_is_reported_as_bypassed_for_a_superuser(pg):
    from rag_assistant.retrieval.pgvector_store import row_security_status

    enforced, detail = row_security_status()
    if _sql("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")[0][0]:
        assert not enforced and "superuser" in detail


def test_the_application_works_when_connected_as_an_ordinary_role(app_role, monkeypatch):
    """Every connection the store hands out sets the scope, so migrations, ingestion,
    retrieval and the keyword index all keep working under an enforcing policy -- and still
    isolate tenants.

    The role *owns* the tables here, having created them, which is the realistic shape: an
    application role running its own migrations. FORCE ROW LEVEL SECURITY is what keeps the
    policy binding on an owner."""
    _reset_database()
    monkeypatch.setenv("DATABASE_URL", app_role)
    get_settings.cache_clear()
    reset_store_cache()
    build_index()
    from rag_assistant.retrieval.pgvector_store import row_security_status

    assert row_security_status() == (True, None)
    docs = get_retriever(k=10, owner="alice").invoke("widgets division revenue plan")
    assert "bob_plan.md" not in _names(docs) and "alice_plan.md" in _names(docs)
    assert get_vector_store()._collection.count() == 4


# ---- generations ----


def test_a_generation_is_its_own_schema_and_the_pointer_is_shared(pg):
    result = reindex.build_generation()

    schema = generations.schema_for_generation(result.generation)
    assert _sql("SELECT COUNT(*) FROM pg_namespace WHERE nspname = %s", (schema,)) == [(1,)]
    assert _sql(f'SELECT COUNT(*) FROM "{schema}".corpus_chunks')[0][0] == 4
    assert generations.active_generation() == generations.LEGACY

    reindex.activate(result.generation, settle=False)

    assert _sql("SELECT generation, previous FROM public.rag_active_index") == [
        (result.generation, "")
    ]
    assert "alice_plan.md" in _names(get_retriever(k=10, owner="alice").invoke("revenue plan"))


def test_gc_drops_the_schema(pg):
    first = reindex.build_generation().generation
    reindex.activate(first, settle=False)
    second = reindex.build_generation().generation
    third = reindex.build_generation().generation
    reindex.activate(second, settle=False)

    assert reindex.garbage_collect() == [third]
    remaining = {
        r[0]
        for r in _sql("SELECT nspname FROM pg_namespace WHERE starts_with(nspname, 'rag_idx_')")
    }
    assert remaining == {
        generations.schema_for_generation(first),
        generations.schema_for_generation(second),
    }


def test_a_backup_archives_the_serving_generation(pg, tmp_path):
    from rag_assistant.backup import create_backup

    result = reindex.build_generation()
    reindex.activate(result.generation, settle=False)
    (pg / "late.md").write_text("A document ingested after the switch.")
    build_index()

    archive = create_backup(output_dir=tmp_path / "backups")

    import tarfile

    with tarfile.open(archive) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("corpus_manifest.jsonl"))
        manifest = tar.extractfile(member).read().decode()
    assert "late.md" in manifest
