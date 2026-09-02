"""Tests for what ingestion actually costs, and for tenant-scoped indexing.

`skipped_files` was always reported honestly -- unchanged files really were skipped for
*embedding*. What it hid is that every one of them was fully parsed first, and then parsed a
second time by the BM25 rebuild. For a PDF with PDF_VISION on, a parse is a vision API call
per figure and per scanned page, so one tenant uploading a small note re-ran paid vision over
every tenant's PDFs, twice. These tests assert on parse counts rather than on skip counts,
because the skip count is the number that looked fine while the cost was real.
"""

from unittest.mock import patch

import pytest

from rag_assistant.ingestion import build_index as build_index_module
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.ingestion.loaders import iter_corpus_files, load_corpus_file
from rag_assistant.ingestion.manifest import load_manifest
from rag_assistant.ingestion.ownership import TENANT_DIR
from rag_assistant.retrieval.vector_store import get_retriever


@pytest.fixture
def multi_tenant_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(4):
        (corpus / f"public{i}.md").write_text(f"# Public {i}\n\n## Body\n\nShared topic {i}.\n")
    for tenant in ("alice", "bob"):
        directory = corpus / TENANT_DIR / tenant
        directory.mkdir(parents=True)
        (directory / "notes.md").write_text(f"# {tenant}\n\n## Body\n\n{tenant} private notes.\n")
    return corpus


def parse_spy():
    """Counts real parses. `load_corpus_file` is the expensive step -- the one that runs
    pymupdf4llm and the vision passes -- so it is what a cost test must count."""
    return patch.object(
        build_index_module, "load_corpus_file", side_effect=load_corpus_file, autospec=True
    )


# ---- parsing only what changed ----


def test_unchanged_files_are_not_parsed_at_all(multi_tenant_corpus, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"
    build_index(source_dir=multi_tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)

    with parse_spy() as spy:
        result = build_index(
            source_dir=multi_tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings
        )

    assert spy.call_count == 0
    assert result.parsed_files == 0
    assert result.skipped_files == 6


def test_only_the_changed_file_is_parsed(multi_tenant_corpus, fake_embeddings, tmp_path):
    persist_dir = tmp_path / "chroma"
    build_index(source_dir=multi_tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)
    (multi_tenant_corpus / "public1.md").write_text("# Public 1\n\n## Body\n\nRewritten.\n")

    with parse_spy() as spy:
        result = build_index(
            source_dir=multi_tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings
        )

    assert result.parsed_files == 1
    assert [call.args[0].source for call in spy.call_args_list] == ["public1.md"]


def test_a_touched_but_unmodified_file_is_not_reparsed(
    multi_tenant_corpus, fake_embeddings, tmp_path
):
    """The change signal is a content fingerprint, not mtime -- rewriting identical bytes
    (which `git checkout` and container rebuilds do constantly) must not trigger re-embedding
    the whole corpus."""
    persist_dir = tmp_path / "chroma"
    build_index(source_dir=multi_tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)
    path = multi_tenant_corpus / "public2.md"
    path.write_text(path.read_text())

    with parse_spy() as spy:
        build_index(
            source_dir=multi_tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings
        )

    assert spy.call_count == 0


def test_a_loader_version_bump_forces_a_reparse(multi_tenant_corpus, fake_embeddings, tmp_path):
    """Improving a loader must re-index affected files, or the collection keeps serving text
    the old loader produced while the code assumes the new one."""
    persist_dir = tmp_path / "chroma"
    build_index(source_dir=multi_tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)

    with patch.object(build_index_module, "LOADER_VERSION", 999):
        result = build_index(
            source_dir=multi_tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings
        )

    assert result.parsed_files == 6
    assert result.skipped_files == 0


# ---- tenant-scoped ingestion ----


def test_a_tenant_scoped_ingest_only_touches_that_tenants_files(
    multi_tenant_corpus, fake_embeddings, tmp_path
):
    persist_dir = tmp_path / "chroma"
    build_index(source_dir=multi_tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)
    alice_dir = multi_tenant_corpus / TENANT_DIR / "alice"
    (alice_dir / "new.md").write_text("# New\n\n## Body\n\nAlice new upload.\n")

    with parse_spy() as spy:
        result = build_index(
            source_dir=multi_tenant_corpus,
            persist_dir=persist_dir,
            embeddings=fake_embeddings,
            owner="alice",
        )

    assert result.parsed_files == 1
    # Only alice's two files are even considered; the public corpus and bob are not scanned.
    assert result.skipped_files == 1
    assert [call.args[0].source for call in spy.call_args_list] == [
        f"{TENANT_DIR}/alice/new.md"
    ]


def test_a_scoped_ingest_does_not_delete_other_tenants_chunks(
    multi_tenant_corpus, fake_embeddings, tmp_path
):
    """The dangerous failure: scoping the *scan* without scoping removal detection reads
    every unscanned file as deleted and drops its chunks."""
    persist_dir = tmp_path / "chroma"
    build_index(source_dir=multi_tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)
    (multi_tenant_corpus / TENANT_DIR / "alice" / "new.md").write_text("# New\n\nAlice upload.\n")

    result = build_index(
        source_dir=multi_tenant_corpus,
        persist_dir=persist_dir,
        embeddings=fake_embeddings,
        owner="alice",
    )

    assert result.removed_files == 0
    manifest = load_manifest(persist_dir)
    assert f"{TENANT_DIR}/bob/notes.md" in manifest
    assert all(f"public{i}.md" in manifest for i in range(4))

    still_there = get_retriever(
        k=20, embeddings=fake_embeddings, persist_dir=persist_dir, owner="bob"
    ).invoke("private notes")
    assert any("bob" in d.metadata["source"] for d in still_there)


def test_a_scoped_ingest_still_removes_that_tenants_deleted_files(
    multi_tenant_corpus, fake_embeddings, tmp_path
):
    persist_dir = tmp_path / "chroma"
    build_index(source_dir=multi_tenant_corpus, persist_dir=persist_dir, embeddings=fake_embeddings)
    (multi_tenant_corpus / TENANT_DIR / "alice" / "notes.md").unlink()

    result = build_index(
        source_dir=multi_tenant_corpus,
        persist_dir=persist_dir,
        embeddings=fake_embeddings,
        owner="alice",
    )

    assert result.removed_files == 1
    assert f"{TENANT_DIR}/alice/notes.md" not in load_manifest(persist_dir)


def test_a_full_rebuild_cannot_be_scoped_to_one_owner(
    multi_tenant_corpus, fake_embeddings, tmp_path
):
    """A full rebuild resets the whole collection, so scoping it would delete every other
    tenant's chunks while re-indexing only one tenant's."""
    with pytest.raises(ValueError, match="cannot be scoped"):
        build_index(
            source_dir=multi_tenant_corpus,
            persist_dir=tmp_path / "chroma",
            embeddings=fake_embeddings,
            incremental=False,
            owner="alice",
        )


def test_public_scope_excludes_tenant_subtrees(multi_tenant_corpus):
    sources = {f.source for f in iter_corpus_files(multi_tenant_corpus, owner="public")}

    assert sources == {f"public{i}.md" for i in range(4)}


def test_unscoped_enumeration_sees_everything(multi_tenant_corpus):
    sources = {f.source for f in iter_corpus_files(multi_tenant_corpus)}

    assert len(sources) == 6
    assert f"{TENANT_DIR}/bob/notes.md" in sources


def test_fingerprint_changes_only_when_bytes_change(multi_tenant_corpus):
    before = {f.source: f.fingerprint for f in iter_corpus_files(multi_tenant_corpus)}
    path = multi_tenant_corpus / "public0.md"
    path.write_text(path.read_text())
    unchanged = {f.source: f.fingerprint for f in iter_corpus_files(multi_tenant_corpus)}
    path.write_text("# Public 0\n\n## Body\n\nDifferent content now.\n")
    changed = {f.source: f.fingerprint for f in iter_corpus_files(multi_tenant_corpus)}

    assert unchanged == before
    assert changed["public0.md"] != before["public0.md"]
    assert changed[f"{TENANT_DIR}/bob/notes.md"] == before[f"{TENANT_DIR}/bob/notes.md"]
