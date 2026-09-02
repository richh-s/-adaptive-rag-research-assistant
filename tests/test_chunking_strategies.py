"""Tests for semantic chunking and small-to-big (parent) retrieval."""

import pytest
from langchain_core.documents import Document

from rag_assistant.graph.nodes.synthesize import expand_to_parents
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.ingestion.semantic_splitter import (
    find_breakpoints,
    semantic_split,
    split_sentences,
)
from rag_assistant.ingestion.splitter import split_with_parents
from rag_assistant.retrieval.parent_store import (
    count_parents,
    get_parents,
    replace_parents_for_source,
)
from rag_assistant.schemas.models import FusedDocument


class TopicEmbeddings:
    """Embeds by topic keyword, so sentences about the same thing are identical vectors and
    sentences about different things are orthogonal. Gives semantic_split a clean, entirely
    offline signal to find breakpoints in."""

    TOPICS = ("funding", "safety", "product")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = [1.0 if topic in lowered else 0.0 for topic in self.TOPICS]
            vectors.append(vector if any(vector) else [0.3, 0.3, 0.3])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ---- sentence segmentation ----


def test_sentences_split_on_terminators():
    assert split_sentences("One thing. Two things! Three things?") == [
        "One thing.",
        "Two things!",
        "Three things?",
    ]


def test_common_abbreviations_do_not_end_a_sentence():
    """'e.g. The model' would otherwise become a boundary mid-clause."""
    sentences = split_sentences("Several labs, e.g. Anthropic and Mistral, publish research.")

    assert len(sentences) == 1


def test_text_without_terminators_is_one_sentence():
    assert split_sentences("no terminator here") == ["no terminator here"]


# ---- breakpoint detection ----


def test_a_clear_topic_shift_produces_a_breakpoint():
    distances = [0.01, 0.02, 0.9, 0.01]

    assert find_breakpoints(distances, percentile=85.0) == [2]


def test_uniform_prose_produces_no_breakpoints():
    """A section genuinely about one thing has tiny, near-equal distances. A pure percentile
    rule would still split it at whichever noise happened to be largest."""
    assert find_breakpoints([0.01, 0.011, 0.009, 0.010], percentile=85.0) == []


def test_no_distances_means_no_breakpoints():
    assert find_breakpoints([], percentile=85.0) == []


# ---- semantic splitting ----


def test_semantic_split_breaks_at_the_topic_change():
    text = (
        "The funding round was large. The funding came from several investors. "
        "The safety team studies alignment. The safety work is published openly."
    )

    chunks = semantic_split(text, TopicEmbeddings(), percentile=50.0, min_chunk_chars=20)

    assert len(chunks) == 2
    assert "funding" in chunks[0] and "safety" not in chunks[0]
    assert "safety" in chunks[1]


def test_semantic_split_honours_the_maximum_size():
    """A section with no detectable shift must still be divided, or one chunk could consume
    the entire context budget."""
    text = " ".join(f"Sentence number {i} about one single topic." for i in range(40))

    chunks = semantic_split(text, TopicEmbeddings(), max_chunk_chars=200, min_chunk_chars=50)

    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)


def test_semantic_split_degrades_to_one_chunk_when_embedding_fails():
    """Chunking runs inside ingestion: losing semantic boundaries is a quality regression,
    failing the ingest is an outage."""

    class Broken:
        def embed_documents(self, texts):
            raise RuntimeError("provider down")

    chunks = semantic_split("One thing. Another thing. A third thing.", Broken())

    assert len(chunks) == 1


def test_semantic_strategy_is_used_when_configured(monkeypatch):
    monkeypatch.setenv("CHUNKING_STRATEGY", "semantic")
    monkeypatch.setenv("SEMANTIC_CHUNK_PERCENTILE", "50")
    funding = (
        "The funding round was large indeed and widely reported at the time. "
        "The funding came from many institutional investors across several regions. "
        "The funding total placed the company among the best capitalised labs anywhere."
    )
    safety = (
        "The safety team studies alignment and publishes its findings openly. "
        "The safety agenda covers interpretability, evaluations and red teaming work. "
        "The safety research is shared with external reviewers before wider release."
    )
    document = Document(
        page_content=f"# Lab\n\n## Notes\n\n{funding} {safety}",
        metadata={"source": "lab.md"},
    )

    result = split_with_parents([document], chunk_size=800, embeddings=TopicEmbeddings())

    assert len(result.chunks) == 2
    assert "funding" in result.chunks[0].page_content.lower()
    assert "safety" in result.chunks[1].page_content.lower()


def test_structural_strategy_remains_the_default(monkeypatch):
    document = Document(
        page_content="# Lab\n\n## Notes\n\nShort body text here.\n", metadata={"source": "lab.md"}
    )

    result = split_with_parents([document], embeddings=TopicEmbeddings())

    assert len(result.chunks) == 1


# ---- parent sections ----


def test_every_chunk_records_its_parent_section():
    document = Document(
        page_content="# Lab\n\n## Funding\n\nRaised money.\n\n## Safety\n\nStudies alignment.\n",
        metadata={"source": "lab.md"},
    )

    result = split_with_parents([document])

    parent_ids = {c.metadata["parent_id"] for c in result.chunks}
    assert len(parent_ids) == 2
    assert set(result.parents) == parent_ids


def test_parent_ids_are_stable_across_reindexing():
    """Re-indexing an unchanged file must reproduce the same ids, or every ingest would
    orphan the previous run's sections."""
    document = Document(
        page_content="# Lab\n\n## Funding\n\nRaised money.\n", metadata={"source": "lab.md"}
    )

    first = split_with_parents([document])
    second = split_with_parents([document])

    assert set(first.parents) == set(second.parents)


def test_parent_store_replaces_rather_than_accumulates(tmp_path):
    """A re-indexed file may produce fewer sections than before; an upsert would leave the
    vanished ones behind as orphans nothing points at."""
    replace_parents_for_source(tmp_path, "lab.md", "public", {"a": "one", "b": "two"})
    replace_parents_for_source(tmp_path, "lab.md", "public", {"a": "one only"})

    assert count_parents(tmp_path) == 1
    assert get_parents(tmp_path, ["a"]) == {"a": "one only"}


def test_build_index_records_parent_sections(tmp_path, fake_embeddings):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "lab.md").write_text(
        "# Lab\n\n## Funding\n\nRaised a round.\n\n## Safety\n\nStudies alignment.\n"
    )
    persist = tmp_path / "chroma"

    build_index(source_dir=corpus, persist_dir=persist, embeddings=fake_embeddings)

    assert count_parents(persist) == 2


def test_deleting_a_source_removes_its_parent_sections(tmp_path, fake_embeddings):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "lab.md").write_text("# Lab\n\n## Funding\n\nRaised a round.\n")
    (corpus / "other.md").write_text("# Other\n\n## Notes\n\nSomething else.\n")
    persist = tmp_path / "chroma"
    build_index(source_dir=corpus, persist_dir=persist, embeddings=fake_embeddings)

    (corpus / "lab.md").unlink()
    build_index(source_dir=corpus, persist_dir=persist, embeddings=fake_embeddings)

    assert count_parents(persist) == 1


# ---- small-to-big expansion ----


def doc(content: str, parent_id: str | None = None, source: str = "a.md") -> FusedDocument:
    metadata = {"parent_id": parent_id} if parent_id else {}
    return FusedDocument(content=content, metadata=metadata, source_id=source, rrf_score=1.0)


def test_expansion_swaps_chunks_for_their_section(tmp_path):
    replace_parents_for_source(tmp_path, "a.md", "public", {"p1": "The whole section body."})

    expanded = expand_to_parents([doc("a fragment", parent_id="p1")], tmp_path)

    assert expanded[0].content == "The whole section body."


def test_expansion_deduplicates_chunks_from_one_section(tmp_path):
    """Several chunks of one section routinely all match. Without collapsing them the section
    would take three slots of the context budget and earn three markers for one passage."""
    replace_parents_for_source(tmp_path, "a.md", "public", {"p1": "The whole section body."})

    expanded = expand_to_parents(
        [doc("frag one", parent_id="p1"), doc("frag two", parent_id="p1")], tmp_path
    )

    assert len(expanded) == 1


def test_expansion_preserves_rank_order(tmp_path):
    replace_parents_for_source(
        tmp_path, "a.md", "public", {"p1": "first section", "p2": "second section"}
    )

    expanded = expand_to_parents([doc("x", parent_id="p2"), doc("y", parent_id="p1")], tmp_path)

    assert [d.content for d in expanded] == ["second section", "first section"]


def test_a_chunk_with_no_stored_parent_keeps_its_own_content(tmp_path):
    """Degrades to ordinary chunk retrieval rather than dropping a document retrieval found."""
    expanded = expand_to_parents([doc("just the chunk", parent_id="missing")], tmp_path)

    assert expanded[0].content == "just the chunk"


def test_synthesis_expands_only_when_parent_context_is_enabled(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from rag_assistant.graph.nodes import synthesize as synthesize_module

    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path))
    replace_parents_for_source(tmp_path, "a.md", "public", {"p1": "THE FULL SECTION"})
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(text="answer")
    monkeypatch.setattr(synthesize_module, "get_chat_model", lambda: fake_llm)
    docs = [doc("small chunk", parent_id="p1")]

    synthesize_module.synthesize_answer(
        {"question": "q", "fused_documents": docs, "route": "vector"}
    )
    assert "THE FULL SECTION" not in fake_llm.invoke.call_args[0][0]

    monkeypatch.setenv("PARENT_CONTEXT", "true")
    from rag_assistant.config import get_settings

    get_settings.cache_clear()
    synthesize_module.synthesize_answer(
        {"question": "q", "fused_documents": docs, "route": "vector"}
    )
    assert "THE FULL SECTION" in fake_llm.invoke.call_args[0][0]


@pytest.mark.parametrize("enabled", ["true", "false"])
def test_parents_are_recorded_regardless_of_the_feature_flag(
    tmp_path, fake_embeddings, monkeypatch, enabled
):
    """Enabling small-to-big later must not require a re-index."""
    monkeypatch.setenv("PARENT_CONTEXT", enabled)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "lab.md").write_text("# Lab\n\n## Funding\n\nRaised a round.\n")
    persist = tmp_path / "chroma"

    build_index(source_dir=corpus, persist_dir=persist, embeddings=fake_embeddings)

    assert count_parents(persist) == 1
