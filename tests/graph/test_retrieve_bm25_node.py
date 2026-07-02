from rag_assistant.graph.nodes.retrieve import retrieve_bm25
from rag_assistant.schemas.models import RetrievedDoc, SubQueryResult


def test_retrieve_bm25_returns_bm25_results_shape(monkeypatch):
    fake_docs = [RetrievedDoc(content="doc a", source_id="a.md", score=1.2)]
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.retrieve.bm25_search",
        lambda sub_query, k=4: fake_docs,
    )

    result = retrieve_bm25({"sub_query": "who founded anthropic"})

    assert result == {
        "bm25_results": [SubQueryResult(sub_query="who founded anthropic", docs=fake_docs)]
    }


def test_retrieve_bm25_handles_no_matches(monkeypatch):
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.retrieve.bm25_search",
        lambda sub_query, k=4: [],
    )

    result = retrieve_bm25({"sub_query": "no matches here"})

    assert result == {"bm25_results": [SubQueryResult(sub_query="no matches here", docs=[])]}
