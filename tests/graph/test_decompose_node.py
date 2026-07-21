from unittest.mock import MagicMock

from rag_assistant.graph.nodes.decompose import decompose_query, dispatch_retrieval
from rag_assistant.schemas.models import SubQueries


def test_decompose_query_returns_sub_queries(monkeypatch):
    fake_result = SubQueries(sub_queries=["Who founded Anthropic?", "Who founded OpenAI?"])
    fake_structured_llm = MagicMock()
    fake_structured_llm.invoke.return_value = fake_result

    monkeypatch.setattr(
        "rag_assistant.graph.nodes.decompose.get_structured_llm", lambda schema: fake_structured_llm
    )

    result = decompose_query({"question": "Who founded Anthropic and OpenAI?"})

    assert result == {"sub_queries": ["Who founded Anthropic?", "Who founded OpenAI?"]}


def test_dispatch_retrieval_fans_out_both_paths_per_subquery():
    sends = dispatch_retrieval({"route": "both", "sub_queries": ["q1", "q2"]})

    targets = sorted(send.node for send in sends)
    assert targets == [
        "retrieve_bm25",
        "retrieve_bm25",
        "retrieve_vector",
        "retrieve_vector",
        "web_search",
        "web_search",
    ]
    args = {(send.node, send.arg["sub_query"]) for send in sends}
    assert args == {
        ("retrieve_vector", "q1"),
        ("retrieve_vector", "q2"),
        ("retrieve_bm25", "q1"),
        ("retrieve_bm25", "q2"),
        ("web_search", "q1"),
        ("web_search", "q2"),
    }


def test_dispatch_retrieval_vector_only():
    sends = dispatch_retrieval({"route": "vector", "sub_queries": ["q1"]})

    targets = sorted(send.node for send in sends)
    assert targets == ["retrieve_bm25", "retrieve_vector"]
    args = {send.arg["sub_query"] for send in sends}
    assert args == {"q1"}


def test_dispatch_retrieval_web_only():
    sends = dispatch_retrieval({"route": "web", "sub_queries": ["q1"]})

    assert len(sends) == 1
    assert sends[0].node == "web_search"
    assert sends[0].arg == {"sub_query": "q1"}
