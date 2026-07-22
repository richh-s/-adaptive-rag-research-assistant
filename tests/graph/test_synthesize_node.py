from unittest.mock import MagicMock

from rag_assistant.graph.nodes.synthesize import synthesize_answer
from rag_assistant.prompts.synthesis_prompt import EMPTY_RETRIEVAL_PROMPT


def _fake_llm(content: str) -> MagicMock:
    fake = MagicMock()
    fake.invoke.return_value = MagicMock(content=content, text=content)
    return fake


def test_empty_retrieval_uses_empty_retrieval_prompt(monkeypatch):
    fake_llm = _fake_llm("No sources found.")
    monkeypatch.setattr("rag_assistant.graph.nodes.synthesize.get_chat_model", lambda: fake_llm)

    result = synthesize_answer(
        {"question": "What is the price of Bitcoin?", "route": "web", "fused_documents": []}
    )

    fake_llm.invoke.assert_called_once_with(
        EMPTY_RETRIEVAL_PROMPT.format(question="What is the price of Bitcoin?")
    )
    assert result == {"final_answer": "No sources found.", "citations": []}


def test_synthesize_returns_cached_answer_without_calling_llm(monkeypatch):
    fake_llm = _fake_llm("should not be used")
    monkeypatch.setattr("rag_assistant.graph.nodes.synthesize.get_chat_model", lambda: fake_llm)
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.synthesize.cache_get",
        lambda key: {
            "final_answer": "Cached answer.",
            "citations": [{"marker": "[1]", "source_id": "doc_a.md"}],
        },
    )

    result = synthesize_answer(
        {"question": "What is X?", "route": "vector", "fused_documents": []}
    )

    fake_llm.invoke.assert_not_called()
    assert result["final_answer"] == "Cached answer."
    assert result["citations"][0].marker == "[1]"
    assert result["citations"][0].source_id == "doc_a.md"


def test_synthesize_caches_result_after_llm_call(monkeypatch):
    fake_llm = _fake_llm("No sources found.")
    monkeypatch.setattr("rag_assistant.graph.nodes.synthesize.get_chat_model", lambda: fake_llm)
    monkeypatch.setattr("rag_assistant.graph.nodes.synthesize.cache_get", lambda key: None)
    captured = {}
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.synthesize.cache_set",
        lambda key, value, ttl: captured.update(key=key, value=value, ttl=ttl),
    )

    synthesize_answer({"question": "What is the price of Bitcoin?", "route": "web", "fused_documents": []})

    assert captured["value"] == {"final_answer": "No sources found.", "citations": []}
    assert captured["ttl"] == 1800
