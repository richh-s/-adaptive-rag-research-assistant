from unittest.mock import AsyncMock, MagicMock

from rag_assistant.graph.nodes.synthesize import synthesize_answer
from rag_assistant.prompts.synthesis_prompt import EMPTY_RETRIEVAL_PROMPT
from rag_assistant.schemas.models import FusedDocument


def _fake_llm(content: str) -> MagicMock:
    fake = AsyncMock()
    fake.ainvoke.return_value = MagicMock(content=content, text=content)
    return fake


async def test_empty_retrieval_uses_empty_retrieval_prompt(monkeypatch):
    fake_llm = _fake_llm("No sources found.")
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.synthesize.get_chat_model", lambda **kw: fake_llm
    )

    result = await synthesize_answer(
        {"question": "What is the price of Bitcoin?", "route": "web", "fused_documents": []}
    )

    fake_llm.ainvoke.assert_called_once_with(
        EMPTY_RETRIEVAL_PROMPT.format(question="What is the price of Bitcoin?", history_block="")
    )
    assert result == {
        "final_answer": "No sources found.",
        "citations": [],
        "context_documents_dropped": 0,
        # Empty, but present: the groundedness check reads this key to know what the answer
        # was written from, and an absent one is indistinguishable from "not recorded".
        "context_documents": [],
    }


async def test_synthesize_returns_cached_answer_without_calling_llm(monkeypatch):
    fake_llm = _fake_llm("should not be used")
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.synthesize.get_chat_model", lambda **kw: fake_llm
    )
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.synthesize.cache_get",
        lambda key: {
            "final_answer": "Cached answer.",
            "citations": [{"marker": "[1]", "source_id": "doc_a.md"}],
        },
    )

    result = await synthesize_answer(
        {"question": "What is X?", "route": "vector", "fused_documents": []}
    )

    fake_llm.ainvoke.assert_not_called()
    assert result["final_answer"] == "Cached answer."
    assert result["citations"][0].marker == "[1]"
    assert result["citations"][0].source_id == "doc_a.md"


async def test_synthesize_caches_result_after_llm_call(monkeypatch):
    fake_llm = _fake_llm("No sources found.")
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.synthesize.get_chat_model", lambda **kw: fake_llm
    )
    monkeypatch.setattr("rag_assistant.graph.nodes.synthesize.cache_get", lambda key: None)
    captured = {}
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.synthesize.cache_set",
        lambda key, value, ttl: captured.update(key=key, value=value, ttl=ttl),
    )

    await synthesize_answer(
        {"question": "What is the price of Bitcoin?", "route": "web", "fused_documents": []}
    )

    assert captured["value"] == {"final_answer": "No sources found.", "citations": []}
    assert captured["ttl"] == 1800


def _doc(content: str, source_id: str = "report.md"):
    return FusedDocument(content=content, source_id=source_id, rrf_score=0.5, metadata={})


async def _key_for(monkeypatch, docs) -> str:
    """The cache key `synthesize_answer` computes for `docs`, captured off `cache_get`."""
    seen = {}
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.synthesize.get_chat_model", lambda **kw: _fake_llm("answer")
    )
    monkeypatch.setattr("rag_assistant.graph.nodes.synthesize.cache_set", lambda *a, **k: None)

    def capture(key):
        seen["key"] = key
        return None

    monkeypatch.setattr("rag_assistant.graph.nodes.synthesize.cache_get", capture)
    await synthesize_answer({"question": "What is X?", "route": "vector", "fused_documents": docs})
    return seen["key"]


async def test_edited_document_does_not_reuse_the_previous_answer(monkeypatch):
    """Re-ingesting an edited file must not serve the answer built from its old text.

    Source ids are stable paths, so keying on them alone made a revision indistinguishable
    from the version it replaced: the cached answer survived for the rest of
    CACHE_TTL_SYNTHESIS, citing a document whose text no longer said that. Nothing in the
    ingest path can reach the Redis key to evict it, so the key has to change on its own.
    """
    before = await _key_for(monkeypatch, [_doc("Revenue was $2.1M in 2023.")])
    after = await _key_for(monkeypatch, [_doc("Revenue was $3.4M in 2023.")])

    assert before != after


async def test_unchanged_documents_still_hit_the_same_key(monkeypatch):
    """The other half: identical context must still cache, or the fix would have disabled it."""
    first = await _key_for(monkeypatch, [_doc("Revenue was $2.1M in 2023.")])
    second = await _key_for(monkeypatch, [_doc("Revenue was $2.1M in 2023.")])

    assert first == second


async def test_the_same_text_under_a_different_source_is_a_different_key(monkeypatch):
    """Content alone is not the identity -- citations name the source, so an answer built
    from `a.md` must not be replayed as one built from `b.md`."""
    first = await _key_for(monkeypatch, [_doc("Shared boilerplate paragraph.", "a.md")])
    second = await _key_for(monkeypatch, [_doc("Shared boilerplate paragraph.", "b.md")])

    assert first != second
