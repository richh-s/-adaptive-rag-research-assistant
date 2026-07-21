from unittest.mock import MagicMock

from rag_assistant.graph.nodes.synthesize import synthesize_answer
from rag_assistant.prompts.synthesis_prompt import EMPTY_RETRIEVAL_PROMPT


def _fake_llm(content: str) -> MagicMock:
    fake = MagicMock()
    fake.invoke.return_value = MagicMock(content=content)
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
