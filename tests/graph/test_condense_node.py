from unittest.mock import MagicMock

from rag_assistant.graph.nodes.condense import condense_question
from rag_assistant.schemas.models import CondensedQuestion

_HISTORY = [
    {"role": "user", "content": "Tell me about Anthropic."},
    {"role": "assistant", "content": "Anthropic is an AI safety company..."},
]


def _fake_llm(standalone: str) -> MagicMock:
    fake = MagicMock()
    fake.invoke.return_value = CondensedQuestion(standalone_question=standalone)
    return fake


def test_no_history_passes_question_through_without_llm_call(monkeypatch):
    fake_factory = MagicMock()
    monkeypatch.setattr("rag_assistant.graph.nodes.condense.get_structured_llm", fake_factory)

    result = condense_question({"question": "Who founded Anthropic?", "chat_history": []})

    fake_factory.assert_not_called()
    assert result == {"original_question": None}


def test_follow_up_is_rewritten_and_original_preserved(monkeypatch):
    fake = _fake_llm("What safety research does Anthropic do?")
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.condense.get_structured_llm", lambda schema: fake
    )

    result = condense_question(
        {"question": "what about their safety research?", "chat_history": _HISTORY}
    )

    assert result["question"] == "What safety research does Anthropic do?"
    assert result["original_question"] == "what about their safety research?"


def test_already_standalone_rewrite_sets_no_original(monkeypatch):
    # The LLM returning the question unchanged means no rewrite happened -- original_question
    # must stay None so the summary panel doesn't show a redundant "interpreted as" row.
    fake = _fake_llm("Who founded Anthropic?")
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.condense.get_structured_llm", lambda schema: fake
    )

    result = condense_question({"question": "Who founded Anthropic?", "chat_history": _HISTORY})

    assert result == {"original_question": None}


def test_llm_failure_degrades_to_original_question(monkeypatch):
    fake = MagicMock()
    fake.invoke.side_effect = RuntimeError("provider down")
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.condense.get_structured_llm", lambda schema: fake
    )

    result = condense_question({"question": "what about pricing?", "chat_history": _HISTORY})

    assert result == {"original_question": None}


def test_history_is_windowed_and_truncated_in_prompt(monkeypatch):
    fake = _fake_llm("rewritten")
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.condense.get_structured_llm", lambda schema: fake
    )

    long_history = [{"role": "user", "content": f"turn {i} " + "x" * 1000} for i in range(20)]
    condense_question({"question": "follow-up?", "chat_history": long_history})

    prompt = fake.invoke.call_args.args[0]
    # Only the most recent 8 turns appear, each truncated well below its 1000-char padding.
    assert "turn 19" in prompt
    assert "turn 11" not in prompt
    assert "x" * 700 not in prompt
