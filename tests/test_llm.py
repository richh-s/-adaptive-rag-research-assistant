from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableWithFallbacks
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_assistant import llm


def test_get_chat_model_is_gemini_only_without_anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    model = llm.get_chat_model()

    assert isinstance(model, ChatGoogleGenerativeAI)


def test_get_chat_model_wraps_anthropic_with_gemini_fallback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    model = llm.get_chat_model()

    assert isinstance(model, RunnableWithFallbacks)
    assert isinstance(model.runnable, ChatAnthropic)
    assert isinstance(model.fallbacks[0], ChatGoogleGenerativeAI)


def test_get_raw_chat_model_is_anthropic_with_no_fallback_when_key_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    model = llm.get_raw_chat_model()

    assert isinstance(model, ChatAnthropic)


def test_get_raw_chat_model_is_gemini_without_anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    model = llm.get_raw_chat_model()

    assert isinstance(model, ChatGoogleGenerativeAI)


def test_anthropic_model_never_receives_temperature(monkeypatch):
    """claude-sonnet-5 rejects `temperature` with an HTTP 400, so it must never be passed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    model = llm.get_raw_chat_model(temperature=0.7)

    assert model.temperature is None


def test_primary_chat_provider_name_reflects_key_presence(monkeypatch):
    from rag_assistant.config import get_settings

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert llm.primary_chat_provider_name() == "Gemini"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    get_settings.cache_clear()
    assert llm.primary_chat_provider_name() == "Anthropic"


def test_get_structured_llm_falls_back_to_gemini_without_anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    from rag_assistant.schemas.models import RouteDecision

    runnable = llm.get_structured_llm(RouteDecision)

    assert not isinstance(runnable, RunnableWithFallbacks)
