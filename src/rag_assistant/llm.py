from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from rag_assistant.config import get_settings


def _gemini_chat_model(temperature: float) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_chat_model,
        temperature=temperature,
        google_api_key=settings.google_api_key,
        timeout=settings.llm_request_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _anthropic_chat_model() -> ChatAnthropic:
    """`temperature` is rejected (HTTP 400) by current Claude models, so it's never passed here."""
    settings = get_settings()
    return ChatAnthropic(
        model=settings.anthropic_chat_model,
        api_key=settings.anthropic_api_key,
        default_request_timeout=settings.llm_request_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    """Primary chat model for plain `.invoke()` calls.

    Anthropic is primary when ANTHROPIC_API_KEY is configured, with Gemini as an
    automatic fallback on error (rate limit, outage, etc). Falls back to Gemini-only
    when no Anthropic key is set.
    """
    settings = get_settings()
    gemini = _gemini_chat_model(temperature)
    if not settings.anthropic_api_key:
        return gemini

    return _anthropic_chat_model().with_fallbacks([gemini])


def get_structured_llm(schema: type, temperature: float = 0.0) -> Runnable:
    """Structured-output runnable with the same Anthropic-primary/Gemini-fallback policy.

    `with_structured_output()` must be bound per-provider before fallbacks are attached —
    `RunnableWithFallbacks` doesn't expose `with_structured_output`, so the fallback has to
    wrap the already-structured runnables rather than the raw chat models.
    """
    settings = get_settings()
    gemini_structured = _gemini_chat_model(temperature).with_structured_output(schema)
    if not settings.anthropic_api_key:
        return gemini_structured

    return _anthropic_chat_model().with_structured_output(schema).with_fallbacks([gemini_structured])


def get_raw_chat_model(temperature: float = 0.0) -> BaseChatModel:
    """The primary chat model with no fallback wrapping.

    `RunnableWithFallbacks` (what `get_chat_model()` returns once Anthropic is configured)
    isn't a `BaseChatModel` -- it has no `.generate_prompt()`/`.temperature` attribute, which
    breaks callers like RAGAS's `LangchainLLMWrapper` that require a real chat model instance.
    """
    settings = get_settings()
    if settings.anthropic_api_key:
        return _anthropic_chat_model()
    return _gemini_chat_model(temperature)


def primary_chat_provider_name() -> str:
    """Which provider `get_chat_model()`/`get_structured_llm()` calls first."""
    return "Anthropic" if get_settings().anthropic_api_key else "Gemini"


def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.google_api_key,
    )
