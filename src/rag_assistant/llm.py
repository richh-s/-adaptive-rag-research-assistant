"""Chat/reasoning model construction and the provider fallback policy.

Providers are tried in a fixed priority order -- local self-hosted first (when configured),
then Anthropic, then Gemini -- and every provider that isn't configured is skipped without
being constructed. See `_provider_chain()` for why "not constructed" matters.
"""

from collections.abc import Callable
from typing import Any

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI

from rag_assistant.config import get_settings
from rag_assistant.metrics import MetricsCallbackHandler


class LocalLLMError(RuntimeError):
    """A local-model response that can't be used. Raised (rather than returned empty) so the
    fallback chain treats it as provider failure and hands the call to Anthropic/Gemini."""


class _LocalChatOpenAI(ChatOpenAI):
    """`ChatOpenAI` pointed at a self-hosted OpenAI-compatible server, with the two quirks
    those servers have that the hosted OpenAI API does not.

    1. Reasoning models (Qwen3-class) sometimes return an empty `content` and put the whole
       answer in a separate `reasoning`/`reasoning_content` field. `langchain_openai` drops
       both fields on the floor -- they're not part of the OpenAI schema it parses against --
       so the answer surfaces downstream as "the model said nothing": an empty synthesis, or
       a structured-output parse failure. Recovering it means reading the *raw* payload here,
       since by the time the parsed result exists the text is already gone.
    2. Truncation at `max_tokens` with nothing in `content` means the model spent its entire
       budget thinking. That's a configuration problem (raise LOCAL_LLM_MAX_TOKENS), not an
       answer, so it's raised instead of silently returning "".
    """

    @staticmethod
    def _reasoning_text(payload: dict) -> str:
        """The answer text a reasoning model filed under the wrong key, if any."""
        for key in ("reasoning_content", "reasoning"):
            value = payload.get(key)
            # Some servers send a structured object here rather than text; only a plain
            # string is safe to treat as the answer.
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def _create_chat_result(self, response: Any, generation_info: dict | None = None) -> ChatResult:
        raw = response if isinstance(response, dict) else response.model_dump()
        result = super()._create_chat_result(response, generation_info)
        for choice, generation in zip(raw.get("choices") or [], result.generations):
            if generation.message.content:
                continue
            reasoning = self._reasoning_text(choice.get("message") or {})
            if reasoning:
                generation.message.content = reasoning
            elif choice.get("finish_reason") == "length":
                raise LocalLLMError(
                    f"Local model {self.model_name!r} hit max_tokens with empty content -- "
                    "it spent the whole budget on reasoning. Raise LOCAL_LLM_MAX_TOKENS."
                )
        return result

    def _convert_chunk_to_generation_chunk(
        self, chunk: dict, default_chunk_class: type, base_generation_info: dict | None
    ) -> ChatGenerationChunk | None:
        """Same recovery for the streaming path, which is the one the synthesis node uses.

        Without this, a reasoning model streams its answer entirely as `delta.reasoning` and
        the SSE endpoint relays a stream of empty tokens. No truncation check here: a stream
        is assembled chunk by chunk, so emptiness can only be judged once it's complete.
        """
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None or generation_chunk.message.content:
            return generation_chunk
        for choice in chunk.get("choices") or []:
            reasoning = self._reasoning_text(choice.get("delta") or {})
            if reasoning:
                generation_chunk.message.content = reasoning
                break
        return generation_chunk


def _local_chat_model(temperature: float, streaming: bool = False) -> _LocalChatOpenAI:
    settings = get_settings()
    return _LocalChatOpenAI(
        model=settings.local_llm_chat_model,
        base_url=settings.local_llm_base_url,
        # ChatOpenAI requires *some* key; Ollama/vLLM ignore it unless a proxy enforces one.
        api_key=settings.local_llm_api_key or "not-needed",
        temperature=temperature,
        max_tokens=settings.local_llm_max_tokens,
        # Long read timeout for slow local generation, short connect timeout so an
        # unreachable box fails over fast instead of hanging -- see config.py.
        timeout=httpx.Timeout(
            settings.local_llm_timeout_seconds,
            connect=settings.local_llm_connect_timeout_seconds,
        ),
        max_retries=settings.local_llm_max_retries,
        streaming=streaming,
        # Per-provider handler: metrics are labelled by whoever actually served the call, so a
        # fallback is visible as an error here plus a success on the provider behind it.
        callbacks=[MetricsCallbackHandler("local", settings.local_llm_chat_model)],
    )


def _gemini_chat_model(temperature: float) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_chat_model,
        temperature=temperature,
        google_api_key=settings.google_api_key,
        timeout=settings.llm_request_timeout_seconds,
        max_retries=settings.llm_max_retries,
        callbacks=[MetricsCallbackHandler("gemini", settings.gemini_chat_model)],
    )


def _anthropic_chat_model(streaming: bool = False) -> ChatAnthropic:
    """`temperature` is rejected (HTTP 400) by current Claude models, so it's never passed here.

    `streaming=True` makes `.invoke()` consume the streaming API and fire token callbacks --
    which LangGraph's `stream_mode="messages"` relays to the SSE endpoint as live answer
    tokens. Only the plain chat model (synthesis path) enables it; structured-output calls
    keep the default, where token streaming buys nothing."""
    settings = get_settings()
    return ChatAnthropic(
        model=settings.anthropic_chat_model,
        api_key=settings.anthropic_api_key,
        default_request_timeout=settings.llm_request_timeout_seconds,
        max_retries=settings.llm_max_retries,
        streaming=streaming,
        callbacks=[MetricsCallbackHandler("anthropic", settings.anthropic_chat_model)],
    )


def _provider_chain(temperature: float, streaming: bool) -> list[Callable[[], BaseChatModel]]:
    """Configured providers in priority order, as *builders* rather than instances.

    Nothing is constructed for an unconfigured provider: the Gemini client validates its API
    key eagerly in `__init__`, so merely building it as an unused fallback crashes a
    deployment that runs Anthropic-only. Returning thunks keeps that decision in one place
    instead of repeating the key guards in every caller.
    """
    settings = get_settings()
    builders: list[Callable[[], BaseChatModel]] = []
    if settings.local_llm_base_url:
        builders.append(lambda: _local_chat_model(temperature, streaming=streaming))
    if settings.anthropic_api_key:
        builders.append(lambda: _anthropic_chat_model(streaming=streaming))
    if settings.google_api_key:
        builders.append(lambda: _gemini_chat_model(temperature))
    if not builders:
        raise RuntimeError(
            "No chat provider configured. Set at least one of LOCAL_LLM_BASE_URL, "
            "ANTHROPIC_API_KEY or GOOGLE_API_KEY (see .env.example)."
        )
    return builders


def _with_fallbacks(runnables: list[Any]) -> Any:
    """`first.with_fallbacks(rest)`, or `first` alone when it's the only provider -- a
    single-element `with_fallbacks` would wrap a plain chat model in a RunnableWithFallbacks
    for no benefit, and callers like `get_raw_chat_model` care about the difference."""
    first, *rest = runnables
    return first.with_fallbacks(rest) if rest else first


def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    """Primary chat model for plain `.invoke()` calls, with every lower-priority configured
    provider attached as an automatic fallback on error (rate limit, outage, unreachable
    local endpoint)."""
    return _with_fallbacks([build() for build in _provider_chain(temperature, streaming=True)])


def _local_structured_runnable(model: _LocalChatOpenAI, schema: type) -> Runnable:
    """Structured output against a self-hosted server.

    `with_structured_output(..., method="json_schema")` can't be used here: langchain_openai
    routes it through OpenAI's Structured Outputs parser, which reads a `parsed` field that
    only hosted OpenAI populates. Against Ollama/vLLM it raises "response does not have a
    'parsed' field" on every call -- even when the server returned perfectly valid JSON --
    so every structured node would fail straight through to the paid fallback.

    Binding `response_format` by hand keeps the half that matters (the server does guided
    decoding against the schema) and parses the JSON out of `content` like the other
    providers do. `PydanticOutputParser` also tolerates the code fences and leading prose
    local models wrap JSON in. If the server ignores `response_format` entirely, parsing
    fails, which is a provider error -- and the fallback chain does its job.
    """
    settings = get_settings()
    method = settings.local_llm_structured_output_method
    if method != "json_schema" or not hasattr(schema, "model_json_schema"):
        # json_mode / function_calling parse from `content` already, so LangChain's own
        # implementations are correct for a self-hosted server.
        return model.with_structured_output(schema, method=method)
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
    }
    return model.bind(response_format=response_format) | PydanticOutputParser(
        pydantic_object=schema
    )


def get_structured_llm(schema: type, temperature: float = 0.0) -> Runnable:
    """Structured-output runnable under the same provider priority.

    `with_structured_output()` must be bound per-provider before fallbacks are attached —
    `RunnableWithFallbacks` doesn't expose `with_structured_output`, so the fallback has to
    wrap the already-structured runnables rather than the raw chat models.
    """
    structured = []
    for build in _provider_chain(temperature, streaming=False):
        model = build()
        if isinstance(model, _LocalChatOpenAI):
            structured.append(_local_structured_runnable(model, schema))
        else:
            structured.append(model.with_structured_output(schema))
    return _with_fallbacks(structured)


def get_raw_chat_model(temperature: float = 0.0) -> BaseChatModel:
    """The highest-priority chat model with no fallback wrapping.

    `RunnableWithFallbacks` (what `get_chat_model()` returns once a second provider is
    configured) isn't a `BaseChatModel` -- it has no `.generate_prompt()`/`.temperature`
    attribute, which breaks callers like RAGAS's `LangchainLLMWrapper` that require a real
    chat model instance.
    """
    return _provider_chain(temperature, streaming=False)[0]()


def primary_chat_provider_name() -> str:
    """Which provider `get_chat_model()`/`get_structured_llm()` calls first."""
    settings = get_settings()
    if settings.local_llm_base_url:
        return f"Local ({settings.local_llm_chat_model})"
    return "Anthropic" if settings.anthropic_api_key else "Gemini"


def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """Always Gemini, even when a local chat provider is configured.

    Embeddings are not interchangeable the way chat models are: the Chroma collection is
    built at one provider's vector dimension, and pointing queries at a different embedding
    model doesn't error -- it silently returns nonsense neighbours. Switching would mean a
    full re-index (`rag-assistant ingest --full`), so it stays a deliberate one-way decision
    rather than a fallback the graph can take at runtime.
    """
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.google_api_key,
    )
