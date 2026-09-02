import httpx
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import RunnableWithFallbacks
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from rag_assistant import llm
from rag_assistant.config import get_settings


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


def test_get_chat_model_is_anthropic_only_when_google_key_blank(monkeypatch):
    """A valid Anthropic key with a blank Google key must yield a working chat model, not
    crash constructing the Gemini fallback (its client validates the key at __init__)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    model = llm.get_chat_model()

    assert isinstance(model, ChatAnthropic)


def test_get_structured_llm_is_anthropic_only_when_google_key_blank(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    from rag_assistant.schemas.models import RouteDecision

    runnable = llm.get_structured_llm(RouteDecision)

    assert not isinstance(runnable, RunnableWithFallbacks)


# --- self-hosted local provider -------------------------------------------------------


LOCAL_URL = "http://gpu-box.example.ts.net:11434/v1"


def test_local_endpoint_becomes_primary_ahead_of_anthropic(monkeypatch):
    """LOCAL_LLM_BASE_URL demotes Anthropic to a fallback rather than sitting behind it --
    the whole point is that configured local hardware serves the calls at $0."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", LOCAL_URL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    model = llm.get_chat_model()

    assert isinstance(model, RunnableWithFallbacks)
    assert isinstance(model.runnable, llm._LocalChatOpenAI)
    assert [type(f) for f in model.fallbacks] == [ChatAnthropic, ChatGoogleGenerativeAI]


def test_local_model_is_configured_from_settings(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", LOCAL_URL)
    monkeypatch.setenv("LOCAL_LLM_CHAT_MODEL", "Qwen/Qwen3.6-35B-A3B")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    model = llm.get_raw_chat_model()

    assert isinstance(model, llm._LocalChatOpenAI)
    assert model.model_name == "Qwen/Qwen3.6-35B-A3B"
    assert str(model.openai_api_base).rstrip("/") == LOCAL_URL


def test_local_connect_timeout_is_short_so_an_unreachable_box_fails_over_fast(monkeypatch):
    """A long read timeout is needed for slow local generation, but the connect timeout must
    stay well under GRAPH_TIMEOUT_SECONDS: off the tailnet there's no route at all, and a
    hanging connect would burn the graph's whole budget before Anthropic ever gets the call."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", LOCAL_URL)

    timeout = llm.get_raw_chat_model().request_timeout

    assert timeout.connect < get_settings().graph_timeout_seconds
    assert timeout.read == get_settings().local_llm_timeout_seconds


def test_local_max_tokens_floor_is_applied(monkeypatch):
    """Reasoning models spend the budget thinking before answering; without the floor the
    response comes back truncated with empty content."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", LOCAL_URL)

    assert llm.get_raw_chat_model().max_tokens == get_settings().local_llm_max_tokens


def test_local_structured_output_parses_content_not_openais_parsed_field(monkeypatch):
    """Regression: `with_structured_output(method="json_schema")` routes through OpenAI's
    Structured Outputs parser, which reads a `parsed` field only hosted OpenAI populates.
    Against Ollama/vLLM that raises on every call even when the JSON is valid, sending every
    structured node to the paid fallback. The schema must still be bound as `response_format`
    (that's the server-side guided decoding), but parsing has to come off `content`."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", LOCAL_URL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    from rag_assistant.schemas.models import RouteDecision

    runnable = llm.get_structured_llm(RouteDecision)

    # A single configured provider must not be wrapped in a pointless fallback chain.
    assert not isinstance(runnable, RunnableWithFallbacks)
    bound, parser = runnable.first, runnable.last
    assert isinstance(parser, PydanticOutputParser)
    response_format = bound.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == RouteDecision.model_json_schema()


def test_local_structured_output_honours_a_non_default_method(monkeypatch):
    """A server with tool support but no guided decoding can be pointed at function calling
    rather than failing every structured node through to Anthropic."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", LOCAL_URL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("LOCAL_LLM_STRUCTURED_OUTPUT_METHOD", "function_calling")

    from rag_assistant.schemas.models import RouteDecision

    runnable = llm.get_structured_llm(RouteDecision)

    assert "response_format" not in getattr(runnable.first, "kwargs", {})


def test_blank_local_url_leaves_anthropic_gemini_behavior_unchanged(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    model = llm.get_chat_model()

    assert isinstance(model.runnable, ChatAnthropic)


def test_primary_provider_name_reports_the_local_model(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", LOCAL_URL)
    monkeypatch.setenv("LOCAL_LLM_CHAT_MODEL", "gemma-4-26b")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    assert llm.primary_chat_provider_name() == "Local (gemma-4-26b)"


def test_no_configured_provider_raises_a_clear_error(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    with pytest.raises(RuntimeError, match="No chat provider configured"):
        llm.get_chat_model()


def test_embeddings_stay_on_gemini_even_with_a_local_chat_provider(monkeypatch):
    """Chat providers are interchangeable; embeddings are not. Pointing queries at a
    different embedding model than the index was built with returns nonsense neighbours
    silently rather than erroring."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", LOCAL_URL)

    assert isinstance(llm.get_embeddings_model(), GoogleGenerativeAIEmbeddings)


# --- local-server response quirks -----------------------------------------------------


def _local_response(content: str, *, reasoning: str | None = None, finish: str = "stop") -> dict:
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "id": "chatcmpl-local",
        "created": 0,
        "model": "Qwen/Qwen3.6-35B-A3B",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_reasoning_field_is_promoted_when_content_is_empty(monkeypatch):
    """Qwen3-class models sometimes return an empty `content` with the whole answer in a
    separate `reasoning` field; stock ChatOpenAI surfaces that as the model saying nothing."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", LOCAL_URL)
    model = llm.get_raw_chat_model()

    result = model._create_chat_result(_local_response("", reasoning="the actual answer"))

    assert result.generations[0].message.content == "the actual answer"


def test_real_content_is_never_replaced_by_reasoning(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", LOCAL_URL)
    model = llm.get_raw_chat_model()

    result = model._create_chat_result(_local_response("answer", reasoning="scratch work"))

    assert result.generations[0].message.content == "answer"


def test_empty_truncated_output_raises_instead_of_returning_nothing(monkeypatch):
    """Burning the entire token budget on reasoning is a config problem, not an answer. It
    raises so the fallback chain hands the call to Anthropic instead of synthesizing ""."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", LOCAL_URL)
    model = llm.get_raw_chat_model()

    with pytest.raises(llm.LocalLLMError, match="LOCAL_LLM_MAX_TOKENS"):
        model._create_chat_result(_local_response("", finish="length"))


def test_local_failure_falls_through_to_anthropic(monkeypatch):
    """End-to-end proof of the degradation path: a host with no route to the tailnet box
    still answers, it just stops being free."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", LOCAL_URL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    def _unreachable(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    def _claude_stream(self, *args, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content="from claude"))

    # Both providers are built with streaming=True on this path (it's what feeds the SSE
    # token stream), so `_stream` -- not `_generate` -- is what invoke() actually calls.
    monkeypatch.setattr(llm._LocalChatOpenAI, "_stream", _unreachable)
    monkeypatch.setattr(ChatAnthropic, "_stream", _claude_stream)

    assert llm.get_chat_model().invoke("hi").content == "from claude"


def test_streamed_reasoning_deltas_are_promoted_to_content(monkeypatch):
    """The synthesis node streams, so the empty-content quirk has to be handled on the
    chunk path too -- otherwise a reasoning model relays a stream of empty SSE tokens."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", LOCAL_URL)
    model = llm.get_raw_chat_model()

    chunk = {
        "id": "chatcmpl-local",
        "created": 0,
        "model": "Qwen/Qwen3.6-35B-A3B",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"reasoning": "tok"}, "finish_reason": None}],
    }
    generation_chunk = model._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)

    assert generation_chunk.message.content == "tok"


def test_non_string_reasoning_payload_is_ignored(monkeypatch):
    """Some servers put a structured object under `reasoning`; only plain text is safe to
    treat as the answer."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", LOCAL_URL)
    model = llm.get_raw_chat_model()

    response = _local_response("")
    response["choices"][0]["message"]["reasoning"] = {"steps": ["a", "b"]}

    result = model._create_chat_result(response)

    assert result.generations[0].message.content == ""


# ---- which provider actually answered ----


class _Response:
    def __init__(self, metadata):
        self.response_metadata = metadata


def test_responding_provider_reads_the_provider_off_the_response():
    """`primary_chat_provider_name()` reports what is configured to be tried first, which is a
    different question -- and a misleading answer whenever the fallback chain fired."""
    from rag_assistant.llm import responding_provider_name

    name = responding_provider_name(
        _Response({"model_provider": "anthropic", "model_name": "claude-sonnet-5"})
    )

    assert name == "anthropic (claude-sonnet-5)"


def test_responding_provider_falls_back_to_whichever_field_is_present():
    from rag_assistant.llm import responding_provider_name

    assert responding_provider_name(_Response({"model_name": "gemini-2.5-flash"})) == (
        "gemini-2.5-flash"
    )
    assert responding_provider_name(_Response({"model_provider": "google"})) == "google"


def test_responding_provider_is_none_when_nothing_is_reported():
    """None means "couldn't tell", not "nobody answered" -- providers aren't obliged to
    populate this."""
    from rag_assistant.llm import responding_provider_name

    assert responding_provider_name(_Response({})) is None
    assert responding_provider_name(object()) is None


def test_hello_names_the_provider_that_answered_not_the_configured_one(monkeypatch, capsys):
    """The bug this replaces: an invalid Anthropic key produced correct answers served by
    Gemini, while the CLI reported "Anthropic says" -- so a dead credential stayed hidden
    behind three successful-looking commands."""
    from unittest.mock import MagicMock

    from rag_assistant import cli

    monkeypatch.setenv("ANTHROPIC_API_KEY", "configured-but-broken")
    monkeypatch.setenv("GOOGLE_API_KEY", "working")
    from rag_assistant.config import get_settings

    get_settings.cache_clear()

    answered_by_gemini = MagicMock()
    answered_by_gemini.text = "Hello!"
    answered_by_gemini.response_metadata = {
        "model_provider": "google_genai",
        "model_name": "gemini-2.5-flash",
    }
    fake = MagicMock()
    fake.invoke.return_value = answered_by_gemini
    monkeypatch.setattr(cli, "get_chat_model", lambda: fake)

    cli.hello()

    output = capsys.readouterr().out
    assert "gemini" in output.lower()
    # And it says so explicitly, rather than leaving the reader to notice.
    assert "did not answer" in output
