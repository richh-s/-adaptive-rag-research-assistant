"""Central configuration. Every other module reads settings through get_settings() —
never os.environ directly — so there's one seam to mock in tests and one place secrets live."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    google_api_key: str = Field(..., description="Google AI Studio API key (free tier)")
    # Required (must be present, even set to "") rather than defaulted to None, so a fresh
    # clone that never touches .env.example fails fast with a clear message instead of
    # silently starting half-configured. Left blank, behavior is unchanged: get_chat_model()
    # etc. still fall back to Gemini-only -- see llm.py's `if not settings.anthropic_api_key`.
    anthropic_api_key: str = Field(
        ..., description="Anthropic API key; when set, becomes the primary chat model"
    )

    # Self-hosted models on an OpenAI-compatible /v1 endpoint (Ollama, vLLM, LM Studio,
    # llama.cpp). Setting LOCAL_LLM_BASE_URL promotes the local box to *primary* chat and
    # reasoning provider, with Anthropic and Gemini demoted to fallbacks behind it -- so every
    # graph node runs at $0 while the endpoint is reachable, and a host that can't reach it
    # (a Render/Vercel deploy that isn't on the tailnet) degrades to Claude rather than failing.
    # Leave blank to disable entirely; nothing about the Anthropic/Gemini path changes.
    local_llm_base_url: str = ""
    local_llm_api_key: str = ""
    local_llm_chat_model: str = "gemma-4-26b"
    # Deliberately NOT a local embeddings switch: the Chroma collection is built at one
    # embedding dimension, and swapping providers under an existing index yields silent
    # garbage retrieval rather than an error. Embeddings stay on Gemini -- see llm.py.

    # Local generation is slow (a 26B on one GPU scores a draft in ~20-25s), so the read
    # timeout is long -- but the connect timeout is short on purpose. Off the tailnet there is
    # no route to the box at all, and a fast connect failure is what makes the Anthropic
    # fallback fire in ~2s instead of burning the whole GRAPH_TIMEOUT_SECONDS budget hanging.
    local_llm_timeout_seconds: float = 180.0
    local_llm_connect_timeout_seconds: float = 2.0
    # Reasoning models spend their budget thinking before answering; without a floor the
    # response comes back truncated with an empty `content`.
    local_llm_max_tokens: int = 4096
    # One retry, not zero: single-model Ollama returns a transient 500 while swapping models.
    local_llm_max_retries: int = 1
    # How structured output is requested. "json_schema" binds the schema as `response_format`
    # so the server constrains decoding, and parses the JSON back off `content` -- see
    # llm.py's _local_structured_runnable for why LangChain's own json_schema path can't be
    # used against a self-hosted server. "function_calling" suits servers with tool support
    # but no guided decoding; "json_mode" suits servers with neither, at the cost of the
    # model never seeing the schema.
    local_llm_structured_output_method: Literal[
        "json_schema", "function_calling", "json_mode"
    ] = "json_schema"

    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    anthropic_chat_model: str = "claude-sonnet-5"

    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    chroma_persist_dir: Path = PROJECT_ROOT / "chroma_db"
    # Server-side conversation history (see conversations/store.py). Lives inside
    # chroma_persist_dir's sibling default so one mounted volume covers both stores.
    conversations_db_path: Path = PROJECT_ROOT / "chroma_db" / "conversations.db"

    # When set (STATIC_DIR env var) and the directory exists, the API serves the built
    # frontend from it at "/" -- used by the Docker image / Render deployment so one
    # container is the whole demo. Unset in development, where Vite serves the frontend.
    static_dir: Path | None = None

    confidence_threshold: float = 0.6

    # Ceiling on how much retrieved context reaches the synthesis prompt (see
    # graph/context_budget.py). Fusion's output size scales with sub-queries x retrieval
    # paths, not with the question, so without a cap prompt cost and latency grow with
    # retrieval breadth and eventually overflow the model's context window. Documents arrive
    # ranked, so the cap drops the ones already judged least useful. 0 disables it.
    synthesis_context_budget_tokens: int = 6000
    # Characters per token, for estimating the above. Deliberately an estimate: Anthropic,
    # Gemini and self-hosted servers don't share a tokenizer, so an exact count for one is
    # wrong for the others. ~4 is a reasonable English average; lower it to be more
    # conservative on token-dense content like code or CJK text.
    synthesis_chars_per_token: float = 4.0

    # PDF vision ingestion (see ingestion/vision.py): describe embedded figures and
    # transcribe scanned pages with the chat provider's vision capability. Costs one
    # vision call per figure/scanned page at ingest time; PDF_VISION=false disables.
    pdf_vision: bool = True

    # caching (Redis) -- see cache.py. `use_cache` lets tests/offline runs disable it outright.
    use_cache: bool = True
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_router: int = 300
    cache_ttl_web_search: int = 600
    cache_ttl_synthesis: int = 1800

    # authentication -- see auth.py. Comma-separated `label:key` (or bare `key`) entries;
    # blank disables auth entirely (open demo mode). Each label is a tenant: conversations
    # are scoped to it and rate limits are keyed by it.
    api_keys: str = ""

    # Browser origins allowed to call this API cross-origin. The defaults cover the Vite dev
    # server; the single-container deploy serves the frontend from this same origin, so it
    # needs none of these. Set CORS_ALLOW_ORIGINS (comma-separated) when the frontend is
    # hosted separately -- e.g. "https://myapp.vercel.app". "*" is accepted but disables
    # credentialed requests per the CORS spec, so prefer explicit origins.
    cors_allow_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5175,http://127.0.0.1:5175"
    )

    # error tracking -- when set, Sentry captures unhandled exceptions (see api.py).
    sentry_dsn: str = ""
    # Fraction of requests sampled for Sentry performance tracing. 0.0 (errors only) is the
    # default because tracing every request on a low-traffic demo is pure quota burn; raise
    # it to ~0.1 once there is enough traffic for latency percentiles to mean anything.
    sentry_traces_sample_rate: float = 0.0

    # Prometheus metrics at GET /metrics (see metrics.py). On by default -- the endpoint is
    # process-local and cheap; set METRICS_ENABLED=false to remove it entirely if the
    # deployment exposes the port publicly and you would rather not publish route timings.
    metrics_enabled: bool = True

    # Conversation retention (see conversations/store.py). Without a ceiling the transcript
    # table grows forever on a mounted volume -- ingest tasks already cap at 500, this is the
    # equivalent bound for durable history. Pruning runs after each appended turn, scoped to
    # the tenant that just wrote, so it costs one indexed DELETE and never scans other owners.
    # Set either to 0 to disable that half of the policy.
    conversation_retention_days: int = 90
    conversation_max_per_owner: int = 500

    # rate limiting -- see api.py's limiter setup.
    rate_limit_rpm: int = 10
    rate_limit_rpm_global: int = 30

    # request timeouts
    web_search_timeout_seconds: float = 10.0
    graph_timeout_seconds: float = 45.0
    # Per-attempt LLM call timeout, with retries capped low. Neither langchain-anthropic nor
    # langchain-google-genai sets a request timeout by default, and Gemini defaults to 6
    # retries -- a slow/rate-limited provider can silently retry with backoff for tens of
    # seconds, which is most of graph_timeout_seconds for a single node. Bounding both keeps
    # a stuck provider from starving the rest of the graph's budget.
    llm_request_timeout_seconds: float = 12.0
    llm_max_retries: int = 1


    def cors_origins(self) -> list[str]:
        """CORS_ALLOW_ORIGINS split into the list CORSMiddleware wants. Blank means no
        cross-origin browser access at all -- correct for the single-container deploy,
        where the frontend is same-origin and CORS never enters the picture."""
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except Exception as exc:
        raise RuntimeError(
            "Missing or invalid configuration. Copy .env.example to .env and fill in "
            "GOOGLE_API_KEY and ANTHROPIC_API_KEY (may be left blank).\n"
            f"Original error: {exc}"
        ) from exc
