"""Central configuration. Every other module reads settings through get_settings() —
never os.environ directly — so there's one seam to mock in tests and one place secrets live."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # One switch that sets the five backend switches coherently.
    #
    # "single-node" is the default and changes nothing: embedded Chroma, SQLite
    # conversations, in-process ingest tasks. No infrastructure, one container.
    #
    # "multi-replica" turns on the shared backends together -- the vector index, the
    # conversation store and the ingest task registry -- because they are not independent
    # choices. A deployment with two replicas and a shared index but per-process ingest tasks
    # serves "unknown ingest task" 404s from whichever replica did not accept the upload; one
    # with shared tasks but a local index has two divergent corpora. Every combination that
    # is half-shared is broken in a way that looks like flakiness rather than
    # misconfiguration, so the individual switches remain available for deliberate overrides
    # and the profile makes the coherent set the easy thing to ask for.
    deployment_profile: Literal["single-node", "multi-replica"] = "single-node"

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
    local_llm_structured_output_method: Literal["json_schema", "function_calling", "json_mode"] = (
        "json_schema"
    )

    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    anthropic_chat_model: str = "claude-sonnet-5"

    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    # Where the vector index lives. "chroma" is the default and is correct for one container:
    # embedded, zero-infrastructure, durable on a mounted volume. "pgvector" (with
    # DATABASE_URL) puts the index in Postgres instead -- worth it when Postgres is already
    # in the deployment for conversations, since it removes embedded Chroma's one-process
    # file lock without operating a second service. See retrieval/pgvector_store.py.
    vector_backend: Literal["chroma", "pgvector"] = "chroma"
    # How often a replica re-checks the shared index version to discover that *another*
    # replica ingested and its in-memory BM25 index is stale (see retrieval/bm25_store.py).
    # Only consulted when vector_backend == "pgvector"; on the single-process default the
    # ingesting process invalidates its own index directly and nothing can race it. The
    # window this opens is stale *keyword ranking*, not stale answers -- vector retrieval and
    # grading read through to the shared store on every query.
    bm25_version_poll_seconds: float = 5.0
    chroma_persist_dir: Path = PROJECT_ROOT / "chroma_db"
    # Point at a Chroma server to share the index across replicas. Embedded Chroma is
    # SQLite-backed and locks its file to one process, which is the single hardest constraint
    # on running more than one worker; server mode removes it. Blank keeps the embedded,
    # zero-infrastructure default.
    chroma_server_host: str = ""
    chroma_server_port: int = 8000
    chroma_server_ssl: bool = False
    # Server-side conversation history (see conversations/store.py). Lives inside
    # chroma_persist_dir's sibling default so one mounted volume covers both stores.
    conversations_db_path: Path = PROJECT_ROOT / "chroma_db" / "conversations.db"
    # Where conversations and feedback live. SQLite is the default and is correct for one
    # container; "postgres" (with DATABASE_URL) is what lets several replicas write, since
    # SQLite's single-writer constraint is the main thing preventing a second one.
    conversations_backend: Literal["sqlite", "postgres"] = "sqlite"
    database_url: str = ""

    # When set (STATIC_DIR env var) and the directory exists, the API serves the built
    # frontend from it at "/" -- used by the Docker image / Render deployment so one
    # container is the whole demo. Unset in development, where Vite serves the frontend.
    static_dir: Path | None = None

    # Per-tenant LLM token allowance per UTC day; 0 disables it entirely (the default).
    # Rate limiting bounds request *count*, which says nothing about cost -- a decomposed,
    # multi-path, corrective run can cost orders of magnitude more than a routed-to-`none`
    # one. Checked before a run and charged after, so a tenant can overshoot by at most one
    # request; see budget.py for why that is the cheaper error than a reservation protocol.
    # Shared across replicas only when Redis is reachable, and logs loudly when it is not.
    tenant_daily_token_budget: int = 0
    # What one vision call counts as against the budget. A real image's cost scales with
    # resolution and is not reported back, so this is a deliberate over-estimate: a cap that
    # under-counts lets a tenant exceed the budget it exists to enforce, while over-counting
    # only makes it conservative.
    vision_call_token_estimate: int = 1500

    # Worker threads available to synchronous request handlers. `POST /api/v1/research` is a
    # sync `def`, so FastAPI runs it on this pool and each in-flight research call occupies
    # one thread for the entire graph -- seconds, not milliseconds. That makes this the real
    # concurrency ceiling of a worker: at the default 40, the 41st concurrent research
    # request waits for a thread rather than starting, no matter how idle the CPU is.
    #
    # Stated as a setting rather than inherited from AnyIO's default so the number is visible
    # and deliberate. Raising it trades memory and context-switching for queueing, and stops
    # helping once the real bottleneck is the provider's own rate limit.
    api_threadpool_size: int = 40

    # OTLP endpoint for distributed traces, e.g. "http://localhost:4318/v1/traces". Blank
    # (the default) leaves tracing off entirely and imports nothing. Needs
    # `uv sync --extra otel`. The per-request trace_id in the logs is unaffected either way --
    # this adds the parent/child span structure that shows where the time actually went.
    otel_exporter_otlp_endpoint: str = ""

    # Shingle containment (|A n B| / min(|A|,|B|)) at or above which fusion treats two
    # retrieved passages as the same one (see fusion/rrf.py, which shows the measurements
    # behind this number). Exact hashing collapsed only byte-identical text, so a local copy
    # and a web copy of one page both reached synthesis and earned separate citation markers
    # pointing at the same words. Measured, every true near-duplicate scores 1.000 and the
    # nearest false positive -- a different chunk of the same document -- scores 0.333, so
    # 0.9 sits in a gap rather than on a slope. 0 disables similarity matching entirely and
    # leaves only exact and normalized hashing.
    fusion_near_duplicate_threshold: float = 0.9

    confidence_threshold: float = 0.6

    # Chunking strategy within a section (see ingestion/semantic_splitter.py). "structural"
    # is fixed-size inside each heading section -- free and deterministic. "semantic" embeds
    # sentences and breaks where similarity drops, which cuts at topic shifts instead of at
    # an arbitrary character offset, at the cost of one embedding call per section at ingest.
    chunking_strategy: Literal["structural", "semantic"] = "structural"
    # Percentile of within-section sentence distances above which a break is placed. Higher
    # means fewer, larger chunks. A percentile rather than an absolute distance because
    # cosine distances aren't comparable across embedding models or prose styles.
    semantic_chunk_percentile: float = 85.0

    # Cross-encoder reranking of fused documents (see retrieval/reranker.py). RRF ranks by
    # retriever consensus and never compares a document against the question; a cross-encoder
    # does. "cohere" needs COHERE_API_KEY and a network call, "cross_encoder" needs the
    # `rerank-local` extra (sentence-transformers + torch). Both are optional extras, so a
    # default install carries neither and a misconfiguration degrades to no reranking.
    reranker: Literal["none", "cohere", "cross_encoder"] = "none"
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-v3.5"
    cross_encoder_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Reranking is per-pair work, so only a shortlist is scored; the tail keeps RRF order.
    rerank_top_n: int = 12

    # Small-to-big retrieval (see retrieval/parent_store.py): retrieve on precise chunks,
    # then hand synthesis the whole section each winner came from. Off by default because it
    # spends noticeably more of the context budget per document.
    parent_context: bool = False

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
    # Where background ingest task state lives (see ingestion/tasks.py). "memory" is correct
    # and free for a single worker; "redis" makes task state visible across replicas, without
    # which a client polling a load-balanced deployment gets 404s from every worker that
    # didn't happen to accept the upload.
    task_backend: Literal["memory", "redis"] = "memory"
    # How long an ingest task may sit in a non-terminal stage before a restart concludes it
    # was orphaned and fails it (see ingestion/tasks.reconcile_stale_tasks). Must exceed the
    # longest legitimate gap between stage updates on a large corpus, or an ingest that is
    # simply slow gets failed out from under itself.
    ingest_stale_after_seconds: float = 900.0
    # How many times an ingest may be attempted before it is failed for good. A file that
    # crashes the parser crashes it again, so without a ceiling a poison upload becomes an
    # infinite restart loop that presents as an unstable deployment rather than a bad file.
    ingest_max_attempts: int = 3
    # Delay between in-process retries of a failed ingest. Fixed rather than exponential: the
    # ceiling is 3 attempts, so the difference between schedules is a few seconds, and a
    # constant is one less thing to reason about when reading a task's timeline.
    ingest_retry_delay_seconds: float = 2.0
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_router: int = 300
    cache_ttl_web_search: int = 600
    cache_ttl_synthesis: int = 1800

    # authentication -- see auth.py. Comma-separated `label:key` (or bare `key`) entries;
    # blank disables auth entirely (open demo mode). Each label is a tenant: conversations
    # are scoped to it and rate limits are keyed by it.
    api_keys: str = ""
    # Richer key definitions -- scopes, expiry, per-key rate limits (see auth.py). Keeps
    # secrets out of the process listing and lets a key be revoked by editing one file.
    api_keys_file: Path | None = None

    # Browser origins allowed to call this API cross-origin. The defaults cover the Vite dev
    # server; the single-container deploy serves the frontend from this same origin, so it
    # needs none of these. Set CORS_ALLOW_ORIGINS (comma-separated) when the frontend is
    # hosted separately -- e.g. "https://myapp.vercel.app". "*" is accepted but disables
    # credentialed requests per the CORS spec, so prefer explicit origins.
    cors_allow_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5175,http://127.0.0.1:5175"
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

    @model_validator(mode="after")
    def _apply_deployment_profile(self) -> "Settings":
        """Fills in the shared backends for `multi-replica`, without overriding anything set
        explicitly.

        `model_fields_set` is what makes that distinction: pydantic-settings records a field
        there when an environment variable supplied it, so an operator who sets
        VECTOR_BACKEND themselves keeps it, and one who sets only DEPLOYMENT_PROFILE gets the
        coherent set. A profile that silently overrode explicit configuration would be the
        opposite of useful -- it would make the individual switches lie.
        """
        if self.deployment_profile != "multi-replica":
            return self
        explicit = self.model_fields_set
        if "vector_backend" not in explicit:
            self.vector_backend = "pgvector"
        if "conversations_backend" not in explicit:
            self.conversations_backend = "postgres"
        if "task_backend" not in explicit:
            self.task_backend = "redis"
        # Checked rather than defaulted: there is no sensible guess for where the database
        # is, and starting without one would fail later, per request, inside a background
        # task -- far from the configuration that caused it.
        if (
            self.vector_backend == "pgvector" or self.conversations_backend == "postgres"
        ) and not self.database_url:
            raise ValueError(
                "DEPLOYMENT_PROFILE=multi-replica needs DATABASE_URL: it puts the vector "
                "index and the conversation store in Postgres. Set DATABASE_URL, or pin the "
                "individual backends explicitly if you meant something narrower."
            )
        return self

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
