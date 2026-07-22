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

    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    anthropic_chat_model: str = "claude-sonnet-5"

    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    chroma_persist_dir: Path = PROJECT_ROOT / "chroma_db"

    confidence_threshold: float = 0.6

    # caching (Redis) -- see cache.py. `use_cache` lets tests/offline runs disable it outright.
    use_cache: bool = True
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_router: int = 300
    cache_ttl_web_search: int = 600
    cache_ttl_synthesis: int = 1800

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
