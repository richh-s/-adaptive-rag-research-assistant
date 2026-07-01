"""Central configuration. Every other module reads settings through get_settings() —
never os.environ directly — so there's one seam to mock in tests and one place secrets live."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_api_key: str = Field(..., description="Google AI Studio API key (free tier)")
    tavily_api_key: str = Field(..., description="Tavily API key (free tier)")

    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"

    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    chroma_persist_dir: Path = PROJECT_ROOT / "chroma_db"

    confidence_threshold: float = 0.6


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except Exception as exc:
        raise RuntimeError(
            "Missing or invalid configuration. Copy .env.example to .env and fill in "
            "GOOGLE_API_KEY and TAVILY_API_KEY.\n"
            f"Original error: {exc}"
        ) from exc
